"""训练/评估入口：威德尔海海冰运动矢量场时空预测。

用法示例：
  # 冒烟测试（合成数据，20步）
  .venv/bin/python src/train.py --synth --max-steps 20 --ds 2 --chid 32
  # 全量训练（后台）
  .venv/bin/python src/train.py --tag simvp-ds2-c64 --epochs 40 --ds 2 --chid 64 \
      --batch 8 --stride 6 --div-reg 0.1
  # 恢复/微调（换分辨率继续训）
  .venv/bin/python src/train.py --tag simvp-ds2-c64 --resume --ds 1 --epochs 10
  # 测试集评估 + 轨迹误差
  .venv/bin/python src/train.py --tag simvp-ds2-c64 --eval-only --ds 1

指标：逐帧掩膜 RMSE (cm/s)、Pearson 相关、矢量 RMSE、对持续性基线的技能分；
轨迹误差：终点误差/路径平均误差（km）。
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import dsutils as du

ROOT = Path(__file__).resolve().parent.parent
sys_path_ok = ROOT / "src" / "dataset.py"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zarr", default="data/processed/weddell_icemotion_daily.zarr")
    p.add_argument("--model", default="simvp", choices=["simvp", "convlstm", "persistence"])
    p.add_argument("--tag", default="run1")
    p.add_argument("--tin", type=int, default=10)
    p.add_argument("--tout", type=int, default=5)
    p.add_argument("--chid", type=int, default=128)
    p.add_argument("--n-s", type=int, default=2)
    p.add_argument("--n-t", type=int, default=4)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-min", type=float, default=1e-5)
    p.add_argument("--warmup", type=float, default=0.05)
    p.add_argument("--mae-weight", type=float, default=0.5)
    p.add_argument("--grad-weight", type=float, default=0.1)
    p.add_argument("--div-reg", type=float, default=0.05,
                   help="TAU式差分散度正则权重（预测场时间差分的散度惩罚）")
    p.add_argument("--ds", type=int, default=1, help="空间降采样倍率(1=原生120x160)")
    p.add_argument("--add-doy", type=int, default=1)
    p.add_argument("--train-years", default="", help="覆盖训练年份段，如 1987-2017（应对1978-86传感器低估的regime shift）")
    p.add_argument("--target-residual", type=int, default=0,
                   help="1=预测相对最后一帧的增量（帧差目标，利于短时效）")
    p.add_argument("--synth", action="store_true")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--init-weights", default="", help="从既有 checkpoint 加载模型参数，重新初始化优化器")
    p.add_argument("--head-only", action="store_true", help="仅微调 SimVP 最后一层预测头")
    p.add_argument("--strict-daily", action="store_true", help="排除缺日或跨训练/验证/测试边界的窗口")
    p.add_argument("--day1-weight", type=float, default=0.0, help="附加第 1 天速度变化监督权重")
    p.add_argument("--selection-windows", type=int, default=0, help="均匀抽取验证窗口数（0=使用全部）")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--ckpt", default="", help="eval-only 用的 checkpoint 路径")
    p.add_argument("--val-windows", type=int, default=64)
    p.add_argument("--patience", type=int, default=0, help="早停耐心(0=关)")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    return p.parse_args()


def pick_device(name: str):
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def masked_mse(pred, target, mask):
    d = (pred - target) ** 2 * mask
    denom = mask.sum() + 1e-8
    return d.sum() / denom


def masked_mae(pred, target, mask):
    d = (pred - target).abs() * mask
    return d.sum() / (mask.sum() + 1e-8)


def grad_loss(pred, target, mask):
    """Sobel 梯度损失（对 u/v 两通道分别计算，仅掩膜内有效）。"""
    dux = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]
    tux = target[:, :, :, :, 1:] - target[:, :, :, :, :-1]
    duy = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
    tuy = target[:, :, :, 1:, :] - target[:, :, :, :-1, :]
    mx = mask[:, :, :, :, 1:] * mask[:, :, :, :, :-1]
    my = mask[:, :, :, 1:, :] * mask[:, :, :, :-1, :]
    l = ((dux - tux).abs() * mx).sum() / (mx.sum() + 1e-8)
    l = l + ((duy - tuy).abs() * my).sum() / (my.sum() + 1e-8)
    return l


def masked_rmse(pred, target, mask):
    return masked_mse(pred, target, mask).clamp_min(0).sqrt()


def pearson(pred, target, mask):
    """掩膜内的 Pearson 相关（按帧平均）。"""
    m = mask.bool()
    p = pred[m].float()
    t = target[m].float()
    if p.numel() < 2:
        return torch.tensor(float("nan"), device=pred.device)
    p = p - p.mean()
    t = t - t.mean()
    denom = (p.pow(2).sum() * t.pow(2).sum()).sqrt() + 1e-8
    return (p * t).sum() / denom


def differential_divergence(y):
    """预测场时间差分 (dy/dt) 的散度惩罚（TAU 的 differential divergence 思想）。"""
    dy = y[:, 1:] - y[:, :-1]                      # (B, T-1, 2, H, W)
    du, dv = dy[:, :, 0], dy[:, :, 1]
    dux = du[:, :, :, 1:] - du[:, :, :, :-1]
    dvy = dv[:, :, 1:, :] - dv[:, :, :-1, :]
    return (dux.abs().mean() + dvy.abs().mean())


def denorm_uv(dataset, u_norm, v_norm):
    """(B,T,H,W) 标准化场 -> cm/s。"""
    u = dataset.unnorm(u_norm, 0)
    v = dataset.unnorm(v_norm, 1)
    return u, v


@torch.no_grad()
def evaluate(model, loader, dataset, device, tout, model_name, persist=None,
             target_residual=False):
    """返回 {rmse_u/v(cm/s), vec_rmse, corr_u/v, acc_u/v, skill_pers,
    逐时效 vec_rmse_d1..dT / skill_pers_d1..dT}。"""
    model.eval()
    su = sv = svec = cu = cv = au = av = 0.0
    spers = 0.0
    n = 0
    lead_mse = [0.0] * tout
    lead_pers = [0.0] * tout
    mean_u, _ = dataset._aligned_stats(0)
    mean_v, _ = dataset._aligned_stats(1)
    mu = mean_u.to(device).unsqueeze(0)
    mv = mean_v.to(device).unsqueeze(0)
    for b in loader:
        x = b["x"].to(device)
        y = b["y"].to(device)
        m = b["mask"].to(device)
        if model_name == "persistence":
            pred = x[:, -1:, :2].expand(-1, tout, -1, -1, -1).contiguous()
        else:
            pred = model(x, tout) if model_name == "convlstm" else model(x)
            if target_residual:
                pred = pred + x[:, -1:, :2].expand_as(pred)
        # 还原为 cm/s
        pu, pv = denorm_uv(dataset, pred[:, :, 0], pred[:, :, 1])
        tu, tv = denorm_uv(dataset, y[:, :, 0], y[:, :, 1])
        mm = m.squeeze(2)   # (B, T, H, W)
        su += masked_rmse(pu, tu, mm).item() * len(x)
        sv += masked_rmse(pv, tv, mm).item() * len(x)
        dvec = (pu - tu) ** 2 + (pv - tv) ** 2
        svec += ((dvec * mm).sum() / (mm.sum() + 1e-8)).item() * len(x)
        # 持续性基线（同一批样本）
        pp = x[:, -1:, :2].expand(-1, tout, -1, -1, -1).contiguous()
        ppu, ppv = denorm_uv(dataset, pp[:, :, 0], pp[:, :, 1])
        dp = (ppu - tu) ** 2 + (ppv - tv) ** 2
        spers += ((dp * mm).sum() / (mm.sum() + 1e-8)).item() * len(x)
        for t in range(tout):
            cu += pearson(pred[:, t, 0], y[:, t, 0], mm[:, t]).item() / tout * len(x)
            cv += pearson(pred[:, t, 1], y[:, t, 1], mm[:, t]).item() / tout * len(x)
            au += pearson(pu[:, t] - mu, tu[:, t] - mu, mm[:, t]).item() / tout * len(x)
            av += pearson(pv[:, t] - mv, tv[:, t] - mv, mm[:, t]).item() / tout * len(x)
            if t < len(lead_mse):
                lead_mse[t] += ((dvec[:, t] * mm[:, t]).sum()
                                / (mm[:, t].sum() + 1e-8)).item() * len(x)
                lead_pers[t] += ((dp[:, t] * mm[:, t]).sum()
                                 / (mm[:, t].sum() + 1e-8)).item() * len(x)
        n += len(x)
    if n == 0:
        out = {"rmse_u": float("nan"), "rmse_v": float("nan"),
               "vec_rmse": float("nan"), "corr_u": float("nan"),
               "corr_v": float("nan"), "acc_u": float("nan"),
               "acc_v": float("nan"), "skill_pers": float("nan")}
        for t in range(tout):
            out[f"vec_rmse_d{t+1}"] = float("nan")
            out[f"skill_pers_d{t+1}"] = float("nan")
        return out
    vec_rmse = math.sqrt(max(svec / n, 0.0))
    vec_pers = math.sqrt(max(spers / n, 0.0))
    skill = 1.0 - vec_rmse / (vec_pers + 1e-8)
    out = {"rmse_u": su / n, "rmse_v": sv / n, "vec_rmse": vec_rmse,
           "corr_u": cu / n, "corr_v": cv / n, "acc_u": au / n,
           "acc_v": av / n, "skill_pers": skill}
    # 逐时效（lead day 1..T_out）矢量 RMSE 与技能分
    for t in range(tout):
        vm = math.sqrt(max(lead_mse[t] / n, 0.0))
        vp = math.sqrt(max(lead_pers[t] / n, 0.0))
        out[f"vec_rmse_d{t+1}"] = vm
        out[f"skill_pers_d{t+1}"] = 1.0 - vm / (vp + 1e-8)
    return out


def evaluate_trajectory(model, dataset, device, tout, n_pts=200, n_windows=8,
                        model_name="simvp"):
    """拉格朗日轨迹误差：用预测场 vs 真值场推演 n_pts 个海冰单元轨迹。"""
    try:
        import trajectory as traj  # datamaster 提供的 src/trajectory.py
    except ImportError:
        print("[traj] src/trajectory.py 不存在，跳过轨迹评估")
        return None
    model.eval()
    res = {}
    rng = np.random.default_rng(0)
    d = dataset
    grid = d.store if not d.synth else None
    # 以最后一个测试窗口作为演示（可扩展）
    t0 = int(d.idx[len(d.idx) // 2])
    b = d[int(len(d) // 2)]
    x = b["x"].unsqueeze(0).to(device)
    with torch.no_grad():
        # (1,T,2,H,W)
        pred = model(x, tout) if model_name == "convlstm" else model(x)
    pu, pv = denorm_uv(d, pred[0, :, 0], pred[0, :, 1])   # (T,H,W) cm/s
    tu, tv = denorm_uv(d, b["y"][:, 0], b["y"][:, 1])     # (T,H,W) cm/s
    m = b["mask"][:, 0].bool()
    valid_cells = torch.nonzero(m.any(0))     # 至少一天有效的像元
    picks = valid_cells[torch.randperm(len(valid_cells), generator=torch.Generator().manual_seed(0))[:n_pts]]
    pts = picks[:, [1, 0]].float()            # (N,2) = (col,row) 像素坐标
    path_pred = traj.advect_torch(pu.cpu(), pv.cpu(), pts.cpu(), tout)
    path_true = traj.advect_torch(tu.cpu(), tv.cpu(), pts.cpu(), tout)
    ep = traj.endpoint_error_km(path_pred, path_true, km_per_cell=25.0 * d.ds)
    pe = traj.path_error_km(path_pred, path_true, km_per_cell=25.0 * d.ds)
    res = {"traj_endpoint_km": float(ep.mean()), "traj_path_km": float(pe.mean()),
           "traj_endpoint_p50_km": float(np.median(ep))}
    # 对比持续性基线轨迹
    xp = b["x"][-1:, :2].expand(tout, -1, -1, -1)
    ppu, ppv = denorm_uv(d, xp[:, 0], xp[:, 1])
    path_pers = traj.advect_torch(ppu.cpu(), ppv.cpu(), pts.cpu(), tout)
    res["pers_traj_endpoint_km"] = float(traj.endpoint_error_km(
        path_pers, path_true, km_per_cell=25.0 * d.ds).mean())
    return res


def main():
    args = parse_args()
    du.set_seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    print(f"[env] device={device} torch={torch.__version__}")
    try:
        import psutil
        print(f"[env] mem_total={psutil.virtual_memory().total/2**30:.1f}GB")
    except ImportError:
        pass

    from dataset import WeddellMotionDataset, SPLITS
    from models import build_model, count_params

    if args.train_years:
        lo, hi = (int(s) for s in args.train_years.split("-"))
        SPLITS["train"] = (lo, hi)
        print(f"[data] 训练段覆盖为 {lo}-{hi}")

    ckpt_dir = ROOT / "models" / args.tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_pt = ckpt_dir / "ckpt_last.pt"
    best_pt = ckpt_dir / "ckpt_best.pt"

    mk = lambda split, shuffle: WeddellMotionDataset(
        args.zarr, split, tin=args.tin, tout=args.tout, stride=args.stride,
        downsample=args.ds, add_doy=bool(args.add_doy), synth=args.synth,
        seed=args.seed, strict_daily=args.strict_daily)
    ds_tr = mk("train", True)
    ds_va = mk("val", False)
    print(f"[data] train={len(ds_tr)} val={len(ds_va)} frames={ds_tr.n_frames}")
    loader_tr = torch.utils.data.DataLoader(ds_tr, batch_size=args.batch,
                                            shuffle=True, num_workers=0)
    val_subset = ds_va
    if 0 < args.selection_windows < len(ds_va):
        indices = np.linspace(0, len(ds_va) - 1, args.selection_windows, dtype=int)
        val_subset = torch.utils.data.Subset(ds_va, indices)
    loader_va = torch.utils.data.DataLoader(val_subset, batch_size=args.batch,
                                            shuffle=False, num_workers=0)

    cin = 4 if args.add_doy else 2
    model = build_model(args.model, tin=args.tin, tout=args.tout, cin=cin,
                        chid=args.chid, n_s=args.n_s, n_t=args.n_t)
    print(f"[model] {args.model} params={count_params(model)/1e6:.2f}M")

    if args.eval_only:
        ckpt = Path(args.ckpt) if args.ckpt else best_pt
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state["model"])
        model = model.to(device)
        out = evaluate(model, loader_va, ds_va, device, args.tout, args.model,
                          target_residual=bool(args.target_residual))
        print("[eval/val]", json.dumps(out))
        # 测试集
        ds_te = mk("test", False)
        loader_te = torch.utils.data.DataLoader(ds_te, batch_size=args.batch,
                                                shuffle=False, num_workers=0)
        te = evaluate(model, loader_te, ds_te, device, args.tout, args.model,
                          target_residual=bool(args.target_residual))
        print("[eval/test]", json.dumps(te))
        traj = evaluate_trajectory(model, ds_te, device, args.tout,
                                   model_name=args.model)
        if traj:
            print("[eval/traj]", json.dumps(traj))
            te.update(traj)
        du.append_metrics(te, experiment=f"{args.tag}-test",
                          params={"model": args.model, "ds": args.ds,
                                  "tin": args.tin, "tout": args.tout,
                                  "ckpt": str(ckpt)})
        return

    if args.init_weights:
        state = torch.load(args.init_weights, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        print(f"[init] {args.init_weights} epoch={state.get('epoch')}")
        del state
    if args.head_only:
        if args.model != "simvp":
            raise ValueError("--head-only 仅支持 SimVP")
        model.requires_grad_(False)
        model.dec.out.requires_grad_(True)
    model = model.to(device)
    if args.init_weights:
        baseline = evaluate(model, loader_va, ds_va, device, args.tout, args.model,
                            target_residual=bool(args.target_residual))
        print("[val/before]", json.dumps(baseline), flush=True)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr,
                            weight_decay=args.weight_decay)
    start_ep = 0
    best_val = float("inf")
    bad_epochs = 0
    if args.resume and last_pt.exists():
        st = torch.load(last_pt, map_location=device)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        start_ep = st["epoch"] + 1
        best_val = st.get("best_val", float("inf"))
        print(f"[resume] epoch={start_ep} best_val={best_val}")

    steps_per_ep = len(loader_tr)
    total_steps = (args.epochs - start_ep) * steps_per_ep
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = int(total_steps * args.warmup)

    def lr_at(step):
        if step < warmup_steps:
            return args.lr * (step + 1) / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return args.lr_min + 0.5 * (args.lr - args.lr_min) * (1 + math.cos(math.pi * p))

    # 持续性基线（验证集，一次算好）
    pers = None
    t0_all = time.time()
    global_step = 0
    last_log = {}
    model.train()
    for ep in range(start_ep, args.epochs):
        t_ep = time.time()
        for b in loader_tr:
            if args.max_steps and global_step >= args.max_steps:
                break
            lr = lr_at(global_step)
            for g in opt.param_groups:
                g["lr"] = lr
            x = b["x"].to(device)
            y = b["y"].to(device)
            m = b["mask"].to(device)
            history_valid = b["last_mask"].to(device)
            base = x[:, -1:, :2].expand_as(y)          # 最后一帧观测
            if args.model == "persistence":
                out_raw = base
            elif args.model == "convlstm":
                out_raw = model(x, args.tout)
            else:
                out_raw = model(x)
            if args.target_residual and args.model != "persistence":
                pred_raw = out_raw                      # 模型输出增量场
                pred = out_raw + base                   # 绝对场
                y_raw = y - base                        # 增量目标
            else:
                pred_raw = pred = out_raw
                y_raw = y
            loss = (masked_mse(pred_raw, y_raw, m)
                    + args.mae_weight * masked_mae(pred_raw, y_raw, m)
                    + args.grad_weight * grad_loss(pred_raw, y_raw, m))
            if args.day1_weight:
                # 与最后一次观测比较漂移变化，突出 24h 时效。
                # 仅在起报日和目标日都有效的像元上计算，排除新生冰/缺测造成的伪变化。
                joint = m[:, 0] * history_valid
                recent = x[:, -1, :2]
                observed_change = y[:, 0] - recent
                predicted_change = pred[:, 0] - recent
                change_weight = 1.0 + observed_change.abs().clamp(max=3.0)
                loss = loss + args.day1_weight * masked_mae(
                    (predicted_change - observed_change) * change_weight,
                    torch.zeros_like(observed_change), joint)
            if args.div_reg > 0 and args.model == "simvp":
                loss = loss + args.div_reg * differential_divergence(pred)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            global_step += 1
            if global_step % 25 == 0:
                dt = time.time() - t_ep
                print(f"[train] ep={ep} step={global_step}/{total_steps} "
                      f"loss={loss.item():.4f} lr={lr:.2e} {dt:.1f}s/25step",
                      flush=True)
                t_ep = time.time()
            if args.max_steps and global_step >= args.max_steps:
                break
        if args.max_steps and global_step >= args.max_steps:
            print(f"[train] 达到 max_steps={args.max_steps}，提前结束")
        # ---- 验证
        val = evaluate(model, loader_va, ds_va, device, args.tout, args.model,
                          target_residual=bool(args.target_residual))
        val["loss_train"] = loss.item()
        print(f"[val] ep={ep} " + json.dumps(
            {k: round(float(v), 4) for k, v in val.items()}), flush=True)
        score = val["vec_rmse_d1" if args.day1_weight else "vec_rmse"]
        is_best = score < best_val
        if is_best:
            bad_epochs = 0
            best_val = score
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "epoch": ep, "best_val": best_val, "args": vars(args)},
                       best_pt)
            du.append_metrics({"vec_rmse_val": score}, experiment=args.tag,
                              params={"epoch": ep, "model": args.model,
                                      "ds": args.ds, "chid": args.chid})
        else:
            bad_epochs += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": ep, "best_val": best_val, "args": vars(args)},
                   last_pt)
        if args.patience and bad_epochs >= args.patience:
            print(f"[early-stop] {bad_epochs} 个 epoch 无改善（best_val={best_val:.4f}），停止")
            break
        if args.max_steps and global_step >= args.max_steps:
            break
    print(f"[done] total={time.time()-t0_all:.0f}s best_val={best_val:.4f} "
          f"ckpt={best_pt}")


if __name__ == "__main__":
    main()
