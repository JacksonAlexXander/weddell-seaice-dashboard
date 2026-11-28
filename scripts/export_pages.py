"""Export predictions from the existing trained models for the GitHub Pages build."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.config import Settings
from backend.app.routers.metrics import get_metrics
from backend.app.routers.models import build_models_payload
from backend.app.routers.overview import get_overview
from backend.app.schemas import BacktestRequest
from backend.app.services.artifact_store import ArtifactStore
from backend.app.services.field_service import grid_info
from backend.app.services.predict_service import PredictService
from backend.app.services.recent_service import MODEL_IDS, RecentService
from src.serving.recent_observations import load_snapshot


def vector_grid(u, v, mask):
    values = np.round(np.stack((u, v), axis=-1).astype(float), 6)
    if not np.isfinite(values[mask]).all():
        raise ValueError("Forecast contains non-finite vectors in valid cells")
    return [
        [pair if valid else None for pair, valid in zip(row, row_mask)]
        for row, row_mask in zip(values.tolist(), mask.tolist())
    ]


def write_json(path, payload):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True, type=Path, help="Unpacked full release root")
    parser.add_argument("--out", required=True, type=Path, help="Snapshots output directory")
    snapshot = parser.add_mutually_exclusive_group(required=True)
    snapshot.add_argument("--snapshot", type=Path, help="Use a verified local OSI SAF batch")
    snapshot.add_argument("--refresh", action="store_true", help="Download the latest OSI SAF batch")
    parser.add_argument("--recent-cache", type=Path, help="Write the downloaded batch outside the runtime package")
    parser.add_argument("--after", type=datetime.fromisoformat,
                        help="Export only the forecast when the observation is newer than this ISO timestamp")
    args = parser.parse_args()
    root = args.runtime.resolve()
    data = root / "data/processed"
    settings = Settings(
        workspace_root=root,
        zarr_path=data / "weddell_icemotion_daily.zarr",
        dashboard_dir=data / "dashboard",
        models_ml_dir=root / "models/ml",
        simvp_ckpt=root / "models/simvp-24h/ckpt_best.pt",
    )
    torch.set_num_threads(2)
    store = ArtifactStore(settings)
    service = RecentService(store)
    if args.snapshot:
        service._batch = load_snapshot(args.snapshot)
    else:
        if args.recent_cache:
            service.path = args.recent_cache
            service._batch = load_snapshot(service.path) if service.path.is_file() else None
        service.refresh()
    recent = service.metadata()
    if not recent["available"] or recent["valid_cells"] == 0:
        raise SystemExit("No usable observations available")
    latest_at = datetime.fromisoformat(recent["latest_at"])
    if args.after and latest_at <= args.after:
        print(f"No newer valid observation; latest available {latest_at.isoformat()}", flush=True)
        return
    if (args.refresh or args.after) and (datetime.now(UTC) - latest_at).total_seconds() >= 72 * 3600:
        raise SystemExit("No new forecast window. Keeping the previously published snapshot.")
    batch = service._require_batch()
    mask = batch["mask"][-1]
    lat, lon = service.grid
    key = f'fields-{recent["latest_date"]}.json'
    recent.update(fields_file=key, published_at=datetime.now(UTC).isoformat())
    fields = {
        "date": recent["latest_date"],
        "grid": grid_info(settings),
        "lat": np.round(lat.astype(float), 7).tolist(),
        "lon": np.round(lon.astype(float), 7).tolist(),
        "observed": vector_grid(batch["u"][-1], batch["v"][-1], mask),
        "forecasts": {},
    }
    for model in MODEL_IDS:
        print(f"Running {model} with {recent['latest_at']}", flush=True)
        u, v = service._forecast(batch, model)
        fields["forecasts"][model] = [vector_grid(day_u, day_v, mask) for day_u, day_v in zip(u, v)]
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / key, fields)
    if args.after is None:
        write_json(args.out / "models.json", build_models_payload(store).model_dump(mode="json"))
        historical = PredictService(store)
        backtest = {
            model: historical.backtest(BacktestRequest(
                model=model, cell={"row": 65, "col": 60},
                start="2025-01-01", end="2025-06-30", max_points=180,
            ))
            for model in ("ridge", "random_forest")
        }
        write_json(args.out / "backtest.json", backtest)
        write_json(args.out / "metrics.json", get_metrics(store))
        write_json(args.out / "overview.json", get_overview(store))
    write_json(args.out / "recent.json", recent)
    print(f"Exported {key} with {recent['valid_cells']} grid cells", flush=True)


if __name__ == "__main__":
    main()
