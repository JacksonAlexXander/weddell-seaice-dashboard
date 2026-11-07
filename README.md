# 南极威德尔海海冰运动轨迹预测大屏

基于 NSIDC-0116 v4 海冰漂移数据的交互式预测大屏。下载完整运行包后，可在自己的电脑上进行预测与历史回测，无需云服务器或 API 密钥。

**[下载 v1.2.0 完整安装包](https://github.com/JacksonAlexXander/weddell-seaice-dashboard/releases/download/v1.2.0/weddell-seaice-dashboard-v1.2.0-full.zip)** · [版本说明和校验文件](https://github.com/JacksonAlexXander/weddell-seaice-dashboard/releases/latest)

完整 ZIP 包含历史数据、已训练权重、近期观测缓存、启动脚本和已构建的大屏。请在 Releases 中选择带
`-full.zip` 的文件。GitHub 自动生成的 `Source code (zip)` 仅供查看源码。
模型选择、训练过程、独立测试结果和论文见[训练与回测记录](docs/训练与回测.md)。

## 快速开始

1. 安装 **Python 3.13（64 位）**：[官方下载](https://www.python.org/downloads/)。Windows 安装时勾选 Python Launcher 与 Add Python to PATH。
2. 将完整安装包解压到本地文件夹，然后进入解压出的 `weddell-seaice-dashboard` 目录。
3. 按系统启动：

| 系统 | 启动方式 |
| --- | --- |
| Windows 10/11（64 位） | 双击 `start-windows.bat` |
| macOS | 在终端进入解压目录，执行 `bash start-macos.command` |
| Linux（64 位） | 在终端进入解压目录，执行 `bash start-linux.sh` |

首次启动会自动创建项目内的 `.venv` 并联网安装依赖，可能需要数分钟。之后会启动本地服务并打开浏览器：**http://127.0.0.1:8000/**。使用时保持启动窗口打开，按 `Ctrl+C` 停止服务。

建议至少 8 GB 内存、预留 6 GB 磁盘空间。无需独立显卡；CPU 可运行。安装依赖后可用本地观测缓存预测和回测。更新近期观测需要联网访问 MET Norway 数据服务。此版本已在 macOS Apple Silicon + Python 3.13 上验证；Windows/Linux 提供启动入口，尚未进行实机验证。

如果 8000 端口已被占用，可选择 8765：

```bash
# macOS / Linux
python3 start.py --port 8765
# Windows
py -3.13 start.py --port 8765
```

## 近期预报

1. 进入「近期预报」，点击「更新近期观测」获取 EUMETSAT OSI SAF 最新的连续 10 日观测。
2. 选 SimVP、Ridge 或 Random Forest，再选 24、48 或 72 小时。通过地图选有近期观测的像元。
3. 点击「一键运行预测」。查看轨迹、每日 u/v 和预报有效截止时刻，或下载 JSON。
4. 切到地图「预测」查看所选时效的区域漂移场。切到「归档回测」可继续用 1978–2025 年历史数据对比。

时效从**实际观测结束时刻**算起。卫星数据有发布时间差，界面会说明所选轨迹是否延伸到当前时刻之后。
SimVP 直接输出多日预测，Ridge 与 Random Forest 使用首日预测递推。点击预测只运行已训练权重。
[近期数据、接口与预报口径](docs/近期预报.md)

## 数据与结果口径

- 数据：NSIDC-0116 v4，威德尔海子集，1978–2025 年，116 × 158 网格。
- 来源为 Tschudi、Meier 和 Stewart（2019）的 NSIDC-0116 v4，由 NASA NSIDC DAAC 分发。包内数据为 60°W–20°E、60°S–83°S 的威德尔海子集。[数据 DOI](https://doi.org/10.5067/INAWUWO7QH7B) · [NSIDC 数据引用指南](https://nsidc.org/data/data-programs/nsidc-daac/citing-nsidc-daac)
- NSIDC 归档覆盖 1978-10-25 至 2025-12-30。近期模式使用独立的 OSI SAF 缓存，更新观测需要联网。
- 轨迹由日平均预测矢量在 EPSG:3409 网格上分段积分，再反投影到经纬度。
- 模型误差、不确定性和历史评估指标在界面中展示。

## 文件结构

```text
start.py / start-*.bat / start-*.command / start-*.sh  启动入口
requirements.txt                                    Python 推理依赖
backend/                                            FastAPI 服务及测试
src/                                                数据处理、模型与推理源代码
dashboard/src/                                      Vue + ECharts 前端源代码
dashboard/dist/                                     已构建前端，无需安装 Node.js
data/processed/                                     历史海冰数据与评估产物（完整包）
models/                                             已训练模型（完整包）
docs/                                               API 契约与设计说明
```

## 开发与排错

前端开发见 `dashboard/README.md`。服务启动后，API 文档位于 `http://127.0.0.1:8000/docs`。

- “Missing runtime assets”：检查 `models/`、`data/processed/` 和 `dashboard/dist/` 是否齐全。
- 安装依赖失败：确认 Python 为 3.13 64 位、网络可连接 PyPI，重试启动即可。不要复制其他电脑的 `.venv`。
- 无效像元：在地图上选有观测数据的海冰像元，或恢复默认行列号。
- 停止后页面不能预测：重新运行启动脚本。
- `127.0.0.1` 指当前电脑。可将本目录复制到其他电脑并在本机启动。

开发者可在已安装运行依赖后，额外安装 `httpx` 与 `pytest`，运行 `scripts/qa/smoke_api.py` 和 `backend/tests`。
