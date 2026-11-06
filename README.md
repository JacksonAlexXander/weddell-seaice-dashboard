# 南极威德尔海海冰运动轨迹预测大屏

基于 NSIDC-0116 v4 海冰漂移数据的交互式预测大屏。下载完整运行包后，可在自己的电脑上进行预测与历史回测，无需云服务器或 API 密钥。

**[下载完整运行包（Releases）](https://github.com/JacksonAlexXander/weddell-seaice-dashboard/releases/latest)**

> 请选择 `weddell-seaice-dashboard-v1.0.0-full.zip`。GitHub 自动生成的 `Source code (zip)` 只包含仓库文件，不含模型与数据，不能单独运行。完整包内含前后端源代码、构建好的页面、五种模型所需文件及历史数据。

## 快速开始

1. 安装 **Python 3.13（64 位）**：[官方下载](https://www.python.org/downloads/)。Windows 安装时勾选 Python Launcher 与 Add Python to PATH。
2. 下载并完整解压 `weddell-seaice-dashboard-v1.0.0-full.zip`，不要直接在压缩包内运行。
3. 按系统启动：

| 系统 | 启动方式 |
| --- | --- |
| Windows 10/11（64 位） | 双击 `start-windows.bat` |
| macOS | 在终端进入解压目录，执行 `bash start-macos.command` |
| Linux（64 位） | 在终端进入解压目录，执行 `bash start-linux.sh` |

首次启动会自动创建项目内的 `.venv` 并联网安装依赖，可能需要数分钟。之后会启动本地服务并打开浏览器：**http://127.0.0.1:8000/**。使用时保持启动窗口打开，按 `Ctrl+C` 停止服务。

建议至少 8 GB 内存、预留 6 GB 磁盘空间。无需独立显卡；CPU 可运行。安装依赖后，预测与历史回测使用本地数据，不依赖在线 API。此版本已在 macOS Apple Silicon + Python 3.13 上验证；Windows/Linux 提供启动入口，尚未进行实机验证。

如果 8000 端口已被占用：

```bash
# macOS / Linux
python3 start.py --port 8001
# Windows
py -3.13 start.py --port 8001
```

## 使用步骤

1. 选择模型与数据范围内的起报日期。
2. 在地图点击像元，或输入行列号；可从默认像元 `row=65, col=60`、日期 `2025-06-26` 开始。
3. 点击“一键运行预测”，查看漂移矢量、24 小时位移曲线与预测路径。
4. 切换“观测 / 预测”漂移场，或选择 90 / 180 / 365 天历史回测。

提供 Ridge 岭回归、Random Forest 随机森林、SimVP 深度学习，以及持续性、气候态两种基线。启动只加载现有模型进行推理，不会重新训练。

## 数据与结果口径

- 数据：NSIDC-0116 v4，威德尔海子集，1978–2025 年，116 × 158 网格。
- 来源：[Polar Pathfinder Daily 25 km EASE-Grid Sea Ice Motion Vectors, Version 4](https://doi.org/10.5067/INAWUWO7QH7B)。引用与使用条件以数据提供方为准。
- 这是**历史数据上的预测与回测工具**，不会自动获取当前日期的新数据，也不是实时业务预报。
- 24 小时逐时轨迹由日平均漂移矢量线性积分得到，不是逐小时实测轨迹。
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

- “Missing runtime assets”：下载了不含数据的源码包，或没有完整解压，请改用完整运行包。
- 安装依赖失败：确认 Python 为 3.13 64 位、网络可连接 PyPI，重试启动即可。不要复制其他电脑的 `.venv`。
- 无效像元：在地图上选有观测数据的海冰像元，或恢复默认行列号。
- 停止后页面不能预测：重新运行启动脚本。
- `127.0.0.1` 永远指当前使用者自己的电脑；分享项目时发送 GitHub 链接。

开发者可在已安装运行依赖后，额外安装 `httpx` 与 `pytest`，运行 `scripts/qa/smoke_api.py` 和 `backend/tests`。
