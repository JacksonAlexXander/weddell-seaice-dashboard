# 数据包整理与验证记录

发布日期：2026-11-06。原始观测块和模型权重未改变。

本地归档目录缺少部分 Zarr v2 元数据。发布副本依据原 ETL 的数组形状、类型及压缩格式，并核对原数据块的解压长度后恢复了组和数组元数据。时间轴使用存储块中原有的 Unix 秒，按 datetime64[s] 解码。

月份数组与缺少的年内日序块从原时间戳确定性重建；缺少的最后一块有效覆盖率由同时间段的原始 valid_mask 计算。所有已有日序块与时间戳逐值一致。没有生成或填补新的海冰运动观测。

时间轴共 17,188 帧，范围 1978-10-25 至 2025-12-30。原说明列出 36 个缺失日；逐日核查实际时间轴有 46 个空缺，额外 10 个为 1985、1987、2017–2024 年的 12 月 31 日。保留原数据与原说明作为来源记录，日期选择与预测使用实际存储的时间轴。

验证：
- Python 3.13 全新环境安装依赖、启动脚本检查通过。
- 后端测试 34 项通过；HTTP 冒烟检查 70 项通过。
- 五种模型在默认日期/像元上的 u、v、speed 与原运行服务完全相同。
- 1978-11-15、2000-06-26、2018-06-26、2025-06-26、2025-12-30 的整场速度与抽样矢量均与原运行服务一致。
- Windows/Linux 尚未实机测试。

恢复的文件：

- `data/processed/weddell_icemotion_daily.zarr/.zgroup`
- `data/processed/weddell_icemotion_daily.zarr/doy/.zarray`
- `data/processed/weddell_icemotion_daily.zarr/doy/67`
- `data/processed/weddell_icemotion_daily.zarr/latitude/.zarray`
- `data/processed/weddell_icemotion_daily.zarr/longitude/.zarray`
- `data/processed/weddell_icemotion_daily.zarr/month/.zarray`
- `data/processed/weddell_icemotion_daily.zarr/month/.zattrs`
- `data/processed/weddell_icemotion_daily.zarr/month/0`
- `data/processed/weddell_icemotion_daily.zarr/month/1`
- `data/processed/weddell_icemotion_daily.zarr/month/10`
- `data/processed/weddell_icemotion_daily.zarr/month/11`
- `data/processed/weddell_icemotion_daily.zarr/month/12`
- `data/processed/weddell_icemotion_daily.zarr/month/13`
- `data/processed/weddell_icemotion_daily.zarr/month/14`
- `data/processed/weddell_icemotion_daily.zarr/month/15`
- `data/processed/weddell_icemotion_daily.zarr/month/16`
- `data/processed/weddell_icemotion_daily.zarr/month/17`
- `data/processed/weddell_icemotion_daily.zarr/month/18`
- `data/processed/weddell_icemotion_daily.zarr/month/19`
- `data/processed/weddell_icemotion_daily.zarr/month/2`
- `data/processed/weddell_icemotion_daily.zarr/month/20`
- `data/processed/weddell_icemotion_daily.zarr/month/21`
- `data/processed/weddell_icemotion_daily.zarr/month/22`
- `data/processed/weddell_icemotion_daily.zarr/month/23`
- `data/processed/weddell_icemotion_daily.zarr/month/24`
- `data/processed/weddell_icemotion_daily.zarr/month/25`
- `data/processed/weddell_icemotion_daily.zarr/month/26`
- `data/processed/weddell_icemotion_daily.zarr/month/27`
- `data/processed/weddell_icemotion_daily.zarr/month/28`
- `data/processed/weddell_icemotion_daily.zarr/month/29`
- `data/processed/weddell_icemotion_daily.zarr/month/3`
- `data/processed/weddell_icemotion_daily.zarr/month/30`
- `data/processed/weddell_icemotion_daily.zarr/month/31`
- `data/processed/weddell_icemotion_daily.zarr/month/32`
- `data/processed/weddell_icemotion_daily.zarr/month/33`
- `data/processed/weddell_icemotion_daily.zarr/month/34`
- `data/processed/weddell_icemotion_daily.zarr/month/35`
- `data/processed/weddell_icemotion_daily.zarr/month/36`
- `data/processed/weddell_icemotion_daily.zarr/month/37`
- `data/processed/weddell_icemotion_daily.zarr/month/38`
- `data/processed/weddell_icemotion_daily.zarr/month/39`
- `data/processed/weddell_icemotion_daily.zarr/month/4`
- `data/processed/weddell_icemotion_daily.zarr/month/40`
- `data/processed/weddell_icemotion_daily.zarr/month/41`
- `data/processed/weddell_icemotion_daily.zarr/month/42`
- `data/processed/weddell_icemotion_daily.zarr/month/43`
- `data/processed/weddell_icemotion_daily.zarr/month/44`
- `data/processed/weddell_icemotion_daily.zarr/month/45`
- `data/processed/weddell_icemotion_daily.zarr/month/46`
- `data/processed/weddell_icemotion_daily.zarr/month/47`
- `data/processed/weddell_icemotion_daily.zarr/month/48`
- `data/processed/weddell_icemotion_daily.zarr/month/49`
- `data/processed/weddell_icemotion_daily.zarr/month/5`
- `data/processed/weddell_icemotion_daily.zarr/month/50`
- `data/processed/weddell_icemotion_daily.zarr/month/51`
- `data/processed/weddell_icemotion_daily.zarr/month/52`
- `data/processed/weddell_icemotion_daily.zarr/month/53`
- `data/processed/weddell_icemotion_daily.zarr/month/54`
- `data/processed/weddell_icemotion_daily.zarr/month/55`
- `data/processed/weddell_icemotion_daily.zarr/month/56`
- `data/processed/weddell_icemotion_daily.zarr/month/57`
- `data/processed/weddell_icemotion_daily.zarr/month/58`
- `data/processed/weddell_icemotion_daily.zarr/month/59`
- `data/processed/weddell_icemotion_daily.zarr/month/6`
- `data/processed/weddell_icemotion_daily.zarr/month/60`
- `data/processed/weddell_icemotion_daily.zarr/month/61`
- `data/processed/weddell_icemotion_daily.zarr/month/62`
- `data/processed/weddell_icemotion_daily.zarr/month/63`
- `data/processed/weddell_icemotion_daily.zarr/month/64`
- `data/processed/weddell_icemotion_daily.zarr/month/65`
- `data/processed/weddell_icemotion_daily.zarr/month/66`
- `data/processed/weddell_icemotion_daily.zarr/month/67`
- `data/processed/weddell_icemotion_daily.zarr/month/7`
- `data/processed/weddell_icemotion_daily.zarr/month/8`
- `data/processed/weddell_icemotion_daily.zarr/month/9`
- `data/processed/weddell_icemotion_daily.zarr/speed/.zarray`
- `data/processed/weddell_icemotion_daily.zarr/stats/.zgroup`
- `data/processed/weddell_icemotion_daily.zarr/stats/u_mean/.zarray`
- `data/processed/weddell_icemotion_daily.zarr/stats/u_std/.zarray`
- `data/processed/weddell_icemotion_daily.zarr/stats/v_mean/.zarray`
- `data/processed/weddell_icemotion_daily.zarr/stats/v_std/.zarray`
- `data/processed/weddell_icemotion_daily.zarr/stats/valid_days_count/.zarray`
- `data/processed/weddell_icemotion_daily.zarr/time/.zarray`
- `data/processed/weddell_icemotion_daily.zarr/valid_fraction/67`
- `data/processed/weddell_icemotion_daily.zarr/x/.zarray`
- `data/processed/weddell_icemotion_daily.zarr/y/.zarray`
