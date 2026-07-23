# Lite 与 Full 模型档位回归

## 结论

2026-07-22 使用 8 个 HYROX 黄金视频逐帧比较官方 MediaPipe Pose Landmarker Full 与 Lite。Lite 推理更快，但按原黄金区间仅 `3/8` 通过，按 Full/Lite 严格产品等价门只有 Farmers Carry 通过，合计 `1/8`。因此产品配置保持：

```yaml
product_pose:
  realtime_model: auto
  analysis_model: full
```

网页仍提供显式 Lite 速度优先档，但 `lite_auto_approved=false`，自动模式不会选择 Lite。设备上 Full 持续过载时，在 Lite 尚未通过精度门的情况下会明确回退服务器兼容姿态，而不会静默把正式规则切到 Lite。

## 测试方法

```powershell
python tools\compare_pose_model_tiers.py --report outputs\validation\pose_model_tiers.json
```

每个视频同时比较：

- image landmarks 归一化二维距离；
- 规则输入关节角差异；
- 候选数、有效数、`NO_REP`、`UNSURE`、分析周期；
- 触地状态和脚部事件；
- 3D Assist 状态与 world landmarks 可用率；
- Full/Lite 推理 P50/P95 和姿态检出率。

## 本机离线结果

下表计数顺序为 `候选/有效/NO_REP/UNSURE/周期`，时间为 Python CPU 离线链路 P95，不代表浏览器 WASM 或 sensor-to-photon 延迟。

| 动作 | Full 计数 | Lite 计数 | 关键点均差 | 角度均差 | Full/Lite P95 | 触地匹配 | 脚部匹配 | 3D Assist 匹配 | 门结果 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Lunge | 1/0/1/0/0 | 1/0/0/1/0 | 0.0560 | 17.90° | 14.2/9.5 ms | 95.5% | 85.3% | 83.5% | 失败 |
| SkiErg | 1/0/0/1/0 | 1/1/0/0/1 | 0.0160 | 7.95° | 14.0/9.5 ms | 100% | 100% | 89.3% | 失败 |
| Burpee Broad Jump | 3/0/0/3/0 | 3/0/0/3/0 | 0.0288 | 17.67° | 13.9/9.7 ms | 100% | 100% | 92.2% | 失败 |
| Sled Push | 5/5/0/0/5 | 8/6/0/2/6 | 0.1290 | 19.50° | 14.4/9.7 ms | 100% | 100% | 49.0% | 失败 |
| Sled Pull | 4/1/0/3/1 | 5/0/0/5/0 | 0.0132 | 13.12° | 14.2/9.6 ms | 100% | 100% | 80.7% | 失败 |
| Wall Ball | 3/0/0/3/0 | 2/0/0/2/0 | 0.0165 | 13.96° | 14.5/9.8 ms | 99.3% | 94.3% | 76.6% | 失败 |
| Rowing | 5/0/0/5/0 | 5/0/0/5/0 | 0.0165 | 6.26° | 14.8/10.0 ms | 100% | 100% | 66.2% | 失败 |
| Farmers Carry | 0/0/0/0/0 | 0/0/0/0/0 | 0.0071 | 9.11° | 14.1/9.5 ms | 100% | 100% | 95.3% | 通过 |

Full 和 Lite 的人体检出率几乎一致，主要回归来自坐标与角度差异进入规则状态机后改变了端点、周期和 3D Assist 结果。因此不能只根据 Lite 更快就自动启用。

## 浏览器启动基准

自动档在当前设备真实摄像头上进行约 3 秒预热：先测 Full，再测 Lite。保存两档推理 P50/P95、姿态 FPS、检出率和主线程长任务数。选择规则为：

- Full P95 `≤20 ms`：Full；
- Full P95 为 `20～33 ms` 且稳定：Full；
- Full P95 `>33 ms`：只有 Lite 已通过离线精度门、当前检出率没有明显下降且确实更快时，才允许自动选择 Lite；
- 当前精度门未通过，所以正式自动档保持 Full。

浏览器实际速度和外部高速录像延迟仍必须在目标设备上重新测量。
