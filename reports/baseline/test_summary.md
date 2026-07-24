# 阶段 0 基线测试汇总

- 状态：`passed`
- 生成时间：`2026-07-24T10:06:17.807410+00:00`

| 检查 | 必需 | 状态 | 耗时（秒） |
|---|---:|---:|---:|
| `python_tests` | 是 | `passed` | 14.955 |
| `web_node_tests` | 是 | `passed` | 0.136 |
| `no_camera_smoke` | 是 | `passed` | 1.332 |
| `camera_startup` | 是 | `passed` | 2.461 |
| `physical_camera_benchmark` | 是 | `passed` | 23.745 |
| `golden_and_latency_baseline` | 是 | `passed` | 41.916 |

详细标准输出和错误输出位于 `logs/`。实际显示延迟与 sensor-to-photon 必须在目标设备上通过 120/240 FPS 外部录像补测。
