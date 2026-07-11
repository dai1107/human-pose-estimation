# HYROX 动作配置

本目录为 8 个 HYROX 动作分别保存独立阈值。实时入口 `main.py` 与离线回放工具
`tools/replay_hyrox_video.py` 共用同一配置和同一分析器；不指定 `--hyrox-config`
时，会按 `--hyrox-action` 自动加载对应文件。自定义文件只需写需要覆盖的字段，缺失
字段会使用代码内安全默认值。

## 文件与建议视角

| 文件 | 用途 | 建议视角 |
|---|---|---|
| `lunge.yaml` | 负重箭步蹲阶段、深度与伸展提示 | 侧面或斜侧面，全身入镜 |
| `wall_ball.yaml` | 深蹲、起身投掷阶段与计数 | 正面或斜前方，手腕与脚踝均入镜 |
| `farmers_carry.yaml` | 搬运姿态稳定性监控 | 正面或斜前方，全身入镜 |
| `rowing.yaml` | 划船周期与划频计数 | 优先侧面 |
| `skierg.yaml` | 上拉、下拉周期与计数 | 正前方或斜前方 |
| `burpee_broad_jump.yaml` | 波比与向前跳的组合计数 | 侧面约 45°，保留落地区域 |
| `sled_push.yaml` | 推动姿态、步态与检测步数 | 侧面或斜侧面 |
| `sled_pull.yaml` | 拉动周期、站姿与左右同步 | 侧面或斜侧面 |

## 通用字段

- `action_name`：配置所属动作，便于检查和追踪。
- `visibility_min`：最低关键点可见度；提高后判断更保守。
- `stable_frames`：阶段切换需连续满足的帧数；提高可减少抖动，但增加延迟。
- `*_cooldown_ms` / `cooldown_ms`：两次计数或事件之间的最短间隔。
- `feedback_limits.max_messages`：单帧最多提示数，默认 2。
- `feedback_limits.low_visibility_exclusive`：可见度不足时只显示取景提示，避免输出不可靠的技术结论。

动作专属字段可按需要微调：角度字段通常以度为单位，`*_margin`、`*_delta_x`、
`*_distance_norm` 是画面或人体尺度归一化值，时间字段以毫秒为单位。建议一次只调整一个
阈值，并用 `--save-debug-csv` 对照阶段和特征变化；不要把不同机位的经验阈值直接混用。

示例：

```powershell
python main.py --camera 0 --mirror --hyrox-action rowing --hyrox-config configs/hyrox/rowing.yaml --hyrox-debug
python tools/replay_hyrox_video.py --video "HYROX视频\划船机.mp4" --hyrox-action rowing --save-debug-csv outputs/rowing_debug.csv
```

## 单摄像头无法可靠判断的项目

这些分析器只提供二维姿态近似指导，不检测器械，也不冒充比赛裁判系统。当前无法可靠
判断器械重量、阻力档位、绳子或雪橇是否过线、lane 合规性、划船机或 SkiErg 屏幕里程，
也不能确认 Farmers Carry 200 m、Burpee Broad Jump 80 m、雪橇 50 m 等官方距离。
二维画面还不能精确测量离地高度、落点距离、胸部真实触地、双脚厘米级误差或关节受力。
