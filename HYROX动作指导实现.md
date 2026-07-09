你现在继续开发一个 Python + OpenCV + MediaPipe 的实时人体姿态检测项目。

当前项目已经可以：
1. 打开摄像头；
2. 用 MediaPipe Pose Landmarker LIVE_STREAM 实时检测人体关键点；
3. 在 OpenCV 窗口显示画面、33 个关键点、骨架线和 FPS；
4. 支持摄像头参数、镜像、录制、平滑等命令行参数。

本轮目标：
在不破坏现有实时摄像头功能的前提下，逐步增加 HYROX 动作实时分析功能。系统最终目标是：用户面对摄像头做 HYROX 动作时，程序能够实时识别动作阶段、统计次数，并给出简单、可解释、低延迟的动作纠正提示。

非常重要：
1. 不要推翻现有项目结构。
2. 不要替换 MediaPipe 主流程。
3. 不要把视频离线分析作为主流程。
4. 摄像头实时分析永远是主流程。
5. 本地 HYROX 视频只作为规则校准、回放测试、对比验证。
6. 所有新增功能必须可关闭。
7. 每次只实现一个小功能。
8. 如果不确定项目原有入口文件名，先读取目录和 README，再按现有结构适配。
9. 不要一次性重构太多文件。
10. 每轮修改后都要给出运行命令和验收标准。

工程设计原则：
- MediaPipe 检测层只负责输出 landmarks。
- 平滑层只负责稳定 landmarks。
- 特征层负责计算角度、距离、速度、身体朝向。
- 动作分析层负责阶段识别、次数统计、错误判断。
- 反馈层负责把提示显示到 OpenCV 画面上。
- 本地视频回放层只复用同一套动作分析逻辑，不另写一套分析逻辑。

第 0 轮任务：项目体检，不允许修改任何文件。

请完成：
1. 读取当前项目目录结构。
2. 找到主入口文件，例如 main.py / app.py / realtime_pose.py。
3. 找到 MediaPipe PoseLandmarker 初始化位置。
4. 找到 OpenCV 摄像头读取循环。
5. 找到绘制关键点、骨架、FPS 的代码。
6. 找到已有 smoothing、record、mirror、camera 参数的实现位置。
7. 输出当前项目结构总结。
8. 输出后续最小改造点建议。

禁止：
- 不要修改文件。
- 不要新增文件。
- 不要重构。
- 不要安装新依赖。

输出格式：
- 当前入口文件
- 当前实时检测流程
- 当前命令行参数
- 可以安全新增的文件
- 下一轮建议修改哪些文件

第 1 轮任务：新增 HYROX 分析模块骨架，但不要影响原有实时检测。

请新增一个独立目录，优先命名为：
hyrox/

建议文件：
hyrox/__init__.py
hyrox/landmark_names.py
hyrox/geometry.py
hyrox/features.py
hyrox/feedback.py
hyrox/base.py

实现内容：
1. landmark_names.py
   - 定义 MediaPipe Pose 33 个关键点的常量名或索引映射。
   - 至少包含 shoulder、hip、knee、ankle、heel、foot_index、elbow、wrist。

2. geometry.py
   - 实现 angle_3pts(a, b, c)，返回 b 点夹角，单位 degree。
   - 实现 safe_distance(p1, p2)。
   - 实现 midpoint(p1, p2)。
   - 所有函数要能处理 None 或 visibility 低的点，不能崩溃。

3. features.py
   - 实现 extract_basic_pose_features(landmarks, image_width, image_height)。
   - 返回 dict，至少包括：
     left_knee_angle
     right_knee_angle
     left_hip_angle
     right_hip_angle
     torso_angle
     shoulder_tilt
     hip_tilt
     visible_score

4. feedback.py
   - 定义 FeedbackMessage 数据结构：
     level: info/warn/error
     code: 字符串
     text: 中文提示
     confidence: 0~1

5. base.py
   - 定义 BaseActionAnalyzer 类：
     reset()
     update(features, timestamp_ms) -> dict
   - 返回 dict 至少包含：
     action
     phase
     rep_count
     feedback_messages
     debug

本轮不要接入摄像头窗口，不要改 main.py，最多只新增模块。
增加最小单元测试或简单自测脚本。

第 2 轮任务：把 hyrox/features.py 接入现有实时摄像头流程，但只显示 debug 信息，不做动作判断。

要求：
1. 保持原有摄像头实时检测完全可用。
2. 新增命令行参数：
   --hyrox-debug
   默认关闭。
3. 当 --hyrox-debug 开启时：
   - 从最新 MediaPipe landmarks 提取 basic pose features。
   - 在 OpenCV 画面左上角显示：
     visible_score
     left_knee_angle
     right_knee_angle
     left_hip_angle
     right_hip_angle
     torso_angle
4. 当 --hyrox-debug 关闭时：
   - 程序表现应与原来一致。
5. 不要新增动作识别。
6. 不要改 MediaPipe 初始化逻辑。
7. 不要影响 FPS 显示。
8. 如果没有检测到人体，显示 “No pose”。

注意：
- overlay 文本要简洁。
- 所有异常都不能导致摄像头窗口崩溃。

第 3 轮任务：实现 HYROX Lunge 动作分析 MVP。

新增文件：
hyrox/actions/lunge.py
hyrox/actions/__init__.py

实现 LungeAnalyzer，继承 BaseActionAnalyzer。

新增命令行参数：
--hyrox-action none/lunge
默认 none。

当 --hyrox-action lunge 时：
1. 实时分析弓步动作。
2. 显示：
   - 当前动作：Lunge
   - 当前阶段：stand / descent / bottom / ascent / unknown
   - rep_count
   - 当前反馈提示

阶段识别规则先用简单阈值：
- stand：
  双膝角度较大，例如 > 150°
- descent：
  膝角从大变小，身体下降
- bottom：
  至少一个膝盖角度较小，例如 < 100°，并且髋/膝位置显示处于低位
- ascent：
  从 bottom 回到 stand
- 完成一次：
  bottom -> stand 形成一次有效 rep

实时纠正提示先做 4 条：
1. LOW_VISIBILITY：
   人体关键点置信度低，提示“请站到画面中间，保证全身入镜”
2. NOT_DEEP_ENOUGH：
   bottom 阶段膝角没有明显变小，提示“下蹲幅度不够，后侧膝盖应接近地面”
3. LEAN_TOO_MUCH：
   torso_angle 过大，提示“躯干前倾过多，保持核心稳定”
4. STAND_EXTENSION：
   回到 stand 时膝髋没有充分伸展，提示“每次站起时膝盖和髋部要伸直”

先不要做：
- 不要判断左右交替。
- 不要判断沙袋位置。
- 不要判断是否真正触地。
- 不要接入语音。
- 不要保存训练数据。

第 4 轮任务：增强 LungeAnalyzer 的稳定性。

问题：
实时姿态检测会抖动，不能一帧满足条件就切阶段，也不能一帧完成就计数。

请实现：
1. 阶段防抖
   - 同一个候选阶段连续出现 N 帧才确认。
   - N 默认 3。

2. rep 冷却时间
   - 完成一次 rep 后，至少 400ms 内不允许再次计数。

3. feedback 优先级
   - error > warn > info。
   - 同一帧最多显示 2 条提示。
   - 如果 LOW_VISIBILITY 出现，只显示 LOW_VISIBILITY，暂停其他判断。

4. 新增 debug 字段
   - raw_phase
   - stable_phase
   - frames_in_phase
   - last_rep_time_ms

5. 新增参数：
   --hyrox-sensitivity low/medium/high
   默认 medium。
   low：阈值更保守，误报少；
   high：阈值更敏感，适合动作幅度小的测试。

不要改动 MediaPipe 主流程。

第 5 轮任务：增加本地视频回放测试工具，用于验证 LungeAnalyzer。

新增工具脚本：
tools/replay_hyrox_video.py

功能：
1. 支持传入视频路径：
   --video path/to/video.mp4
2. 支持动作：
   --hyrox-action lunge
3. 逐帧读取视频。
4. 对每帧执行 MediaPipe 姿态检测。
5. 把 landmarks 输入同一个 LungeAnalyzer。
6. 显示 OpenCV 回放窗口，包括：
   - 原视频画面
   - 骨架
   - 当前阶段
   - rep_count
   - feedback
7. 支持按 q 退出。
8. 支持 --speed 1.0 / 0.5 / 2.0 控制回放速度。
9. 支持 --save-debug-csv path.csv，把每帧特征和分析结果保存下来。

重要：
视频回放只是测试工具，不要替代实时摄像头主流程。
不要把视频分析逻辑单独写一套。
必须复用 hyrox/features.py 和 hyrox/actions/lunge.py。

第 6 轮任务：把 LungeAnalyzer 里的阈值移到配置文件。

新增：
configs/hyrox/lunge.yaml

配置内容至少包括：
visibility_min
stand_knee_angle_min
bottom_knee_angle_max
torso_lean_warn
stable_frames
rep_cooldown_ms

要求：
1. LungeAnalyzer 支持从配置 dict 初始化。
2. main.py 新增参数：
   --hyrox-config configs/hyrox/lunge.yaml
3. 如果配置文件不存在，使用默认配置，不崩溃。
4. replay_hyrox_video.py 也支持同样的 --hyrox-config。
5. 在 overlay debug 中显示当前配置名。

目的：
用本地 HYROX 视频回放来调阈值，但不要改代码。