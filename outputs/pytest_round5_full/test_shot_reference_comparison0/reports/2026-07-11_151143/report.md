# 篮球投篮专项分析报告

投篮类型：set_shot
投篮侧：right
机位：side

出手代理时刻：600 ms
出手代理置信度：0.90
依据：shooting wrist speed local high value + shooting elbow extension signal + wrist near or above shooting shoulder + short follow-through trend after proxy frame

EVENT SEQUENCE

- pelvis_upward_speed_peak -> shooting_side_knee_extension_peak: 0 ms (overlap)
- shooting_side_knee_extension_peak -> shooting_side_hip_extension_peak: 0 ms (overlap)
- shooting_side_hip_extension_peak -> shoulder_elevation_peak: 700 ms (in_order)
- shoulder_elevation_peak -> shooting_side_elbow_extension_peak: -300 ms (early)
- shooting_side_elbow_extension_peak -> shooting_side_wrist_speed_peak: 0 ms (overlap)
- shooting_side_wrist_speed_peak -> release_proxy_time: 0 ms (overlap)

与个人参考投篮相比：
- shooting_knee_angle 平均差异 0。
- shooting_hip_angle 平均差异 0。
- trunk_tilt_proxy 平均差异 0。
- shooting_shoulder_angle 平均差异 0。
该报告描述的是与指定参考动作之间的运动学差异，不代表投篮动作正确性、命中概率或真实发力质量。

固定限制说明：
本报告基于单目视频人体关键点与运动学代理指标生成。

它不能直接测量：
- 地面反作用力；
- 真实关节力矩；
- 肌肉发力；
- 球离手的精确时刻（除非未来接入并验证篮球检测）；
- 投篮命中率；
- 医学风险；
- 投篮技术是否绝对标准。
