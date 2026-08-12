# HYROX 第 12 轮角度准确性验证

生成日期：2026-08-11

## 数据与通道

- 人工 2D 角度标注：150 条，其中 149 条有可匹配姿态缓存。
- 必选动作：Lunge、Wall Ball、Burpee Broad Jump、Rowing 均有标注。
- 同时输出：raw 2D、filtered 2D、raw 3D、filtered 3D、canonical 3D、selected rule angle。
- 正式规则仍使用 causal filtered 2D；3D/canonical 3D 只用于验证，没有替换 HYROX 阈值。

## 当前总体结果

| 通道 | MAE | Median AE | P90 | P95 |
|---|---:|---:|---:|---:|
| raw 2D | 15.5292° | 11.6921° | 32.7615° | 38.4556° |
| filtered 2D | 17.5325° | 10.7511° | 38.1830° | 45.2030° |
| raw 3D projection gap | 30.1792° | 23.9894° | 64.5302° | 71.4387° |
| filtered 3D projection gap | 31.7783° | 25.4356° | 63.4468° | 79.6146° |
| canonical 3D projection gap | 31.7783° | 25.4356° | 63.4468° | 79.6146° |
| selected rule angle | 17.5325° | 10.7511° | 38.1830° | 45.2030° |

人工标签是视频像素上的投影 2D 角，因此 3D 数字是投影一致性差距，不是空间 3D 真值 MAE。三点关节角在刚性人体坐标旋转前后保持不变，所以当前 canonical 3D 与 filtered world 3D 数值一致。

## 旧版与新版

当前可复现代理基线定义为 raw-lite unfiltered 2D，新版候选定义为 causal-full filtered 2D selected-rule angle。

| 指标 | 旧版 | 新版 | 变化 | 非回归 |
|---|---:|---:|---:|---|
| MAE | 17.3965° | 17.5325° | +0.1360° | 否 |
| Median AE | 10.9921° | 10.7511° | -0.2410° | 是 |
| P90 | 38.8965° | 38.1830° | -0.7135° | 是 |
| P95 | 43.6998° | 45.2030° | +1.5032° | 否 |

结论：严格的四指标非回归当前未通过。因此本轮没有把 3D/canonical 角提升为正式规则输入，也没有修改现有阈值。

## 尚缺证据

- 当前视角只有 front、side、oblique_back；缺少明确标定的 30° 和 45°斜侧素材。
- 人工报告包含最低点/完全伸展标签，但缓存没有成对的旧版/新版程序事件帧，无法诚实计算版本间 event timing error。
- `compare_manual_angles.py --baseline-report ... --report ...` 已支持在取得成对帧报告后计算最低点和完全伸展事件误差及非回归结果。

机器可读结果见 `reviewed_angle_evaluation.json`，逐条六通道数据见 `reviewed_angle_rows.csv`。
