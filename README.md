# 动漫脸剪辑器 — 数据流程说明

> 本文说明 `src/` 下这套代码的数据流向、各阶段处理逻辑,以及输入/输出。
> 结合了真实运行产物 [`output/1/`](./output/1/) 中的数据。

---

## 一、总览

这是一个**动漫脸剪辑器**:给一个动漫视频,自动找出"人脸事件密集"的 15 秒片段并切出来。

判定标准:一个 15 秒窗口里如果有 **≥13 个不同人脸轨迹的"起点"**,就算合格片段。

入口链路(见 [`src/main.py`](./src/main.py)):

```
main()  →  run_pipeline()  →  process_video()
```

整个流程被注释切成 6 个阶段:

```
抽帧 → 检测 → 过滤 → 跟踪 → 选段 → 截取
```

---

## 二、输入与输出

### 输入

| 项 | 说明 |
|----|------|
| 视频文件 | [`data/1.mp4`](./data/1.mp4),时长 **1403.948 秒**(约 23 分钟) |
| 默认命令 | `python src/main.py`(不传参即处理 `data/1.mp4`,输出到 `output/`) |
| 带可视化 | `python src/main.py data/1.mp4 --viz 8`(额外导出 8 张标注样本帧) |

### 输出(全部落在 `output/1/`,目录名 `1` 来自视频文件名 stem)

| 产物 | 内容 | 本次实际数量 |
|------|------|--------------|
| `detections.json` | 每个原始检测框一条记录(含 `kept` 是否通过过滤) | **2895 条** |
| `tracks.json` | 每条人脸轨迹一条记录 | **285 条** |
| `windows.json` | 总摘要 + 选中片段 + 本次所用参数 | 5 个片段 |
| `crops/track_<id>.jpg` | 每条轨迹一张代表裁剪图 | **285 张** |
| `clips/clip_<NNN>.mp4` | 切出的 15 秒视频 | **5 个** |
| `viz/*.jpg` | 标注样本帧,仅 `--viz N` 时生成 | 8 张 |

---

## 三、数据流向(一图串起来)

```
data/1.mp4
  │ ① ffmpeg 抽帧 (0.3s/帧)
  ▼ List[(idx, time, path)]                  ← 临时目录,用完删除
  │ ② 逐帧: HSV直方图→is_cut[] ; YOLOv8检测→Detection[]
  │ ③ 三道质量门槛 (置信度 / 人脸大小 / 清晰度)
  ▼ frame_detections[][]  +  detection_records ──► detections.json (2895)
  │ ④ IoU 匹配 + 镜头切换断轨 → 串成轨迹
  ▼ tracks (285) ──► tracks.json + crops/*.jpg (285)
  │ ⑤ 取每条轨迹 start_time,15s 滑窗计数 ≥13
  ▼ segments (5) ──► windows.json
  │ ⑥ ffmpeg 从原视频帧精确切片
  ▼ clips/clip_00X.mp4 (5)
```

---

## 四、逐阶段详解

### 阶段 1:抽帧 — `extract_frames`([main.py:73](./src/main.py#L73))

- 用 ffmpeg 以 `fps = 1/0.3`(`frame_interval=0.3`,即每 0.3 秒一帧)抽帧,JPEG 写入一个**临时目录**(`tempfile.mkdtemp`)。
- 产出数据结构:`List[(frame_index, time_seconds, frame_path)]`。
- 时间戳计算 `time = index * 0.3`,因此 frame 0 → 0.0s,frame 84 → 25.2s(与 `detections.json` 第一条 `time: 25.2` 吻合)。
- 临时帧默认在结束时 `shutil.rmtree` 删除,除非加 `--keep-frames`。
- `--limit-seconds N` 在此处截断帧列表(只取前 N 秒,用于快速调参)。

### 阶段 2:检测 + 镜头切换标记 — 主循环([main.py:495](./src/main.py#L495))

对每一帧依次做两件事:

**(a) 镜头切换检测** — `compute_hsv_hist`([main.py:104](./src/main.py#L104))
计算本帧 HSV(H、S)直方图,与**上一帧**直方图算相关性 `compareHist`。相关性 `< scene_cut_threshold`(0.6)则标记 `is_cut[i]=True`。
- 产出 `is_cut: List[bool]`,`is_cut[i]` 表示第 i-1 帧和第 i 帧之间发生了切换。
- 第一帧无前帧,固定为 `False`。

**(b) 人脸检测** — `detector.detect`([detectors.py:122](./src/detectors.py#L122))
调用 `imgutils.detect.detect_faces`(YOLOv8 动漫脸模型,`level='s'`、`version='v1.4'`、`conf_threshold=0.5`)。每个框包装为 `Detection` 对象(frame_index、time、bbox、confidence、label="anime_face")。

### 阶段 3:过滤 — 三道质量门槛 `passes_quality`([main.py:130](./src/main.py#L130))

对每个原始检测,先补 `blur_var = laplacian_variance(...)`(裁剪区域的拉普拉斯方差,衡量清晰度),再过三关:

1. `confidence ≥ 0.5`(`conf_threshold`)
2. 人脸框高度 `≥ 0.045 × 帧高`(`min_face_height_ratio`,丢弃太小/太远的脸)
3. `blur_var ≥ 50.0`(`blur_var_threshold`,丢弃模糊/运动拖影)

- **每个原始检测**(无论是否通过)都写入 `detection_records`,带 `kept: true/false` → 即 `detections.json` 的 2895 条。（4680 帧抽出来 → 其中 2074 帧有脸 → 总共 2895 个人脸框）
- **仅通过的**进入 `frame_detections: List[List[Detection]]`(每帧一个列表)送去跟踪。
- 通过过滤的帧顺带存入 `viz_candidates`,供 `--viz` 随机采样画框。

### 阶段 4:跟踪 — `track_faces`([main.py:165](./src/main.py#L165)) + `assign_representatives`([main.py:246](./src/main.py#L246))

把逐帧人脸框沿时间串成**轨迹(Track)**。一条轨迹 = 同一张脸的一次连续出现。

逐帧推进,核心逻辑:

- **断轨**([main.py:203](./src/main.py#L203)):某活跃轨迹丢帧超过 `track_gap_tolerance`(1 帧),**或**它最后一帧到当前帧之间发生了镜头切换,即封存(finalize)。
  → 这就是"镜头切换强制断轨"的来源:同一位置切到不同角色会被算作新事件。
- **贪心 IoU 匹配**([main.py:211](./src/main.py#L211)):当前帧的框与活跃轨迹最后一个框算 IoU,`≥0.3`(`iou_threshold`)且 label 相同才能接上;按 IoU 从高到低贪心配对,每条轨迹/每个框只用一次，避免“一个框被两条轨迹抢”或“一条轨迹接两个框”。
- **新轨迹**([main.py:233](./src/main.py#L233)):未匹配任何轨迹的框,说明是一张新出现的脸,开一条新轨迹。

封存的 Track 记录起止时间与所有成员检测,最后按 `start_time` 排序 → **285 条轨迹**。

`assign_representatives`:每条轨迹挑一个"代表帧"——使 `blur_var × confidence` 最大的检测(最清晰且最自信),从对应采样帧裁出脸,写成 `crops/track_<id>.jpg`。

- 产出 → `tracks.json`(285 条)+ `crops/`(285 张)。
- 例:track_1 起于 25.2s、止于 26.1s、4 个检测、代表帧 86(25.8s),与 `detections.json` 开头连续帧吻合。

### 阶段 5:选段 — `select_segments`([main.py:281](./src/main.py#L281))

只用每条轨迹的 `start_time`(轨迹"起点" = 一次人脸事件)。

- 所有 start_time 排序,候选窗口起点 `t = k × 0.3（抽帧间隔）` 步进。
- 对窗口 `[t, t+15)`,用 `bisect` 数有多少轨迹起点落入([main.py:307](./src/main.py#L307))。
- `count ≥ min_events_per_window`(13)→ 窗口合格,输出片段 `[t, t+15]`,记录 `event_count` 与 `track_ids`;随后**跳到 ≥ t+15** 保证片段不重叠([main.py:323](./src/main.py#L323));否则 `k += 1` 继续滑动。

产出 `segments` 列表 + `num_qualified` 计数 → 写入 `windows.json`。

本次结果:285 条轨迹扫出 **5 个合格窗口/片段**,例如:

| 片段 | 区间 | 事件数 | track_ids |
|------|------|--------|-----------|
| 1 | 66.3 – 81.3s | 13 | 11–23 |
| 2 | 414.0 – 429.0s | 13 | 50–62 |
| 3 | 551.7 – 566.7s | 21 | 89–109 |
| 4 | 1061.1 – 1076.1s | 15 | 215–229 |
| 5 | 1296.9 – 1311.9s | 13 | 273–285 |

### 阶段 6:截取 — `clip_segments`([main.py:349](./src/main.py#L349))

对每个片段用 ffmpeg 从**原视频**(非抽出的帧)重新编码切 15 秒:

- `-ss start -t 15 -c:v h264_nvenc -c:a aac`,帧精确。
- 优先 GPU 编码器 `h264_nvenc`,失败自动回退 CPU `libx264`([main.py:357](./src/main.py#L357))。
- 产出 → `clips/clip_001.mp4` … `clip_005.mp4`(对应 `windows.json` 的 `clips` 字段)。

---

## 五、参数速查表(`src/config.py`)

| 参数 | 默认值 | 作用 | CLI 覆盖 |
|------|--------|------|----------|
| `frame_interval` | 0.3 | 抽帧间隔(秒),决定事件时间分辨率 | `--frame-interval` |
| `scene_cut_threshold` | 0.6 | HSV 相关性低于此值判为镜头切换 | `--scene-cut` |
| `conf_threshold` | 0.5 | 检测置信度下限 | `--conf` |
| `min_face_height_ratio` | 0.045 | 人脸最小高度占比 | — |
| `blur_var_threshold` | 50.0 | 清晰度下限(拉普拉斯方差) | `--blur-var` |
| `iou_threshold` | 0.3 | 相邻帧连成同一轨迹的 IoU 下限 | — |
| `track_gap_tolerance` | 1 | 轨迹关闭前允许连续丢帧数 | — |
| `window_seconds` | 15.0 | 片段长度(秒) | — |
| `min_events_per_window` | 13 | 窗口合格所需的轨迹起点数 | `--min-events` |
| `encoder` / `encoder_fallback` | h264_nvenc / libx264 | 视频编码器及回退 | `--encoder` |

调试用参数:`--limit-seconds`(只处理前 N 秒)、`--viz N`(导出 N 张标注帧)、`--keep-frames`(保留临时帧)。

---

## 六、值得注意的细节

1. **两个时间精度互不影响**:抽帧间隔 0.3s 决定所有"事件起点"的时间分辨率,但最终切片是从**原视频**帧精确截取,所以片段画质不受抽帧影响。

2. **`detections.json` 从 frame 84 开始**:前约 25 秒(片头/黑屏等)检测器没有返回任何框(或返回的都低于 `conf_threshold` 被检测器内部丢弃),因此没有记录。这是正常现象。

3. **`min_events_per_window=13` 是最关键的调参旋钮**:285 条轨迹只产出 5 个片段,说明门槛偏严。调低可出更多片段(`--min-events`)。

4. **GPU 依赖**:检测走 ONNX(可能用 GPU),编码默认 `h264_nvenc`(NVIDIA GPU)。无 GPU 时编码自动回退 `libx264`,检测则取决于 onnxruntime 安装的 provider(`_report_providers` 会打印一次,见 [main.py:413](./src/main.py#L413))。

5. **可扩展性**:检测器通过注册表(`@register`)插拔,下游跟踪/选段/截取只消费 `Detection` 对象,换成检测动物/物体等只需新增一个 `Detector` 子类(见 [detectors.py](./src/detectors.py) 模块说明)。
