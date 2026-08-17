# 动漫脸剪辑器 — 数据流程说明

> 本文说明 `src/` 下这套代码的数据流向、各阶段处理逻辑,以及输入/输出。
> 结合了真实运行产物 [`output/1/`](./output/1/) 中的数据。

---

## 一、总览

这是一个**动漫脸剪辑器**:给一个动漫视频,自动找出"角色密集出场"的 15 秒片段并切出来。

判定标准:一个 15 秒窗口里如果**出现过 ≥13 个不同角色**,就算合格片段。角色身份由 CCIP(动漫角色相似度模型)对轨迹代表裁剪图做 complete-linkage 层次聚类得到;"出现过"指轨迹时间区间与窗口相交(包括窗口开始前就在画面中的角色),同一角色的多条轨迹只计一次。

入口链路(见 [`src/main.py`](./src/main.py)):

```
main()  →  run_pipeline()  →  process_video()  →  scan_video()   ← 唯一的抽帧循环
                                                     └─ detect_cuts()  ← 另一趟全帧率解码,只出切镜时刻表
```

整个流程分为 7 个阶段:

```
流式解码 → 检测 → 过滤 → 跟踪 → 角色识别 → 选段 → 截取
```

**流式处理**:帧由 `cv2.VideoCapture` 顺序解码,用完即弃,**全程不落盘**。内存中同时存在的图像只有"当前帧 + 每条活跃轨迹一张人脸裁剪图",与视频长度无关(实测帧数 ×6 峰值内存仅 +3%)。

---

## 二、输入与输出

### 输入

| 项 | 说明 |
|----|------|
| 视频文件 | [`data/1.mp4`](./data/1.mp4),时长 **1403.948 秒**(约 23 分钟) |
| 默认命令 | `python src/main.py`(不传参即处理 `data/1.mp4`,输出到 `output/`) |
| 带可视化 | `python src/main.py data/1.mp4 --viz 8`(额外导出 8 张标注样本帧) |

### 输出(全部落在 `output/1/`,目录名 `1` 来自视频文件名 stem)

> 片段数量为当前默认参数(`ccip_threshold=0.05`)下的实测;检测/轨迹/裁剪图数量与选段规则无关。

| 产物 | 内容 | 本次实际数量 |
|------|------|--------------|
| `detections.json` | 每个原始检测框一条记录(含 `kept` 是否通过过滤) | **2925 条**(2408 条通过) |
| `tracks.json` | 每条人脸轨迹一条记录(含 `character_id`) | **289 条** |
| `windows.json` | 总摘要 + 选中片段 + 本次所用参数 | 5 个片段 |
| `crops/track_<id>.jpg` | 每条轨迹一张代表裁剪图 | **289 张**(共 14MB) |
| `clips/clip_<NNN>.mp4` | 切出的 15 秒视频 | **5 个** |
| `viz/sample_<NNN>.jpg` | 标注样本帧(全片蓄水池采样),仅 `--viz N` 时生成 | 8 张 |
| `sweep.csv` | 阈值敏感性扫描结果,仅 `sweep.py` 生成(落在 `output/` 根下) | — |

---

## 三、数据流向(一图串起来)

```
data/1.mp4
  │ ⓪ ffmpeg scdet 全帧率扫一趟 ← 独立的一趟解码,不抽帧,只出时刻表
  ▼ cuts = [t1, t2, ...]  (切镜时刻,秒)
  │
  │ ① cv2.VideoCapture 顺序解码,每 0.3s 取一帧
  ▼ (idx, time, frame_bgr)                   ← 生成器,一次只有一帧在内存
  │ ┌──────────── 同一趟循环内完成 ②③④ ────────────┐
  │ ② 查 cuts 表(上一采样帧~本帧之间有切镜?)→cut ; YOLOv8检测→Detection[]
  │ ③ 三道质量门槛 (置信度 / 人脸大小 / 清晰度) + 当帧就地裁脸
  │    detection_records ──────────────────────────► detections.json (2925)
  │ ④ FaceTracker.update(): IoU 匹配 + 镜头切换断轨 + 代表裁剪图擂台
  │ └──────── 帧在此失去引用,内存立即回收 ────────┘
  ▼ tracks (289) + crops/*.jpg (289)
  │ ⑤ CCIP 分批提取代表裁剪图特征 → complete-linkage 层次聚类 → 每条轨迹得 character_id
  ▼ tracks (带 character_id) ──► tracks.json
  │ ⑥ 15s 滑窗统计"出现过"(区间相交)的不同角色数 ≥13
  │    合格窗口的起点吸附到 cuts 里最近的切镜点(≤2s),并在新位置复核角色数
  ▼ segments ──► windows.json
  │ ⑦ ffmpeg 从原视频帧精确切片
  ▼ clips/clip_00X.mp4
```

**⓪ 为什么要单独一趟**:scdet 必须看到每一帧才能建立"帧间差异的跳变"这个状态量,0.3s 抽帧下镜头内的正常演进已与真切镜混叠。代价实测 6.4s/300s(24 分钟一集约 +31s),相对主循环里 YOLOv8 跑约 5000 帧属于噪音。

---

## 四、逐阶段详解

### 阶段 1:流式解码 — `iter_frames`([main.py:93](./src/main.py#L93))

- 用 `cv2.VideoCapture` 顺序解码,按 `frame_interval=0.3` 采样,**产出内存中的 BGR 数组**,不写任何中间文件。
- 用 `grab()`/`retrieve()` 而非 seek:长 GOP 的 H.264 上 seek 会被吸附到关键帧,既不准也不快。`grab()` 跳过不需要的帧的色彩转换与内存拷贝,`retrieve()` 只在命中采样点时才真正取出图像。
- 时间戳取 `raw_index / fps`(真实 PTS),而不是"采样序号 × 间隔"——后者在 fps 非整除时会随视频长度线性漂移。例:23.976fps 的素材第一条检测在 **25.234s**,旧算法给的是 25.2s。
- `--limit-seconds N` 在此处**停止解码**(而非解码完再过滤),调参时省下的是真实时间。
- 打不开视频或拿不到有效 fps 时直接抛 `RuntimeError`,不静默产出空序列。

### 阶段 2:检测 + 镜头切换标记 — `scan_video` 主循环([main.py:781](./src/main.py#L781))

**(a) 镜头切换检测** — `detect_cuts`([main.py:644](./src/main.py#L644)) + `_cut_between`([main.py:688](./src/main.py#L688))

主循环开始之前,先用 ffmpeg 的 **`scdet` 滤镜**全帧率扫一趟,拿到一张切镜时刻表:

```
ffmpeg -hide_banner -nostats -loglevel info -i <video> -an -vf scdet=threshold=10 -f null -
```

从 stderr 里按 `lavfi.scd.time: <秒>` 正则抓出时刻,排序后返回 `List[float]`。主循环里每帧只做一次二分查找:上一采样帧与本帧之间(区间 `(prev_time-tol, time+tol]`)夹着切镜时刻则 `cut=True`;第一帧无前帧,固定为 `False`。

- **fail fast**:ffmpeg 返回非零直接抛 `RuntimeError` 并带上日志尾部,不静默降级成空列表。空列表本身是合法结果——素材里确实存在整整 30 秒零切镜的片段。
- **为什么需要 `cut_time_tolerance`(0.1s)**:scdet 报的是真实 PTS,而 `iter_frames` 的时间戳是 `raw_index / fps`。实测这批素材上两者末帧误差 ≤0.18s 且**不累积**(`cv2.CAP_PROP_FPS` 返回的是平均帧率而非 `r_frame_rate`,VFR 不会造成累积漂移);叠加溶解转场本身也没有单帧答案,需要一点容差。往灵敏侧放是对的,理由见下方"误差不对称"。
- **阈值默认 10**:手绘作画素材上逐帧人工验证过(魔女之旅 9/9、EVA_1730 11/11 全真)。CG 动作素材(如凡人修仙传)在 10 上偏保守,用 `--scdet-threshold 5`。
- 这一节替换掉了旧的 HSV 直方图方案,原因与实测数字见下文"六、值得注意的细节"第 8 条。

**(b) 人脸检测** — `detector.detect`([detectors.py:123](./src/detectors.py#L123))
调用 `imgutils.detect.detect_faces`(YOLOv8 动漫脸模型,`level='s'`、`version='v1.4'`、`conf_threshold=0.5`)。每个框包装为 `Detection` 对象(frame_index、time、bbox、confidence、label="anime_face")。
- 检测器**吃的是 `PIL.Image.Image`,不是路径**——流式下帧不落盘,只能传内存对象。`to_pil`([main.py:145](./src/main.py#L145)) 负责 BGR→RGB 转换;`PIL.Image.Image` 是 imgutils `ImageTyping` 的成员,NumPy 数组不是。

### 阶段 3:过滤 — 三道质量门槛 `passes_quality`([main.py:195](./src/main.py#L195))

对每个原始检测,先用 `crop_bbox`([main.py:168](./src/main.py#L168)) **当帧就地裁出脸**,再补 `blur_var = laplacian_variance(crop)`,然后过三关:

1. `confidence ≥ 0.5`(`conf_threshold`)
2. 人脸框高度 `≥ 0.045 × 帧高`(`min_face_height_ratio`,丢弃太小/太远的脸)
3. `blur_var ≥ 50.0`(`blur_var_threshold`,丢弃模糊/运动拖影)

- **裁剪必须在当帧完成**:帧一丢就没有第二次机会回头读。裁剪图挂在 `Detection.crop` 上带进跟踪阶段。
- `crop_bbox` 返回的是 `.copy()` 而非 numpy 切片视图——切片会拖住整帧不放,一个引用就抵消掉全部流式内存优势。
- 没过质量门槛的框立刻 `det.crop = None` 释放(不会成为代表)。
- **每个原始检测**(无论是否通过)都写入 `detection_records`,带 `kept: true/false` → 即 `detections.json` 的 2925 条。（4680 帧采样 → 其中 2087 帧有脸 → 总共 2925 个人脸框,2408 个通过三道门槛）
- `--viz N` 用**蓄水池采样**从全片均匀抽 N 张标注帧;流式下不知道总帧数,且样本当场压成 JPEG 字节(≈200KB/张)才进池子,不留整帧(1080p ≈ 6MB/张)。

### 阶段 4:跟踪 — `FaceTracker`([main.py:225](./src/main.py#L225)) + `save_representatives`([main.py:366](./src/main.py#L366))

把逐帧人脸框沿时间串成**轨迹(Track)**。一条轨迹 = 同一张脸的一次连续出现。

跟踪器是**在线**的(`update()` 每帧调用一次),这是流式化的关键:代表裁剪图要挑"该轨迹内 `blur_var × confidence` 最大"的那张,而这在轨迹结束前无从判定。批处理版可以等全部跑完再回头读那一帧,流式下帧已经丢了。于是改为**边跟踪边擂台**——每条活跃轨迹只留当前最优的一张裁剪图(`_offer`,[main.py:260](./src/main.py#L260)),来了更好的就换掉,落选的立刻解引用。内存占用因此只与**同时活跃的轨迹数**成正比,与视频长度无关。

逐帧推进,核心逻辑:

- **断轨**([main.py:286](./src/main.py#L286)):某活跃轨迹丢帧超过 `track_gap_tolerance`(1 帧),**或**它最后一帧到当前帧之间发生了镜头切换,即封存(finalize)。切换标记用一个 `blocked` 标志在线传播,等价于批处理版回看 `is_cut[]` 区间。
  → 这就是"镜头切换强制断轨"的来源:同一角色每次重新出场都是一条新轨迹(角色去重由阶段 5 负责)。
- **贪心 IoU 匹配**([main.py:309](./src/main.py#L309)):当前帧的框与活跃轨迹最后一个框算 IoU,`≥0.3`(`iou_threshold`)且 label 相同才能接上;按 IoU 从高到低贪心配对,每条轨迹/每个框只用一次，避免“一个框被两条轨迹抢”或“一条轨迹接两个框”。
- **新轨迹**([main.py:244](./src/main.py#L244)):未匹配任何轨迹的框,说明是一张新出现的脸,开一条新轨迹。

封存的 Track 记录起止时间与所有成员检测,最后按 `start_time` 排序 → **289 条轨迹**。

`track_faces`([main.py:345](./src/main.py#L345)) 是喂满整个列表的批处理入口,只给单测用;主流程直接逐帧调 `update()`。

`save_representatives`:把跟踪时已选定的裁剪图写成 `crops/track_<id>.jpg`,然后释放数组引用。写入前先清空目录,保证重跑幂等(换参数往往产生更少轨迹,留着上一轮的 `track_N.jpg` 会与 `tracks.json` 对不上)。

- **中文路径**:落盘走 `imwrite_unicode`([main.py:150](./src/main.py#L150)) 而非 `cv2.imwrite`。后者在 Windows 上按 ANSI 代码页处理路径,遇到中文目录名**静默返回 `False`**——中文片名的视频会导致所有裁剪图丢失、所有 `character_id` 变成 `None`、最终一个片段都选不出来。绕开方式是 `cv2.imencode` + `numpy.tofile`。
- 产出 → `tracks.json`(289 条)+ `crops/`(289 张)。
- 例:track_1 起于 25.234s、止于 25.817s、3 个检测、代表帧 86(25.817s)。

### 阶段 5:角色识别 — `assign_characters`([main.py:477](./src/main.py#L477)) + `_cluster_by_difference`([main.py:390](./src/main.py#L390))

给每条轨迹一个**角色身份**(`character_id`),让后续选段能按"不同角色"去重计数:

- 对每条有代表裁剪图的轨迹,用 `imgutils.metrics` 的 **CCIP**(动漫角色相似度模型)提取 `crops/track_<id>.jpg` 的特征,再算两两差异矩阵(`compute_ccip_differences`,[main.py:429](./src/main.py#L429))。特征**分批提取**(每批 32 张),避免一次性送入几百张图导致 ONNX 推理内存分配失败。
- 特征提取与合并阈值无关,因此单独拆成一个函数——阈值扫描(见 `sweep.py`)可以只算一次、复用到所有阈值上。
- 对差异矩阵做 **complete-linkage(全连接)层次聚类**(scipy `linkage` + `fcluster`):簇内**任意两张**裁剪图差异都 `< ccip_threshold`(0.05)才允许同簇,同一簇 = 同一角色,簇编号即 `character_id`。**不做传递合并**——a~b 且 b~c 但 a、c 差异超阈值时,a、c 不会同簇(旧实现用并查集传递合并,差异链会把全片轨迹塌缩进一个簇,已弃用)。
- 没有代表裁剪图(或文件缺失)的轨迹保持 `character_id=None`,**不参与角色计数**(无法确认身份就不算一个角色)。
- 首次运行会从 HuggingFace 下载 CCIP ONNX 模型(一次性)。

产出:每条轨迹带 `character_id` → 写入 `tracks.json`;日志打印识别出的角色总数。（默认 `ccip_threshold=0.05` 下为 **198**——注意这与 289 条轨迹已经很接近,说明这个阈值下角色去重几乎没有生效,详见下文敏感性扫描）

### 阶段 6:选段 — `select_segments`([main.py:528](./src/main.py#L528))

统计窗口内**出现过的不同角色数**("出现过" = 轨迹时间区间与窗口相交,包括窗口开始前就在画面中的角色)。

- 轨迹按 `start_time` 排序,候选窗口起点 `t = k × 0.3（抽帧间隔）` 步进。
- 对窗口 `[t, t+15)`,用 `bisect` 取所有 `start_time < t+15` 的前缀,再过滤 `end_time ≥ t`,得到与窗口相交的轨迹(`_characters_in_window`,[main.py:498](./src/main.py#L498))。
- 相交轨迹中不同 `character_id` 的数量(`None` 不计)`≥ min_events_per_window`(13)→ 窗口合格,输出片段,记录 `character_count`、`character_ids` 与 `track_ids`(窗口内相交的全部轨迹);随后**跳到 ≥ 片段终点** 保证片段不重叠([main.py:585](./src/main.py#L585));否则 `k += 1` 继续滑动。

**边界吸附**(`_nearest_cut`,[main.py:518](./src/main.py#L518)):候选起点只是任意的 `k × 0.3s`,大概率切在镜头中间。合格窗口拿到后,起点会吸附到 `clip_snap_max_shift`(2.0s)以内最近的切镜点——实测平均镜头长 2.8~3.7s,默认值意味着绝大多数片段都会被吸附。

- 吸附后**必须在新位置重新统计角色数**(复用同一个 `_characters_in_window`),仍达标才采用,否则保持原起点。这不是可选的:`windows.json` 会记录每个片段的 `character_count` 和 `character_ids`,不重算就会写出与实际片段不符的数字。
- 吸附点会让窗口越过视频末尾时同样放弃。
- 贪心跳转用的是**吸附后**的终点,片段仍然互不重叠。
- `clip_snap_max_shift = 0` 关闭吸附;`cuts` 为空(未传或素材确实无切镜)时行为与吸附引入前完全一致。

产出 `segments` 列表 + `num_qualified` 计数 → 写入 `windows.json`。

**默认参数实测**(`ccip_threshold=0.05`、`min_events_per_window=13`):289 条轨迹聚成 **198 个角色**,扫出 **5 个合格窗口/片段**:

| 片段 | 区间 | 窗口内角色数 |
|------|------|------|
| 1 | 63.3 – 78.3s | 13 |
| 2 | 551.7 – 566.7s | 13 |
| 3 | 566.7 – 581.7s | 18 |
| 4 | 1061.1 – 1076.1s | 14 |
| 5 | 1297.2 – 1312.2s | 13 |

5 个片段中有 4 个刚好卡在门槛线上(13~14),说明结果对 `min_events_per_window` 的取值极为敏感——门槛动一格,出片数量就大幅变化。

### 阶段 7:截取 — `clip_segments`([main.py:611](./src/main.py#L611))

对每个片段用 ffmpeg 从**原视频**重新编码切 15 秒:

- `-ss start -t 15 -c:v h264_nvenc -c:a aac`,帧精确。
- `start` 已在阶段 6 吸附到镜头边界,所以切出来的片段以一个完整镜头开头,而不是从某个镜头的中间起。
- 优先 GPU 编码器 `h264_nvenc`,失败自动回退 CPU `libx264`([main.py:620](./src/main.py#L620))。
- 产出 → `clips/clip_001.mp4` … `clip_003.mp4`(对应 `windows.json` 的 `clips` 字段)。

---

## 四之二、阈值敏感性扫描 — [`src/sweep.py`](./src/sweep.py)

本流水线的两个关键参数会**互相补偿**:`ccip_threshold` 调严 → 同一角色被拆成多个身份 → 角色数虚高 → 更容易越过 `min_events_per_window` 门槛。两个参数一起动时,最终片段数可以在几倍范围内摆动,而这与"画面里到底有几个角色"没有必然关系。

`sweep.py` 在 `ccip_threshold × min_events_per_window` 的网格上重跑聚类与选段,把这个波动量化出来:

```powershell
$py = "D:\Programs\DevEnvironments\Anaconda\anaconda3\envs\myenv\python.exe"
& $py src/sweep.py data/1.mp4
& $py src/sweep.py data/1.mp4 --limit-seconds 300
& $py src/sweep.py data/1.mp4 --ccip 0.05,0.1,0.178 --min-events 5,8,13
```

视频只解码一次、CCIP 特征只提取一次,网格上每个格子只是重跑一次层次聚类和滑窗计数,几乎免费——所以格子多不影响耗时。产出控制台表格 + `output/sweep.csv`。

**全片实测(`data/1.mp4`,1404 秒,289 条轨迹)**,格内为片段数:

|   ccip | 角色数 | ME=5 | ME=8 | ME=13 |
|-------:|------:|-----:|-----:|------:|
|  0.050 |   198 |   33 |   16 |     5 |
|  0.100 |   114 |   30 |   12 |     2 |
|  0.178 |    53 |   26 |    7 |     1 |
|  0.250 |    25 |   21 |    4 |     1 |

**只动这两个旋钮,角色数在 25~198 之间摆动(8 倍),片段数 1~33(33 倍)。** 而 289 条轨迹在当前默认 `ccip_threshold=0.05` 下被判成 198 个角色——几乎每条轨迹一个身份,等于角色去重基本没生效;把阈值放到模型自带的 0.178,同一批轨迹只剩 53 个角色。

跨度越大,说明当前参数越是在**支配结论本身**,而不是在测量画面里的角色数。换素材时先跑这张表,再定参数。

---

## 五、参数速查表(`src/config.py`)

| 参数 | 默认值 | 作用 | CLI 覆盖 |
|------|--------|------|----------|
| `frame_interval` | 0.3 | 采样间隔(秒),决定事件时间分辨率 | `--frame-interval` |
| `scdet_threshold` | 10.0 | ffmpeg scdet 阈值(0-100),调低检测到的切换更多;CG 动作素材建议 5 | `--scdet-threshold` |
| `cut_time_tolerance` | 0.1 | 切镜时刻与采样区间匹配的容差(秒) | — |
| `clip_snap_max_shift` | 2.0 | 片段起点吸附到切镜点的最大位移(秒),设 0 关闭 | — |
| `conf_threshold` | 0.5 | 检测置信度下限 | `--conf` |
| `min_face_height_ratio` | 0.045 | 人脸最小高度占比 | — |
| `blur_var_threshold` | 50.0 | 清晰度下限(拉普拉斯方差) | `--blur-var` |
| `iou_threshold` | 0.3 | 相邻帧连成同一轨迹的 IoU 下限 | — |
| `track_gap_tolerance` | 1 | 轨迹关闭前允许连续丢帧数 | — |
| `ccip_threshold` | 0.05 | 簇内任意两张裁剪图 CCIP 差异都低于此值才视为同一角色;设为 None 用模型自带阈值(≈0.178) | `--ccip-threshold` |
| `window_seconds` | 15.0 | 片段长度(秒) | — |
| `min_events_per_window` | 13 | 窗口合格所需的不同角色数 | `--min-events` |
| `encoder` / `encoder_fallback` | h264_nvenc / libx264 | 视频编码器及回退 | `--encoder` |

调试用参数:`--limit-seconds`(只处理前 N 秒,到点即停止解码)、`--viz N`(蓄水池采样导出 N 张标注帧)。

参数校准建议先跑 [`src/sweep.py`](./src/sweep.py)(见上文"阈值敏感性扫描"),而不是凭单次运行的数字定 `ccip_threshold` 与 `min_events_per_window`。

---

## 六、值得注意的细节

1. **两个时间精度互不影响**:采样间隔 0.3s 决定所有"事件起点"的时间分辨率,但最终切片是从**原视频**帧精确截取,所以片段画质不受采样影响。

1b. **流式化省掉了什么**:旧实现先用 ffmpeg 把全片抽成 JPEG 写入临时目录,再逐张 `imread` 读回。本素材实测 141 KB/张,4680 帧 ≈ **0.66GB 临时文件**(体积随分辨率变化),外加一轮多余的 JPEG 有损编码/解码(会轻微劣化送进检测器和 CCIP 的画质)。现在解码一次直接用,磁盘只剩 14MB 裁剪图。实测 `scan_video` 在 200 / 600 / 1200 帧下峰值 RSS 为 1215 / 1246 / 1253 MB——帧数增至 6 倍,内存仅增约 3%,增量基本只是 ONNX 模型的固定开销。

2. **`detections.json` 从 frame 84 开始**:前约 25 秒(片头/黑屏等)检测器没有返回任何框(或返回的都低于 `conf_threshold` 被检测器内部丢弃),因此没有记录。这是正常现象。

3. **`ccip_threshold` 与 `min_events_per_window` 是最关键的两个调参旋钮,且它们互相补偿**:CCIP 是按"角色图"训练的,这里输入的是纯脸部裁剪,模型自带阈值(≈0.178)偏松。聚类已改为 complete-linkage(簇内任意两两都要达标,不做传递合并),杜绝了差异链把全片轨迹塌缩成一个簇的问题。但在此基础上,`ccip_threshold` 单独一项就能让角色数摆动 8 倍(见上文扫描表:0.05→198 个角色,0.25→25 个),而 `min_events_per_window` 的门槛是拿角色数来比的——阈值调严会让角色数虚高,反过来更容易越过门槛。**两个参数中的任何一个都无法单独解释最终片段数**。换素材时先跑 `sweep.py` 看波动范围,再校准角色数,最后用 `--min-events` 控制出片多少(调低出片更多)。

4. **在没有逐窗口人工真值的情况下,单次运行的数字不是测量**。上面那张扫描表说明输出在很大程度上是所选阈值的函数。这套流水线适合用来**找候选片段供人工筛选**,不适合用来断言"这段视频里有 N 个角色"或"本片不存在满足条件的片段"——后者尤其做不到:检测、身份去重、计数任一环节漏一次就会放过真实片段,系统只能给出"未发现",无法证明"不存在"。

5. **GPU 依赖**:检测走 ONNX(可能用 GPU),编码默认 `h264_nvenc`(NVIDIA GPU)。无 GPU 时编码自动回退 `libx264`,检测则取决于 onnxruntime 安装的 provider(`_report_providers` 会打印一次,见 [main.py:744](./src/main.py#L744))。

6. **Windows 中文路径**:`cv2.imwrite` 遇到中文目录名会静默失败(返回 `False`,不抛异常),写图一律走 `imwrite_unicode`。控制台输出走 `force_utf8_stdout`([main.py:701](./src/main.py#L701)),否则中文片名在 UTF-8 终端下是乱码。`cv2.VideoCapture` 本身对中文路径正常。

7. **可扩展性**:检测器通过注册表(`@register`)插拔,下游跟踪/选段/截取只消费 `Detection` 对象,换成检测动物/物体等只需新增一个 `Detector` 子类(见 [detectors.py](./src/detectors.py) 模块说明)。新增检测器的 `detect()` 收到的是 `PIL.Image.Image`,不是路径。

8. **切镜检测为什么不用直方图**:旧实现在 0.3s 抽帧上比较相邻两帧的 HSV(H、S)直方图相关性。在 `data/` 下 9 个片段上以 scdet 为参照实测(其中魔女之旅、EVA_1730、凡人修仙传_695s 三段已逐帧人工验证过对照图):

   | 指标 | 结果 |
   |---|---|
   | 整体召回 / 精确 | **51% / 51%**(71 个真切镜,命中 36、漏 35、误报 35) |
   | EVA_1270s / EVA_1730_30s | **0% / 0%**(0/12、0/12,完全失效) |
   | 爱情神话_925s | **0%**(0/3) |

   三个根因,受控实验逐条验证:

   - **无状态**:只比较相邻两帧,无法区分"变化大是因为切镜"和"变化大是因为整个镜头本来就在剧变"。合成测试中每帧全新随机纹理(0 个真切镜),HSV 报 47/47,scdet 报 1。真实素材上就是魔女的 16 个闪光误报、凡人的 83 个运动模糊误报。
   - **丢了 V 通道**:`calcHist([hsv], [0,1], ...)` 只取 H、S。纯白与纯黑落进同一 bin,相关性恒为 1.000——对亮度变化完全失明,这是 EVA 全灭的直接原因。补 V 能把 EVA 全帧率召回从 45% 提到 73%,但同时让魔女误报 20→25、凡人 83→147,且 0.3s 抽帧下依然 0%。两个失效方向互相拉扯,无解。
   - **采样率**:0.3s(约 7 帧)间隔下镜头内正常演进已与真切镜混叠。

   scdet 的 score 取「帧间差异」与「帧间差异相对上一次的跳变量」的**较小值**,持续剧变被压制、孤立尖峰才上报——这是第 1 条无法靠改通道或改阈值补上的**结构性**差异。实测切换后 EVA_1730_30s 的轨迹数 17 → 23(镜头切换强制断轨,原本粘连的轨迹被正确拆开)。

9. **依赖 ffmpeg 的非结构化日志**:`detect_cuts` 从 stderr 正则抓 `lavfi.scd.time`。这个 metadata key 自 scdet 滤镜引入以来稳定(当前环境 ffmpeg 8.1.1),但它终究是日志文本。returncode 非零即抛,不静默降级;**升级 ffmpeg 后若切镜数突然变 0,先查这条正则**。另:旧的 `--scene-cut` 参数已直接移除,旧命令行会报错退出而非静默忽略——这是刻意的。

10. **误差不对称,所以切镜宁多勿少**:漏一刀会把两个角色粘成一条轨迹、**静默丢掉一个角色**;多切一刀只是把一条轨迹断成两段,阶段 5 的 CCIP 聚类会把它们还原成同一个 `character_id`。`cut_time_tolerance` 因此往灵敏侧放。
