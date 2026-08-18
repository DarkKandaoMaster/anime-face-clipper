# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

回答用简体中文。

## 项目定位

本仓库是 `自动化识别项目不可行性说明.docx`（仓库根目录，git 忽略）里被判"技术不可达"的那个需求的**简化实现**：判断视频里是否存在**时长 ≤30 秒、出镜主体达到 X 人**的片段。该项目已重启，方向是**做一个粗糙的版本**，不是重新论证不可行。

两者的技术路线不同：不可行性说明里的实测用的是**行人 ReID**（OSNet / OSNet-IBN / PersonViT，四轮换后端+阈值，同一视频主体数预测在 1~10 之间摆动）；本仓库用的是**人脸检测 + 人脸/角色身份特征**，且按画风分三条支路（见下）。

文档列出的五环链路是本项目的技术地图，改代码时要清楚自己在动哪一环：

```
人物检测 → 镜头内跟踪/关联 → 跨镜头身份去重 → 滑窗内唯一身份计数 → 存在性判断
```

- 公开数据集不可用、自建数据集成本过高 → 真值只能靠**文件名里的人工标注**（见下）。
- 模型不跨风格泛化（真人 / 2D 动画 / 3D 动画）→ **已落地分风格路由**：`style.py` 判 2d / 3d / real，`config.StyleProfile` 成套换检测器 + 身份特征 + 阈值 + 裁剪口径 + 尺寸门槛。
- "谁算主体"无可计算依据 → 当前用**画面显著程度替代剧情重要程度**（清晰度 + 人脸框占比达标 = 主体，其余当路人）。长期思路是统计出镜时长 + 出现频次，但这依赖跨镜头身份去重准确。
- **跨镜头身份去重是当前最大难点**，也是"筛掉非正脸"的动机：非正脸的 CCIP 特征最不可靠。
- 存在性判断逻辑上无法证明"不存在"，只能给"未发现"。不要在输出或文档里把它说成"证明不存在"。

## 真值与素材

`data/` 下 9 段素材（git 忽略），**每段都恰好 30 秒**——即"一个 30 秒窗口"本身就是一个样本。文件名即人工标注：

```
魔女之旅_760s（人脸数期望：2+5+1+0+3+1）（跨镜头身份去重后期望：9）.mp4
         ↑源片位置    ↑逐镜头人脸数：6 个镜头       ↑全片不同角色数
```

- `人脸数期望` 按 `+` 分段 = 逐镜头的主体数，段数 = 镜头数。标注口径：正脸、出镜 >1s、表情清晰、人脸框宽度占画面宽度达标。标注时有大量模棱两可的脸，真值本身有噪声。
- `跨镜头身份去重后期望` = 该 30 秒内不同角色总数，**这是召回率评估的主要对照值**。
- 风格分布（分风格路由的现成测试集）：**真人** = 爱情神话 ×2；**3D** = 凡人修仙传 ×2；**2D** = JOJO / 高木同学 / EVA ×2 / 魔女之旅。

## 环境（本机已验证，2026-08-17）

```powershell
$py = "D:\Programs\DevEnvironments\Anaconda\anaconda3\envs\myenv\python.exe"   # 不要用 python，PATH 上不是它
```

- Python 3.12.13 / opencv 4.11 / numpy 1.26.4 / scipy 1.17.1 / onnxruntime 1.18.1（providers: Tensorrt、**CUDA**、CPU）/ ffmpeg + ffprobe 8.1.1。`h264_nvenc` 实测可用。
- HF 模型缓存在 `~/.cache/huggingface/hub`，已缓存：`deepghs/anime_face_detection`（仅 `face_detect_v1.4_s`）、`deepghs/ccip_onnx`、`deepghs/anime_classification`、**`public-data/insightface`（`models/buffalo_l/det_10g.onnx` = SCRFD-10G，`models/buffalo_l/w600k_r50.onnx` = ArcFace）**。改 `detector_level='n'` 或换 `detector_version` 会触发新下载。
- **直连 HuggingFace 当前是通的**（本次实测下载成功）。README 与旧记忆里"必须设 `HF_ENDPOINT=hf-mirror.com` + `NO_PROXY='*'` 否则 SSL 失败"的结论只在系统代理开启时成立；遇到 `SSL: UNEXPECTED_EOF_WHILE_READING` 再加这两个变量。
- 模型已缓存时加 `$env:HF_HUB_OFFLINE = '1'` 最稳（imgutils 每个新进程首次检测都会联网列模型清单，离线模式免掉这次请求）。实测离线可跑通全流程。
- PowerShell 读文件一律用 UTF-8，否则中文乱码。

### 命令

```powershell
& $py src/main.py "data/魔女之旅_760s（人脸数期望：2+5+1+0+3+1）（跨镜头身份去重后期望：9）.mp4" --min-events 5
& $py src/main.py <video> --limit-seconds 60 --viz 8   # 调参：只跑前 60s + 导出 8 张标注帧
& $py src/sweep.py <video>                             # 阈值敏感性网格扫描 → output/sweep.csv
& $py src/evaluate.py data/*.mp4                       # 召回率评估（出厂配置）→ output/recall.csv
& $py src/evaluate.py data/*.mp4 --threshold 0.178,0.2,0.85,0.95   # 阈值网格上扫召回
& $py src/evaluate.py data/*.mp4 --eyes 2              # 打开正脸过滤做对照
# 受控对照：关掉路由，把同一套模型/门槛压到全部素材，再看分风格那几列
& $py src/evaluate.py data/*.mp4 --no-style-routing --detector real_face_scrfd --embedder arcface --min-face-height 0.09
& $py src/main.py <video> --no-clip                     # 只分析不编码片段（长片批量实测用）
& $py src/main.py <video> --frontal-weight 1.0         # 正脸分给代表图擂台加权（默认 0=关）
& $py src/evaluate.py data/*.mp4 --frontal-weight 1.0 --min-frontal 0.3   # 正脸的两种用法做对照
& $py src/evaluate_long.py                             # 片长尺度对照：全片聚类 vs 窗口内聚类（真值 = 9 段切片在片源里的原位窗口）
& $py src/evaluate_long.py --titles 魔女之旅 --no-clip-baseline   # 只跑一部
& $py src/main.py <video> --cluster-scope window       # 聚类范围压回单个 30 秒窗口
& $py src/montage.py output/<stem> --clusters 18       # 按 character_id 拼接触印相表，肉眼核对聚类
& $py src/summarize.py output_raw                      # 汇总一批 windows.json：片段数 / 不重叠窗口上限
& $py -m pytest tests -q                               # 141 个纯逻辑单测，3s，不需要 GPU/模型/网络
& $py -m pytest tests/test_streaming.py::TestDetectCuts -q
& $py -m pytest "tests/test_main.py::TestSelectSegments::test_single_qualified_window" -q
```

**CLI 默认参数 `data/1.mp4` 已不存在**（README 里的全部实测数字都出自那个 23 分钟视频，现已不在 `data/`），必须显式传视频路径。

`tests/` 全部用合成数据（tmp_path 下现造小视频），改 IoU/聚类/选段/切镜逻辑先跑它。`tests/conftest.py` 把 `src/` 塞进 `sys.path`，因为 `main.py` 用**平铺导入**（`from config import Config`），**不是包**——新增模块沿用平铺导入。

## 架构

7 个阶段，主体在 `src/main.py`（约 1000 行，用 `# === N. 阶段名 ===` 分节）：

```
流式解码 → 检测 → 过滤 → 跟踪 → 角色识别 → 选段 → 截取
main() → run_pipeline() → process_video() → scan_video()   ← 唯一的抽帧循环
                                              └─ detect_cuts()  ← 另一趟全帧率解码，只出切镜时刻表
```

| 文件 | 职责 |
|---|---|
| `src/config.py` | 唯一的参数集中地（`Config` + `StyleProfile` dataclass），不做 I/O。新参数加在这里，不要散进逻辑 |
| `src/detectors.py` | `Detection` 数据契约 + `Detector` 抽象基类 + `@register` 名称注册表 + 裁剪几何（`crop_bbox` / `expand_bbox`）+ 两个实现：imgutils YOLOv8 动漫脸、InsightFace SCRFD-10G 真人脸（含 5 点关键点与 ArcFace 对齐）。**新检测器加在这里** |
| `src/embedders.py` | 身份特征注册表：`ccip`（动漫角色）与 `arcface`（真人脸）。统一接口 `differences(crop_paths) -> N×N 矩阵`，聚类阶段不关心是谁算的 |
| `src/main.py` | 七阶段流水线 + CLI |
| `src/sweep.py` | 在 `identity_threshold × min_events_per_window` 网格上重跑聚类与选段。视频只解码一次、身份特征只提一次，加格子几乎免费 |
| `src/style.py` | 画风路由：抽帧判 **2d / 3d / real** 三分类 + `apply_style` 成套覆盖 Config（检测器 / 特征 / 阈值 / 外扩 / 尺寸门槛）|
| `src/frontal.py` | 正脸评分（纯几何、无 I/O）：真人/3D 走 SCRFD 5 点关键点算偏航，2D 走动漫眼检测器的**框位置**。判据挂在 `Detector.frontal_score` 上（和检测器绑死）。**加权与硬筛实测都无收益，默认全关** |
| `src/montage.py` | 把一次运行的代表图按 `character_id` 拼成接触印相表（`--tracks` 可指向另一份聚类结果）。没有真值的完整片源只能这么核对聚类 |
| `src/summarize.py` | 汇总一批 `windows.json`：轨迹/角色/片段数，以及**片段数 ÷ 不重叠窗口上限**——长片上最该盯的一列 |
| `src/groundtruth.py` | 纯解析：文件名标注 → `GroundTruth`。不做 I/O、不碰视频 |
| `src/evaluate_long.py` | **片长尺度**的对照评估。9 段切片都是从 `data/原始数据/` 里按整秒原位截出的（逐帧比对确认 shift=0），所以文件名真值同时也是**完整片源上 `[offset, offset+30)` 这个窗口的真值**——这补上了 README 第六之五节说的"缺片长尺度标注"。画风强制用真值，避开长片路由判错这个已知缺陷 |
| `src/evaluate.py` | 召回率评估。复用 `sweep.py` 同款切分：扫一次视频、提一次特征，网格上只重跑聚类 |
| `README.md` | 逐阶段数据流 + 决策理由 + 多个被否掉的旧方案及其实测数据。**流水线行为变了要同步更新它**；但其中的实测数字属于已不存在的 `data/1.mp4`，别当现状引用 |

`scan_video` 停在轨迹层（不聚类、不选段），因为后续阶段与阈值无关、可在不重扫视频的前提下反复重跑——`sweep.py` 正是这么用的。改动时保持这个切分。

### 流式处理是硬约束

帧由 `cv2.VideoCapture` 顺序解码，用完即弃，**全程不落盘**。内存中同时只有"当前帧 + 每条活跃轨迹一张人脸裁剪图"，与视频长度无关（帧数 ×6 峰值内存 +3%）。由此产生的约束：

- **裁剪必须在检测的当帧就地完成**（`crop_bbox` 挂到 `Detection.crop`），帧一丢就没有第二次机会回头读。
- `crop_bbox` 必须返回 `.copy()`，numpy 切片视图会拖住整帧，一个引用抵消掉全部内存优势。
- 代表裁剪图靠**在线擂台**（`FaceTracker._offer`，比 `blur_var × confidence`）选出，不能等轨迹结束再回看。
- `--viz N` 用蓄水池采样，样本当场压成 JPEG 字节才进池子。
- 任何"需要回看某一帧"的新功能（**正脸/眼睛判定就是**）必须**在当帧算完、结果挂在 `Detection` 上带下去**，不能留到后面阶段再取像素。

### 其他必须遵守的点

- **fail fast**：打不开视频、拿不到 fps、ffmpeg 返回非零 → 直接 `RuntimeError` 带日志尾部，不静默降级成空结果。这是刻意的，不要加兜底分支。
- **Windows 中文路径**：写图一律走 `imwrite_unicode`（`cv2.imencode` + `numpy.tofile`）。`cv2.imwrite` 遇中文目录名**静默返回 False 不抛异常**，后果是裁剪图全丢 → `character_id` 全 None → 一个片段都选不出来。控制台输出走 `force_utf8_stdout`。`cv2.VideoCapture` 对中文路径正常。素材文件名全是中文，这条随时会踩。
- **切镜检测用 ffmpeg `scdet` 滤镜**，从 stderr 正则抓 `lavfi.scd.time`。别改回 HSV 直方图方案（实测召回/精确只有 51%，EVA 素材 0%，根因见 README 第七节第 8 条）。升级 ffmpeg 后切镜数突然变 0，先查这条正则。
- **切镜宁多勿少**：漏一刀会把两个角色粘成一条轨迹、静默丢掉一个角色；多切一刀只是把轨迹断成两段，身份聚类会还原。
- **读裁剪图必须用 `imread_unicode`**（`embedders.py`）：`cv2.imread` 遇中文目录名静默返回 None，和 `cv2.imwrite` 是同一个坑。
- **聚类是 complete-linkage，不做传递合并**：簇内任意两张裁剪图差异都要低于阈值。旧的并查集传递合并会让差异链把全片轨迹塌缩成一个簇，已弃用，不要回退。
- **`identity_threshold` 与 `min_events_per_window` 互相补偿**，任一个都无法单独解释最终片段数（旧素材上 0.05→198 个角色，0.25→25 个；片段数摆动 33 倍）。换素材或改门槛前先跑 `sweep.py`，不要凭单次运行的数字定参。
- **阈值跨 embedder 不可比**：CCIP ≈0.1~0.2，ArcFace（`1 - 余弦`）≈0.6~1.0。开着路由扫网格时，汇总表只有分风格那几列可读。
- **`min_face_height_ratio` 也是按画风走的**，不是全局常数。SCRFD 的框只到眉毛~下巴，比动漫框矮一大截，共用一个比例会让真人素材放进大量远景路人。
- 输出目录名取视频文件名 stem，`save_representatives` 写入前先清空 `crops/`，保证重跑幂等。

## 已落地的改造

### 第一轮：重启后的四项（2026-08-18 上午）

1. **窗口与门槛**：`window_seconds = 30`（需求口径，别再当可调参数）、`min_events_per_window = 4`（真值下界）。
2. **画风路由**：`src/style.py`。
3. **正脸过滤（第一代）**：`config.require_eyes` + `--eyes N`，已实现但**默认关闭（0），因为实测有害**。第三轮用更好的判据重做了一遍，结论不变，见下。detect_eyes 是动漫眼检测器，对真人/3D 几乎不触发（爱情神话_1525s 23 条轨迹 → 0），2D 上各自调优后 F1 也是 0.89 vs 0.82。**不要因为它"看起来该开"就把默认值改回 2**，README 第六之二节有完整 A/B。
4. **召回评估**：`src/evaluate.py` + `src/groundtruth.py`。

结果：macro 召回 0.96、macro F1 0.90。

### 第二轮：真人支路 + 分档门槛（2026-08-18 下午）

1. **真人与 3D CG 换整套模型**：`real_face_scrfd`（InsightFace SCRFD-10G）+ `arcface`（w600k_r50），代表裁剪图按 5 点关键点做 Umeyama 相似变换对齐到 112×112。SCRFD 后处理是自己写的（约 40 行），没引 `insightface` 包（它会拖进 albumentations/scikit-image 一整条依赖链）。
2. **画风路由扩到三分类**：2d / 3d / real。第一步仍是 `anime_classify` 投票分 2D / 非 2D（9/9）；第二步用**两个人脸检测器在抽样帧上的置信度之和之比**（anime / scrfd）分 3D CG 与真人，实测 2D ≥1.15、3D CG 0.59~0.64、真人 0.29~0.30。`anime_real` 分不开（把凡人修仙传_1040s 判成 real）。
3. **2D 代表裁剪图外扩 `crop_margin = 0.6`**：CCIP 按角色图训练，而检测框紧贴五官，发型（动漫身份最强线索）被裁在框外。
4. **`min_face_height_ratio` 按画风分档**：2D 0.03（救远景，EVA_1270s 从 5/7 变 7/7）、3D CG 0.045、真人 0.09（街景路人太多，爱情神话_925s 128 条轨迹 → 59 条）。
5. **`crops_per_track`（每轨迹多张代表图 + 中位数聚合）**：已实现但**默认 1，因为实测中性**（2D 0.89 vs 0.91、3D 0.97 vs 1.00）。榜单按 blur×conf 排序，前三名来自相邻几帧、高度冗余。README 第六之三节有 A/B。

### 第三轮：完整片源实测 + 正脸检测（2026-08-18 晚）

1. **`--no-clip` + `src/montage.py`**：长片实测的两件工具。前者跳过片段编码只写 JSON，后者把代表图按 `character_id` 拼成接触印相表——没有真值时唯一能核对聚类的手段。
2. **`src/frontal.py` 正脸评分**：真人/3D 用 SCRFD 5 点关键点纯几何算偏航（零成本、目视校准很干净），2D 用动漫眼检测器的**框位置**（不是旧的数眼睛）。主用法是给代表图擂台加权（`frontal_weight`），**不丢轨迹**。
   - **实测无收益，默认全关**：加权在 9 段素材上 2d 0.95→0.94、3d 0.97→0.97、real 0.94→0.92（抖动量级），长片上 `魔女之旅` 54→57 个角色、耗时 +27%；硬门槛（`min_frontal_score=0.3`）real F1 0.94→**0.50**，和 `require_eyes` 同一个失败模式。
   - **不要因为"判据这次是对的"就把默认值打开**。判据确实是对的（真人支路 0.9+ 全正脸、0.0 全侧脸），崩掉的是**用法**：真值只数"有几个不同角色"，某个角色可能全片只有侧脸出镜，丢掉它 = 直接少一个真值身份。正脸判定不该出现在任何会丢弃轨迹的位置。README 第六之四节有完整 A/B。

### ⚠️ 片长尺度上这套参数会塌（第三轮实测，README 第六之五节）

**上面所有数字都出自 9 段各 30 秒的素材**。压到 `data/原始数据/` 下 **17 段完整片源（共 7.3 小时）**上（`src/main.py data/原始数据/*.mp4 --no-clip --output-dir output_raw` + `src/summarize.py output_raw`）：

- 一趟跑完，**无崩溃 / 无内存问题**。吞吐 2D 约 **4 倍实时**（24 分钟 1080p 片 ≈ 363 秒），真人/CG 约 **8 倍**；流式内存约束在片长上成立。
- **`min_events_per_window = 4` 形同虚设：676 / 884 个不重叠窗口合格 = 76% 的片长被判为合格片段**（`亚托莉` 46/47）。唯一正常的是双人剧 `高木同学`（19/48），说明门槛没在筛选、只在反映素材的角色密度。
- **同一个角色被拆成 4~5 个身份**（`魔女之旅` 整集 54 个角色，白发主角一人占 5 个簇），拆分边界是**画质与姿态**（暗/糊/侧脸各自成簇），不是身份。
- 机制：complete-linkage 对离群点零容忍。788 条轨迹的差异矩阵上，1431 个簇对里有 **106 对的簇间中位距离已低于阈值**，只因极少数离群图的 max 越界而被拒绝合并。轨迹越多越容易撞上离群图，**所以这个失败模式随片长恶化，30 秒素材上根本不出现**。
- **画风路由在片长上判错**：`爱情神话`（真人实拍）整片证据比 0.77 → 判成 3d（30 秒切片是 0.29 → real）。整片探针会抽到大量远景/空镜/暗场，SCRFD 找不到脸导致分母塌陷、比值被推高，真人(0.77) 与 3D CG(0.92) 的间隔消失。后果是尺寸门槛用错一档。**别去重调 0.5 这个门槛，间隔已经窄到没有安全取值**，要换判据。
- 单调调阈值解决不了：0.32 重聚时同一张印相表上同时出现"仍在拆"和"已合错"；single-linkage 一步塌成 1 个簇，average 从 0.25 起过度合并。

### ⚠️ 四条修法已经实测过，三条无效——不要重复试

| 修法 | 结果 |
|---|---|
| 正脸分给代表图加权（`frontal_weight`） | ❌ 魔女之旅 54 → **57** 个角色，2D 还慢 27% |
| `crops_per_track = 3` | ❌ 54 → **60**。擂台前三名来自**相邻帧**（冗余非多视角），且 complete-linkage 卡的是**簇间**最大值，动不了轨迹内部的稳健性。为此加了 `crop_min_gap_frames`（默认 0）可以让 K 张图在时间上分开，但没解决根因 |
| 两段式分位数 linkage（严阈值出微簇 → 按簇间 90 分位合并） | ❌ 同等簇数下与"直接调高 complete 阈值"犯同一批错。**卡点在 CCIP 特征本身，不在 linkage 规则** |
| `min_track_seconds`（最短出镜时长） | ⚠️ 长片上唯一有效的（54 → **34**，主角从 5 簇合成 1 簇），但 30 秒真值集上单调变差（2d 0.95→0.91、3d 0.97→0.62、real 0.94→0.68），**默认 0** |
| **聚类范围压回 30 秒窗口**（`cluster_scope`） | ✅ **有效，已设为默认**。片长尺度 macro F1 0.79 → 0.90，平均 ratio 1.57 → 0.99；30 秒真值集恒等无变化。见第四轮 |

`min_track_seconds` 失败的原因指向一个具体设计错误：**轨迹被切镜强制打断，"一条轨迹的时长"不是"这个角色的出镜时长"**。角色演满 3 秒中间切两刀 = 3 条各 1 秒的轨迹，门槛 1.2 秒会把这个真值身份整个抹掉。标注写的"出镜 >1s"是**累计**出镜时长，是聚类之后才知道的量。按累计口径重算：魔女之旅 54 → 48（≥1s），爱情神话 146 → 125（≥1s）/ 81（≥3s）——**清的是垃圾（后脑勺、纯黑图那些簇），治不了过拆**。

还没试、值得试：⓪ 先修画风路由（唯一一个判错就全盘用错参数的环节，且已实测翻车，成本最低）；① 补代表图质量门槛（真人对齐图的黑边占比可直接从仿射矩阵算，不用看像素；以及整体亮度过低）；② `min_events_per_window` 按片长重定或改成相对口径；③ 把最短出镜时长改成**按角色**算（聚类之后、选段之前）；④ ~~先在场景/镜头邻域内合并、再跨场景合并~~ → **已按更粗的粒度做掉了（窗口内聚类，见第四轮）**；更细的场景邻域粒度仍可试，尤其针对窗内轨迹稀疏时合过头的那类误差。

**另一条结论同样重要：现在这套真值测不出片长尺度的问题。** 9 段各 30 秒的样本里几乎每条轨迹都对应一个真值身份，任何过滤都只会掉分。要继续往前走，先得有片长尺度的标注。→ **第四轮解决了这一条**：9 段切片本来就是片源的原位窗口，`src/evaluate_long.py` 直接拿它当片长尺度真值。

### 第四轮：聚类范围压回单个 30 秒窗口（2026-08-18 晚，README 第六之六节）

**这是目前唯一在片长尺度上真正见效、且在 30 秒真值集上零代价的改动，已设为默认。**

1. **先白捡了片长尺度的真值**：9 段切片都是从 `data/原始数据/` 按整秒**原位**截出的（逐帧比对确认全部 `shift=0`），所以文件名里的「跨镜头身份去重后期望」**同时就是完整片源上 `[offset, offset+30)` 这个窗口的真值**。这补上了第三轮小结里"现在这套真值测不出片长尺度的问题"缺的那一块，不用新标注。工具是 `src/evaluate_long.py`。
2. **`config.cluster_scope`（默认 `"window"`）**：需求只问"这 30 秒里有几个不同角色"，从不需要全片一致的身份编号。complete-linkage 被离群点卡住的概率随轨迹数增长，把 N 从"全片 788~1100 条"降到"窗内 6~91 条"，过拆就消失了。差异矩阵仍只算一次（全片），窗口聚类只是取子矩阵重跑 linkage（`IdentityIndex.count()`，按行号元组缓存），耗时无可测增加。
3. **`track.character_id` 仍是全片口径**（`crops/` 与 `montage.py` 唯一的身份线索）；片段里 `character_ids` 是全片 id 只作溯源，`character_count` 才是计数口径，两者在窗口口径下可以不等长。

结果（9 个片长尺度窗口）：**macro F1 0.785 → 0.902**（clip30 上限 0.955），**平均 ratio 1.57 → 0.99**（过拆基本消失），9 个里 8 个变好、1 个略差。合格片段占比 67% → 48%（6 部片源、2.4 小时）。

要点：
- **30 秒素材上两种口径恒等**（整段就是一个窗口），已实跑对比 `windows.json` 确认。所以 `evaluate.py` 的出厂基线一个数字都没动——**这次改动没法用旧真值集验证，必须跑 `evaluate_long.py`**。
- **失效方向反转了**：唯一变差的 `爱情神话_1525s` 窗内只有 6 条轨迹（4 → 3），是**合过头**而不是拆过头。轨迹越稀疏，子矩阵越小，约束越松。别把它和过拆混为一谈。
- **不要用"片段数下降"当成功指标**：门槛没变，只是喂进去的角色数不再虚高。问题一（`min_events_per_window` 在片长上不筛东西）没解决，48% 仍偏高。

### 当前基线（2026-08-18 实测，`src/evaluate.py data/*.mp4`）

出厂配置下 9 段素材：**macro 召回 0.98、micro 0.97、macro F1 0.96、画风路由 9/9（三分类）**，9 段里 6 段角色数与真值完全一致。分风格 F1：2D 0.95 / 3D CG 1.00 / 真人 0.92。逐次运行有 ±1 个角色的抖动（cuDNN 算法选择非确定，边界上的一对轨迹会翻面），量级在真值噪声之下。逐视频表格与阈值网格在 [README.md](./README.md) 第六之二节。要点：

- **选阈值看 F1 列，不看召回列**。数量召回被截断在 1.0，把一个角色拆成十个也是满分；必须和 `ratio`（未截断的 检出/真值）一起读。
- **选参数看平台宽度，不看单点峰值**。真人的 `min_face_height_ratio=0.09` 在阈值 0.80~0.95 一整段都 ≈0.92，比 0.06 上只在单点成立的 0.92 稳。
- **真人与 3D CG 的最优 ArcFace 阈值相同（0.85）**；此前"非 2D 共用一个阈值只能到 F1 0.83"是**尺寸门槛**造成的假象。
- 剩下没打满的三段是两个方向的误差：`JOJO_635s` 7/9 与 `高木同学_325s` 9/7（同一个 2D 阈值一边合过头一边拆过头），`爱情神话_925s` 14~15/11（街景路人）。
- 所有 profile 数字都是对 9 个样本拟合的（非 2D 只有 2+2 段），**换素材必须重跑 `evaluate.py --threshold` 网格重新定值**。
