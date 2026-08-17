# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

回答用简体中文。

## 项目定位

本仓库是 `自动化识别项目不可行性说明.docx`（仓库根目录，git 忽略）里被判"技术不可达"的那个需求的**简化实现**：判断视频里是否存在**时长 ≤30 秒、出镜主体达到 X 人**的片段。该项目已重启，方向是**做一个粗糙的版本**，不是重新论证不可行。

两者的技术路线不同：不可行性说明里的实测用的是**行人 ReID**（OSNet / OSNet-IBN / PersonViT，四轮换后端+阈值，同一视频主体数预测在 1~10 之间摆动）；本仓库用的是**动漫脸检测 + CCIP 角色相似度**。

文档列出的五环链路是本项目的技术地图，改代码时要清楚自己在动哪一环：

```
人物检测 → 镜头内跟踪/关联 → 跨镜头身份去重 → 滑窗内唯一身份计数 → 存在性判断
```

- 公开数据集不可用、自建数据集成本过高 → 真值只能靠**文件名里的人工标注**（见下）。
- 模型不跨风格泛化（真人 / 2D 动画 / 3D 动画）→ 需要**分风格路由**。
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
- HF 模型缓存在 `~/.cache/huggingface/hub`，**`deepghs/anime_face_detection`（仅 `face_detect_v1.4_s`）与 `deepghs/ccip_onnx` 已缓存**。改 `detector_level='n'` 或换 `detector_version` 会触发新下载。
- **直连 HuggingFace 当前是通的**（本次实测下载成功）。README 与旧记忆里"必须设 `HF_ENDPOINT=hf-mirror.com` + `NO_PROXY='*'` 否则 SSL 失败"的结论只在系统代理开启时成立；遇到 `SSL: UNEXPECTED_EOF_WHILE_READING` 再加这两个变量。
- 模型已缓存时加 `$env:HF_HUB_OFFLINE = '1'` 最稳（imgutils 每个新进程首次检测都会联网列模型清单，离线模式免掉这次请求）。实测离线可跑通全流程。
- PowerShell 读文件一律用 UTF-8，否则中文乱码。

### 命令

```powershell
& $py src/main.py "data/魔女之旅_760s（人脸数期望：2+5+1+0+3+1）（跨镜头身份去重后期望：9）.mp4" --min-events 5
& $py src/main.py <video> --limit-seconds 60 --viz 8   # 调参：只跑前 60s + 导出 8 张标注帧
& $py src/sweep.py <video>                             # 阈值敏感性网格扫描 → output/sweep.csv
& $py -m pytest tests -q                               # 82 个纯逻辑单测，1.7s，不需要 GPU/模型/网络
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
| `src/config.py` | 唯一的参数集中地（`Config` dataclass），不做 I/O。新参数加在这里，不要散进逻辑 |
| `src/detectors.py` | `Detection` 数据契约 + `Detector` 抽象基类 + `@register` 名称注册表 + imgutils YOLOv8 动漫脸实现。**风格路由要新增的检测器加在这里** |
| `src/main.py` | 七阶段流水线 + CLI |
| `src/sweep.py` | 在 `ccip_threshold × min_events_per_window` 网格上重跑聚类与选段。视频只解码一次、CCIP 特征只提一次，加格子几乎免费 |
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
- **切镜检测用 ffmpeg `scdet` 滤镜**，从 stderr 正则抓 `lavfi.scd.time`。别改回 HSV 直方图方案（实测召回/精确只有 51%，EVA 素材 0%，根因见 README 第六节第 8 条）。升级 ffmpeg 后切镜数突然变 0，先查这条正则。
- **切镜宁多勿少**：漏一刀会把两个角色粘成一条轨迹、静默丢掉一个角色；多切一刀只是把轨迹断成两段，CCIP 聚类会还原。
- **聚类是 complete-linkage，不做传递合并**：簇内任意两张裁剪图差异都要低于阈值。旧的并查集传递合并会让差异链把全片轨迹塌缩成一个簇，已弃用，不要回退。
- **`ccip_threshold` 与 `min_events_per_window` 互相补偿**，任一个都无法单独解释最终片段数（旧素材上 0.05→198 个角色，0.25→25 个；片段数摆动 33 倍）。换素材或改门槛前先跑 `sweep.py`，不要凭单次运行的数字定参。
- 输出目录名取视频文件名 stem，`save_representatives` 写入前先清空 `crops/`，保证重跑幂等。

## 重启后待做的改造（当前代码还没做到）

1. **窗口与门槛**：`window_seconds` 15.0 → **30**；`min_events_per_window` 13 → 目标 **X 人（素材实测落在 1~10）**。当前默认值在 30 秒素材上几乎选不出片段。
2. **风格路由**：均匀抽帧 → 投票判断真人 / 2D / 3D → 路由到对应模型与参数集。判断条件待定；`imgutils.validate.anime_real` 已验证可用（返回 `('anime', 0.98)` 这类二分类），**2D vs 3D 还没有现成分类器**，需要另想（`anime_style_age` 或自定手工特征）。插拔点是 `detectors.py` 的 `@register` 注册表。
3. **正脸过滤**：脸上没检出两只眼睛就排除。`imgutils.detect.detect_eyes` 已验证可用（对 crops 返回 2 只眼）。这是"粗糙化"的核心手段，目的是提高跨镜头去重的准确率。注意上面的流式约束：要在当帧算、结果挂在 `Detection` 上。
4. **召回率测量**：解析文件名里的标注当真值，算召回。这是重启后唯一有意义的验收指标，目前**没有任何评估脚本**。

### 当前基线（2026-08-17 实测，默认参数）

`魔女之旅_760s`（真值：6 个镜头 2+5+1+0+3+1，去重后 9 个角色）→ scdet 检出 **5 刀**（真值 6 镜头，对上了）、100 采样帧、**35 条轨迹**、`ccip_threshold=0.05` 下 **20 个角色**。跨镜头去重把 9 个角色拆成了 20 个（≈2.2 倍），这就是第 3 项改造要压下去的数字。
