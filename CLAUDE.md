# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

回答用简体中文。

## 项目定位

这套代码是 `自动化识别项目不可行性说明.docx`（仓库根目录，git 忽略）里被判"技术不可达"的那个需求的**简化实现**：判断一段视频里是否存在**时长 ≤30 秒、出镜主体达到 X 人**的片段。该项目现已重启，方向是**做一个粗糙但能量化召回率的版本**，而不是重新论证不可行。

不可行性说明列出的五个障碍是本项目的技术地图，改代码时要清楚自己在动哪一环：

```
人物检测 → 镜头内跟踪/关联 → 跨镜头身份去重 → 滑窗内唯一身份计数 → 存在性判断
```

- 公开数据集不可用、自建数据集成本过高 → 所以真值只能靠**文件名里的人工标注**（见下）。
- 模型不跨风格泛化（真人 / 2D 动画 / 3D 动画）→ 所以要**分风格路由**。
- "谁算主体"无可计算依据 → 当前用**画面显著程度替代剧情重要程度**（表情清晰 + 人脸框宽度占比达标 = 主体，其余当路人）。长期思路是统计出镜时长 + 出现频次，但这依赖跨镜头身份去重准确。
- **跨镜头身份去重是当前最大难点**，也是"筛掉非正脸"这个想法的动机：非正脸的 CCIP 特征最不可靠。
- 存在性判断逻辑上无法证明"不存在"，只能给"未发现"——README 第六节第 4 条已经写明，不要在输出或文档里把它说成"证明不存在"。

### 重启后待做的改造（当前代码还没做到）

1. 窗口从 `window_seconds=15.0` 改到 **30 秒**；合格门槛从"≥13 个不同角色"改到目标 **X 人（素材实测落在 1~10）**。
2. **风格路由**：均匀抽帧 → 投票判断真人 / 2D / 3D → 路由到对应的模型与参数集。判断条件待定，检测器注册表（`detectors.py` 的 `@register`）已经是为此准备的插拔点。
3. **正脸过滤**：人脸上没有检出两只眼睛就排除。这是"粗糙化"的核心手段，目的是提高跨镜头去重的准确率。
4. **召回率测量**：拿文件名标注当真值，算召回。这是重启后唯一有意义的验收指标。

## 环境与运行

Python 固定用这个解释器（不要用 `python`，PATH 上不是它）：

```powershell
$py = "D:\Programs\DevEnvironments\Anaconda\anaconda3\envs\myenv\python.exe"
```

**跑任何需要 HuggingFace 的命令前必须设这三个环境变量**，否则 imgutils 下载/校验检测与 CCIP 模型时会 `SSL: UNEXPECTED_EOF_WHILE_READING`（本机代理失效 + 直连被墙），整条流水线直接崩：

```powershell
$env:HF_ENDPOINT = 'https://hf-mirror.com'; $env:NO_PROXY = '*'; $env:no_proxy = '*'
$env:HF_HUB_OFFLINE = '1'   # 模型已缓存时加这条最稳：imgutils 每个新进程首次检测都会联网列模型清单
```

常用命令（项目根目录下）：

```powershell
& $py src/main.py "data/JOJO的奇妙冒险_635s（人脸数期望：1+2+2+1+1+3+2+4+1）（跨镜头身份去重后期望：9）.mp4"
& $py src/main.py <video> --limit-seconds 60 --viz 8   # 调参：只跑前 60s + 导出 8 张标注帧
& $py src/main.py <video> --min-events 5 --ccip-threshold 0.178 --encoder libx264
& $py src/sweep.py <video>                             # 阈值敏感性网格扫描 → output/sweep.csv
& $py -m pytest tests -q                               # 全部单测（不需要 GPU / 模型 / HF）
& $py -m pytest tests/test_streaming.py::TestDetectCuts -q          # 单个测试类
& $py -m pytest "tests/test_main.py::TestSelectSegments::test_single_qualified_window" -q
```

`tests/` 有 130+ 个纯逻辑单测，全部用合成数据（tmp_path 下现造小视频），跑完只要几秒——改 IoU/聚类/选段/切镜逻辑先跑它。`tests/conftest.py` 把 `src/` 塞进 `sys.path`，因为 `main.py` 用的是平铺导入（`from config import Config`），**不是包**。新增模块沿用平铺导入。

PowerShell 里读文件一律用 UTF-8，否则中文乱码。

## 架构

7 个阶段，全部在 `src/main.py`（约 1000 行，用 `# === N. 阶段名 ===` 分节）：

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
| `README.md` | 逐阶段的详细数据流说明 + 实测数字 + 决策理由。**改流水线行为要同步更新它**，它是这个项目的主文档 |

各阶段的完整逻辑、参数速查表、以及"为什么这么做"（含多个被否掉的旧方案及其实测数据）见 [README.md](./README.md)。下面只列改代码时最容易踩的不变量。

### 流式处理是硬约束

帧由 `cv2.VideoCapture` 顺序解码，用完即弃，**全程不落盘**。内存中同时只有"当前帧 + 每条活跃轨迹一张人脸裁剪图"，与视频长度无关（帧数 ×6 峰值内存 +3%）。由此产生的约束：

- **裁剪必须在检测的当帧就地完成**（`crop_bbox` 挂到 `Detection.crop`），帧一丢就没有第二次机会回头读。
- `crop_bbox` 必须返回 `.copy()`，numpy 切片视图会拖住整帧，一个引用抵消掉全部内存优势。
- 代表裁剪图靠**在线擂台**（`FaceTracker._offer`，比 `blur_var × confidence`）选出，不能等轨迹结束再回看。
- `--viz N` 用蓄水池采样，样本当场压成 JPEG 字节才进池子。
- 任何新增的"需要回看某一帧"的功能（比如正脸/眼睛判定）都要**在当帧算完、把结果挂在 `Detection` 上带下去**，不能留到后面阶段再取像素。

### 其他必须遵守的点

- **fail fast**：打不开视频、拿不到 fps、ffmpeg 返回非零 → 直接 `RuntimeError` 带日志尾部，不静默降级成空结果。这是刻意的，不要加兜底分支。
- **Windows 中文路径**：写图一律走 `imwrite_unicode`（`cv2.imencode` + `numpy.tofile`）。`cv2.imwrite` 遇中文目录名**静默返回 False 不抛异常**，后果是裁剪图全丢 → `character_id` 全 None → 一个片段都选不出来。控制台输出走 `force_utf8_stdout`。`cv2.VideoCapture` 对中文路径正常。
- **切镜检测用 ffmpeg `scdet` 滤镜**，从 stderr 正则抓 `lavfi.scd.time`。别改回 HSV 直方图方案（实测召回/精确只有 51%，EVA 素材 0%，根因见 README 第六节第 8 条）。升级 ffmpeg 后切镜数突然变 0，先查这条正则。
- **切镜宁多勿少**：漏一刀会把两个角色粘成一条轨迹、静默丢掉一个角色；多切一刀只是把轨迹断成两段，CCIP 聚类会还原。
- **聚类是 complete-linkage，不做传递合并**：簇内任意两张裁剪图差异都要低于阈值。旧的并查集传递合并会让差异链把全片轨迹塌缩成一个簇，已弃用，不要回退。
- **`ccip_threshold` 与 `min_events_per_window` 互相补偿**，任一个都无法单独解释最终片段数（0.05→198 个角色，0.25→25 个；片段数摆动 33 倍）。换素材或改门槛前先跑 `sweep.py`，不要凭单次运行的数字定参。
- 输出目录名取视频文件名 stem，`save_representatives` 写入前先清空 `crops/`，保证重跑幂等。

## 素材与人工标注约定

`data/` 与 `output/` 都在 `.gitignore` 里，不进 git。

- `data/原始数据/*.mp4` 是整片；`data/<片名>_<起始秒>s.mp4` 是从整片该秒切出的测试片段（`新世纪福音战士_1730_30s.mp4` = 1730s 起、30s 长）。
- 文件名里的 `（人脸数期望：1+2+2+...）` 是**人工标注的真值**：按镜头顺序列出每个镜头的期望人脸数，`1+2+2` = 三个镜头，各期望 1、2、2 张脸。`（跨镜头身份去重后期望：9）` 是整段去重后的主体数真值。
- **人工标注的镜头判定口径**：只有出现硬切等真实转场才算换镜头。镜头内的剧烈画面变化不算——例如人物从镜头前走过短暂遮住画面，前后仍算**同一个镜头**。
  这与 `scdet` 的行为有偏差：这类遮挡会造成孤立的帧间差异尖峰，正是 scdet 会上报的形态，于是一个人工镜头被拆成两个检出镜头，加号分段数就对不上。所以**按镜头逐项比对人脸数时，先做镜头对齐（多对一），不要按序号硬配**；跨镜头去重后的总主体数不受影响（多切一刀由 CCIP 聚类还原，见上文"切镜宁多勿少"）。
- 标注口径（人工判定时已考虑）：**是正脸、出镜时长 >1 秒、表情清晰、人脸框宽度不小于画面宽度某个比例**。写召回率评测时必须对齐这个口径，否则算出来的召回没有意义。
- 素材覆盖三种风格，正好对应要做的风格路由：真人（爱情神话）、2D 动画（JOJO、高木同学、EVA、魔女之旅）、3D/CG 动画（凡人修仙传）。
- 参数已知的风格差异：`scdet_threshold` 默认 10 在手绘作画素材上逐帧人工验证过，CG 动作素材（凡人修仙传）在 10 上偏保守，要用 `--scdet-threshold 5`。
- 现成的反例素材：`新世纪福音战士` 566.5–568.1s 是绫波丽克隆体群像，单帧检出 26~27 张脸且全部是真检测，但语义上是同一个角色——"算几个主体"没有统一口径的极端案例。该片全片任意半分钟窗口主体数上界只有 4。

## 已知不一致

`main.py` 里 `--ccip-threshold` 的行内注释写"默认用模型自带阈值 ≈0.178"，但 `config.py` 的实际默认是 `0.05`。以 `config.py` 为准。
