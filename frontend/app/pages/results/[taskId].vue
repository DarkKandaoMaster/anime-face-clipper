<script setup lang="ts">
// 结果页。照 design/Dashboard.dc.html 翻过来。
// 唯一可调的参数就是这里的 X：用户对 X 取值没有直觉，事前猜不如事后调。
// 拖滑块只重新打一次 /segments，后端在 windows_scan.json 上重放选段，不重跑模型。
const route = useRoute()
const authed = useAuthed()
const taskId = route.params.taskId as string

type Segment = {
  start: number
  end: number
  count: number
  snap_shift: number
  characters: number[]
  faces: number[]
}
type Item = {
  item_id: string
  asset_id: string
  stem: string
  title: string
  style: string | null
  duration: number
  num_characters: number | null
  curve: number[]
  segments: Segment[]
  all_faces: number[]
}
type Payload = {
  x: number
  window_seconds: number
  sensitivity: { x: number; ratio: number }[]
  items: Item[]
}

const x = ref(4)
const data = ref<Payload | null>(null)
const selected = ref(0)
const loading = ref(false)
const sheetFor = ref<Item | null>(null)
const player = ref<HTMLVideoElement | null>(null)

const WIN = computed(() => data.value?.window_seconds ?? 30)

/** 全部片源的片段拉平成一条列表，选中态就是它的下标（画板里的 flat）。 */
const flat = computed(() =>
  (data.value?.items ?? []).flatMap((item) =>
    item.segments.map((segment) => ({ item, segment })),
  ),
)
const hits = computed(() => flat.value.length)
const current = computed(() => flat.value[Math.min(selected.value, hits.value - 1)] ?? null)
const totalDuration = computed(() =>
  (data.value?.items ?? []).reduce((s, i) => s + i.duration, 0),
)
const hitPct = computed(() =>
  totalDuration.value ? Math.round((hits.value * WIN.value * 100) / totalDuration.value) + '%' : '0%',
)

async function load() {
  loading.value = true
  try {
    data.value = await apiGet<Payload>(`/api/tasks/${taskId}/segments?x=${x.value}`)
  } catch (e: any) {
    if (e?.status !== 401) data.value = null
  } finally {
    loading.value = false
  }
}

let timer: ReturnType<typeof setTimeout> | null = null
function setX(value: number) {
  x.value = value
  selected.value = 0
  if (timer) clearTimeout(timer)
  timer = setTimeout(load, 120)
}

watch(authed, (ok) => ok && load(), { immediate: true })

/** 曲线：画板的 viewBox 是 1000×56，顶点按该片源曲线的最大值归一。 */
const H = 56
function shape(item: Item) {
  const values = item.curve.length ? item.curve : [0]
  const top = Math.max(x.value + 2, Math.ceil(Math.max(...values)))
  const points = values.map(
    (v, i) => `${((i / Math.max(1, values.length - 1)) * 1000).toFixed(1)},${(H - (v / top) * H).toFixed(1)}`,
  )
  const line = 'M' + points.join(' L')
  return { line, area: `${line} L1000,${H} L0,${H} Z`, xy: (H - (x.value / top) * H).toFixed(1) }
}

const groups = computed(() =>
  (data.value?.items ?? [])
    .filter((item) => item.segments.length)
    .map((item) => {
      const geometry = shape(item)
      const base = flat.value.findIndex((f) => f.item.item_id === item.item_id)
      return {
        item,
        ...geometry,
        hitText: `${item.segments.length} 段 · ${clock(item.segments.length * WIN.value)}`,
        ticks: [0, 1, 2, 3, 4].map((t) => clock((item.duration * t) / 4)),
        entries: item.segments.map((segment, i) => ({
          segment,
          index: base + i,
          left: ((segment.start / item.duration) * 100).toFixed(2) + '%',
          width: ((WIN.value / item.duration) * 100).toFixed(2) + '%',
        })),
      }
    }),
)

const nav = computed(() =>
  (data.value?.items ?? []).map((item) => {
    const first = flat.value.findIndex((f) => f.item.item_id === item.item_id)
    return {
      item,
      hits: item.segments.length,
      on: current.value?.item.item_id === item.item_id,
      first: first < 0 ? selected.value : first,
    }
  }),
)

const cropUrl = (item: Item, characterId: number) =>
  `/api/tasks/${taskId}/${encodeURIComponent(item.stem)}/crops/${characterId}`
const frameUrl = (item: Item, t: number) => `/api/assets/${item.asset_id}/frame?t=${t.toFixed(1)}`

/** 预览不编码：直接播原视频，跳到片段起点，播满 30 秒停。 */
function syncPlayer() {
  const video = player.value
  const entry = current.value
  if (!video || !entry) return
  video.currentTime = entry.segment.start
}
watch(current, () => nextTick(syncPlayer))

function onTimeUpdate() {
  const video = player.value
  const entry = current.value
  if (!video || !entry) return
  if (video.currentTime >= entry.segment.end || video.currentTime < entry.segment.start - 0.5) {
    video.pause()
    video.currentTime = entry.segment.start
  }
}

function locate() {
  const video = player.value
  if (!video || !current.value) return
  video.currentTime = current.value.segment.start
  video.play()
}

function downloadOne() {
  const entry = current.value
  if (!entry) return
  const stem = encodeURIComponent(entry.item.stem)
  window.location.href =
    `/api/tasks/${taskId}/clip?stem=${stem}&start=${entry.segment.start}`
}

function downloadAll() {
  if (!hits.value) return
  window.location.href = `/api/tasks/${taskId}/download?x=${x.value}`
}
</script>

<template>
  <div class="page">
    <AccessGate />
    <SideBar active="results" :results-to="`/results/${taskId}`">
      <template #middle>
        <div class="srclist">
          <span class="label" style="padding:4px 7px 6px">片源</span>
          <div
            v-for="n in nav" :key="n.item.item_id"
            class="srcrow" :class="{ on: n.on }"
            @click="selected = n.first"
          >
            <span class="dot" :style="{ background: n.hits ? (n.on ? 'var(--am)' : 'var(--am-soft)') : 'var(--line)' }" />
            <span class="title" :style="{ color: n.on ? 'var(--fg-hi)' : 'var(--mut)' }">{{ n.item.title }}</span>
            <span class="num" style="font-size:11px" :style="{ color: n.hits ? 'var(--fg)' : 'var(--dim)' }">{{ n.hits }}</span>
          </div>
        </div>
      </template>
      <template #footer>
        <span class="label">本次汇总</span>
        <div class="kv"><span>总片长</span><span class="num">{{ clock(totalDuration) }}</span></div>
        <div class="kv"><span>命中片段</span><span class="num" style="color:var(--am)">{{ hits }} 段</span></div>
        <div class="kv"><span>占总片长</span><span class="num">{{ hitPct }}</span></div>
      </template>
    </SideBar>

    <div class="center">
      <header>
        <span style="color:var(--fg-hi);font-size:13.5px;font-weight:600;letter-spacing:-0.012em">结果</span>
        <span class="pill">{{ data?.items.length ?? 0 }} 个片源 · 分析完成</span>
        <span style="flex:1" />
        <button class="gbtn" :disabled="!current" @click="sheetFor = current?.item ?? null">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 2 9 5-9 5-9-5 9-5Z" /><path d="m3 12 9 5 9-5" /></svg>
          角色印相表
        </button>
        <button class="abtn" :disabled="!hits" @click="downloadAll">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round"><path d="M12 15V3" /><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><path d="m7 10 5 5 5-5" /></svg>
          打包导出 {{ hits }} 段
        </button>
      </header>

      <div class="ctrl">
        <div style="display:flex;align-items:center;gap:12px;flex:none">
          <div style="display:flex;flex-direction:column;gap:1px">
            <span class="label">合格门槛</span>
            <span style="color:var(--mut);font-size:11.5px">{{ WIN }} 秒窗口内至少出现</span>
          </div>
          <span class="xnum">{{ x }}</span>
          <div style="display:flex;flex-direction:column;gap:3px">
            <input
              class="sl" type="range" min="2" max="10" step="1"
              :value="x" @input="setX(Number(($event.target as HTMLInputElement).value))"
            >
            <div style="display:flex;align-items:center;justify-content:space-between;color:var(--dim);font-size:9.5px;font-family:var(--mono)">
              <span>2</span><span>个不同角色</span><span>10</span>
            </div>
          </div>
        </div>

        <span style="width:1px;height:40px;background:var(--line)" />

        <div style="display:flex;flex-direction:column;gap:4px;flex:none">
          <span class="label">门槛敏感度 · 点柱可直接切换</span>
          <!-- 柱子与刻度分成两排：画板里它们同在一个 28px 的行里，靠柱子永远矮
               才没露馅；本项目的比例能顶到 100%（X 小的时候几乎每个窗口都合格），
               同一个盒子会把数字顶到标题上去。 -->
          <div class="bars">
            <div
              v-for="c in data?.sensitivity ?? []" :key="c.x"
              class="bar"
              :title="`X=${c.x} → ${Math.round(c.ratio * 100)}% 的窗口合格`"
              @click="setX(c.x)"
            >
              <div
                style="width:100%;border-radius:1px"
                :style="{ height: Math.max(2, Math.round(c.ratio * 28)) + 'px', background: c.x === x ? 'var(--am)' : 'var(--line2)' }"
              />
            </div>
          </div>
          <div class="bars ticks">
            <span
              v-for="c in data?.sensitivity ?? []" :key="c.x"
              class="bar" style="font-size:9px;font-family:var(--mono);text-align:center"
              :style="{ color: c.x === x ? 'var(--am)' : 'var(--dim)' }"
              @click="setX(c.x)"
            >{{ c.x }}</span>
          </div>
        </div>

        <span style="flex:1" />

        <div style="display:flex;align-items:stretch;gap:22px;flex:none">
          <div class="stat"><span class="label">命中</span><span class="big">{{ hits }}</span></div>
          <div class="stat"><span class="label">合计时长</span><span class="big">{{ clock(hits * WIN) }}</span></div>
        </div>
      </div>

      <div class="body">
        <div class="scroll">
          <div v-if="loading && !data" class="empty">
            <span style="color:var(--mut);font-size:13px;font-weight:600">正在读取结果…</span>
          </div>
          <div v-else-if="!hits" class="empty">
            <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#39414d" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8" /><path d="m21 21-4.3-4.3" /></svg>
            <span style="color:var(--mut);font-size:13px;font-weight:600">当前门槛下没有命中任何片段</span>
            <span style="color:var(--dim);font-size:11.5px;font-family:var(--mono)">把滑块往左拖</span>
          </div>

          <section v-for="g in groups" :key="g.item.item_id">
            <div style="display:flex;align-items:center;gap:9px">
              <span style="width:4px;height:13px;flex:none;border-radius:1px;background:var(--am)" />
              <span class="stitle">{{ g.item.title }}</span>
              <span class="tag">{{ styleLabel(g.item.style) }}</span>
              <span style="color:var(--dim);font-size:11px;font-family:var(--mono)">{{ clock(g.item.duration) }}</span>
              <span style="flex:1" />
              <span style="color:var(--am);font-size:11.5px;font-family:var(--mono);font-variant-numeric:tabular-nums">{{ g.hitText }}</span>
            </div>

            <div style="display:flex;flex-direction:column;gap:4px">
              <div class="timeline">
                <svg viewBox="0 0 1000 56" preserveAspectRatio="none" style="position:absolute;inset:0;width:100%;height:100%">
                  <path :d="g.area" fill="rgb(121 130 143 / 0.16)" />
                  <path :d="g.line" fill="none" stroke="#79828f" stroke-width="1" vector-effect="non-scaling-stroke" />
                  <line x1="0" x2="1000" :y1="g.xy" :y2="g.xy" stroke="#e8a33d" stroke-width="1" stroke-dasharray="5 4" vector-effect="non-scaling-stroke" opacity="0.65" />
                </svg>
                <div
                  v-for="b in g.entries" :key="b.index"
                  class="blk"
                  :style="{
                    left: b.left, width: b.width,
                    background: b.index === selected ? 'var(--am)' : 'var(--am-soft)',
                    border: b.index === selected ? '1px solid var(--fg-hi)' : '1px solid var(--am-edge)',
                  }"
                  :title="range(b.segment.start, b.segment.end)"
                  @click="selected = b.index"
                >
                  <span
                    style="font-size:9.5px;font-weight:600;font-family:var(--mono)"
                    :style="{ color: b.index === selected ? 'var(--am-ink)' : 'var(--am-label)' }"
                  >{{ b.segment.count }}</span>
                </div>
              </div>
              <div style="display:flex;align-items:center;justify-content:space-between;color:var(--dim);font-size:9.5px;font-family:var(--mono)">
                <span v-for="(t, i) in g.ticks" :key="i">{{ t }}</span>
              </div>
            </div>

            <div class="cards">
              <div
                v-for="c in g.entries" :key="c.index"
                class="card" :class="{ on: c.index === selected }"
                @click="selected = c.index"
              >
                <div class="shot">
                  <img :src="frameUrl(g.item, c.segment.start)" alt="" loading="lazy">
                  <span class="tc">{{ range(c.segment.start, c.segment.end) }}</span>
                  <span class="cnt">{{ c.segment.count }} 人</span>
                </div>
                <div style="display:flex;align-items:center;gap:4px">
                  <div v-for="f in c.segment.faces.slice(0, 5)" :key="f" class="face sm">
                    <img :src="cropUrl(g.item, f)" alt="" loading="lazy">
                  </div>
                  <span v-if="c.segment.faces.length > 5" style="margin-left:1px;color:var(--dim);font-size:10px;font-family:var(--mono)">
                    +{{ c.segment.faces.length - 5 }}
                  </span>
                </div>
              </div>
            </div>
          </section>
        </div>

        <aside class="inspector">
          <div class="ihead">片段详情</div>
          <div class="ibody">
            <div class="preview">
              <video
                v-if="current"
                ref="player"
                :src="`/api/assets/${current.item.asset_id}/stream`"
                controls preload="metadata"
                @loadedmetadata="syncPlayer"
                @timeupdate="onTimeUpdate"
              />
              <div v-else class="noplay">没有选中的片段</div>
            </div>

            <div style="display:flex;flex-direction:column">
              <div class="detail">
                <span>来源</span>
                <span class="v">{{ current?.item.title ?? '—' }}</span>
              </div>
              <div class="detail">
                <span>时间码</span>
                <span class="v num">{{ current ? range(current.segment.start, current.segment.end) : '—' }}</span>
              </div>
              <div class="detail">
                <span>窗口角色数</span>
                <span class="v num" style="color:var(--am);font-size:13px;font-weight:600">{{ current ? current.segment.count + ' 个' : '—' }}</span>
              </div>
              <div class="detail" style="border-bottom:0">
                <span>起点吸附</span>
                <span class="v num" style="color:var(--lime);font-size:11.5px">
                  {{ current ? (current.segment.snap_shift > 0 ? `向前 ${current.segment.snap_shift.toFixed(1)}s 到切镜点` : '无（原起点即切镜点）') : '—' }}
                </span>
              </div>
            </div>

            <div style="display:flex;flex-direction:column;gap:7px">
              <span class="label">这 {{ WIN }} 秒出现的角色</span>
              <div class="facegrid">
                <div v-for="f in current?.segment.faces ?? []" :key="f" class="face">
                  <img :src="cropUrl(current!.item, f)" alt="" loading="lazy">
                </div>
              </div>
              <span style="color:var(--dim);font-size:10.5px;font-family:var(--mono)">代表裁剪图取自 crops/，按 character_id 去重</span>
            </div>
          </div>

          <div class="ifoot">
            <button class="abtn" :disabled="!current" @click="downloadOne">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round"><path d="M12 15V3" /><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><path d="m7 10 5 5 5-5" /></svg>
              下载此片段
            </button>
            <button class="gbtn wide" :disabled="!current" @click="locate">在原片中定位</button>
          </div>
        </aside>
      </div>
    </div>

    <div v-if="sheetFor" class="sheet" @click.self="sheetFor = null">
      <div class="sheetbox">
        <div class="sheethead">
          <span class="stitle">{{ sheetFor.title }} · 角色印相表</span>
          <span style="color:var(--dim);font-size:11px;font-family:var(--mono)">
            全片 {{ sheetFor.all_faces.length }} 个角色（character_id 全片口径）
          </span>
          <span style="flex:1" />
          <button class="gbtn sm" @click="sheetFor = null">关闭</button>
        </div>
        <div class="sheetgrid">
          <div v-for="f in sheetFor.all_faces" :key="f" class="sheetcell">
            <img :src="cropUrl(sheetFor, f)" alt="" loading="lazy">
            <span class="num">{{ f }}</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.page { height: 100%; display: flex; background: var(--k0); color: var(--fg); overflow: hidden; }
.center { flex: 1; display: flex; flex-direction: column; min-width: 0; }

header {
  display: flex; align-items: center; gap: 11px;
  height: var(--topbar); flex: none; padding: 0 18px;
  border-bottom: 1px solid var(--line); background: var(--k1);
}
.pill {
  padding: 2px 8px; border: 1px solid var(--line2); border-radius: var(--r);
  color: var(--mut); font-size: 11px; font-family: var(--mono);
}

.ctrl {
  display: flex; align-items: center; gap: 26px;
  height: var(--ctrlbar); flex: none; padding: 0 18px;
  border-bottom: 1px solid var(--line); background: var(--k2); box-shadow: var(--lift);
}
.xnum {
  color: var(--am); font-size: 30px; font-weight: 600; font-family: var(--mono);
  line-height: 1; letter-spacing: -0.02em;
}
.sl {
  -webkit-appearance: none; appearance: none;
  width: 180px; height: 3px; border-radius: 0; background: var(--line2);
  outline: none; cursor: pointer;
}
.sl::-webkit-slider-thumb {
  -webkit-appearance: none; appearance: none;
  width: 9px; height: 18px; border-radius: 1px; background: var(--am); cursor: pointer;
}
.sl::-moz-range-thumb {
  width: 8px; height: 16px; border-radius: 1px; background: var(--am); border: 0; cursor: pointer;
}
.bars { display: flex; align-items: flex-end; gap: 5px; height: 28px; }
.bars.ticks { height: auto; margin-top: 3px; }
.bar { width: 15px; flex: none; cursor: pointer; }
.stat { display: flex; flex-direction: column; gap: 1px; text-align: right; }
.big {
  color: var(--fg-hi); font-size: 19px; font-weight: 600;
  font-family: var(--mono); font-variant-numeric: tabular-nums;
  line-height: 1.2; letter-spacing: -0.02em;
}

.body { flex: 1; min-height: 0; display: flex; }
.scroll {
  flex: 1; min-width: 0; overflow: auto;
  display: flex; flex-direction: column; gap: 13px; padding: 15px 18px 20px;
}
.empty {
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  gap: 7px; height: 400px;
}
section {
  display: flex; flex-direction: column; gap: 9px; padding: 11px;
  border: 1px solid var(--line); border-radius: var(--r); background: var(--k1);
}
.stitle { color: var(--fg-hi); font-size: 12.5px; font-weight: 600; }
.tag {
  padding: 0 6px; border: 1px solid var(--line2); border-radius: var(--r-sm);
  color: var(--dim); font-size: 10px; font-family: var(--mono);
}

.timeline {
  position: relative; height: 56px;
  border: 1px solid var(--line); border-radius: var(--r-sm);
  background: var(--k0); overflow: hidden;
}
.blk {
  position: absolute; top: 8px; bottom: 8px; border-radius: var(--r-sm); cursor: pointer;
  display: flex; align-items: center; justify-content: center;
  transition: filter 0.12s ease;
}
.blk:hover { filter: brightness(1.35); }

.cards { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 9px; }
.card {
  display: flex; flex-direction: column; gap: 6px; padding: 6px;
  border: 1px solid var(--line); border-radius: var(--r);
  background: var(--k2); cursor: pointer;
  transition: border-color 0.12s ease, background 0.12s ease;
}
.card:hover { border-color: var(--line3); }
.card.on { border-color: var(--am); background: rgb(232 163 61 / 0.08); }
.shot {
  position: relative; width: 100%; aspect-ratio: 16 / 9; border-radius: var(--r-sm);
  background: var(--k0); border: 1px solid var(--line); overflow: hidden;
}
.shot img { width: 100%; height: 100%; object-fit: cover; display: block; }
.tc {
  position: absolute; left: 5px; bottom: 5px; padding: 0 5px; border-radius: var(--r-sm);
  background: rgb(10 12 15 / 0.9); color: var(--mut); font-size: 10px; font-family: var(--mono);
}
.cnt {
  position: absolute; right: 5px; top: 5px; padding: 0 6px; border-radius: var(--r-sm);
  background: var(--am-dim); border: 1px solid var(--am-edge);
  color: var(--am); font-size: 10px; font-weight: 600; font-family: var(--mono);
}
.face {
  width: 100%; aspect-ratio: 1; border-radius: var(--r-sm);
  background: var(--k3); border: 1px solid var(--line2); overflow: hidden;
}
.face.sm { width: 24px; height: 24px; flex: none; aspect-ratio: auto; }
.face img { width: 100%; height: 100%; object-fit: cover; display: block; }

.inspector {
  width: var(--inspector); flex: none; display: flex; flex-direction: column;
  border-left: 1px solid var(--line); background: var(--k1);
}
.ihead {
  display: flex; align-items: center; height: 34px; flex: none; padding: 0 13px;
  border-bottom: 1px solid var(--line);
  color: var(--dim); font-size: 10px; font-weight: 600;
  letter-spacing: 0.16em; text-transform: uppercase;
}
.ibody {
  flex: 1; min-height: 0; overflow: auto;
  display: flex; flex-direction: column; gap: 13px; padding: 13px;
}
.preview {
  position: relative; width: 100%; aspect-ratio: 16 / 9; border-radius: var(--r-sm);
  background: #06080a; border: 1px solid var(--line);
  display: flex; align-items: center; justify-content: center; overflow: hidden;
}
.preview video { width: 100%; height: 100%; }
.noplay { color: var(--dim); font-size: 11.5px; font-family: var(--mono); }
.detail {
  display: flex; align-items: center; justify-content: space-between; gap: 10px;
  padding: 7px 0; border-bottom: 1px solid var(--line);
}
.detail > span:first-child { flex: none; color: var(--dim); font-size: 11px; }
.detail .v {
  min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  color: var(--fg-hi); font-size: 12px; font-weight: 500;
}
.facegrid { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 5px; }
.ifoot {
  display: flex; flex-direction: column; gap: 6px; flex: none;
  padding: 11px 13px; border-top: 1px solid var(--line);
}

.srclist { display: flex; flex-direction: column; gap: 1px; padding: 0 9px; overflow: auto; }
.srcrow {
  display: flex; align-items: center; gap: 8px; height: var(--h); padding: 0 8px;
  border-radius: var(--r); cursor: pointer; transition: background 0.12s ease;
}
.srcrow:hover { background: var(--k3); }
.srcrow.on { background: var(--k2); }
.dot { width: 4px; height: 14px; flex: none; border-radius: 1px; }
.srcrow .title {
  flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  font-size: 12px;
}
.kv { display: flex; align-items: baseline; justify-content: space-between; }
.kv > span:first-child { color: var(--mut); font-size: 11px; }
.kv .num { color: var(--fg); font-size: 12px; }

.gbtn {
  display: inline-flex; align-items: center; justify-content: center; gap: 6px;
  height: var(--h); padding: 0 10px;
  border: 1px solid var(--line2); border-radius: var(--r); background: transparent;
  color: var(--mut); font-size: 11.5px; font-weight: 500; cursor: pointer;
  transition: border-color 0.12s ease, color 0.12s ease, background 0.12s ease;
}
.gbtn:hover:not(:disabled) { border-color: var(--line4); color: var(--fg-hi); background: var(--k3); }
.gbtn:disabled { opacity: 0.45; cursor: default; }
.gbtn.wide { height: var(--h-sm); }
.gbtn.sm { height: 24px; padding: 0 8px; font-size: 11px; }
.abtn {
  display: inline-flex; align-items: center; justify-content: center; gap: 7px;
  height: var(--h-lg); padding: 0 13px;
  border: 1px solid var(--am); border-radius: var(--r); background: var(--am);
  color: var(--am-ink); font-size: 12.5px; font-weight: 600; cursor: pointer;
  transition: background 0.12s ease, border-color 0.12s ease;
}
header .abtn { height: var(--h); font-size: 12px; }
.abtn:hover:not(:disabled) { background: var(--am-2); border-color: var(--am-2); }
.abtn:disabled { opacity: 0.45; cursor: default; }

.sheet {
  position: fixed; inset: 0; z-index: 40;
  background: rgb(10 12 15 / 0.82);
  display: flex; align-items: center; justify-content: center; padding: 40px;
}
.sheetbox {
  display: flex; flex-direction: column; gap: 11px;
  width: min(1000px, 100%); max-height: 100%;
  padding: 14px; border: 1px solid var(--line2); border-radius: var(--r);
  background: var(--k1);
}
.sheethead { display: flex; align-items: center; gap: 10px; }
.sheetgrid {
  overflow: auto;
  display: grid; grid-template-columns: repeat(auto-fill, minmax(88px, 1fr)); gap: 7px;
}
.sheetcell {
  position: relative; aspect-ratio: 1; border-radius: var(--r-sm);
  border: 1px solid var(--line2); background: var(--k3); overflow: hidden;
}
.sheetcell img { width: 100%; height: 100%; object-fit: cover; display: block; }
.sheetcell .num {
  position: absolute; left: 3px; bottom: 3px; padding: 0 4px; border-radius: var(--r-sm);
  background: rgb(10 12 15 / 0.9); color: var(--mut); font-size: 9px;
}
</style>
