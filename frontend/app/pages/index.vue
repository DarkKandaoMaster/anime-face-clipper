<script setup lang="ts">
// 工作台。逐元素照 design/Main.dc.html 翻过来：内联样式原样保留，
// sc-for → v-for，renderVals() 里的派生量 → computed。
// 画板里的数据全是假的，这里换成 /api/assets 与 SSE 的真数据。
const w = useWorkbench()
const authed = useAuthed()
const fileInput = ref<HTMLInputElement | null>(null)
const dragging = ref(false)
const now = ref(Date.now())

type Row = {
  key: string
  kind: 'asset' | 'upload'
  id?: string
  name: string
  meta: string
  dur: string
  posterUrl?: string
  style: string
  styleColor: string
  chars: string
  segs: string
  stateText: string
  stateColor: string
  pctText: string
  barWidth: string
  action: string
  actionColor: string
}

const STATE: Record<string, { text: string; color: string }> = {
  idle: { text: '待分析', color: 'var(--mut)' },
  queued: { text: '排队中', color: 'var(--dim)' },
  running: { text: '分析中', color: 'var(--am)' },
  done: { text: '已完成', color: 'var(--lime)' },
  failed: { text: '失败', color: 'var(--err)' },
  cancelled: { text: '已取消', color: 'var(--dim)' },
}

const stageLabel = (key: string | null) =>
  w.stages.value.find((s) => s.key === key)?.label ?? ''

const rows = computed<Row[]>(() => {
  const fromAssets: Row[] = w.assets.value.map((a) => {
    const state = STATE[a.status] ?? STATE.idle
    const queuePos = w.queued.value.findIndex((q) => q.id === a.id)
    let text = state.text
    if (a.status === 'running' && a.stage) text = `分析中 · ${stageLabel(a.stage)}`
    if (a.status === 'queued') text = `排队中 · 前面 ${queuePos < 0 ? 0 : queuePos} 个`
    if (a.status === 'failed' && a.error) text = `失败 · ${a.error}`
    return {
      key: a.id,
      kind: 'asset',
      id: a.id,
      name: a.filename,
      meta: `${(a.size_bytes / 1073741824).toFixed(2)} GB · ${a.width}×${a.height} · ${a.codec.toUpperCase()}`,
      dur: clock(a.duration),
      posterUrl: `/api/assets/${a.id}/poster`,
      style: styleLabel(a.style),
      styleColor: a.style ? 'var(--mut)' : 'var(--dim)',
      chars: a.num_characters == null ? '—' : String(a.num_characters),
      segs: a.num_segments == null ? '—' : String(a.num_segments),
      stateText: text,
      stateColor: state.color,
      pctText: a.status === 'running' ? `${a.percent}%` : '',
      barWidth: a.status === 'done' ? '100%' : `${a.percent}%`,
      action: a.status === 'done' ? '查看结果'
        : ['queued', 'running'].includes(a.status) ? '取消' : '移除',
      actionColor: a.status === 'done' ? 'var(--am)' : 'var(--mut)',
    }
  })
  const fromUploads: Row[] = w.uploads.value.map((u) => ({
    key: u.tempId,
    kind: 'upload',
    name: u.filename,
    meta: `${(u.size_bytes / 1073741824).toFixed(2)} GB · 上传中`,
    dur: '—',
    style: '—',
    styleColor: 'var(--dim)',
    chars: '—',
    segs: '—',
    stateText: u.error ? `无法接收 · ${u.error}` : '上传中',
    stateColor: u.error ? 'var(--err)' : 'var(--fg)',
    pctText: u.error ? '' : `${u.percent}%`,
    barWidth: `${u.percent}%`,
    action: u.error ? '移除' : '上传中',
    actionColor: 'var(--mut)',
  }))
  return [...fromAssets, ...fromUploads]
})

const library = computed(() => ({
  count: w.assets.value.length,
  duration: clock(w.assets.value.reduce((s, a) => s + a.duration, 0)),
  size: gb(w.assets.value.reduce((s, a) => s + a.size_bytes, 0)),
}))

// 2D 支路实测约 4 倍实时（README 第五轮），拿它当预估口径。
const REALTIME = 4
const pendingSeconds = computed(() =>
  [...w.pending.value, ...w.queued.value].reduce((s, a) => s + a.duration, 0) / REALTIME,
)
const etaText = computed(() => {
  const minutes = Math.max(1, Math.round(pendingSeconds.value / 60))
  return `${w.pending.value.length} 个视频待分析 · 2D 支路约 ${REALTIME} 倍实时，预计 ${minutes} 分钟`
})
const finishAt = computed(() => {
  const at = new Date(now.value + pendingSeconds.value * 1000)
  return `${pad(at.getHours())}:${pad(at.getMinutes())}`
})
const elapsed = computed(() =>
  w.runningSince.value ? clock((now.value - w.runningSince.value) / 1000) : '0:00',
)

const gpus = computed(() =>
  Array.from({ length: w.gpuCount.value }, (_, i) => {
    const active = i === 0 && w.running.value
    return {
      name: `GPU ${i}`,
      state: active ? '分析中' : '空闲',
      color: active ? 'var(--am)' : 'var(--dim)',
      load: active ? `${w.running.value!.percent}%` : '3%',
      job: active ? `${w.running.value!.filename} · 已用 ${elapsed.value}` : '等待下一个任务',
    }
  }),
)

const stageRows = computed(() => {
  const current = w.stageKey.value
  const index = w.stages.value.findIndex((s) => s.key === current)
  return w.stages.value.map((s, i) => {
    const done = index < 0 ? false : i < index
    const on = i === index
    return {
      key: s.key,
      name: s.label,
      mark: done ? '✓' : on ? '▸' : '·',
      color: done ? 'var(--lime)' : on ? 'var(--am)' : 'var(--dim)',
      time: on ? elapsed.value : done ? '✓' : '—',
    }
  })
})

function pick(row: Row) {
  if (row.kind === 'upload') return w.dismissUpload(row.key)
  const asset = w.assets.value.find((a) => a.id === row.id)!
  if (asset.status === 'done') return navigateTo(`/results/${asset.task_id}?asset=${asset.id}`)
  if (['queued', 'running'].includes(asset.status) && asset.task_id) return w.cancel(asset.task_id)
  return w.remove(asset.id)
}

function onDrop(e: DragEvent) {
  dragging.value = false
  const files = Array.from(e.dataTransfer?.files ?? []).filter((f) => f.size > 0)
  if (files.length) w.addFiles(files)
}

function onPick(e: Event) {
  const input = e.target as HTMLInputElement
  if (input.files?.length) w.addFiles(Array.from(input.files))
  input.value = ''
}

let poll: ReturnType<typeof setInterval> | null = null
let tick: ReturnType<typeof setInterval> | null = null

watch(authed, async (ok) => {
  if (!ok) return
  await w.loadHealth()
  await w.refresh()
  if (w.activeTaskId.value) w.watch(w.activeTaskId.value)
}, { immediate: true })

onMounted(() => {
  poll = setInterval(() => authed.value && w.refresh(), 4000)
  tick = setInterval(() => (now.value = Date.now()), 1000)
})
onBeforeUnmount(() => {
  if (poll) clearInterval(poll)
  if (tick) clearInterval(tick)
  w.close()
})
</script>

<template>
  <div class="page">
    <AccessGate />
    <SideBar active="work">
      <template #footer>
        <span class="label">素材库</span>
        <div class="kv"><span>视频</span><span class="num">{{ library.count }}</span></div>
        <div class="kv"><span>总时长</span><span class="num">{{ library.duration }}</span></div>
        <div class="kv"><span>占用</span><span class="num">{{ library.size }}</span></div>
      </template>
    </SideBar>

    <div class="center">
      <header>
        <span style="color:var(--fg-hi);font-size:13.5px;font-weight:600;letter-spacing:-0.012em">工作台</span>
        <span v-if="w.busy.value" class="pill">
          <span style="width:5px;height:5px;border-radius:50%;background:var(--am)" />
          1 个任务运行中
        </span>
        <span style="flex:1" />
        <span style="color:var(--dim);font-size:11px;font-family:var(--mono)">分析完成后到结果页调整门槛，改门槛不重跑模型</span>
      </header>

      <main>
        <div
          class="drop" :class="{ hot: dragging }"
          @click="fileInput?.click()"
          @dragover.prevent="dragging = true"
          @dragleave.prevent="dragging = false"
          @drop.prevent="onDrop"
        >
          <div class="dropmark">
            <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#e8a33d" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v12" /><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><path d="m7 8 5-5 5 5" /></svg>
          </div>
          <div style="flex:1;display:flex;flex-direction:column;min-width:0;line-height:1.4">
            <span style="color:var(--fg-hi);font-size:13px;font-weight:600">拖拽视频到此处，或点击选择文件</span>
            <span style="color:var(--dim);font-size:11px;font-family:var(--mono)">H.264 / MP4 · 可一次多选，也可以分多次添加</span>
          </div>
          <button class="gbtn" @click.stop="fileInput?.click()">选择文件</button>
          <input ref="fileInput" type="file" accept="video/mp4,video/*" multiple hidden @change="onPick">
        </div>

        <div class="table">
          <div class="thead">
            <span>预览</span>
            <span>文件</span>
            <span>画风</span>
            <span style="text-align:right">角色</span>
            <span style="text-align:right">片段</span>
            <span>状态</span>
            <span style="text-align:right">操作</span>
          </div>

          <div class="tbody">
            <div v-if="!rows.length" class="empty">素材库是空的，先拖一个 H.264 / MP4 进来。</div>
            <div v-for="row in rows" :key="row.key" class="trow">
              <div class="thumb">
                <img v-if="row.posterUrl" :src="row.posterUrl" alt="">
                <svg v-else width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#242a33" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="3" rx="2" /><path d="M7 3v18" /><path d="M17 3v18" /><path d="M3 12h18" /></svg>
                <span class="badge">{{ row.dur }}</span>
              </div>

              <div style="display:flex;flex-direction:column;min-width:0;line-height:1.4">
                <span class="fname">{{ row.name }}</span>
                <span style="color:var(--dim);font-size:10.5px;font-family:var(--mono)">{{ row.meta }}</span>
              </div>

              <span style="font-size:11px;font-family:var(--mono)" :style="{ color: row.styleColor }">{{ row.style }}</span>
              <span class="cell-num">{{ row.chars }}</span>
              <span class="cell-num">{{ row.segs }}</span>

              <div style="display:flex;flex-direction:column;gap:5px;min-width:0">
                <div style="display:flex;align-items:center;justify-content:space-between;gap:8px">
                  <span class="statetext" :style="{ color: row.stateColor }">{{ row.stateText }}</span>
                  <span style="flex:none;color:var(--dim);font-size:10.5px;font-family:var(--mono);font-variant-numeric:tabular-nums">{{ row.pctText }}</span>
                </div>
                <div style="height:3px;background:var(--k3);overflow:hidden">
                  <div style="height:100%;transition:width .2s ease" :style="{ width: row.barWidth, background: row.stateColor }" />
                </div>
              </div>

              <div style="display:flex;justify-content:flex-end">
                <button class="gbtn sm" :style="{ color: row.actionColor }" @click="pick(row)">{{ row.action }}</button>
              </div>
            </div>
          </div>

          <div class="tfoot">
            <span style="color:var(--mut);font-size:11.5px;font-family:var(--mono)">{{ etaText }}</span>
            <button class="abtn" :disabled="!w.pending.value.length" @click="w.startAnalysis()">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14" /><path d="M12 5v14" /></svg>
              开始分析
            </button>
          </div>
        </div>
      </main>
    </div>

    <aside class="inspector">
      <div class="ihead">运行状态</div>
      <div class="ibody">
        <div class="block">
          <span class="label">计算资源</span>
          <div v-for="g in gpus" :key="g.name" class="card">
            <div style="display:flex;align-items:center;gap:8px">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#565f6c" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex:none"><rect width="16" height="16" x="4" y="4" rx="2" /><rect width="6" height="6" x="9" y="9" rx="1" /><path d="M15 2v2" /><path d="M15 20v2" /><path d="M2 15h2" /><path d="M20 15h2" /><path d="M2 9h2" /><path d="M20 9h2" /><path d="M9 2v2" /><path d="M9 20v2" /></svg>
              <span style="color:var(--fg-hi);font-size:12px;font-weight:600;font-family:var(--mono)">{{ g.name }}</span>
              <span style="flex:1" />
              <span style="font-size:11px;font-family:var(--mono)" :style="{ color: g.color }">{{ g.state }}</span>
            </div>
            <div style="height:3px;background:var(--k3);overflow:hidden">
              <div style="height:100%" :style="{ width: g.load, background: g.color }" />
            </div>
            <span class="job">{{ g.job }}</span>
          </div>
        </div>

        <div class="block">
          <span class="label">当前任务</span>
          <div class="card">
            <span class="fname" style="font-size:12px">{{ w.running.value?.filename ?? '没有正在跑的任务' }}</span>
            <div v-for="s in stageRows" :key="s.key" style="display:flex;align-items:center;gap:8px">
              <span style="width:14px;flex:none;text-align:center;font-size:10px;font-family:var(--mono)" :style="{ color: s.color }">{{ s.mark }}</span>
              <span style="flex:1;font-size:11.5px" :style="{ color: s.color }">{{ s.name }}</span>
              <span style="flex:none;font-size:10.5px;font-family:var(--mono);color:var(--dim)">{{ s.time }}</span>
            </div>
            <span v-if="!stageRows.length" style="color:var(--dim);font-size:11px;font-family:var(--mono)">开始分析后这里显示逐阶段进度</span>
          </div>
        </div>

        <div class="block">
          <span class="label">队列</span>
          <div class="card">
            <div class="kv"><span>等待中</span><span class="num">{{ w.queued.value.length }}</span></div>
            <div class="kv"><span>并行度</span><span class="num">{{ w.gpuCount.value }}（= 显卡数）</span></div>
            <div class="kv">
              <span>预计全部完成</span>
              <span class="num" style="color:var(--am)">{{ w.busy.value ? finishAt : '—' }}</span>
            </div>
          </div>
        </div>
      </div>
    </aside>
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
  display: inline-flex; align-items: center; gap: 6px; padding: 2px 8px;
  border: 1px solid var(--line2); border-radius: var(--r);
  color: var(--mut); font-size: 11px; font-family: var(--mono);
}

main {
  flex: 1; min-height: 0; display: flex; flex-direction: column; gap: 14px;
  padding: 16px 18px 20px; overflow: hidden;
}

.drop {
  display: flex; align-items: center; gap: 14px; flex: none;
  padding: 13px 16px; border: 1px dashed var(--line2); border-radius: var(--r);
  background: var(--k1); cursor: pointer; transition: border-color 0.12s ease;
}
.drop.hot { border-color: var(--am); }
.dropmark {
  width: 34px; height: 34px; flex: none; border-radius: var(--r);
  background: var(--am-dim); display: flex; align-items: center; justify-content: center;
}

.table {
  flex: 1; min-height: 0; display: flex; flex-direction: column;
  border: 1px solid var(--line); border-radius: var(--r); background: var(--k1); overflow: hidden;
}
.thead, .trow {
  display: grid;
  grid-template-columns: 88px minmax(0, 1fr) 56px 64px 64px 236px 84px;
  gap: 12px; align-items: center; padding: 9px 16px;
}
.thead {
  flex: none; border-bottom: 1px solid var(--line); background: var(--k2);
  color: var(--dim); font-size: 9.5px; font-weight: 600;
  letter-spacing: 0.16em; text-transform: uppercase;
}
.tbody { flex: 1; min-height: 0; overflow: auto; display: flex; flex-direction: column; }
.trow { border-bottom: 1px solid var(--line); transition: background 0.12s ease; }
.trow:hover { background: var(--k1); }
.empty { padding: 28px 16px; color: var(--dim); font-size: 12px; font-family: var(--mono); }

.thumb {
  position: relative; width: 88px; height: 50px; border-radius: var(--r-sm);
  background: var(--k0); border: 1px solid var(--line);
  display: flex; align-items: center; justify-content: center; overflow: hidden;
}
.thumb img { width: 100%; height: 100%; object-fit: cover; }
.badge {
  position: absolute; right: 3px; bottom: 3px; padding: 0 4px; border-radius: var(--r-sm);
  background: rgb(13 15 18 / 0.9); color: var(--mut); font-size: 9px; font-family: var(--mono);
}
.fname {
  color: var(--fg-hi); font-size: 12.5px; font-weight: 500;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.cell-num {
  text-align: right; color: var(--fg); font-size: 12.5px;
  font-family: var(--mono); font-variant-numeric: tabular-nums;
}
.statetext {
  font-size: 11px; font-weight: 500;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}

.tfoot {
  display: flex; align-items: center; justify-content: space-between; gap: 16px;
  flex: none; padding: 11px 16px; border-top: 1px solid var(--line); background: var(--k2);
}

.gbtn {
  display: inline-flex; align-items: center; justify-content: center;
  height: var(--h); padding: 0 12px;
  border: 1px solid var(--line2); border-radius: var(--r); background: transparent;
  color: var(--mut); font-size: 12px; font-weight: 500; cursor: pointer;
  transition: border-color 0.12s ease, color 0.12s ease;
}
.gbtn:hover { border-color: var(--line4); color: var(--fg-hi); }
.gbtn.sm { height: 26px; padding: 0 9px; font-size: 11.5px; }
.abtn {
  display: inline-flex; align-items: center; justify-content: center; gap: 7px;
  height: 32px; padding: 0 14px;
  border: 1px solid var(--am); border-radius: var(--r); background: var(--am);
  color: var(--am-ink); font-size: 12.5px; font-weight: 600; cursor: pointer;
  transition: background 0.12s ease, border-color 0.12s ease;
}
.abtn:hover:not(:disabled) { background: var(--am-2); border-color: var(--am-2); }
.abtn:disabled { opacity: 0.45; cursor: default; }

.inspector {
  width: var(--inspector); flex: none; display: flex; flex-direction: column;
  border-left: 1px solid var(--line); background: var(--k1);
}
.ihead {
  display: flex; align-items: center; height: var(--topbar); flex: none; padding: 0 14px;
  border-bottom: 1px solid var(--line);
  color: var(--dim); font-size: 10px; font-weight: 600;
  letter-spacing: 0.16em; text-transform: uppercase;
}
.ibody {
  flex: 1; min-height: 0; overflow: auto;
  display: flex; flex-direction: column; gap: 14px; padding: 14px;
}
.block { display: flex; flex-direction: column; gap: 8px; }
.card {
  display: flex; flex-direction: column; gap: 7px; padding: 11px;
  border: 1px solid var(--line); border-radius: var(--r); background: var(--k2);
  box-shadow: var(--lift);
}
.job { color: var(--dim); font-size: 10.5px; font-family: var(--mono);
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.kv { display: flex; align-items: baseline; justify-content: space-between; }
.kv > span:first-child { color: var(--mut); font-size: 11px; }
.kv .num { color: var(--fg); font-size: 12px; }
</style>
