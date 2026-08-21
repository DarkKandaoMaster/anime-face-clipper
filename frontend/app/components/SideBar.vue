<script setup lang="ts">
// 两个页面共用的左栏：品牌 + 工作区导航 + 片源列表 + 底部汇总卡（插槽）。
// 尺寸与配色全部照 design/Main.dc.html、design/Dashboard.dc.html 的原值。
import type { AssetRow } from '~/composables/useWorkbench'

// counts：结果页按当前 X 现算的命中数，覆盖 /api/assets 里按默认 X 存的那份。
const props = defineProps<{
  active: 'work' | 'results'
  activeAsset?: string | null
  counts?: Record<string, number>
}>()

const router = useRouter()
const w = useWorkbench()
const authed = useAuthed()

// 片源列表两个页面都要，所以数据源放在这里，不再由结果页现拉。
const sources = computed(() => w.assets.value.filter((a) => a.status === 'done' && a.task_id))

// 工作台自己会拉素材列表，结果页不会——空了才补一次。
watch(authed, (ok) => {
  if (ok && !w.assets.value.length) w.refresh()
}, { immediate: true })

const hitsOf = (a: AssetRow) => props.counts?.[a.id] ?? a.num_segments ?? 0
// 结果页里片源名是不带扩展名的 stem，左栏跟它保持一致。
const titleOf = (a: AssetRow) => a.filename.replace(/\.[^.]+$/, '')

const open = (a: AssetRow) => router.push(`/results/${a.task_id}?asset=${a.id}`)

function goResults() {
  if (props.active === 'results') return
  // 从工作台进结果页：自动选中第一个片源。
  const first = sources.value[0]
  if (first) open(first)
}
</script>

<template>
  <aside class="side">
    <div class="brand">
      <div class="mark">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#e8a33d" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2" /><circle cx="9" cy="7" r="4" /><path d="M22 21v-2a4 4 0 0 0-3-3.87" /></svg>
      </div>
      <div style="display:flex;flex-direction:column;line-height:1.25">
        <b style="color:var(--fg-hi);font-size:12.5px;font-weight:600;letter-spacing:-0.012em">主体片段检测</b>
        <small style="color:var(--dim);font-size:9.5px;font-family:var(--mono);letter-spacing:0.02em">headcount-30s</small>
      </div>
    </div>

    <div class="nav">
      <span class="label" style="padding:4px 6px 6px">工作区</span>
      <button
        class="navrow" :class="{ on: active === 'work' }"
        @click="$router.push('/')"
      >
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" :stroke="active === 'work' ? '#e8a33d' : '#565f6c'" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex:none"><path d="M12 3v12" /><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><path d="m7 8 5-5 5 5" /></svg>
        <span>工作台</span>
      </button>
      <button class="navrow" :class="{ on: active === 'results' }" @click="goResults">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" :stroke="active === 'results' ? '#e8a33d' : '#565f6c'" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex:none"><rect width="7" height="7" x="3" y="3" rx="1" /><rect width="7" height="7" x="14" y="3" rx="1" /><rect width="7" height="7" x="14" y="14" rx="1" /><rect width="7" height="7" x="3" y="14" rx="1" /></svg>
        <span>结果</span>
      </button>
    </div>

    <div class="middle">
      <div class="srclist">
        <span class="label" style="padding:4px 7px 6px">片源</span>
        <div v-if="!sources.length" class="srcempty">还没有分析完成的片源</div>
        <div
          v-for="s in sources" :key="s.id"
          class="srcrow" :class="{ on: s.id === activeAsset }"
          @click="open(s)"
        >
          <span class="dot" :style="{ background: hitsOf(s) ? (s.id === activeAsset ? 'var(--am)' : 'var(--am-soft)') : 'var(--line)' }" />
          <span class="title" :style="{ color: s.id === activeAsset ? 'var(--fg-hi)' : 'var(--mut)' }">{{ titleOf(s) }}</span>
          <span class="num" style="font-size:11px" :style="{ color: hitsOf(s) ? 'var(--fg)' : 'var(--dim)' }">{{ hitsOf(s) }}</span>
        </div>
      </div>
    </div>

    <div class="foot">
      <div class="card">
        <slot name="footer" />
      </div>
    </div>
  </aside>
</template>

<style scoped>
.side {
  width: var(--sidebar);
  flex: none;
  display: flex;
  flex-direction: column;
  border-right: 1px solid var(--line);
  background: var(--k1);
}
.brand {
  display: flex;
  align-items: center;
  gap: 9px;
  height: var(--topbar);
  flex: none;
  padding: 0 14px;
  border-bottom: 1px solid var(--line);
}
.mark {
  width: 23px; height: 23px; flex: none; border-radius: var(--r);
  background: var(--am-dim); display: flex; align-items: center; justify-content: center;
}
.nav { display: flex; flex-direction: column; gap: 1px; flex: none; padding: 11px 9px; }
.navrow {
  display: flex;
  align-items: center;
  gap: 9px;
  width: 100%;
  height: var(--h);
  padding: 0 8px;
  border: 0;
  border-left: 2px solid transparent;
  border-radius: 0 var(--r) var(--r) 0;
  background: transparent;
  color: var(--mut);
  font-size: 12.5px;
  font-weight: 500;
  text-align: left;
  cursor: pointer;
  transition: background 0.12s ease;
}
.navrow:hover { background: var(--k3); }
.navrow.on {
  border-left-color: var(--am);
  background: var(--k2);
  color: var(--fg-hi);
}
.middle { flex: 1; min-height: 0; display: flex; flex-direction: column; }
.srclist { display: flex; flex-direction: column; gap: 1px; padding: 0 9px; overflow: auto; }
.srcempty { padding: 4px 8px; color: var(--dim); font-size: 11px; }
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
.foot { flex: none; padding: 11px 9px; border-top: 1px solid var(--line); }
.card {
  display: flex;
  flex-direction: column;
  gap: 7px;
  padding: 11px;
  border: 1px solid var(--line);
  border-radius: var(--r);
  background: var(--k2);
  box-shadow: var(--lift);
}
</style>
