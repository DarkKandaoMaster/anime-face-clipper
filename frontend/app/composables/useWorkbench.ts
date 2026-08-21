// 工作台的数据源：素材列表（轮询）+ 当前任务的 SSE 进度。
// 页面只管画，状态机放在这里。

export type AssetRow = {
  id: string
  filename: string
  duration: number
  width: number
  height: number
  codec: string
  size_bytes: number
  task_id: string | null
  status: string // idle | queued | running | done | failed | cancelled
  style: string | null
  num_characters: number | null
  num_segments: number | null
  error: string | null
  percent: number
  stage: string | null
}

export type UploadRow = {
  tempId: string
  filename: string
  size_bytes: number
  percent: number
  error: string | null
}

export type Stage = { key: string; label: string }

const ACTIVE = ['queued', 'running']

export function useWorkbench() {
  const assets = useState<AssetRow[]>('assets', () => [])
  const uploads = useState<UploadRow[]>('uploads', () => [])
  const gpuCount = useState<number>('gpu_count', () => 1)
  const stages = useState<Stage[]>('stages', () => [])
  const stageKey = useState<string | null>('stage_key', () => null)
  // 阶段耗时只能在前端量：后端不记每阶段用了多久，SSE 一变阶段就在这里打点。
  const stageSeconds = useState<Record<string, number>>('stage_seconds', () => ({}))
  const runningSince = useState<number | null>('running_since', () => null)
  const error = useState<string>('workbench_error', () => '')

  const activeTaskId = computed(
    () => assets.value.find((a) => ACTIVE.includes(a.status))?.task_id ?? null,
  )
  const running = computed(() => assets.value.find((a) => a.status === 'running') ?? null)
  const pending = computed(() =>
    assets.value.filter((a) => ['idle', 'failed', 'cancelled'].includes(a.status)),
  )
  const queued = computed(() => assets.value.filter((a) => a.status === 'queued'))
  const busy = computed(() => assets.value.some((a) => ACTIVE.includes(a.status)))

  async function refresh() {
    try {
      assets.value = await apiGet<AssetRow[]>('/api/assets')
    } catch (e: any) {
      if (e?.status !== 401) error.value = e?.detail ?? '取素材列表失败。'
    }
  }

  async function loadHealth() {
    const h = await apiGet<{ gpu_count: number }>('/api/health').catch(() => ({ gpu_count: 1 }))
    gpuCount.value = h.gpu_count
  }

  /** 把 SSE 推来的一帧合并进列表——进度以 SSE 为准，轮询只兜底。 */
  function applyState(state: any) {
    stages.value = state.stages ?? []
    for (const item of state.items ?? []) {
      const row = assets.value.find((a) => a.id === item.asset_id)
      if (!row) continue
      row.status = item.status
      row.percent = item.percent
      row.stage = item.stage
      row.style = item.style ?? row.style
      row.num_characters = item.num_characters ?? row.num_characters
      row.error = item.error
      if (item.status === 'running') {
        if (stageKey.value !== item.stage) {
          stageKey.value = item.stage
          stageSeconds.value = { ...stageSeconds.value, [item.stage]: 0 }
        }
        if (runningSince.value === null) runningSince.value = Date.now()
      }
    }
    if (['done', 'failed', 'cancelled'].includes(state.status)) {
      stageKey.value = null
      runningSince.value = null
      refresh()
    }
  }

  let source: EventSource | null = null
  function watch(taskId: string) {
    close()
    // EventSource 带不了自定义头，但会带 cookie——后端鉴权走 cookie 正是为此。
    source = new EventSource(`/api/tasks/${taskId}/events`)
    source.onmessage = (e) => applyState(JSON.parse(e.data))
    source.onerror = () => close()
  }
  function close() {
    source?.close()
    source = null
  }

  async function startAnalysis() {
    const ids = pending.value.map((a) => a.id)
    if (!ids.length) return
    stageSeconds.value = {}
    const res = await apiPost<{ task_id: string }>('/api/tasks', { asset_ids: ids })
    await refresh()
    watch(res.task_id)
    return res.task_id
  }

  async function cancel(taskId: string) {
    await apiPost(`/api/tasks/${taskId}/cancel`)
    await refresh()
  }

  async function remove(assetId: string) {
    await apiDelete(`/api/assets/${assetId}`)
    await refresh()
  }

  async function addFiles(files: File[]) {
    for (const file of files) {
      const row: UploadRow = {
        tempId: `${file.name}-${file.size}-${Math.random().toString(36).slice(2, 8)}`,
        filename: file.name,
        size_bytes: file.size,
        percent: 0,
        error: null,
      }
      uploads.value = [...uploads.value, row]
      try {
        await uploadFile(file, (sent, total) => {
          row.percent = Math.round((sent / total) * 100)
        })
        uploads.value = uploads.value.filter((u) => u.tempId !== row.tempId)
        await refresh()
      } catch (e: any) {
        // 格式不对（非 H.264 / 非 MP4）就停在这一行上，把后端的原话显示出来。
        row.error = e?.detail ?? '上传失败。'
        row.percent = 100
      }
    }
  }

  function dismissUpload(tempId: string) {
    uploads.value = uploads.value.filter((u) => u.tempId !== tempId)
  }

  return {
    assets, uploads, gpuCount, stages, stageKey, stageSeconds, runningSince, error,
    activeTaskId, running, pending, queued, busy,
    refresh, loadHealth, watch, close, startAnalysis, cancel, remove, addFiles, dismissUpload,
  }
}
