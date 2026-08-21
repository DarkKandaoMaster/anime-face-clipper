// 画板里所有数字的显示口径都在这里，页面里不再各写一份。

export const pad = (n: number) => (n < 10 ? '0' + n : String(n))

/** 秒 → mm:ss（画板里的时长、时间码全是这个口径，超过一小时也不进位到 hh） */
export const clock = (sec: number) => {
  const s = Math.max(0, Math.round(sec))
  return pad(Math.floor(s / 60)) + ':' + pad(s % 60)
}

/** 片段时间码：00:00 – 00:30 */
export const range = (start: number, end: number) => `${clock(start)} – ${clock(end)}`

export const gb = (bytes: number) => (bytes / 1073741824).toFixed(2) + ' GB'

export const styleLabel = (style?: string | null) => {
  if (style === '2d') return '2D'
  if (style === '3d') return '3D CG'
  if (style === 'real') return '真人'
  return '—'
}
