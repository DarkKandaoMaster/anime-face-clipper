// 一律用相对路径打 /api，由 nitro.devProxy 转到 127.0.0.1:8000，不要写死端口。
// 鉴权是后端种的 HttpOnly cookie，所以每个请求都得带上 credentials。

export type ApiError = { status: number; detail: string }

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await fetch(path, { credentials: 'include', ...init })
  if (res.status === 401) {
    useAuthed().value = false
    // 后端对"码不对"和"没带码"回的是不同的话，别在这里统一盖掉。
    let detail = '需要访问码。'
    try {
      detail = (await res.json()).detail ?? detail
    } catch {
      /* 不是 JSON 就用默认句 */
    }
    throw { status: 401, detail } as ApiError
  }
  if (!res.ok) {
    let detail = `${res.status}`
    try {
      detail = (await res.json()).detail ?? detail
    } catch {
      /* 后端 fail fast 时也可能不是 JSON，保留状态码就够 */
    }
    throw { status: res.status, detail } as ApiError
  }
  return res.status === 204 ? (undefined as T) : await res.json()
}

export const useAuthed = () => useState<boolean>('authed', () => false)

export const apiGet = <T>(path: string) => request<T>(path)

export const apiPost = <T>(path: string, body?: unknown) =>
  request<T>(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  })

export const apiDelete = <T>(path: string) => request<T>(path, { method: 'DELETE' })

/** 分片上传：后端给 chunk_size，前端照着切。整个 POST 一个大文件会撞代理的体积上限。 */
export async function uploadFile(
  file: File,
  onProgress?: (sent: number, total: number) => void,
) {
  const init = await apiPost<{ upload_id: string; chunk_size: number }>('/api/uploads/init', {
    filename: file.name,
    size: file.size,
  })
  let sent = 0
  for (let index = 0; index * init.chunk_size < file.size; index++) {
    const blob = file.slice(index * init.chunk_size, (index + 1) * init.chunk_size)
    const form = new FormData()
    form.append('index', String(index))
    form.append('chunk', blob, 'part')
    const res = await fetch(`/api/uploads/${init.upload_id}/chunk`, {
      method: 'POST',
      body: form,
      credentials: 'include',
    })
    if (!res.ok) throw { status: res.status, detail: '分片上传失败。' } as ApiError
    sent += blob.size
    onProgress?.(sent, file.size)
  }
  return await apiPost<{ id: string; filename: string }>(`/api/uploads/${init.upload_id}/complete`)
}
