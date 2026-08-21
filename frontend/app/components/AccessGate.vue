<script setup lang="ts">
// 未授权时盖在页面上的一层。整个鉴权就一个共享访问码：公网上传 + GPU + 下载，
// 无门槛等于开放转码服务器。
const authed = useAuthed()
const code = ref('')
const error = ref('')
const busy = ref(false)

async function submit() {
  if (!code.value || busy.value) return
  busy.value = true
  error.value = ''
  try {
    await apiPost('/api/auth/verify', { code: code.value })
    authed.value = true
    code.value = ''
  } catch (e: any) {
    error.value = e?.detail || '校验失败。'
  } finally {
    busy.value = false
  }
}

onMounted(async () => {
  const res = await apiGet<{ ok: boolean }>('/api/auth/session').catch(() => ({ ok: false }))
  authed.value = res.ok
})
</script>

<template>
  <div v-if="!authed" class="gate">
    <form class="box" @submit.prevent="submit">
      <div class="brand">
        <div class="mark">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#e8a33d" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2" /><circle cx="9" cy="7" r="4" /><path d="M22 21v-2a4 4 0 0 0-3-3.87" /></svg>
        </div>
        <div style="display:flex;flex-direction:column;line-height:1.25">
          <b style="color:var(--fg-hi);font-size:12.5px;font-weight:600;letter-spacing:-0.012em">主体片段检测</b>
          <small style="color:var(--dim);font-size:9.5px;font-family:var(--mono);letter-spacing:0.02em">headcount-30s</small>
        </div>
      </div>
      <span class="label">访问码</span>
      <input v-model="code" type="password" autofocus placeholder="输入访问码" />
      <span v-if="error" class="err">{{ error }}</span>
      <!-- 故意不是 type=submit：hydrate 完成前点它会走浏览器原生提交，
           把访问码带进 URL 和历史记录。回车仍然可用（form 上的 @submit.prevent）。 -->
      <button type="button" :disabled="busy" @click="submit">{{ busy ? '校验中…' : '进入' }}</button>
    </form>
  </div>
</template>

<style scoped>
.gate {
  position: fixed;
  inset: 0;
  z-index: 50;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--k0);
}
.box {
  display: flex;
  flex-direction: column;
  gap: 9px;
  width: 292px;
  padding: 18px;
  border: 1px solid var(--line);
  border-radius: var(--r);
  background: var(--k1);
  box-shadow: var(--lift);
}
.brand { display: flex; align-items: center; gap: 9px; padding-bottom: 6px; }
.mark {
  width: 23px; height: 23px; flex: none; border-radius: var(--r);
  background: var(--am-dim); display: flex; align-items: center; justify-content: center;
}
input {
  height: var(--h);
  padding: 0 9px;
  border: 1px solid var(--line2);
  border-radius: var(--r);
  background: var(--k0);
  color: var(--fg-hi);
  font-family: var(--mono);
  font-size: 12.5px;
  outline: none;
}
input:focus { border-color: var(--am); }
button {
  height: var(--h-lg);
  border: 1px solid var(--am);
  border-radius: var(--r);
  background: var(--am);
  color: var(--am-ink);
  font-size: 12.5px;
  font-weight: 600;
  cursor: pointer;
}
button:disabled { opacity: 0.6; cursor: default; }
.err { color: #d97b6c; font-size: 11.5px; font-family: var(--mono); }
</style>
