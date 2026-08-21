// 后端是 FastAPI，开发时跑在 127.0.0.1:8000；/api 由 nitro 代理过去，前端一律用相对路径请求。
export default defineNuxtConfig({
  compatibilityDate: '2025-07-15',
  devtools: { enabled: true },
  modules: ['@nuxt/fonts'],

  css: ['~/assets/css/tokens.css'],

  // 字体在构建时下载并自托管，运行时不请求 Google（离线可跑，上公网也不依赖外部）
  fonts: {
    families: [
      { name: 'IBM Plex Sans', provider: 'google', weights: [400, 500, 600, 700] },
      { name: 'IBM Plex Mono', provider: 'google', weights: [400, 500, 600] },
    ],
  },

  // 不指定 host 时 Nuxt 只绑 ::1（IPv6），curl 127.0.0.1:3000 会连不上。锁成 IPv4。
  devServer: { host: '127.0.0.1', port: 3000 },

  nitro: {
    devProxy: {
      '/api': { target: 'http://127.0.0.1:8000/api', changeOrigin: true },
    },
  },
})
