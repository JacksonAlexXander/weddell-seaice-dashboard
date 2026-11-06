# 前端开发

普通使用者请阅读根目录 README，无需安装 Node.js。

开发需要 Node.js 22+、pnpm。先在根目录启动 Python 后端，然后：

```bash
cd dashboard
pnpm install --frozen-lockfile
pnpm run dev
```

构建：`pnpm run build`，输出 `dashboard/dist/`。
生产前端请求同源 `/api`，由 Python 后端同时提供页面及预测 API。
仅开发环境支持 `?mock=1`；发布构建使用真实模型。
