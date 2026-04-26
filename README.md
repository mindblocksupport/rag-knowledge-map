# RAG 知识地图 — GitHub Pages 部署

> 完整 RAG (Retrieval-Augmented Generation) 知识体系, 17,000+ 行 / 12,000+ XMind 节点 / 60 张架构图 / 60+ 面试题 / 22 真实生产案例.

## 在线访问

🌐 **https://mindblocksupport.github.io/custom-agent/**

## 内容覆盖

| 模块 | 内容 |
|---|---|
| §0 速览 | 三存储 / Hybrid / 9 步流程 / 60+ 术语速查 |
| §1 基础原理 | LLM 三大局限 / In-Context Learning 理论 / 4 代演进 |
| §2-§3 业务流程+架构 | 5 大流程 / 5 层架构 / 70/20/10 投资 |
| §4 L1 数据治理 | 7 道防线 / 14.5K 行最深章节 / 企业级架构 |
| §5 L2 索引 | 8 种 Chunking / Embedder 7 维度选型 / HNSW/IVF/DiskANN |
| §6 L3 检索 | BM25 完整推导 / 8 种 Reranker / Hybrid + RRF |
| §7-§9 路由+Agent+生成 | Modular Router / 7 RAG-Agent 模式 / vLLM/SGLang |
| §10 横切 | ACL 三层 / Cache 5 层 / Cost 公式 / 5 部署模式 |
| §13 真实案例 | 25 个生产案例完整复盘 (Air Canada/Klarna/Uber/Mercari/...) |
| §14 评估 | RAGAS 4 指标 / VECTARA HEM / AutoNugget / 5 工具对比 |
| §15 面试题 | 60+ 题 + 完整答案 + 追问链 |
| §16 Failure Mode | 7 大失败类型 (含 Prompt Injection) + 诊断决策树 |
| §17 学习路径 | 4 级路径 + 16 道验证题 |
| §18 源码 | 完整工程目录 + 5 接口 + 6 算法实现 |
| §19 Modular RAG | 7 模块完整数据流 + Yunfan Gao 2024 综述 |
| §20 Agent RAG (Gen 4) | 2025-2026 最新发展 (MCP/Reasoning/Computer Use/Multi-Agent) |

## 特色

- **60 张 Mermaid 架构图** — 可缩放可点击
- **124 个发音按钮** — 鼠标悬停看 IPA 音标 + 朗读 (Web Speech API)
- **暗色/亮色自动切换** — 跟随系统
- **响应式布局** — 移动端可读
- **SEO 优化** — 含 Open Graph / Twitter Card / Sitemap

## 本地预览

```bash
# 用任意 HTTP server 起本地服务
cd docs
python3 -m http.server 8000
# 访问 http://localhost:8000/
```

## 重新生成

```bash
# Markdown → HTML (含 Mermaid 注入 + 发音按钮)
python3 scripts/md_to_html.py docs/rag-knowledge-map.md docs/rag-knowledge-map.html

# Markdown → XMind (12,000+ 节点)
python3 scripts/md_to_xmind.py docs/rag-knowledge-map.md docs/rag-knowledge-map.xmind

# HTML → PDF (A4, 含页眉页脚, 9.6MB)
DYLD_LIBRARY_PATH=/opt/homebrew/lib python3 scripts/html_to_pdf.py \
  docs/rag-knowledge-map.html docs/rag-knowledge-map.pdf
```

## 自动化部署

`.github/workflows/deploy-pages.yml` 监听 `docs/rag-knowledge-map.md` 和 `scripts/md_to_html.py` 的变更, 自动重新生成 HTML 并部署到 GitHub Pages.

手动触发: GitHub Actions → "Deploy RAG Knowledge Map to GitHub Pages" → "Run workflow".

## 文件清单

```
docs/
├── index.html                    # 入口 (重定向到主文档)
├── rag-knowledge-map.html        # 主文档 (~950KB)
├── rag-knowledge-map.md          # Markdown 源 (~17K 行)
├── rag-knowledge-map.xmind       # XMind 思维导图
├── rag-knowledge-map.opml        # OPML (兼容 macOS Outliner)
├── rag-knowledge-map.pdf         # PDF (A4, 9.6MB)
├── sitemap.xml                   # SEO sitemap
├── robots.txt                    # 爬虫规则
└── .nojekyll                     # GitHub Pages 跳过 Jekyll
```

## 反馈 / 贡献

欢迎在 GitHub Issues 提建议. 错别字 / 事实错误 / 新案例 都欢迎 PR.
