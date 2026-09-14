# MewHelp

> 电商场景的 AI 智能客服系统。**确定的事交给 Workflow，需要判断的事交给 Agent。**

一句话定位：把一条用户消息（"那它能退吗"）从进来到吐出回答的全过程做成一条**可预测、可拒答、可追溯、可回灌**的流水线 —— 不是一个"套了 RAG 的聊天框"，而是一套把**幻觉、乱承诺、答非所问**当成工程问题来解的客服系统。

---

## 它解决什么问题

电商客服有四类问题，用同一种方法解决都会出事：

| 问题 | 常见做法 | 会出什么事 | MewHelp 的做法 |
| --- | --- | --- | --- |
| "我要退款" | 让 Agent 自由发挥 | 模型用"帮用户一次"的善意绕开规则，真的把钱退了 | 高风险意图走**确定性子流程**：规则写在工具里，模型只能提议，不能执行 |
| "生鲜能退吗" | 定长切分 + 向量检索 | 规则被从中间劈开，模型只看到半个条件就自信地答"可以退" | 按**标题层级 + 表格结构**切分，条件句永不断开 |
| 知识库里没有的问题 | 硬编一个答案 | 用户拿着编造的时效去投诉 | 生成**之前**过置信度闸门，证据不够就拒答并转人工 |
| 答不上的问题 | 悄悄溜走 | 同一个问题被问 60 遍，运营完全不知道 | 落进**数据飞轮**，并带上当时的检索快照，人工按被问频次排队处理 |

---

## 架构

### 主流水线

```mermaid
flowchart TD
    START([用户消息 / SSE 接入]) --> resolve["resolve 指代消解<br/>那它能退吗 → 七天无理由可以退货吗"]
    resolve --> classify["classify 意图识别 + 槽位抽取<br/>规则高速通道 → 模型 → 兜底"]
    classify --> route{"route 条件边<br/>9 类意图 → 5 个出口"}

    route -->|"knowledge（product_faq / policy_faq / 分类降级）"| retrieve["retrieve 双路召回<br/>dense + BM25 → RRF → 精排"]
    retrieve --> gate{"gate 置信度闸门<br/>生成之前，零模型调用"}
    gate -->|通过| generate["generate 生成<br/>只根据证据作答"]
    gate -->|不通过| handoff["handoff 转人工<br/>拒答话术逐条可解释"]

    route -->|"business_tool（order_query / logistics_query）"| agent["agent ReAct 循环<br/>工具调用，上限 6 步"]
    agent --> generate

    route -->|"high_risk（refund / return / cancel / change_address）"| highrisk["high_risk_flow 确定性子流程<br/>模型无自由发挥空间"]
    highrisk -->|已给出确定答复| generate
    highrisk -->|"业务拒绝需复核 / 缺槽位"| handoff
    highrisk -.->|"缺槽位 + 挂了 checkpointer"| paused(["interrupt 原地挂起<br/>用户补充后从原位置继续"])

    route -->|"handoff（human_handoff）"| handoff
    handoff --> flywheel["flywheel 落问题池<br/>连同检索快照一起存"]
    flywheel --> END([END])
    generate --> END
```

三个刻意的设计（都在 `src/mewhelp/workflow/graph.py` 的模块 docstring 里）：

1. **`route` 是条件边，不是 if/else。** 九类意图在 `domain.INTENT_TO_EXIT` 里收敛成五个出口，图因此只有五条边。以后加一类意图，只要它落进已有出口，这张图一行都不用改。
2. **handoff 汇到 flywheel。** 答不上的问题不会白白溜走 —— 这是数据飞轮的入口，也是"这套系统"和"调 API 的 demo"的分界线。
3. **所有路径最终汇合到 `generate`。** 主干只有一条出口，答复的格式、埋点、SSE 推送都只需要实现一次。

> 与 `CONTRACT.md` §2.F 的一处出入：契约图里画的是 `retrieve → rerank → gate`，实际代码里**没有独立的 rerank 节点** —— 精排发生在 `RetrievalPipeline.retrieve()` 内部（`retrieval/pipeline.py`），`route` 只有四条出边。`domain.NodeName.RERANK` / `TOOL` 两个常量当前未被使用。

### 数据飞轮

```mermaid
flowchart LR
    A["线上答不上 / 用户点踩"] --> B["落池<br/>FlywheelStore.record()"]
    B --> C["同题去重<br/>ask_count += 1"]
    C --> D["按 ask_count 降序排队<br/>pending()"]
    D --> E{"分诊<br/>triage_hint(快照)"}
    E -->|"一条证据都没召回"| F["backfill_kb<br/>补文档 / 补 FAQ"]
    E -->|"召回了但分数不足"| G["tune_retrieval<br/>调 embedding / 精排 / 阈值"]
    E -->|"分数够却没通过"| H["review_answer<br/>查生成与闸门判据"]
    F --> I["人工审核 review()"]
    G --> I
    H --> I
    I --> J["to_kb_chunk() 回灌知识库<br/>chunk_id 由归一化问题哈希，幂等"]
    J -.->|"下一个人问同样的问题时答得上来了"| A
```

分诊的判据是**落池那一刻的 `RetrievalResult` 快照**，不是重跑一次检索 —— 审核时知识库已经变了，重跑出来的是今天的现场，会把"没检索到"误判成"真缺知识"。这就是 `FlywheelRecord.retrieval_snapshot` 必须存在的理由。

---

## 快速开始

默认**全套 mock**：不装 Docker、不配 API Key、不联网，也能把整条链路跑通。

```bash
cd E:/临时/projects/mewhelp

# 1) 建环境（可选：当前仓库用 PYTHONPATH=src 也能直接跑）
python -m venv .venv && .venv\Scripts\activate

# 2) 装依赖
python -m pip install -e .
#    或只装运行时依赖：python -m pip install -r requirements.txt

# 3) 跑分流演示：九类意图 → 五个出口，逐个打印路径、证据、答复
PYTHONPATH=src python -m mewhelp.cli demo
```

> Windows PowerShell 把 `PYTHONPATH=src` 写成 `$env:PYTHONPATH='src'`。
> 控制台是 GBK 时中文会乱码，加 `PYTHONIOENCODING=utf-8`（**乱码不是 bug**）。

其它 CLI 子命令（**全部实测可用**）：

```bash
PYTHONPATH=src python -m mewhelp.cli index              # 重建知识库索引并打印 chunk 统计
PYTHONPATH=src python -m mewhelp.cli chat "运费怎么算"    # 单轮问答，打印证据、tokens、耗时
PYTHONPATH=src python -m mewhelp.cli flywheel           # 问题池统计与待审列表
PYTHONPATH=src python -m mewhelp.cli serve              # 启动 FastAPI 服务（含用户界面）
```

不想带 argparse 的最小入口是 `examples/smoke_demo.py`（四个问题，四条不同的边）。

### HTTP 接口

```bash
PYTHONPATH=src python -m mewhelp.cli serve --port 8123
# 用户界面：GET /      接口文档：GET /docs      健康检查：GET /api/health
```

> **请求体字段是 `message`，不是 `question`**；接口都在 `/api` 前缀下
> （`/api/chat` 流式、`/api/chat/sync` 非流式、`/api/orders` 订单快捷选择）。
> 这两点是实际踩过的坑。

### 前端（React）

界面是 React 19 + TypeScript + Vite 写的，源码在 `web/`。**产物已提交**，
所以上面的 `serve` 不需要 Node —— 断网、没装 Node 都能打开。

```bash
cd web
npm install          # 只在首次 / 改依赖时需要（需联网）
npm run build        # 产物输出到 src/mewhelp/api/static/
```

开发时用 `npm run dev`（5173 端口，`/api` 自动代理到 8000），改完记得
`npm run build` 重新产出提交 —— 否则线上跑的还是旧界面。

**布局是左右分栏**：左边对话，右边工作区。工作区把这一轮"办的事"陈列出来 ——
订单卡（商品明细 / 合计 / 收货地址）、物流轨迹、政策依据、操作结果。
窄屏（≤1024px）下工作区收成抽屉，由页头按钮开合。

工作区的卡片是**按数据形状**认出来的（有 `items` 就是订单、有 `events` 就是
物流），不依赖工具名 —— 工具改名不该让界面塌掉。这份逻辑在
`web/src/workspace.ts`，是纯函数，测试会用后端真实推的数据跑它
（`test_frontend_builds_workspace_cards_from_real_payloads`）。

### 界面上的快捷功能

| 功能 | 为什么有它 |
| --- | --- |
| **订单快捷选择**（输入框上方一排订单 chip） | 订单号是用户最记不住的东西。让他为了查物流去翻订单号再手打 `ORD-1007`，是把系统的记忆成本转嫁给用户。点一下填进输入框（**只填不发** —— 他多半还想补一句"这单能退吗"） |
| **右上角计数** | 工作区里有什么、有几条，不用切过去看 |
| **复制答复 / 复制卡片** | 拿到订单信息的下一步多半是粘到聊天窗口或工单系统。让他用鼠标划选一张带排版的卡片是折磨 |
| **重问一次** | 只给最后一条答复 —— 中间那些重问没有意义 |
| **回到底部** | 往上翻历史时才浮出来。用户翻历史时被拽回底部，是聊天界面里最恼人的一类交互 |
| **主题切换**（跟随系统 / 浅色 / 深色） | 选择记在 localStorage；`data-theme` 由 `index.html` 里的内联脚本在**首次绘制前**写好，否则每次进页面都会先闪一下白 |
| **快捷键** `Enter` 发送 / `Shift+Enter` 换行 / `Esc` 停止 / `Ctrl+K` 聚焦 | 中文输入法选词时的回车不算发送（判 `isComposing`） |

答复正文按**轻量 Markdown** 排版（段落 / 表格 / `**加粗**`），实现在
`web/src/RichText.tsx`：走"拆成 React 元素"而不是 `dangerouslySetInnerHTML`，
答复里就算混进 HTML 也只会被当字面量显示。刻意不引 markdown 库 ——
一个客服答复用不到完整语法，少一份依赖就少一处断网风险。
配套地，`_mock_generate` 也改成**空行分块**输出（表格保持成块），
否则整张表会被压成一行竖线塞进气泡。

**界面只呈现用户该看的东西**：意图、节点路径、证据分数、工具名、耗时这些
内部信息一律不渲染（开发者走 `/docs` 和 `/api/trace/{request_id}`）。
知识库里的 Markdown 记号（`**加粗**`、表格分隔行）也会在推给前端前收拾干净，
不会原样露在卡片上。这几条约束都有测试守着。

实测（本机真实运行）：

```
GET /api/health → 200
{"status":"ok","components":{"llm":"mock","embedding":"mock","vector_store":"memory",
 "reranker":"mock","database":"sqlite","observability":"local"},
 "fully_mocked":true,"kb_chunks":70,"db_ready":true, ...}

POST /api/chat/sync {"user_id":"u1","session_id":"s1","message":"运费怎么算"} → 200
  intent policy_faq | exit knowledge | path ['resolve','classify','retrieve','gate','generate']
  answer 根据平台规则：\n\n运费与损失承担：无理由退货的往返运费由用户承担…
  （**空行分块**：段落之间空一行，表格逐行保留 —— 前端据此渲染成段落与表格）
  usage  {'input_tokens': 524, 'output_tokens': 363, 'total_tokens': 887, 'cost_usd': 0.0}

GET /api/orders → 200
  {"count":2,"demo":true,"orders":[{"order_id":"ORD-1001","status_label":"待付款/待处理",
   "title":"MewSound Air 无线蓝牙耳机","total_amount":728.0,"item_count":2}, …]}
  （界面上的订单快捷选择用它 —— 没传 user_id 时 demo 为 true，标明是演示数据）
```

SSE 事件流在 `POST /api/chat`，事件名按契约冻结：`meta` / `token` / `tool` / `evidence` / `done` / `error`。

### 实测输出 ①：`python -m mewhelp.cli demo`（本机真实运行，节选）

```
════════════════════════MewHelp 分流演示════════════════════════
  意图映射表：
    order_query     查订单     → business_tool
    logistics_query 查物流     → business_tool
    refund_request  申请退款    → high_risk
    return_request  申请退货    → high_risk
    cancel_order    取消订单    → high_risk
    change_address  修改收货地址  → high_risk
    product_faq     商品咨询    → knowledge
    policy_faq      规则政策咨询  → knowledge
    human_handoff   转人工     → handoff

【知识检索】生鲜类可以退款吗
  意图 policy_faq → 出口 knowledge
  路径：resolve → classify → retrieve → gate → generate
  证据（5 条）：
    [1] 1.000  生鲜类  (退款政策#0003)
    [2] 0.998  生鲜类  (退款政策#0005)
    [3] 0.993  生鲜类  (退款政策#0004)
  答复（已答复）：
    根据平台规则：

    生鲜类：生鲜类商品（蔬果、肉禽、水产、乳品、冷冻食品）不适用七天无理由退款，
    只受理**品质问题**退款，且必须在签收后 24 小时内提交，同时上传商品照片。

    生鲜类：
    商品状态 | 申请时间窗 | 退款比例 | 是否需要寄回
    未发货 | 支付后任意时间 | 100% | 否
    已发货未签收 | 拦截成功后 | 95% | 否
    …（表格逐行保留，前端渲染成真正的表格，而不是一行竖线）

【高风险（槽位齐备）】帮我取消订单 ORD-1001
  意图 cancel_order → 出口 high_risk
  路径：resolve → classify → high_risk_flow → generate
  工具：
    [OK ] cancel_order
  答复（已答复）：
    已为你取消订单 ORD-1001。退款将按原支付方式退回，1 到 3 个工作日到账。

【知识库外（应拒答转人工）】你们卖不卖火箭
  意图 policy_faq → 出口 knowledge
  路径：resolve → classify → retrieve → gate → handoff → flywheel
  证据（5 条）：
    [1] 0.100  积分规则  (会员与优惠#0008)
    [2] 0.098  开票时效与限制  (发票与售后#0007)
  答复（转人工（weak_evidence））：
    抱歉，我找到的资料相关性只有 0.10（低于 0.35），不足以支撑一个确定答复，帮你转人工确认。

【转人工】我要找人工
  意图 human_handoff → 出口 handoff
  路径：resolve → classify → handoff → flywheel
```

### 实测输出 ②：`python -m mewhelp.cli index`

```
  文件                      chunk    表格      平均字数
  会员与优惠.md                   11     3       123
  发票与售后.md                   16     5       108
  商品与尺码.md                   13     5       183
  物流与配送.md                   10     3       173
  退款政策.md                    12     3       135
  退货政策.md                     8     2       158
  合计：70 个 chunk（其中表格 21 个），平均 144 字，最长 485 字
  索引完成：写入 70 条，检索器现有 70 条
```

### 实测输出 ③：`python -m mewhelp.cli chat "运费怎么算"`（尾部）

```
  状态：正常答复   耗时 43.544 ms   tokens 524/367
  答复：
    根据平台规则，运费与损失承担：无理由退货的往返运费由用户承担，除非商品页明确标注"退货包运费"…
```

### 中断 / 恢复（高风险子流程）

```python
from langgraph.types import Command
from mewhelp.config import get_settings
from mewhelp.workflow.graph import build_default_graph, make_sqlite_checkpointer
from mewhelp.workflow.state import new_state

graph, deps = build_default_graph(checkpointer=make_sqlite_checkpointer("ck.db"))
deps.retriever.load_from_dir(get_settings().knowledge_dir)

cfg = {"configurable": {"thread_id": "t-1"}}
graph.invoke(new_state(request_id="r1", session_id="s1", user_id="u1", question="我要退款"), cfg)
# → 图挂在 high_risk_flow 里，__interrupt__ = {'missing': ['order_id'], 'prompt': '可以帮你申请退款。请把订单号发我…'}

graph.invoke(Command(resume="ORD-1007"), cfg)
# → ['resolve', 'classify', 'high_risk_flow', 'generate']
# → "订单 ORD-1007 的退款申请已提交，审核通过后 1 到 3 个工作日退回原支付方式。"
```

实测结论：`interrupt()` 真的把图停在原地，`Command(resume=...)` 之后从 `high_risk_flow` 继续，**不是重跑整条流水线**。代价是每个线程要有独立的 `thread_id`，且状态会落盘。

---

## RAG 评估结果

评估集 300 题，五个桶各 60 题（`data/eval/golden.jsonl`），四种检索策略跑同一套题。

跑法：

```bash
PYTHONPATH=src python eval/run_eval.py          # 产物写到 eval/artifacts/ 与 data/eval/ablation.json
PYTHONPATH=src python eval/build_golden.py      # 重建评估集
```

### 四策略对照（本机真实运行结果）

| 策略 | Recall@5 | MRR | HitRate@5 | Precision@5 | P50(ms) | 拒答准确率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | 0.8743 | 0.7951 | 0.9625 | 0.2375 | 0.2 | 0.0000 |
| sparse | 0.9181 | 0.8624 | 0.9792 | 0.3100 | 0.1 | 0.9833 |
| hybrid | 0.9181 | 0.8522 | 0.9833 | 0.2533 | 0.3 | 0.0000 |
| **hybrid_rerank** | **0.9389** | **0.8914** | **1.0000** | 0.2592 | 5.2 | **1.0000** |

每一步的增量都能读出来：

- **双路融合**：`dense` → `hybrid` 把 HitRate@5 从 0.9625 提到 0.9833、Recall@5 从 0.8743 提到 0.9181。
- **精排**：`hybrid` → `hybrid_rerank` 把 HitRate@5 打到 1.0000、Recall@5 提到 0.9389、MRR 从 0.8522 提到 0.8914 —— **精排的收益主要在排序位置（MRR）而不是"能不能召回"**。
- **代价**：P50 从 0.3ms 涨到 5.2ms（mock 精排）。这个数量级相对于一次 LLM 调用可以忽略，但它说明"精排不是免费的"。
- **拒答**：`dense` / `hybrid` 的拒答准确率是 **0.0000**，而 `sparse` 是 0.9833、`hybrid_rerank` 是 1.0000。原因在 §"已知限制"里解释 —— 这是一个真实且有意义的发现，不是 bug。

### 分桶结果（`hybrid_rerank`）

| 桶 | 题数 | Recall@5 | MRR | HitRate@5 | 拒答准确率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| fact_lookup | 60 | 1.0000 | 0.8750 | 1.0000 | — |
| multi_condition | 60 | 0.8750 | 0.8806 | 1.0000 | — |
| multi_turn | 60 | 0.8806 | 0.8317 | 1.0000 | — |
| table_lookup | 60 | 1.0000 | 0.9783 | 1.0000 | — |
| no_answer | 60 | — | — | — | 1.0000 |

**分桶才看得出来问题在哪**：`fact_lookup` 和 `table_lookup` 已经满分（说明切分保住了表格结构），而 `multi_condition`（0.8750）和 `multi_turn`（0.8806）是短板 —— 前者是切分的边界，后者是指代消解。

### 指代消解的增量（multi_turn 桶单独对照）

| 做法 | Recall@5 | MRR | HitRate@5 |
| --- | ---: | ---: | ---: |
| 消解后再检索 | 0.8806 | 0.8317 | 1.0000 |
| 半截话直接检索 | 0.8222 | 0.7554 | 0.9333 |
| **增量** | **+0.0584** | **+0.0763** | **+0.0667** |

这是"指代消解这一层值不值"的直接证据 —— 不消解，`multi_turn` 桶的 HitRate 会掉 6.7 个百分点。

---

## 切真栈

每个组件都是"mock / 真实"二选一，**代码路径完全一致**，只改 `.env`。

```bash
docker compose up -d          # Milvus + etcd + MinIO + MySQL + Langfuse
```

改 `.env` 的这几行：

```ini
# 1) 真实 LLM（任何 OpenAI 兼容端点：官方 / 中转 / vLLM / Ollama）
MEWHELP_LLM_MODE=openai
MEWHELP_LLM_MODEL=gpt-5-mini
MEWHELP_OPENAI_API_KEY=sk-...
MEWHELP_OPENAI_BASE_URL=          # 留空走官方

# 2) 真实向量库（Milvus）
MEWHELP_VECTOR_STORE_MODE=milvus
MEWHELP_MILVUS_URI=http://localhost:19530

# 3) 真实业务库（MySQL）
MEWHELP_DATABASE_URL=mysql+pymysql://mewhelp:mewhelp@localhost:3306/mewhelp

# 4) 可观测（Langfuse）
MEWHELP_LANGFUSE_ENABLED=true
MEWHELP_LANGFUSE_PUBLIC_KEY=pk-...
MEWHELP_LANGFUSE_SECRET_KEY=sk-...

# 5) 真实 Embedding / 精排（约 2GB 模型下载）
MEWHELP_EMBEDDING_MODE=bge            # BAAI/bge-m3
MEWHELP_RERANKER_MODE=bge             # BAAI/bge-reranker-v2-m3
#    需要先装：python -m pip install -e ".[bge]"
```

判定标准在 `config.Settings.is_fully_mocked`：只要 `llm_mode=mock`、`embedding_mode=mock`、`vector_store_mode=memory` 三者同时成立就是纯离线模式。

---

## 目录结构

```
src/mewhelp/
  domain.py          领域常量：9 类 Intent、5 个 Exit、拒答原因、错误码、节点名（纯数据，零依赖）
  config.py          集中配置：所有可调参数都在这里，模块不各自读 os.environ
  llm.py             LLMClient 抽象 + MockLLMClient（确定性规则）+ OpenAILLMClient（任何兼容端点）
  schemas.py         跨模块数据契约（Pydantic v2）—— 检索/NLU/编排/API 之间唯一的接缝
  cli.py             命令行入口：demo / chat / index / flywheel / serve
  retrieval/         RAG：结构化切分、双路召回、RRF 融合、Cross-Encoder 精排、numpy/Milvus 两套向量库
  nlu/               指代消解 + 意图识别 + 槽位抽取（规则高速通道优先，模型兜底，异常不外泄）
  quality/           置信度闸门（生成前）+ 事实自评（生成后，模型与规则取更保守的一份）
  tools/             业务工具注册表（超时/重试/审计）+ 订单物流退款等 6 个业务工具
  memory/            双层会话上下文：近处滑窗保原文，更早的增量压成摘要
  workflow/          LangGraph 状态图：state / nodes / graph / high_risk 子流程
  flywheel/          数据飞轮：落池（带检索快照）→ 按频次排队 → 审核 → 回灌知识库
  observability/     Tracer：本地 TraceStep 始终收集，Langfuse 关闭时零开销 no-op
  storage/           SQLAlchemy 2.0 模型与引擎（SQLite / MySQL）
  finetune/          RoBERTa 全参微调的旁路主题分类（含不训练的基线对照）
  eval/              评估体系：5 个桶、4 种检索策略、Recall@5 / MRR / HitRate / 拒答准确率
  api/               FastAPI + SSE 服务层：app.py（路由）/ service.py（编排）/ static/（前端构建产物）

web/                 前端源码（React 19 + TypeScript + Vite）。只在开发/构建时需要 Node；
                     产物输出到 src/mewhelp/api/static/，所以**运行时不装 Node 也能跑**。

data/knowledge/      6 篇知识库原文（按标题层级 + 表格写，共切出 70 个 chunk，其中 21 个表格）
data/fixtures/       12 笔模拟订单（状态覆盖 5 种）+ 11 个商品参数
data/eval/           300 题评估集 golden.jsonl + 消融结果 ablation.json
eval/                build_golden.py（造题）/ run_eval.py（跑四策略对照）/ artifacts/（每策略一份报告）
examples/smoke_demo.py   最小离线冒烟：建库 → 跑图 → 打印路径
docs/                ARCHITECTURE.md（架构详解）/ INTERVIEW.md（面试问答）
tests/               466 项 pytest，全部离线可跑
finetune/            RoBERTa 微调产物目录
```

---

## 技术栈

| 层 | 技术 | 职责 / 为什么选它 |
| --- | --- | --- |
| 服务层 | FastAPI + `sse-starlette` + uvicorn | SSE 逐字推送；事件格式（`meta`/`token`/`tool`/`evidence`/`done`/`error`）在契约里冻结 |
| 编排层 | LangGraph 1.x（`StateGraph` + 条件边 + reducer） | 业务铁律画在图上；`interrupt()` + `SqliteSaver` 提供真正的中断/恢复语义 |
| NLU | 自研规则通道 + OpenAI 兼容 LLM | 规则先跑（零成本、可解释、确定性），模型只处理规则判不出来的句子 |
| 检索 | jieba + `rank_bm25` + numpy / Milvus 2.5 | Milvus 2.5 内建 BM25 稀疏向量，**一个 collection 同时扛稠密和稀疏两路**，不必再引一套 ES |
| 融合 | RRF（`k=60`） | 扔掉分数只看排名，免疫余弦分与 BM25 分的量纲差异 |
| 精排 | Cross-Encoder（`bge-reranker-v2-m3`）/ mock 伪 IDF 覆盖度 | 多条件规则题最容易出错；实测把 HitRate@5 打到 1.0000、MRR +0.039 |
| 闸门 | 纯规则（`quality/gate.py`） | 零模型调用、微秒级、同输入同输出，所以可以每次请求都跑 |
| 自评 | 模型判断 ∪ 确定性规则校验 | 只盯数字类声明（天数/金额/时效/折扣），取两者中更保守的一份 |
| 工具层 | 线程池超时 + 重试 + 审计 | 异常一律变成 `ToolCall(ok=False)`，重试次数如实落进记录 |
| 存储 | SQLAlchemy 2.0（SQLite / MySQL） | 审计日志、会话、飞轮记录、反馈 |
| 观测 | Langfuse 3（可选）+ 本地 `TraceStep` | 关闭时**不 import langfuse、不建连接**，连函数调用都不进 |
| 评估 | 自研 `eval/metrics.py` | 5 个桶 + 4 种策略消融，指标口径写死在代码里 |
| 微调 | transformers + torch，`hfl/chinese-roberta-wwm-ext` | 旁路统计需求（17 类主题），全参微调，CPU 十几分钟跑完 |

---

## 测试

```bash
PYTHONPATH=src python -m pytest tests/ -q
```

**实测结果（本机真实输出）：**

```
........................................................................ [ 94%]
........................                                                 [100%]
456 passed in 15.60s
```

（`pyproject.toml` 里已配 `pythonpath = ["src"]`，所以在本仓库根目录直接 `python -m pytest tests/ -q` 也能跑通。）

分布：

| 文件 | 用例数 | 覆盖 |
| --- | ---: | --- |
| `test_tools.py` | 84 | 工具注册表 + 6 个业务工具的业务铁律 |
| `test_nlu.py` | 56 | 指代消解、意图识别、槽位抽取 |
| `test_quality.py` | 53 | 闸门判据、实体覆盖率、自评合并 |
| `test_retrieval.py` | 49 | 切分、RRF、四种策略、向量库 |
| `test_flywheel.py` | 35 | 去重合并、排队、审核、回灌幂等、分诊 |
| `test_observability.py` | 35 | Tracer、摘要裁剪、关闭态的零开销 |
| `test_workflow.py` | 35 | 图的分流、Agent 步数上限、高风险出口 |
| `test_api.py` | 29 | HTTP 路由、SSE 事件格式、反馈与飞轮接口 |
| `test_memory.py` | 29 | 双层上下文、增量摘要、token 度量 |
| `test_finetune.py` | 22 | 数据构造、分层切分、指标计算、基线 |
| `test_eval.py` | 17 | 指标口径、分桶、消融表渲染 |
| `test_storage.py` | 7 | 建表、约束、唯一键 |
| `test_interrupt.py` | 5 | `interrupt()` / `Command(resume=...)` 的中断恢复语义 |
| **合计** | **456** | |

---

## 设计取舍

按"不这么做会怎样"来读，每条都对应一处代码。更详细的版本见 `docs/ARCHITECTURE.md`。

### 1. Workflow + Agent 混合，而不是纯 Agent

**为什么**：客服场景里"退款、取消、改地址"是会真的动钱的动作。纯 Agent 的失败模式是**不可枚举**的 —— 你没法用测试证明"模型不会心软"。混合架构把决策分了两类：能枚举的先验规则（意图 → 出口 → 状态机）交给图，需要理解的（话术、指代消解、槽位抽取）交给模型。

**不这么做会怎样**：模型可以"帮用户一次"，直接承诺一个不符合政策的退款。而在混合架构里，模型只能*提议*，真正生效要过 `tools/business.py` 的 `request_refund` 这道门，门后面的规则是代码，可以被单测钉死。三处具体体现：

- `nodes.py::make_agent_node` 里，Agent 拿到 `high_risk_actions()` 里的工具名时，直接写一条 `ToolCall(ok=False, error="HIGH_RISK_REQUIRES_FLOW")` 并中断循环 —— 不是"提示模型不要调"，是**物理上不给这个能力**。
- `high_risk.py::run_high_risk_flow` 里没有任何模型调用，流程是线性的：查槽位 → 调工具 → 如实转达。
- 工具拒绝的话术（`_REJECTION_TEXT`）刻意不交给模型写 —— "做了什么"是事实陈述，措辞可以润色，事实不能。

### 2. 九类意图收敛到五个出口

**为什么**：收敛带来三个具体好处。

- **成本可以按出口统计**：五条边 = 五种成本结构（知识出口有检索+2 次模型调用，转人工出口 0 次）。九条边的话每个意图都要单独建一套成本模型。
- **相同流程只开发一次**：四条高风险意图（退款/退货/取消/改地址）共用一条子流程和一套中断/恢复逻辑。
- **图只有五条边**：`route_by_exit()` 就是一张 5 项的 dict 查表。新增一类意图（比如"催发货"）只要落进 `business_tool`，`graph.py` 一行都不用改。

**不这么做会怎样**：加一类意图就要动一次图的结构，图会随业务线性膨胀；成本报表的维度会爆炸；四套几乎一样的高风险流程各自演化，最后行为不一致（"退款要订单号但取消不要"这种没人能解释的差异）。

### 3. 置信度闸门必须卡在生成**之前**

**为什么**：`ConfidenceGate.check()` 的签名是 `(question, retrieval, intent, slots)` —— **没有 `answer`**，这是刻意的。顺序不同，代价差一个数量级：

- **先判后答**：预测都没发出去，最坏结果是多一次转人工。用户从没看到错的东西。
- **先答后判**：模型已经把话说完了，用户已经读到了。此时再"发现"证据不足，你只能追加一句"以上仅供参考"—— **幻觉已经送达，撤回不存在**。

闸门是纯规则的，零模型调用、微秒级（判据只有正则抽实体 + 几个算术）；它替代掉的是"一次生成调用 + 一次自评调用"。**先判后答省掉的不是钱，是用户看到幻觉的机会。**

**不这么做会怎样**：`RefusalReason.SELF_EVAL_FAILED` 会被大量触发 —— 模型已经写完回答，自评才发现证据不支撑，用户读到了一段没有依据的话，然后系统补一句"帮你转人工"。这正是 `nodes.py::make_generate_node` 里那道自评的**兜底定位**：它拦不住已经发生的幻觉，只能拦住"即将发出去"的那一次。

### 4. RAG 按标题层级 + 表格结构切分

**为什么**：电商规则几乎都是"条件 + 例外"的结构。以 `data/knowledge/退款政策.md` 的生鲜小节为例，一条完整规则横跨三处：**正文的条件（"必须在签收后 24 小时内提交"）→ 三档比例的表格（24h 内 100% / 24–48h 50% / 部分变质 80%）→ 段末的兜底（"超过 48 小时才反馈品质问题的生鲜订单，原则上不予退款，由客服视情况发放等值优惠券作为补偿"）**。

定长滑窗（比如每 500 字一刀）很容易把表格与"超过 48 小时"那一句切开。检索到正文的模型看到的是"生鲜只受理品质问题退款，且必须在签收后 24 小时内提交"——它会答"生鲜需要 24 小时内提交"。这句话漏掉了"24 至 48 小时可退 50%"，得到的是一个**比真实政策更严格**的答案（用户本该能拿回 50% 却被告知不能退）。**半个条件是错误答案，不是不完整的答案**，而且它读起来完全通顺，没有任何可察觉的错误信号。

**不这么做会怎样**：所以切分的第一原则是**语义边界优先于长度边界**：① 标题是天然边界，跨标题绝不合并、也绝不切开；② 表格是原子单位，整张表一个 chunk（拆了就丢列头，单元格变成孤儿）；③ 只有一个小节超长（>800 字）时才在**段落**边界细分，且**单个段落再长也不切**（`_pack_paragraphs`）。

配套的两个细节：`KbChunk.prompt_text()` 渲染表格时会补回表头；`chunk_id` 用 `doc_id#序号`（四位补零）而不是内容哈希 —— 序号可读、可定位（"退款政策#0004 就是生鲜那张表"），且在文档内容不变时天然稳定，评估集里写死的 `expected_chunk_ids` 不会随重建索引失效。

### 5. 双路召回 + RRF，而不是加权求和

**为什么**：稠密的余弦分和 BM25 的分数量纲根本不可比 —— 余弦挤在 0.6~1.0，BM25 可以是 0~30 的任意值，**归一化也救不了，因为它们的分布形状不同**。RRF（`score(d) = Σ w/(k + rank(d))`）直接扔掉分数、只看排名，天然免疫量纲问题。

`k=60` 是原论文默认值，作用是压住"第 1 名和第 2 名"的差距，让多路都靠前的文档胜出，而不是被某一路的第一名一票否决。这正是要的性质：**两路都排第 3 的文档，通常比只有一路排第 1 的更靠谱。**

**不这么做会怎样**：加权求和需要为每一路调一个归一化方案（min-max？z-score？按查询归一化？）和一组权重，而这些参数会随知识库规模漂移 —— 换一批文档就要重调一次。RRF 没有需要随数据分布重调的参数。

实测支持：`dense` → `hybrid` 把 HitRate@5 从 0.9625 提到 0.9833。代码里还留了 `fusion.weighted_fuse()` 作为对照组，专门用来回答"为什么不用更简单的加权投票"。

### 6. 双层上下文：滑窗 + 摘要

**为什么**：多轮问答里最早的几轮往往携带"订单号 / 结论 / 用户诉求"这类关键实体，直接截断会让模型在第 10 轮突然失忆；但把全部历史塞进 prompt，token 成本随轮数**线性增长**，且越靠后的信噪比越低。所以分两层：近处 `window_turns=3` 轮保原文，更早的压成一段**可增量更新**的摘要，成本从 O(轮数) 降到 O(1)。

两个非平凡的取舍：

- **滑窗保原文是为了做字面替换。** 指代消解要靠它做精确匹配，摘要过的文字没法做 —— 所以不能反过来"全用摘要"。
- **摘要必须增量。** `ContextManager.build(turns, prev=上一轮的 bundle)` 只把"这次新滑出滑窗"的轮次并进旧摘要（prompt 里分成 `【已有摘要】` / `【本次新增轮次】` 两段）。不传 `prev` 就退化成"每次全量重摘"，token 成本又回到线性。
- **覆盖范围只信绝对下标，且必须"没有越界"**（`_coverage_index`）。历史被重放/截断时下标会失配，继续增量就会把实际没被摘要的轮次**永久跳过**，那是静默的信息丢失，比多花一次模型调用严重得多。

实测支持：指代消解让 `multi_turn` 桶的 Recall@5 从 0.8222 提到 0.8806（+0.0584），而不消解的半截话直搜正是滑窗拿不到实体时的表现。

**不这么做会怎样**：`ContextBundle` 带着 `summarized_turns` / `token_estimate` / `summary_generated` 三个可度量字段，就是为了让"省了多少 token、丢了几轮"能被度量，而不是藏在一段拼字符串的代码里。

### 7. 高风险子流程的中断 / 恢复

场景：

```
用户：我要退款
系统：请把订单号发我（形如 ORD-1007）…    ← 流程在这里暂停，而不是编一个订单号继续
用户：ORD-1007
系统：（从暂停的位置继续）订单 ORD-1007 的退款申请已提交…
```

**为什么用 `interrupt()` + checkpointer**：朴素做法是把"等用户补充"存成一次问答再重新走一遍全流程，代价是重复检索、重复调模型，而且第二轮很容易丢失第一轮的意图（"ORD-1007"这四个字进分类器可能被判成 `order_query`）。LangGraph 的 `interrupt()` 让流程真的"停在原地"，状态原样还在，`Command(resume=...)` 之后从 `high_risk_flow` 继续。

一个刻意的降级：**没挂 checkpointer 时不允许挂起。** `nodes.py::make_high_risk_node` 会检查 `deps.checkpointer`，没挂时把追问话术当作答复返回（`handoff=False`）—— 因为 `interrupt()` 在没有 checkpointer 时会直接抛异常，把整条请求打成 500。用户的诉求没被满足，但他明确知道下一步该给什么。这个判断显式地从 `GraphDeps.checkpointer` 读，而不是让节点自己回想"我编译时挂了没"。

**另一条安全线**：`high_risk.py` 的所有槽位抽取函数**只吃当轮文本**（`extract_order_id` 不扫历史）。如果用正则去历史里"扫"订单号，上一轮讨论过的订单会被当成这一轮的 —— 用户说"我要退款"（本轮无订单号）时，系统会拿上轮的订单**直接退款**。`tests/test_workflow.py:291` 有一条测试专门钉这件事。

### 8. 数据飞轮必须带检索快照

**为什么**：人工审核最容易犯的错，是把所有没答上来的问题都当成"缺知识"去写文档 —— 结果文档越写越多，检索还是捞不出来。事实是失败至少分两种：

| 快照状态 | 分诊结论 | 该做什么 |
| --- | --- | --- |
| 一条证据都没召回 | `backfill_kb` | 补文档 / 补 FAQ |
| 召回了但 `top_score < 0.35` | `tune_retrieval` | 调 embedding / 精排 / 阈值 —— **补文档完全无用** |
| 分数够了却没通过 | `review_answer` | 查生成与闸门判据，检索侧不用动 |
| 没有快照 | `unknown` | 无法判定 —— 这本身就是"必须存快照"的反面教材 |

区分它们的唯一依据就是落池那一刻的 `RetrievalResult`。**事后再去重跑检索是复现不出当时的现场的**（知识库已经变了、会话上下文没了）。

另外三个运营层面的取舍：

- **同题去重、`ask_count` 累加**：人工审核是稀缺资源，100 条记录里有 60 条是同一个问题，就必须先审那一条。去重键是问题的**归一化文本**（全角转半角、转小写、删所有标点和空白），`pending()` 按 `ask_count` 降序 —— 排队顺序本身就是运营优先级。
- **快照只补空、不覆盖**：第一次落池的现场才是现场，后来的重复提问盖掉它等于把证据毁掉。
- **排除流程问题**：`missing_slot` / `high_risk_unapproved` 不算知识侧问题（`is_knowledge_candidate`）—— 写一百篇文档也解决不了"用户没给订单号"，混进分诊表只会让"该写几篇文档"这个数字虚高。

### 9. 什么时候才轮到微调

顺序是 **Prompt → RAG → 规则 → 微调**，三个问题都答"能"就别微调：① Prompt 能不能解决？改措辞、加 few-shot，成本几乎为零。② RAG 能不能解决？知识不够就补知识，别拿模型参数去记事实。③ 规则能不能解决？有限类别、关键词明确时，规则比模型更准更便宜。

所以 `finetune/` 训的**不是意图识别**（9 类已经被 LLM + 规则解决了，再训一个模型是重复投资），而是**17 类主题分类** —— 运营要看"这个月用户在抱怨什么"，这类旁路统计不需要推理能力，只需要又快又便宜又稳定，正是判别式小模型的甜区。

**为什么 RoBERTa 全参而不是 LoRA**：LoRA / QLoRA 是给"全参训不起"的大模型（7B+）准备的 PEFT，冻结原参数只训一小撮低秩矩阵。而 RoBERTa 只有约 1 亿参数，**全参微调在 CPU 上十几分钟就能跑完**，根本不需要 PEFT 那套复杂度 —— 在这个量级上引入 LoRA 是增加了一个没必要的抽象，而不是优化。

配套的两件事让结论可信：`baseline_keyword()` 提供一个**不训练**的关键词/字符 bigram 基线（没有基线的"准确率 92%"没有意义，你分不清是模型的功劳还是任务本来就简单）；`_compute_metrics` 手写混淆矩阵 + 每类 F1（只看 Macro-F1 会掩盖"某个类完全没学会"）。

### 10. 可观测层：trace 是产品功能，不是调试开关

**为什么**：`langfuse_enabled` 只决定"要不要往远端导出"，**不决定"要不要留痕"**。`GET /api/trace/{request_id}` 要能返回调用树，所以就算不开 Langfuse，`TraceStep` 也照收（实测空 step 约 12µs，一条请求十几步 ≈ 0.2ms，相对一次 LLM 调用可以忽略）。反过来，关闭时必须**零成本**：不 import langfuse、不建 HTTP 连接、不碰 OpenTelemetry —— import 全部关在 `_LangfuseBridge` 里，Tracer 关闭时连这个类都不会被实例化。

**不这么做会怎样**：没做这件事的项目，离线用户会被一个观测 SDK 拖慢甚至拖挂。另外 `_StepScope.__exit__` 永远返回 `False` —— **观测代码的职责是记录异常，不是处理异常**。

### 11. 一切失败都降级，异常绝不外泄

**为什么**：流水线的价值在于"部分失败也能给出有用结果"。整条链路里每一层都有明确的降级路径：

| 挂掉的东西 | 降级成什么 |
| --- | --- |
| 精排器 | 保留 RRF 融合结果（`pipeline.py`） |
| 向量库 | 退化成纯 BM25 单路仍然可用；BM25 也挂了退化成"零证据"→ 闸门拒答 |
| 嵌入模型 | 原文仍留在内存里，BM25 单路可用 |
| 闸门自己 | 抽出 `NO_EVIDENCE` 拒答（键位与正常路径完全一致，下游不会 KeyError） |
| 自评模型 | 降级为纯规则校验（数字能不能在证据里逐字找到） |
| 意图分类 | 兜底成 `policy_faq`（knowledge 出口；给高风险意图会无中生有地向用户要订单号） |
| 指代消解 | 返回**原文**（没补全顶多让检索差一点，抛异常会让用户什么都拿不到） |
| 摘要生成 | 无摘要 + 完整滑窗继续 |
| 工具调用 | `ToolCall(ok=False, error=<ToolError>)`，不抛异常 |
| 飞轮落池 | 记一笔 error，不影响这次答复 |
| Langfuse | 只留本地 trace + 一条 warning，业务无感 |

**不这么做会怎样**：异常冒到图外面，用户拿到的就是一个 500，而不是一句"我暂时查不到，帮你转人工"。文件级的一致约定写在 `workflow/nodes.py` 的模块 docstring 里：**节点不抛异常、每步都写 trace 和 path、节点之间不共享可变状态。**

### 12. 确定性

**为什么**：同输入同输出不是洁癖，是"能跑消融实验"和"能复现线上问题"的前提。

- 不用内置 `hash()`（CPython 对 str 有进程级随机盐）：`flywheel/store.py::_stable` 和 `retrieval/embedders.py::_stable_hash` 都用 `hashlib`（sha1 / blake2b）。
- 排序永远有确定性的二级排序键：RRF 同分按 `chunk_id` 升序（`fusion.py`），`pending()` 并列时按 `(created_at, record_id)` 再排（`store.py`），关键词按 `(词频, 词)` 排（`chunking.py`）。
- 工具夹具里 `as_of: 2026-08-20` 声明了基准日期，`_resolve_today()` 在没有显式注入时用它兜底 —— 否则"正好第 30 天可退"这条用例明天就失效，CI 变成薛定谔的红。生产实现去掉这个兜底。
- `TokenUsage.cost_usd` 在 mock 下恒为 0：离线跑不能有"假成本"。

### 13. 已知限制与在途项

诚实写在这里，避免读者用错误的预期去验证：

- **业务工具出口（`business_tool`）的答复没有接上 Agent 的工具返回。** `nodes.py::make_agent_node` 不写 `answer`，`make_generate_node` 的 prompt 只拼 `RetrievalResult.to_context()`，**不含 Agent 的工具结果**。实测（`mewhelp demo` 的第一题）："帮我查下订单 ORD-1002 到哪了" → `track_logistics` / `get_order` 都返回 `[OK]` → 但答复是通用的"抱歉，我暂时没有找到可以回答这个问题的依据"，状态为 `转人工（self_eval_failed）`。`tests/test_workflow.py` 只断言了这条路径的 `tool_calls`，没有断言答复内容，所以这个缺口没有被测试暴露。
- **`dense` / `hybrid` 的拒答准确率是 0.0000，`hybrid_rerank` 是 1.0000 —— 这不是评测 bug。** 原因：除精排外的三条策略，证据分都来自 RRF 的相对归一化（`raw_score / prior_max`），也就是"本查询最好的一条"恒等于 1.0，必然高于阈值 0.35，所以 `no_answer` 桶永远不会被分数拦下；`sparse` 之所以能到 0.9833，是因为那批问题在 BM25 侧零词面命中（`s > 0` 过滤后无证据）→ 走 `no_evidence` 拒答。**这个对比说明：精排除了排序，还提供了闸门赖以工作的绝对分数标尺。** 换句话说，"要不要精排"不只是效果问题，也是"闸门能不能工作"的问题。
- **mock 精排的绝对分在个别查询上仍偏宽松。** 例如"你们在火星有仓库吗"实测拿到 `top_score = 0.3906`（略高于 0.35，因 "仓库" 与退货流程的片段有字面重叠），而 `low_coverage` 判据因该问题抽不出可加权实体而跳过。这类边界案例是接真实 embedding/reranker 后需要重新校准阈值的地方。
- **契约与代码的两处出入**：`CONTRACT.md` §2.F 的图里有独立的 `rerank` 节点，实际没有（精排在 `RetrievalPipeline.retrieve()` 内部）；`domain.NodeName.RERANK` / `TOOL` 两个常量当前未被使用。
- **`nodes.py:494` 有一行自赋值 `usage = usage`**（自评调用的 except 分支里），是个无效果的死语句，属可读性问题，不影响行为。

### 本文档的验证范围

本文里每条命令都在本机实际跑过，输出是从终端**直接复制**的（不是示意）。唯一两处**没有**验证的：

- `python -m pip install -e .` —— 本机依赖已装，直接用 `PYTHONPATH=src` 运行，没有执行 editable 安装（它会重新解析 `pyproject.toml` 的依赖，可能改动现有环境）。入口 `mewhelp = "mewhelp.cli:main"` 已在 `pyproject.toml` 里声明。
- `docker compose up -d` —— 需要 Docker，且本项目承诺"离线可跑"，所以没有为了写文档去启动真栈。
