# MewHelp 模块契约（冻结版）

> 所有并行开发的模块都按这份契约实现。**不要修改 `domain.py` / `config.py` / `llm.py` / `schemas.py`
> 已定义的公共接口**；如果需要新增字段，只能往后追加且必须有默认值。

## 0. 项目根

```
E:\临时\projects\mewhelp\
  pyproject.toml  requirements.txt  docker-compose.yml  .env.example
  src/mewhelp/
    __init__.py     # 已冻结
    domain.py       # 已冻结 —— 常量：Intent / Exit / RefusalReason / ToolError / FlywheelSource / NodeName
    config.py       # 已冻结 —— Settings + get_settings()
    llm.py          # 已冻结 —— LLMClient / MockLLMClient / OpenAILLMClient / build_llm() / extract_json()
    schemas.py      # 已冻结 —— 全部 Pydantic 契约
  tests/
  data/knowledge/   data/eval/   data/fixtures/
  eval/             # 评估脚本
  finetune/         # 第 10 章 RoBERTa 微调
```

安装运行依赖（已装好，无需重装）：
```
python -m pip install -e .          # 或 -r requirements.txt
```

跑命令时统一用：
```
cd E:/临时/projects/mewhelp
PYTHONPATH=src python -m pytest -q
```

**Windows 控制台是 GBK**，打印中文请加 `PYTHONIOENCODING=utf-8`，否则看到乱码不是 bug。

---

## 1. 已冻结的核心契约（直接 import，不要重定义）

### domain.py
```python
class Intent(StrEnum): order_query / logistics_query / refund_request / return_request /
                       cancel_order / change_address / product_faq / policy_faq / human_handoff
class Exit(StrEnum):   knowledge / business_tool / high_risk / handoff / fallback
INTENT_TO_EXIT: dict[Intent, Exit]        # 9 类 → 5 出口（knowledge 2 / business_tool 2 / high_risk 4 / handoff 1）
HIGH_RISK_INTENTS: frozenset[Intent]      # refund/return/cancel/change_address
TOOL_INTENTS: frozenset[Intent]           # order_query / logistics_query
INTENT_SPECS: tuple[IntentSpec, ...]      # 每项含 intent/label/keywords/examples
INTENT_KEYWORDS: dict[Intent, tuple[str,...]]
INTENT_LABELS: dict[Intent, str]
class RefusalReason(StrEnum): no_evidence / weak_evidence / low_coverage /
                              self_eval_failed / missing_slot / high_risk_unapproved
class ToolError(StrEnum): ORDER_NOT_FOUND / ORDER_NOT_CANCELLABLE / ORDER_ADDRESS_LOCKED /
                          REFUND_NOT_ELIGIBLE / TIMEOUT / UPSTREAM_ERROR
class FlywheelSource(StrEnum): refused / self_eval / thumbs_down
class NodeName(StrEnum): resolve / classify / route / retrieve / rerank / gate / tool /
                         high_risk_flow / agent / generate / handoff / flywheel
```

### llm.py
```python
class LLMClient(ABC):
    model: str
    def chat(self, messages, *, tools=None, temperature=None, max_tokens=None,
             json_mode=False, task="generic") -> LLMResponse: ...

@dataclass
class LLMResponse:
    text: str; usage: TokenUsage; model: str
    tool_calls: list[dict]   # [{"id","name","arguments"}]
    finish_reason: str; latency_ms: float; raw: Any
    def json(self) -> Any    # 容错解析（剥 ```json 围栏）

build_llm(settings=None) -> LLMClient        # mock | openai
extract_json(text) -> Any
MOCK_TASKS: frozenset[str]
```

**`task` 取值（mock 按它分派规则，真实实现忽略）：**
| task | 期望返回 JSON |
|---|---|
| `resolve` | `{"resolved": str, "rewritten": bool, "substitutions": {pron: entity}}` |
| `classify` | `{"intent": <Intent.value>, "confidence": float, "matched": [str], "fallback": bool}` |
| `slots` | `{"order_id": str\|null, "address": str\|null, "reason": str\|null, "product": str\|null}` |
| `self_eval` | `{"grounded": bool, "coverage": float, "unsupported_claims": [str], "note": str}` |
| `generate` | 纯文本回答（不是 JSON） |
| `plan_tool` | `{"thought": str, "action": <tool_name\|"finish">, "action_input": {}}` |
| `summarize` | 纯文本摘要 |

### schemas.py（全部 Pydantic v2）
`KbChunk` `Evidence` `RetrievalResult` `ResolveResult` `ClassifyResult` `SlotExtraction`
`GateDecision` `SelfEvalResult` `ToolCall` `ToolSpec` `Turn` `ContextBundle`
`TraceStep` `TokenUsage` `ConversationResult` `FlywheelRecord`

关键方法：
- `KbChunk.prompt_text() -> str`（表格会补回表头）
- `RetrievalResult.to_context(max_chars=4000) -> str`（渲染成 `[1] [标题]\n正文` 的编号块）
- `RetrievalResult.top_score -> float`
- `SlotExtraction.missing_for(intent) -> list[str]`
- `GateDecision.ok(conf, checks)` / `GateDecision.reject(conf, reason, checks, message="")`
- `ContextBundle.as_messages() -> list[dict]`

---

## 2. 各模块要实现的接口

### A. 检索层 `retrieval/`
```python
# retrieval/base.py
class Embedder(ABC):
    dim: int
    def encode(self, texts: list[str]) -> list[list[float]]: ...

class VectorStore(ABC):
    def upsert(self, chunks: list[KbChunk], vectors: list[list[float]]) -> int: ...
    def search(self, vector: list[float], top_k: int) -> list[tuple[str, float]]: ...  # (chunk_id, score)
    def delete_all(self) -> None: ...
    def count(self) -> int: ...

class Reranker(ABC):
    def rerank(self, query: str, evidences: list[Evidence], top_k: int) -> list[Evidence]: ...

# retrieval/chunking.py
def chunk_document(doc_id: str, title: str, markdown: str, source_path: str = "") -> list[KbChunk]

# retrieval/pipeline.py
class RetrievalPipeline:
    def __init__(self, embedder, store, reranker, settings=None): ...
    def index(self, chunks: list[KbChunk]) -> int: ...
    def retrieve(self, query: str, *, strategy: str = "hybrid_rerank") -> RetrievalResult: ...
    def load_from_dir(self, path: Path) -> int: ...   # 读 data/knowledge/*.md 建库

# retrieval/fusion.py
def rrf_fuse(rankings: list[list[str]], k: int = 60, weights: list[float] | None = None) -> list[tuple[str, float]]
```
`strategy` 四选一（消融实验用）：`"dense"` `"sparse"` `"hybrid"` `"hybrid_rerank"`。
BM25 用 `rank_bm25.BM25Okapi`，中文分词用 `jieba`。

### B. NLU 层 `nlu/`
```python
# nlu/coreference.py
class CoreferenceResolver:
    def __init__(self, llm, settings=None): ...
    def resolve(self, question: str, history: list[Turn]) -> ResolveResult: ...

# nlu/intent.py
class IntentClassifier:
    def __init__(self, llm, settings=None): ...
    def classify(self, text: str) -> ClassifyResult: ...
    def extract_slots(self, text: str, history: list[Turn]) -> SlotExtraction: ...
```
两者都必须：先走规则高速通道（高置信直接返回，`method="rule"`），未命中再调 `llm.chat(..., task="classify"/"resolve")`，任何异常都兜底成低置信结果（`method="fallback"`）而不是抛出去。

### C. 会话上下文 `memory/`
```python
# memory/context.py
class ContextManager:
    def __init__(self, llm, settings=None): ...
    def build(self, turns: list[Turn]) -> ContextBundle: ...
    def append(self, session: list[Turn], turn: Turn) -> list[Turn]: ...
```
双层：近 `window_turns` 轮保原文进 `window`，更早的经 `llm.chat(task="summarize")` 压成 `summary`。
要维护 `summarized_turns` / `token_estimate`，让"省了多少 token、丢了几轮"可被度量。

### D. 工具层 `tools/`
```python
# tools/registry.py
class ToolRegistry:
    def register(self, spec: ToolSpec, fn: Callable[..., Any]) -> None: ...
    def specs(self) -> list[dict]                 # OpenAI function-calling schema
    def call(self, name: str, args: dict) -> ToolCall   # 带超时/重试/审计
    def describe(self) -> str                     # 给 prompt 用的工具清单

# tools/business.py  —— 业务工具，基于 data/fixtures/orders.json
get_order(order_id) / track_logistics(order_id) / search_faq(query)
request_refund(order_id) / cancel_order(order_id) / change_address(order_id, address)
```
后三个是 `high_risk=True`：**Agent 不能直接调用**，只能提议，由高风险子流程决定。
错误一律返回 `ToolCall(ok=False, error=<ToolError 值>)`，不抛异常。

### E. 闸门 `quality/`
```python
# quality/gate.py
class ConfidenceGate:
    def __init__(self, settings=None): ...
    def check(self, question: str, retrieval: RetrievalResult,
              intent: Intent, slots: SlotExtraction) -> GateDecision: ...

# quality/selfeval.py
class SelfEvaluator:
    def __init__(self, llm): ...
    def evaluate(self, answer: str, context: str) -> SelfEvalResult: ...
```
`check()` 必须**在生成之前**调用。判据写进 `checks` 字典，逐条可解释：
`top_score` / `evidence_count` / `score_margin` / `entity_coverage`。阈值来自 `settings.confidence_threshold`。

### F. 工作流 `workflow/`
```python
# workflow/state.py
class AgentState(TypedDict, total=False):
    # 输入
    request_id: str; session_id: str; user_id: str; question: str
    history: list[Turn]
    # NLU
    resolve: ResolveResult; intent: Intent; intent_confidence: float
    slots: SlotExtraction; exit: Exit
    # 检索
    retrieval: RetrievalResult
    # 闸门
    gate: GateDecision
    # 工具 / Agent
    tool_calls: list[ToolCall]
    # 输出
    answer: str; refusal_reason: RefusalReason | None; handoff: bool
    # 子流程中断
    interrupt_prompt: str | None; missing_slots: list[str]
    # 记分
    trace: list[TraceStep]; usage: TokenUsage; path: list[str]

# workflow/graph.py
def build_graph(deps: GraphDeps, checkpointer=None) -> CompiledGraph
class GraphDeps:  # 依赖注入容器，测试时全换成 mock
    llm; resolver; classifier; retriever; gate; self_eval; tools; context; flywheel
```
图的形状：
```
START → resolve → classify → route ─┬→ retrieve → rerank → gate ─┬→ generate → END
                                     │                            └→ handoff  → flywheel → END
                                     ├→ agent(ReAct) ────────────────→ generate → END
                                     ├→ high_risk_flow ─┬→ generate → END
                                     │                  └→ [interrupt] → 恢复后继续
                                     └→ handoff → flywheel → END
```
`route` 是条件边，按 `INTENT_TO_EXIT` 分流。`interrupt` 用 LangGraph 的 `interrupt()` +
checkpointer（`langgraph.checkpoint.sqlite.SqliteSaver`），缺槽位时暂停，用户补充后从原位置继续。

### G. 数据飞轮 `flywheel/`
```python
# flywheel/store.py
class FlywheelStore:
    def record(self, rec: FlywheelRecord) -> FlywheelRecord   # 同题去重，ask_count += 1
    def pending(self, limit=50) -> list[FlywheelRecord]        # 按 ask_count 降序（被问频次优先）
    def review(self, record_id: str, approved: bool, reviewer: str, answer: str | None = None) -> FlywheelRecord
    def stats(self) -> dict

# flywheel/feedback.py
def source_for(reason: RefusalReason | None, thumbs_down: bool) -> FlywheelSource
```
`record()` 必须**把当时的 `RetrievalResult` 一起存进 `retrieval_snapshot`** —— 没有快照，人工
就无法判断"真缺知识"还是"没检索到"，而这两种情况处理方式完全不同。

### H. 可观测 `observability/`
```python
# observability/tracing.py
class Tracer:
    def __init__(self, settings=None, enabled: bool | None = None): ...
    def step(self, node: str, **details) -> AbstractContextManager[TraceStep]: ...
    def flush(self) -> None
def build_tracer(settings=None) -> Tracer
```
`langfuse_enabled=False` 时必须是**零开销的 no-op**，且不 import langfuse。
挂载方式为零侵入：不改业务代码，用 LangGraph callback / 手动 span 包装节点。

### I. 服务层 `api/`
```python
# api/app.py
def create_app(deps: GraphDeps | None = None) -> FastAPI
# 路由
POST /api/chat          {user_id, session_id, message} → SSE 流（逐字）
POST /api/chat/sync     同上但返回完整 JSON
POST /api/feedback      {session_id, request_id, thumbs_down, comment}
GET  /api/flywheel/pending
POST /api/flywheel/review  {record_id, approved, reviewer, answer}
GET  /api/health
GET  /api/trace/{request_id}
# 会话历史（所有权靠 user_id 查询参数，没有鉴权 —— 见 README 已知限制）
GET    /api/sessions?user_id=            → 列表（标题 / 轮数 / 预览 / 更新时间）
GET    /api/sessions/{id}?user_id=       → 恢复对话（只回用户视角字段）
PATCH  /api/sessions/{id}?user_id=       {title}  重命名
DELETE /api/sessions/{id}?user_id=       → 删除（不动长期记忆）
# 长期记忆
GET    /api/memories?user_id=            → 全部档案（钉住的排前面）
POST   /api/memories                     {user_id, text, kind} 用户自己写一条（默认钉住）
PATCH  /api/memories/{id}                {user_id, pinned}
DELETE /api/memories/{id}?user_id=
GET  /                  服务 src/mewhelp/api/static/index.html
```
SSE 事件格式（**前端按这个解析，不要改**）：
```
event: meta     data: {"request_id":..,"intent":..,"exit":..,"path":[..]}
event: evidence data: {"chunks":[{"chunk_id":..,"title":..,"score":..,"snippet":..}]}
event: memory   data: {"items":[{"id":..,"kind":..,"kind_label":..,"text":..,"pinned":..,"hits":..}]}
event: tool     data: {"name":..,"ok":..,"result":..}
event: token    data: {"text": "..."}
event: done     data: {"answer":..,"usage":{..},"latency_ms":..,"handoff":..}
event: error    data: {"message":..}
```
顺序是硬约束：`meta → evidence → memory? → tool* → token* → done`。
`memory` 只在**该用户确实有档案**时才推 —— 前端因此可以把"收到这个事件"
直接当成"记忆面板该更新了"的信号，不必自己判断空列表。
推的是**全量**档案而不是增量：增量一旦漏掉一次事件，面板和真实记忆就会
永久对不上，而且没有任何报错。

### J. 存储 `storage/`
```python
# storage/db.py
def get_engine(settings=None) -> Engine        # sqlite 或 mysql+pymysql
def init_db(engine) -> None                    # 建表 + 给老库补新列（ALTER TABLE ADD COLUMN）
def ensure_utc(dt) -> datetime                 # 读出来的朴素时间补上 UTC 时区
# storage/models.py  —— SQLAlchemy 2.0 DeclarativeBase
class KbChunkRow   (chunk_id PK, doc_id, title, breadcrumb_json, text, keywords_json, is_table, source_path)
class AuditLogRow  (id PK, request_id, session_id, user_id, node, payload_json, tokens_in, tokens_out, created_at)
class SessionRow   (session_id PK, user_id, title, summary, turns_json, updated_at)
class UserMemoryRow(id PK, user_id, kind, mem_key, text, source, pinned, hits, created_at, updated_at,
                    UNIQUE(user_id, mem_key))
class FlywheelRow  (record_id PK, session_id, question, resolved_question, intent, source,
                    refusal_reason, snapshot_json, ask_count, status, reviewer, answer, created_at, reviewed_at)
class FeedbackRow  (id PK, session_id, request_id, thumbs_down, comment, created_at)
```

`session` 与 `user_memory` **必须是两张表**：前者的生命周期由用户决定
（"清空聊天记录"），后者跨会话保留。混在一张表里，一句级联删除就会让用户
发现自己写下的备忘跟着聊天记录一起没了。

### K. 长期记忆 `memory/`
```python
# memory/context.py —— 会话内上下文（不在这里展开，见 §2.C）
# memory/store.py
class MemoryStore:
    def list(user_id, *, limit=None) -> list[MemoryItem]
    def for_prompt(user_id, *, limit=None) -> list[MemoryItem]   # 进 prompt 的那几条（默认 8）
    def upsert(user_id, *, kind, text, key="", source="auto", pinned=False) -> MemoryItem | None
    def remember(user_id, candidate, *, source="auto", pinned=False) -> MemoryItem | None
    def delete(user_id, memory_id) -> bool          # 必须带 user_id（所有权校验）
    def set_pinned(user_id, memory_id, pinned) -> bool
    def mark_used(user_id, memory_ids) -> int       # "真的被用上"才计数
# memory/extract.py
def extract_candidates(llm, question) -> list[MemoryCandidate]   # task="memory"，失败退回规则
def extract_by_rules(question) -> list[MemoryCandidate]
def used_memory_ids(memories, question, resolved_question) -> list[int]
```

契约要点：
* 抽取走 `LLMClient`（`task="memory"`），mock 与真模型**共用同一张规则表** ——
  两处各写一套会出现"离线能记住、接上真模型就忘"。
* 只记**稳定的事实**（尺码 / 称呼 / 联系偏好 / 会员 / 关注的订单 / 用户写的备忘），
  不记一次性诉求（"我要退款"）和任何提问本身。
* `(user_id, mem_key)` 唯一：同键是**更新**（去重靠数据库约束，不是"先查再写"）。
* `hits` 记的是"帮上忙几次"：判据是某个值出现在**补全后**的问题里、却不在用户
  原话里。每轮被塞进 prompt 不算。
* 上限：每用户 50 条，超出淘汰最久未更新的**未钉住**条目。

### L. 数据文件
```
data/knowledge/*.md              # 知识库原文（按标题层级 + 表格写，供切分）
data/fixtures/orders.json        # 模拟订单，至少 8 单，覆盖各状态（待发货/已发货/已签收/已取消）
data/fixtures/products.json      # 商品参数，供商品咨询
data/eval/golden.jsonl           # 300 题评估集，分 5 桶
data/eval/ablation.json          # 四种检索策略的对照结果（脚本产出）
```

`orders.json` 字段：`{order_id, user_id, status, shipping_address, delivered_on, refund_status, items[], created_at}`
状态取值：`pending` / `processing` / `shipped` / `delivered` / `cancelled`

`golden.jsonl` 每行：
```json
{"qid":"Q001","bucket":"fact_lookup","question":"...","expected_intent":"...",
 "expected_chunk_ids":["..."],"golden_answer":"...","multi_turn":false}
```
5 个桶：`fact_lookup`(事实查询) `multi_condition`(多条件) `multi_turn`(多轮指代)
`table_lookup`(表格/结构化) `no_answer`(知识库没有，应正确拒答)

---

## 3. 硬性要求

1. **所有模块必须能纯 mock 跑通**，不依赖 Docker、不依赖 API Key。
2. **确定性**：同输入同输出。不要用 `random` 不设种子，不要在业务逻辑里用当前时间做判断分支。
3. **异常不外泄**：每个模块自己兜底，返回降级结果 + 记录 `error`，不要让异常冒到 API 层。
4. **不要在业务代码里直接读 `os.environ`**，一律走 `get_settings()`。
5. 每个模块配 `tests/` 下的 pytest 用例，**必须能在离线跑过**。
6. 中文注释，但注释解释"为什么这么做"，不要复述代码在做什么。
