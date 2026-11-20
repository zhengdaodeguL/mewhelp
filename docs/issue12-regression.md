# Issue #12 回归记录

基线：`1351d8bf7e5df438b25c4efd38c9352b387ca5ef`，分支 `fix/mewhelp-issue-12-dialog-handoff`。
先新增测试，未修改生产代码时运行：

```bash
MEWHELP_LANGFUSE_ENABLED=false MEWHELP_DATABASE_URL=sqlite:////tmp/mewhelp-issue12-red.db \
  .venv/bin/python -m pytest tests/test_issue12.py tests/test_issue13.py -q --tb=short
```

结果：**33 failed, 106 passed in 17.63s**。其中 #12 为 **33 failed / 6 passed**，#13 为 **100 passed**。
原始输出保存在 `/tmp/mewhelp-issue12-red.log`。

失败模式：

- 正确规格复述被模型自评否定，或生成中夹带转接句，最终出现多余转接。
- “无法确认”等未回答正文被自评放行；自评接口没有接收当前问题。
- 黑土豆表格及 15% 政策已覆盖，但商品槽位的逐字匹配导致闸门误拒。
- 模型误读重量问题或生成超时，已有规格、包装及价格证据没有确定性作答路径。
- “是不是退20%”“按数量算吗”没有被确定性政策规则识别，模型可将其送入退款流程。
- 会话保存的是展示用摘要，下一轮检索不到时没有完整表格/证据可以沿用。
- 完整包装规则被低精排分一刀切过滤；冲突区间、未知档位或赔偿问题不能套用重量比较。
- 真正生成失败没有拒答原因，并重复追加转接通知。

生鲜报告未附完整知识库。测试只转录报告明确给出的事实，另使用仓库
`data/fixtures/products.json` 和 `data/knowledge/退款政策.md` 验证跨商品、表格、政策行为；
未编造报告中未给出的红运道价格。

## 修复

- 新增共享的证据规则：按文档主体定位规格、包装定义、价格表与政策条款，
  用 `Decimal` 比较明确的净重区间。答案及数字来自原始片段，不存商品答案模板。
  闸门与检索共用覆盖结果；原始分数不变，`content_supported` 记录内容通过原因。
  无完整规则的证据仍须满足原有分数、条数及主题要求。
- 常见商品属性与售后比例咨询确定性分类；保留显式订单号、操作申请及“这单”
  的原有优先级，未修改 Agent 当轮订单范围约束与高风险工具流程。
- 生成先使用已核验的规则；其余生成与自评使用温度 0。自评新增当前问题与
  `answered` 判据，本地拦截空回答、“无法确认”等未回答结果；保留已回答问题时
  对未被询问事项的合理保留意见。
- 生成失败、自评未通过都明确拒答、记录原因后转接，只保留一次转接通知。
  模型自行追加的转接确认在发送前过滤，同步和流式正文遵守同一规则。
  已发出的流式内容自评未通过时补充明确更正，非流式不交付未通过的原答案。
- 会话内部保存完整 `Evidence`（正文、表格、来源），不改变界面的摘要载荷。
  仅沿用最近一轮相关商品的证据，允许上一轮拒答但仍有检索证据；
  跨商品、跨会话及显式订单不借用。当前检索返回的同 ID 片段优先。
  按名称追问价格只摘录对应行。

## Green

修复中追加 6 个边界用例，并分别先记录失败：同句逗号分隔的冲突区间、
同步/流式的模型擅自转接、名称追问的价格行选择、未覆盖的复合问题、无价格的价格表。
日志为 `/tmp/mewhelp-issue12-review-red.log`、`/tmp/mewhelp-issue12-price-red.log`、
`/tmp/mewhelp-issue12-partial-red.log`、`/tmp/mewhelp-issue12-no-price-red.log`。

```bash
MEWHELP_LANGFUSE_ENABLED=false MEWHELP_DATABASE_URL=sqlite:////tmp/mewhelp-issue12-final-related.db \
  .venv/bin/python -m pytest tests/test_issue12.py tests/test_issue13.py -q --tb=short
```

结果：**145 passed in 25.60s**（#12 的 45 例 + 未改动的 #13 原有 100 例）。
原始输出：`/tmp/mewhelp-issue12-green.log`。

扩大验证还覆盖了 quality、workflow、NLU、retrieval、issue4、interrupt 模块；
该阶段为 **501 passed**。最终以全量运行结果为准。

```bash
MEWHELP_LANGFUSE_ENABLED=false MEWHELP_DATABASE_URL=sqlite:////tmp/mewhelp-issue12-full.db \
  .venv/bin/python -m pytest -q --tb=short
```

全量结果：**891 passed, 1 skipped in 75.84s**，包含 #13 原有 100 例。
原始输出：`/tmp/mewhelp-issue12-full.log`。
唯一跳过的是既有 `tests/test_observability.py::test_real_langfuse_points_at_dead_host`：
当前环境无法初始化 Langfuse，降级路径已由其他用例覆盖；以单独 `-q -rs` 运行
确认了这一原因（`/tmp/mewhelp-issue12-skip.log`）。未修改或跳过任何 #13 用例。

测试模型/向量/精排由 `tests/conftest.py` 固定为 mock，检索使用内存存储；
新增夹具显式关闭 Langfuse 并使用 pytest 临时目录下的 SQLite。
全量运行也通过环境变量隔离默认 SQLite，未重建虚拟环境或补装依赖。

## 改动文件

| 文件 | 说明 |
|---|---|
| `src/mewhelp/knowledge.py` | 共享证据覆盖、确定性答复及相关历史证据选择。 |
| `src/mewhelp/quality/gate.py` | 完整内容规则可通过闸门，避免粗槽位和低分误拒。 |
| `src/mewhelp/quality/selfeval.py` | 核对当前问题是否被回答，并拦截明确未回答结果。 |
| `src/mewhelp/nlu/intent.py` | 确定性区分商品/比例咨询，保留原订单请求边界。 |
| `src/mewhelp/retrieval/pipeline.py` | 与闸门共用内容判定，保留低分但明确覆盖的规则。 |
| `src/mewhelp/workflow/nodes.py` | 接入规则与跨轮证据，统一拒答、流式转接过滤及转接去重。 |
| `src/mewhelp/workflow/state.py` | 携带当前问题可复用的历史证据。 |
| `src/mewhelp/api/service.py` | 在会话内部保存完整证据快照。 |
| `src/mewhelp/schemas.py` | 为自评结果增加 `answered` 字段。 |
| `tests/test_issue12.py` | 45 个固定模型/原有数据回归用例。 |
| `docs/issue12-regression.md` | 记录根因、改动、red→green、全量结果及限制。 |

## 范围与限制

- 报告七项对话问题均有离线回归覆盖；未调用真实模型或访问 8125/8126 实例。
  完整生鲜数据集不在仓库，因此不能把固定事实回放当作原实例的实机复测。
- 确定性只覆盖可从当前证据明确核验的规则。复杂语义仍由生成与自评模型处理，
  不承诺模型在规则范围外的绝对稳定，也不降低无依据问题的拒答要求。
- 旧会话如果只保存过展示摘要，无法凭摘要补造完整表格；需要重新检索。
  本次修改之后保存的轮次可在重开会话/重建服务后沿用完整证据。
- 未改动报告中的其他议题（订单流程、整体性能、停止生成后的历史保存、界面布局）。
