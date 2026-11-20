# Issue #15 验证记录

基线：`02277b7573a1235f2c9e19bd4341aa71f951639c`，分支 `fix/mewhelp-issue-15-interaction-perf`。
仅修改本 issue 相关代码与测试，不提交、push、开 PR 或评论 issue。

## Red（先写测试，未改生产代码）

命令：

```sh
MEWHELP_LANGFUSE_ENABLED=false MEWHELP_DATABASE_URL=sqlite:////tmp/mewhelp-issue15-red-suite.db .venv/bin/python -m pytest tests/test_issue15.py -q -s
```

结果：**15 failed in 39.44s**。覆盖发货日期 3 例、默认预算 1 例、逐步骤超时/可观测性 6 例、HTTP SSE 时序 2 例、异步 answered 校验 1 例、首字前/出字后停止保存 2 例。
原始日志：`/tmp/mewhelp-issue15-red.log`。

本地真实回环 HTTP，固定 mock：槽位 80ms、分片间隔 60ms、自评 400ms、记忆 450ms、每次落库 220ms。

| 路径 | 首 token | 最后 token | done | done − 最后 token | HTTP EOF | token 数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 模型生成（基线） | 585.13ms | 677.79ms | 1380.59ms | 702.80ms | 1381.24ms | 2 |
| evidence_rule（基线） | 495.11ms | 495.11ms | 723.19ms | 228.08ms | 726.54ms | 1 |

这不是 8125 的性能承诺；未调用 gpt-6-astra 中转，也未复测其排队、网络抖动和重试。

## 数据来源核查

- `data/fixtures/orders.json` 中 ORD-1007 有 `created_at=2026-08-12…`、`delivered_on=2026-08-14`；物流事件含 2026-08-13 已揽收，但没有 `shipped_at` 或 `shipped_on`。回答路径是 `get_order → note → _render_tool_answer`，不是生成模型或退款高风险流程。明确标注发货日期暂无记录，不将揽收时间当成发货日期。
- 仓库中没有报告使用的 fresh-food 数据集，全文检索未发现“日式小圆饼干”及报告括号原句。已请求本机数据目录；不能在没有原文时假装完成知识库修订。

## Green（监督者复核）

codex 运行约 70 分钟后被运行时以审批超时终止，未输出最终总结；以下由监督者（coordinator）对遗留工作区独立验证，未重新启动 codex：

- `tests/test_issue15.py`：31/31 通过；基线对照（仅 stash 生产代码）30 失败、1 通过。
- 全量：`.venv/bin/python -m pytest -q` → **922 passed、1 skipped**（#12 的 45 例与 #13 的 100 例均在内，无回归；跳过项与之前一致，为 Langfuse 初始化受环境限制）。
- 前端：`npm run typecheck` 通过；`npm run build` 通过，重建产物哈希与提交的产物一致（`index-D5bx0cwM.js` / `index-D57B2bQj.css`）；`node --test tests/issue4.test.mjs tests/issue15.test.mjs` → 14 通过、0 失败。
- 时序说明：SSE 时序测试断言的是行为（done 不等收尾、首字前停止仍保存），不是毫秒数；本记录只有基线的毫秒对照表，未测修复后的毫秒表——本地 mock 的毫秒数不能当作 8125 中转链路的承诺。
- 未修复项：饼干括号自相矛盾——fresh-food 数据集不在仓库内，无原文可修订；需报告人提供数据原文或在实机数据侧修正，不在本 PR 内。
