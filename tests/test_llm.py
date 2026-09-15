"""真实 LLM 客户端（`OpenAILLMClient`）与配置优先级的离线测试。

这里是整条链路上唯一"和外部端点打交道"的一段，也是最容易出**静默错误**的一段：
接错了 key、模型返回了空正文、推理模型把预算烧在思维链上 —— 三种故障都不会抛异常，
表现出来都是"客服答非所问"或者"总是转人工"。所以这一批测试不碰网络，用假
`openai` 模块把请求与响应都钉死，逐个断言客户端**发了什么**、**怎么理解回来的东西**。

分成两组：

  1. **客户端接缝**：请求参数（预算 / response_format / 重试 / 超时）与响应解析
     （正文、思维链、工具调用、空正文）。
  2. **配置优先级**：`MEWHELP_OPENAI_API_KEY` 必须压过环境里的 `OPENAI_API_KEY`。
     这条曾经真的反了 —— 机器上遗留的 `OPENAI_API_KEY` 让 `.env` 里配的 key 失效，
     而症状只是"认证失败"，看不出是配置被谁盖住了。
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest

from mewhelp.config import Settings
from mewhelp.llm import LLMError, OpenAILLMClient

# ---------------------------------------------------------------- 假 openai


class _Completions:
    def __init__(self, mod: types.ModuleType) -> None:
        self._mod = mod

    def create(self, **kwargs: Any) -> Any:
        self._mod.last_kwargs = kwargs
        result = self._mod.result
        if isinstance(result, Exception):
            raise result
        return result


@pytest.fixture
def fake_openai(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """把 `openai` 换成一个只实现 `OpenAI` 的假模块。

    客户端是在 `__init__` 里 `from openai import OpenAI` 的（延迟导入），
    所以只要在构造之前塞进 `sys.modules` 就能接管。
    """
    mod = types.ModuleType("openai")
    mod.init_kwargs = {}  # type: ignore[attr-defined]
    mod.last_kwargs = None  # type: ignore[attr-defined]
    mod.result = None  # type: ignore[attr-defined]

    class OpenAI:  # noqa: N801 — 刻意和真实类同名
        def __init__(self, **kwargs: Any) -> None:
            mod.init_kwargs = kwargs  # type: ignore[attr-defined]
            self.chat = SimpleNamespace(completions=_Completions(mod))

    mod.OpenAI = OpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", mod)
    return mod


def reply(
    content: str | None,
    *,
    reasoning: str | None = None,
    finish: str = "stop",
    tool_calls: list[Any] | None = None,
    prompt_tokens: int = 11,
    completion_tokens: int = 7,
) -> SimpleNamespace:
    """一个最小可用的 ChatCompletion 返回值。"""
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    if reasoning is not None:
        message.reasoning_content = reasoning
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish)],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def make_settings(**over: Any) -> Settings:
    """不读 .env、不读环境变量的干净配置 —— 测试必须自己能跑。"""
    base: dict[str, Any] = {
        "_env_file": None,
        "llm_mode": "openai",
        "llm_model": "test-model",
        "openai_api_key": "sk-test",
        "openai_base_url": "https://example.invalid/v1",
    }
    base.update(over)
    return Settings(**base)


def make_client(fake_openai: types.ModuleType, **over: Any) -> OpenAILLMClient:
    return OpenAILLMClient(make_settings(**over))


# ---------------------------------------------------------------- 响应解析


def test_content_is_returned_as_text(fake_openai):
    fake_openai.result = reply("您的订单已签收。")
    resp = make_client(fake_openai).chat([{"role": "user", "content": "hi"}])
    assert resp.text == "您的订单已签收。"
    assert resp.finish_reason == "stop"
    assert resp.usage.total_tokens == 18


def test_reasoning_is_captured_but_never_leaks_into_text(fake_openai):
    """思维链单独存放 —— 它常含模型对用户的臆测，混进正文就是事故。"""
    fake_openai.result = reply(
        "已为您转接人工。",
        reasoning="用户在试探能不能白拿，先稳住他。",
    )
    resp = make_client(fake_openai).chat([{"role": "user", "content": "hi"}])
    assert resp.text == "已为您转接人工。"
    assert resp.reasoning == "用户在试探能不能白拿，先稳住他。"
    assert "试探" not in resp.text


def test_surrounding_whitespace_is_trimmed(fake_openai):
    fake_openai.result = reply("\n\n  已处理。  \n")
    resp = make_client(fake_openai).chat([{"role": "user", "content": "hi"}])
    assert resp.text == "已处理。"


@pytest.mark.parametrize(
    "content, reasoning",
    [
        (None, None),  # 模型什么都没说
        ("", "想了半天但没有结论"),  # 推理模型预算耗尽
        ("   \n ", None),  # 只有空白
    ],
)
def test_empty_content_is_a_failure_not_an_answer(fake_openai, content, reasoning):
    """空正文必须抛错。

    返回空字符串的话，下游只会把它渲染成一条空气泡（界面上看着像"转人工"），
    而日志里没有任何异常 —— 排查时只会看到"模型没报错也没说话"。
    """
    fake_openai.result = reply(content, reasoning=reasoning, finish="length")
    with pytest.raises(LLMError) as excinfo:
        make_client(fake_openai).chat([{"role": "user", "content": "hi"}], task="generate")
    message = str(excinfo.value)
    assert "空正文" in message
    assert "finish_reason=length" in message
    assert "task=generate" in message


def test_budget_exhausted_by_reasoning_says_so(fake_openai):
    """光看 "空正文" 不知道该调什么；点明是思维链吃掉了预算，才知道去开推理模式。"""
    fake_openai.result = reply("", reasoning="The user wants ...", finish="length")
    with pytest.raises(LLMError, match="思维链已耗尽输出预算"):
        make_client(fake_openai).chat([{"role": "user", "content": "hi"}])


def test_empty_content_with_tool_calls_is_legal(fake_openai):
    """要调工具的那一轮，正文本来就是空的 —— 这不是失败。"""
    fake_openai.result = reply(
        None,
        finish="tool_calls",
        tool_calls=[
            SimpleNamespace(
                id="call_1",
                function=SimpleNamespace(name="query_order", arguments='{"order_id":"ORD-1"}'),
            )
        ],
    )
    resp = make_client(fake_openai).chat(
        [{"role": "user", "content": "hi"}], tools=[{"type": "function"}], task="plan_tool"
    )
    assert resp.text == ""
    assert resp.tool_calls == [
        {"id": "call_1", "name": "query_order", "arguments": {"order_id": "ORD-1"}}
    ]


def test_truncated_content_is_a_failure_not_an_answer(fake_openai):
    """被预算截断的正文必须抛错，不能当完整答复发出去。

    半句话比空正文更危险：空正文看着像"模型没说话"，半句话看着像一句完整答复。
    用户会一字一字收到"款项将在 1 到 3 个工"然后流就结束了。
    """
    fake_openai.result = reply("订单 ORD-1002 申请退款后，款项将在 1 到 3 个工", finish="length")
    with pytest.raises(LLMError, match="截断"):
        make_client(fake_openai).chat([{"role": "user", "content": "hi"}], task="generate")


def test_truncated_but_tool_calling_turn_is_still_legal(fake_openai):
    """finish_reason=length 且带着 tool_calls 时不能判失败 —— 动作是完整的，
    缺的只是那句可有可无的旁白。"""
    fake_openai.result = reply(
        "I'll check ",
        finish="length",
        tool_calls=[
            SimpleNamespace(
                id="c1", function=SimpleNamespace(name="get_order", arguments='{"order_id":"ORD-1"}')
            )
        ],
    )
    resp = make_client(fake_openai).chat([{"role": "user", "content": "hi"}], tools=[{}])
    assert resp.tool_calls[0]["name"] == "get_order"


def test_broken_tool_arguments_degrade_to_empty_object(fake_openai):
    fake_openai.result = reply(
        None,
        finish="tool_calls",
        tool_calls=[
            SimpleNamespace(
                id="call_1", function=SimpleNamespace(name="query_order", arguments="{not json")
            )
        ],
    )
    resp = make_client(fake_openai).chat([{"role": "user", "content": "hi"}], tools=[{}])
    assert resp.tool_calls[0]["arguments"] == {}


def test_client_exception_becomes_llmerror(fake_openai):
    """上层只认 LLMError —— 裸的 openai 异常漏出去会变成 500 而不是兜底话术。"""
    fake_openai.result = RuntimeError("connection reset")
    with pytest.raises(LLMError, match="connection reset"):
        make_client(fake_openai).chat([{"role": "user", "content": "hi"}])


def test_missing_api_key_fails_fast(monkeypatch, fake_openai):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMError, match="没找到 API Key"):
        OpenAILLMClient(make_settings(openai_api_key=None))


# ---------------------------------------------------------------- 请求参数


def test_plain_mode_asks_for_json_when_json_mode(fake_openai):
    fake_openai.result = reply('{"intent": "logistics"}')
    make_client(fake_openai).chat([{"role": "user", "content": "hi"}], json_mode=True)
    assert fake_openai.last_kwargs["response_format"] == {"type": "json_object"}
    assert fake_openai.last_kwargs["max_tokens"] == 1024


def test_reasoning_mode_drops_response_format_and_raises_budget(fake_openai):
    """推理模型不认 response_format，且思维链要占预算 —— 两件事一起改。"""
    fake_openai.result = reply('{"intent": "logistics"}', reasoning="...")
    client = make_client(fake_openai, llm_reasoning=True)
    client.chat([{"role": "user", "content": "hi"}], json_mode=True)
    assert "response_format" not in fake_openai.last_kwargs
    assert fake_openai.last_kwargs["max_tokens"] == 4096


def test_explicit_max_tokens_still_wins(fake_openai):
    """调用方显式指定的预算不被推理模式的默认值覆盖。"""
    fake_openai.result = reply("ok")
    make_client(fake_openai, llm_reasoning=True).chat(
        [{"role": "user", "content": "hi"}], max_tokens=64
    )
    assert fake_openai.last_kwargs["max_tokens"] == 64


def test_connect_timeout_is_far_shorter_than_read_timeout(fake_openai):
    """连接阶段必须快速失败，否则重试根本没机会发生。

    中转域名下挂着多个 IP，其中个别是黑洞：如果连接也等满 60 秒，第一次挑中
    坏 IP 就把整次调用的预算烧光，"重试 3 次"实际上一次都不会发生。
    """
    make_client(fake_openai, llm_timeout_s=60.0, llm_connect_timeout_s=8.0)
    timeout = fake_openai.init_kwargs["timeout"]
    assert timeout.connect == 8.0
    assert timeout.read == 60.0
    assert timeout.write == 60.0


def test_retries_are_llm_settings_not_tool_settings(fake_openai):
    """模型重试和工具重试是两件事，共用一个开关迟早会有人改错一边。"""
    make_client(fake_openai, tool_max_retries=0, llm_max_retries=5)
    assert fake_openai.init_kwargs["max_retries"] == 5


def test_request_carries_model_and_temperature(fake_openai):
    fake_openai.result = reply("ok")
    make_client(fake_openai, llm_temperature=0.3).chat([{"role": "user", "content": "hi"}])
    assert fake_openai.last_kwargs["model"] == "test-model"
    assert fake_openai.last_kwargs["temperature"] == 0.3


# ---------------------------------------------------------------- 配置优先级


def test_plain_openai_env_is_used_as_fallback(monkeypatch):
    monkeypatch.delenv("MEWHELP_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    settings = Settings(_env_file=None)
    assert settings.openai_api_key == "sk-from-env"


def test_mewhelp_prefix_beats_plain_env_var(monkeypatch):
    """这条曾经是反的：环境变量优先级高于 .env，机器上遗留的 OPENAI_API_KEY
    会盖住本项目 .env 里配的 key，症状却只是"认证失败"。"""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-别人的")
    monkeypatch.setenv("MEWHELP_OPENAI_API_KEY", "sk-本项目配的")
    assert Settings(_env_file=None).openai_api_key == "sk-本项目配的"


def test_mewhelp_prefix_in_dotenv_beats_plain_env_var(monkeypatch, tmp_path):
    """真实场景：`MEWHELP_...` 写在 .env 里，而 `OPENAI_API_KEY` 在环境里。"""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-别人的")
    monkeypatch.delenv("MEWHELP_OPENAI_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("MEWHELP_OPENAI_API_KEY=sk-本项目配的\n", encoding="utf-8")

    assert Settings(_env_file=env_file).openai_api_key == "sk-本项目配的"


def test_explicit_argument_beats_every_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-别人的")
    monkeypatch.setenv("MEWHELP_OPENAI_API_KEY", "sk-环境里的")
    assert Settings(_env_file=None, openai_api_key="sk-显式传的").openai_api_key == "sk-显式传的"


def test_empty_env_value_falls_through_instead_of_blanking(monkeypatch):
    """`.env.example` 里 key 留空是常态 —— 空串必须当成"没配"，而不是"配成了空"。"""
    monkeypatch.setenv("MEWHELP_OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    assert Settings(_env_file=None).openai_api_key == "sk-from-env"


def test_base_url_follows_the_same_rule(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://别人家的/v1")
    monkeypatch.setenv("MEWHELP_OPENAI_BASE_URL", "https://本项目配的/v1")
    assert Settings(_env_file=None).openai_base_url == "https://本项目配的/v1"


def test_output_budget_follows_reasoning_flag():
    assert Settings(_env_file=None, llm_max_tokens=512).llm_output_budget == 512
    assert (
        Settings(
            _env_file=None, llm_reasoning=True, llm_max_tokens=512, llm_max_tokens_reasoning=8192
        ).llm_output_budget
        == 8192
    )
