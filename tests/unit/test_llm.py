import httpx
import pytest

from querypilot.config import Settings
from querypilot.llm import LlamaCppLLM, LLMError, OllamaLLM, create_llm


def client_with(handler) -> httpx.Client:
    return httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler))


def test_llamacpp_returns_message_content_and_sends_prompt():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = request.read().decode()
        return httpx.Response(200, json={"choices": [{"message": {"content": "SELECT 1"}}]})

    llm = LlamaCppLLM("http://test", client=client_with(handler))

    assert llm.invoke("write sql") == "SELECT 1"
    assert seen["path"] == "/v1/chat/completions"
    assert "write sql" in seen["body"]


def test_llamacpp_http_error_becomes_llm_error():
    llm = LlamaCppLLM("http://test", client=client_with(lambda r: httpx.Response(500)))
    with pytest.raises(LLMError):
        llm.invoke("x")


def test_llamacpp_unexpected_payload_becomes_llm_error():
    llm = LlamaCppLLM(
        "http://test", client=client_with(lambda r: httpx.Response(200, json={"oops": 1}))
    )
    with pytest.raises(LLMError):
        llm.invoke("x")


def test_llamacpp_connection_failure_becomes_llm_error():
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(LLMError):
        LlamaCppLLM("http://test", client=client_with(refuse)).invoke("x")


def test_create_llm_picks_backend_from_settings():
    assert isinstance(create_llm(Settings(llm_backend="llamacpp")), LlamaCppLLM)
    assert isinstance(create_llm(Settings(llm_backend="ollama")), OllamaLLM)
