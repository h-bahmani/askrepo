import httpx
import openai
import pytest

from askrepo.llm import LLMError, OpenAICompatibleClient


def test_provider_error_is_translated_to_a_clean_llm_error(monkeypatch: pytest.MonkeyPatch) -> None:
    client = OpenAICompatibleClient(model="test-model", api_key="fake-key")

    response = httpx.Response(
        status_code=413,
        request=httpx.Request("POST", "https://example.invalid/v1/chat/completions"),
    )
    error = openai.APIStatusError(
        "Error code: 413",
        response=response,
        body={"error": {"message": "Request too large for model (8692 > 8000 TPM)"}},
    )

    def raise_error(*args, **kwargs):
        raise error

    monkeypatch.setattr(client._client.chat.completions, "create", raise_error)

    with pytest.raises(LLMError, match="8692"):
        client.complete(messages=[{"role": "user", "content": "hi"}], tools=[])
