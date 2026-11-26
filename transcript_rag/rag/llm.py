
# llm.py

"""LM Studio chat model + robust helpers for small local models."""

from __future__ import annotations
import re
from functools import lru_cache

from .config import Settings, SETTINGS


_THINK = re.compile(r"<think>.*?</think>", re.S)


@lru_cache(maxsize=1)
def get_llm(cfg: Settings = SETTINGS):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=cfg.llm_model,
        base_url=cfg.lmstudio_url,
        api_key=cfg.lmstudio_api_key,
        temperature=0,
        seed=42,
        max_tokens=cfg.llm_num_predict,
    )


def text_of(msg) -> str:
    """Plain text of a model reply, with any leaked <think> block removed."""
    content = getattr(msg, "content", msg)
    return _THINK.sub(
        "",
        content if isinstance(content, str) else str(content)
    ).strip()


def structured(llm, schema, prompt: str, default, retries: int = 1):
    """Schema-constrained output through JSON Schema.

    Falls back to default so one malformed reply from a small
    local model cannot crash the graph.
    """
    runner = llm.with_structured_output(
        schema,
        method="json_schema"
    )

    for _ in range(retries + 1):
        try:
            out = runner.invoke(prompt)
            if out is not None:
                return out
        except Exception:
            continue

    return default


def check_lmstudio(cfg: Settings) -> None:
    """Fail fast if LM Studio is down or the model is not loaded."""

    from openai import OpenAI

    try:
        client = OpenAI(
            base_url=cfg.lmstudio_url,
            api_key=cfg.lmstudio_api_key,
        )

        models = client.models.list()
        names = [m.id for m in models.data]

    except Exception as e:
        raise RuntimeError(
            f"Cannot reach LM Studio at {cfg.lmstudio_url}. "
            "Open LM Studio, start the local server, "
            "and check that the server is running."
        ) from e

    if cfg.llm_model not in names:
        raise RuntimeError(
            f"Model '{cfg.llm_model}' is not available in LM Studio. "
            f"Load the model in LM Studio and check its model identifier. "
            f"Available models: {names}"
        )
