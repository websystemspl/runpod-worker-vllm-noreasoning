import os
import json
from typing import Any

import runpod

from utils import JobInput
from engine import vLLMEngine, OpenAIvLLMEngine


# -----------------------------------------------------------------------------
# "Kuloodporny" filtr: usuwa reasoning z każdego payloadu zanim trafi do klienta
# -----------------------------------------------------------------------------

# Klucz, który psuje Twój pipeline:
# - OpenAI-compat streaming: choices[].delta.reasoning_content
# - czasem non-stream: choices[].message.reasoning_content
_REASONING_KEYS = {
    "reasoning_content",
}

def _drop_reasoning(obj: Any) -> Any:
    """
    Rekurencyjnie usuwa klucze reasoning z dict/list.
    Nie modyfikuje prymitywów (str/int/None).
    """
    if isinstance(obj, dict):
        # usuń reasoning_content na KAŻDYM poziomie
        return {k: _drop_reasoning(v) for k, v in obj.items() if k not in _REASONING_KEYS}
    if isinstance(obj, list):
        return [_drop_reasoning(x) for x in obj]
    return obj


def _filter_sse_line(line: str) -> str:
    """
    Filtruje pojedynczą linię SSE w formacie:
      "data: {json}\n\n"
      "data: [DONE]\n\n"
    Jeżeli nie da się sparsować JSON, przepuszcza bez zmian.
    """
    if not isinstance(line, str):
        return line

    # Najczęstszy format vLLM OpenAI streaming
    if not line.startswith("data:"):
        return line

    payload = line[len("data:"):].strip()

    # sentinel końca streamu
    if payload == "[DONE]":
        return line

    try:
        j = json.loads(payload)
    except Exception:
        # jeśli to nie jest JSON (albo uszkodzone), nie ryzykujemy crusha
        return line

    j = _drop_reasoning(j)

    # zachowaj standard SSE: "data: ...\n\n"
    return "data: " + json.dumps(j, ensure_ascii=False) + "\n\n"


def _filter_batch(batch: Any) -> Any:
    """
    Batch może być:
      - dict / list (gdy worker zwraca ustrukturyzowany obiekt)
      - str (pojedynczy chunk SSE)
      - list[str] (agregacja streamu w RunPod)
    Zwraca batch po filtracji reasoning_content.
    """
    if isinstance(batch, str):
        return _filter_sse_line(batch)

    if isinstance(batch, dict):
        return _drop_reasoning(batch)

    if isinstance(batch, list):
        # lista może zawierać dicty albo stringi SSE
        filtered = []
        for item in batch:
            if isinstance(item, str):
                filtered.append(_filter_sse_line(item))
            else:
                filtered.append(_drop_reasoning(item))
        return filtered

    # inne typy – przepuść bez zmian
    return batch


# -----------------------------------------------------------------------------
# Engine init
# -----------------------------------------------------------------------------
vllm_engine = vLLMEngine()
openai_engine = OpenAIvLLMEngine(vllm_engine)


# -----------------------------------------------------------------------------
# Handler
# -----------------------------------------------------------------------------
async def handler(job):
    job_input = JobInput(job["input"])
    engine = openai_engine if job_input.openai_route else vllm_engine

    results_generator = engine.generate(job_input)

    async for batch in results_generator:
        # HARD GUARANTEE: reasoning_content nigdy nie wyjdzie z workera
        yield _filter_batch(batch)


runpod.serverless.start(
    {
        "handler": handler,
        "concurrency_modifier": lambda x: vllm_engine.max_concurrency,
        "return_aggregate_stream": True,
    }
)
