import json
from typing import Any

import runpod

from utils import JobInput
from engine import vLLMEngine, OpenAIvLLMEngine

# Keys to remove from OpenAI-compatible payloads
_REASONING_KEYS = {"reasoning_content"}


def _drop_reasoning(obj: Any) -> Any:
    """Recursively remove reasoning keys from dict/list JSON-like structures."""
    if isinstance(obj, dict):
        return {k: _drop_reasoning(v) for k, v in obj.items() if k not in _REASONING_KEYS}
    if isinstance(obj, list):
        return [_drop_reasoning(x) for x in obj]
    return obj


def _filter_sse_line(line: Any) -> Any:
    """
    Filter a single SSE chunk line like:
      'data: {...}\\n\\n' or 'data: [DONE]\\n\\n'
    If not parseable JSON, return unchanged.
    """
    if not isinstance(line, str):
        return line

    if not line.startswith("data:"):
        return line

    payload = line[len("data:"):].strip()
    if payload == "[DONE]":
        return line

    try:
        j = json.loads(payload)
    except Exception:
        return line

    j = _drop_reasoning(j)
    return "data: " + json.dumps(j, ensure_ascii=False) + "\n\n"


def _filter_batch(batch: Any) -> Any:
    """
    Batch can be:
      - dict/list (structured output)
      - str (single SSE chunk)
      - list[str|dict] (aggregated stream chunks)
    """
    if isinstance(batch, str):
        return _filter_sse_line(batch)

    if isinstance(batch, dict):
        return _drop_reasoning(batch)

    if isinstance(batch, list):
        out = []
        for item in batch:
            if isinstance(item, str):
                out.append(_filter_sse_line(item))
            else:
                out.append(_drop_reasoning(item))
        return out

    return batch


# Engine init (keep original behavior)
vllm_engine = vLLMEngine()
openai_engine = OpenAIvLLMEngine(vllm_engine)


async def handler(job):
    job_input = JobInput(job["input"])
    engine = openai_engine if job_input.openai_route else vllm_engine

    results_generator = engine.generate(job_input)

    async for batch in results_generator:
        yield _filter_batch(batch)


runpod.serverless.start(
    {
        "handler": handler,
        "concurrency_modifier": lambda _: vllm_engine.max_concurrency,
        "return_aggregate_stream": True,
    }
)
