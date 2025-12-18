# encoding: utf-8
"""
handler.py - RunPod serverless handler with lazy vLLM/OpenAI engine init
- opóźniona inicjalizacja silników (nie blokuje importu)
- inicjalizacja wykonana w wątku by uniknąć wywoływania asyncio.run() w działającym loopie
- zachowany "kuloodporny" filtr usuwający reasoning_content z odpowiedzi
"""

import os
import json
import logging
import asyncio
from typing import Any, Optional

import runpod

from utils import JobInput

from engine import vLLMEngine, OpenAIvLLMEngine

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# "Kuloodporny" filtr: usuwa reasoning z każdego payloadu zanim trafi do klienta
# -----------------------------------------------------------------------------
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
# Lazy engine init: nie tworzymy silników przy imporcie (unikamy blokowania)
# -----------------------------------------------------------------------------
vllm_engine: Optional[vLLMEngine] = None
openai_engine: Optional[OpenAIvLLMEngine] = None
_engines_lock: Optional[asyncio.Lock] = None


async def _ensure_engines():
    """
    Zapewnia jednokrotną inicjalizację vllm_engine i openai_engine.
    Heavy-lifting wykonujemy w wątku (run_in_executor), żeby nie blokować loopa
    i żeby konstruktor OpenAIvLLMEngine mógł użyć asyncio.run() bez konfliktu.
    """
    global vllm_engine, openai_engine, _engines_lock

    if _engines_lock is None:
        # lock musi być powiązany z aktualnym event loopem
        _engines_lock = asyncio.Lock()

    async with _engines_lock:
        loop = asyncio.get_running_loop()

        if vllm_engine is None:
            logger.info("Initializing vllm_engine (this may take time)...")
            # Utworzenie vLLMEngine jest blocking – róbemy to w executorze
            try:
                vllm_engine = await loop.run_in_executor(None, vLLMEngine)
                logger.info("vllm_engine initialized")
            except Exception as e:
                logger.exception("Failed to initialize vllm_engine: %s", e)
                raise

        if openai_engine is None:
            logger.info("Initializing openai_engine (this may take time)...")
            # OpenAIvLLMEngine.__init__ używa asyncio.run(...) -> musi być zrobione poza bieżącym loopem
            try:
                openai_engine = await loop.run_in_executor(None, OpenAIvLLMEngine, vllm_engine)
                logger.info("openai_engine initialized")
            except Exception as e:
                logger.exception("Failed to initialize openai_engine: %s", e)
                raise


def _default_max_concurrency() -> int:
    try:
        return int(os.getenv("MAX_CONCURRENCY", "1"))
    except Exception:
        return 1


def _concurrency_modifier(default_value):
    """
    Bezpieczny concurrency_modifier dla runpod: jeżeli silnik jest już zainicjalizowany,
    zwróci jego wartość, w przeciwnym razie odczyta env var.
    """
    try:
        if vllm_engine is not None and hasattr(vllm_engine, "max_concurrency"):
            return int(vllm_engine.max_concurrency)
    except Exception:
        logger.exception("Error while reading vllm_engine.max_concurrency")
    return _default_max_concurrency()


# -----------------------------------------------------------------------------
# Handler
# -----------------------------------------------------------------------------
async def handler(job):
    """
    Główny handler - najpierw upewniamy się, że silniki są zainicjalizowane,
    potem delegujemy generowanie do odpowiedniego engine i filtrujemy reasoning.
    """
    # Lazy init (pierwsze żądanie uruchomi stworzenie silników)
    await _ensure_engines()

    job_input = JobInput(job["input"])

    engine = openai_engine if job_input.openai_route else vllm_engine
    if engine is None:
        # powinno się nigdy nie zdarzyć, ale na wszelki wypadek
        logger.error("Engine not available after initialization")
        raise RuntimeError("Engine not initialized")

    try:
        results_generator = engine.generate(job_input)

        async for batch in results_generator:
            # HARD GUARANTEE: reasoning_content nigdy nie wyjdzie z workera
            yield _filter_batch(batch)

    except Exception as e:
        # Logujemy wyjątek i zwracamy prosty error object (RunPod obsłuży to)
        logger.exception("Unhandled exception in handler: %s", e)
        # Zwracamy dict z błędem — zachowujemy kompatybilność z dotychczasową obsługą
        yield {"error": {"message": str(e)}}


# -----------------------------------------------------------------------------
# Start serverless
# -----------------------------------------------------------------------------
runpod.serverless.start(
    {
        "handler": handler,
        "concurrency_modifier": _concurrency_modifier,
        "return_aggregate_stream": True,
    }
)
