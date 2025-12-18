import os
import logging
import runpod
from utils import JobInput
from engine import vLLMEngine, OpenAIvLLMEngine

# Ensure logs are visible on RunPod serverless
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    force=True,
)
logging.info("Starting worker initialization")

logging.info("Creating vLLMEngine")
vllm_engine = vLLMEngine()
logging.info("Creating OpenAI vLLM Engine wrapper")
OpenAIvLLMEngine = OpenAIvLLMEngine(vllm_engine)

async def handler(job):
    logging.info("Received job: id=%s route=%s", job.get("id"), job.get("input", {}).get("openai_route"))
    job_input = JobInput(job["input"])
    engine = OpenAIvLLMEngine if job_input.openai_route else vllm_engine
    results_generator = engine.generate(job_input)
    async for batch in results_generator:
        yield batch

runpod.serverless.start(
    {
        "handler": handler,
        "concurrency_modifier": lambda x: vllm_engine.max_concurrency,
        "return_aggregate_stream": True,
    }
)
