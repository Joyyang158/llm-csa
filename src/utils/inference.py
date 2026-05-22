"""vLLM inference helpers shared across data-generation and CSA scripts.

Wraps :class:`vllm.LLM` with a single function that handles chat-template
application (including the Qwen-family ``enable_thinking`` flag) and returns
a list of generations per call.
"""

import time
from typing import List, Optional, Tuple, Union

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_vllm(
    model_name: str,
    gpu_memory_utilization: float = 0.95,
    max_model_len: Optional[int] = None,
) -> Tuple[AutoTokenizer, LLM]:
    """Load a model and tokenizer for vLLM inference."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    kwargs = dict(
        model=model_name,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
    )
    if max_model_len is not None:
        kwargs["max_model_len"] = max_model_len
    model = LLM(**kwargs)
    return tokenizer, model


# ---------------------------------------------------------------------------
# Chat template helper
# ---------------------------------------------------------------------------

def apply_chat_template(
    tokenizer,
    prompt: str,
    model_type: str,
    enable_thinking: bool,
) -> str:
    """Apply the model's chat template, handling Qwen's thinking-mode flag."""
    messages = [{"role": "user", "content": prompt}]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if "qwen" in model_type.lower():
        kwargs["enable_thinking"] = enable_thinking
    return tokenizer.apply_chat_template(messages, **kwargs)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def generate_vllm(
    model: LLM,
    tokenizer,
    prompt: str,
    model_type: str = "qwen",
    max_tokens: int = 32768,
    enable_thinking: bool = False,
    num_generations: int = 1,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = -1,
    return_time: bool = False,
) -> Union[List[str], Tuple[List[str], float]]:
    """Generate one or more completions with vLLM.

    Always returns a list of strings (one per generation). When
    ``return_time=True``, also returns the wall-clock time taken.
    """
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        n=num_generations,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    text = apply_chat_template(tokenizer, prompt, model_type, enable_thinking)

    start = time.time()
    outputs = model.generate([text], sampling_params, use_tqdm=False)
    elapsed = time.time() - start

    generations = [out.text for out in outputs[0].outputs]
    return (generations, elapsed) if return_time else generations
