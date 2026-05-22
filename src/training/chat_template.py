"""Chat template helpers shared by SFT and GRPO training."""


def render_user_prompt(tokenizer, user_content: str, model_type: str) -> str:
    """Render a single user turn with the appropriate chat template.

    ``add_generation_prompt`` is set to True so the returned string ends with
    the assistant-turn opening tokens, ready for the model (or the
    SFT/completion loss) to continue generating from.
    """
    messages = [{"role": "user", "content": user_content}]
    kwargs = {"tokenize": False, "add_generation_prompt": True}

    # Qwen3-family models accept an ``enable_thinking`` flag in their chat
    # template. We always disable thinking mode for routing decisions —
    # routing should be a fast, structured output, not a reasoning trace.
    if "qwen" in model_type.lower():
        kwargs["enable_thinking"] = False

    return tokenizer.apply_chat_template(messages, **kwargs)
