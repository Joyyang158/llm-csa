"""Centralized prompt templates used across the project.

Two task domains are supported:

  * ``math``    — open-ended mathematical reasoning. Models are asked to
                  produce a final answer in ``\\boxed{...}``.
  * ``science`` — multiple-choice questions (MMLU-Pro Science style). Models
                  are asked to produce a single option letter in
                  ``\\boxed{<letter>}``.
"""

# ---------------------------------------------------------------------------
# Benchmark answering prompts (used when collecting model answers on datasets)
# ---------------------------------------------------------------------------

BENCHMARK_MATH = (
    "Question: {question}\n"
    "Answer the question and provide the final answer in a box format: "
    "\\boxed{{...}}."
)

BENCHMARK_SCIENCE = (
    "Question: {question}\n"
    "Choices:\n"
    "{choices}\n\n"
    "Answer the question and provide the final selected option letter "
    "(e.g., A, B, C, D...) in the box format: \\boxed{{<option letter>}}."
)


def get_benchmark_prompt(domain: str) -> str:
    """Return the benchmark answering prompt template for a given domain."""
    if domain == "math":
        return BENCHMARK_MATH
    if domain == "science":
        return BENCHMARK_SCIENCE
    raise ValueError(f"Unknown domain: {domain!r}. Expected 'math' or 'science'.")


# ---------------------------------------------------------------------------
# CSA decision prompts (used at inference time by the CSA model)
# ---------------------------------------------------------------------------

CSA_DECISION_WITH_ANALYSIS = """
Decide whether you can reliably and correctly answer the user's query.
You do not need to actually solve the problem. Just assess whether you are
capable of solving it.

- Choose SELF_SOLVE if you believe you can solve it by yourself.
- Choose DELEGATE if you believe it requires a more powerful model.

# Output Format
<analysis>
Explain why you chose SELF_SOLVE or DELEGATE.
</analysis>

<decision>
SELF_SOLVE or DELEGATE
</decision>

# Query:
{query}
""".strip()


CSA_DECISION_BINARY = """
Decide whether you can reliably and correctly answer the user's query.
You do not need to actually solve the problem. Just assess whether you are
capable of solving it.

- Choose SELF_SOLVE if you believe you can solve it by yourself.
- Choose DELEGATE if you believe it requires a more powerful model.

# Output Format
<decision>
SELF_SOLVE or DELEGATE
</decision>

# Query:
{query}
""".strip()


# ---------------------------------------------------------------------------
# Routing analysis generation prompts (used to build SFT training data)
# ---------------------------------------------------------------------------

ANALYSIS_TEACHER = """
You are analyzing the behavior of a target language model.

You are given a user query, along with a **ground-truth label** that correctly
reflects the target model's capability:
- SELF_SOLVE: the target model can solve the query by itself.
- DELEGATE: the query requires a more powerful model.

For the following user query:
{query}

The correct label is: {decision}

Your goal is to explain **why this query should be classified as {decision} for
the target model**.
Write the explanation from the **first-person perspective of the target model**,
as if it is assessing its own capability.

You do NOT need to actually solve the query.
""".strip()


ANALYSIS_SELF = """
You are given a user query, along with a **ground-truth label** that correctly
reflects your capability.

The label is:
- SELF_SOLVE: you can solve the query by yourself.
- DELEGATE: the query requires a more powerful model.

For the following query: {query}

The ground-truth label is: {decision}

Write a coherent self-assessment explaining **why this query should be
classified as {decision}**. You do NOT need to actually solve the query.
""".strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def label_to_decision(is_correct: int) -> str:
    """Map a binary correctness label to a routing decision string."""
    return "SELF_SOLVE" if int(is_correct) == 1 else "DELEGATE"
