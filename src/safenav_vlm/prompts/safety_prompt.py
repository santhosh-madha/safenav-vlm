"""Prompts for SafeNav-VLM safety evaluation."""


def build_safety_prompt(question: str) -> str:
    """Build the prompt used by the direct-VLM baseline."""

    return f"""You are assisting a blind or visually impaired user.

User question: {question}

Use only the visible evidence in the image.

Respond using exactly this format:

Decision: SAFE, UNSAFE, or UNCERTAIN
Reason: One short sentence explaining the visible evidence."""
