"""Utility functions for DeepScaler.

This module contains various utility functions for making API calls to LLMs,
implementing RAG functionality, and managing network ports.
"""
import time
from typing import List, Union
import openai

from graders.verl_grader.reward_score.deepscaler.globals import GCP_PROJECT_ID, GCP_LOCATION, GEMINI_MODEL, OAI_RM_MODEL


def call_oai_rm_llm(
    prompt: str,
    system_prompt: str,
    n: int = 1,
    temperature: float = 1.0,
    model_id: str = OAI_RM_MODEL,
    retry_count: int = 1000000000
) -> Union[str, List[str]]:
    """Call OpenAI API with retry logic.

    Args:
        prompt: The text prompt to send to the model
        system_prompt: System instruction for the model
        n: Number of completions to generate
        temperature: Sampling temperature
        model_id: OpenAI model ID to use
        retry_count: Number of retries on rate limit errors

    Returns:
        Generated text(s) from the model
    """
    client = openai.OpenAI()
    backoff = 1
    retry_count = int(retry_count)

    for _ in range(retry_count):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                n=n,
            )
            break
        except Exception as exc:
            if "429" in str(exc):
                print("Retry due to rate limit: ", exc)
                time.sleep(backoff)
                backoff = min(backoff * 2, 64)  # Exponential backoff up to 64s
                continue
            print("Exception: ", exc)
            return []

    if n == 1:
        return response.choices[0].message.content
    return [choice.message.content for choice in response.choices]
