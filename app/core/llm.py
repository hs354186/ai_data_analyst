import os
from openai import OpenAI

# Read key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize client only if key exists
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


def llm_chat(messages, model="gpt-4o-mini", temperature=0):
    """
    Centralized LLM call (OpenAI >= 1.0, compatible with 2.x).
    - messages: [{"role": "user|system|assistant", "content": "..."}]
    - returns: string content ONLY
    """
    if client is None:
        # Safe fallback (never crash the app)
        return '{"domain":"unknown","columns":[],"confidence":0.0}'

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )

    return response.choices[0].message.content
