from __future__ import annotations

import os
import requests
import gradio as gr

API_BASE = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
MODEL = os.environ.get("MODEL", "be-local")
API_KEY = os.environ.get("OPENAI_API_KEY", "")
SYSTEM_HINT = "Адказвай толькі па-беларуску. Не называй сябе ChatGPT або OpenAI."


def chat(message: str, history: list[tuple[str, str]]):
    messages = [{"role": "system", "content": SYSTEM_HINT}]
    for user, assistant in history[-8:]:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": message})
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    payload = {"model": MODEL, "messages": messages, "temperature": 0.4, "max_tokens": 256, "stream": False}
    try:
        r = requests.post(f"{API_BASE}/chat/completions", headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "") or str(data)
    except Exception as exc:
        return f"Памылка сэрвера мадэлі: {exc}"


demo = gr.ChatInterface(fn=chat, title="Беларускамоўная nanochat-мадэль", description="Дэманстрацыйны інтэрфейс. Мадэль павінна адказваць па-беларуску.")

demo.launch()
