def extract_text(response: dict, provider: str) -> str:
    
    if isinstance(response, dict) and "text" in response:
        return response["text"]

    provider = provider.strip().lower()

    try:
        if provider == "openai":
            return response["choices"][0]["message"]["content"]
        elif provider == "anthropic":
            return response["completion"]
        elif provider == "cohere":
            return response["generations"][0]["text"]
        elif provider == "google":
            return response["candidates"][0]["content"]["parts"][0]["text"]
        elif isinstance(response, dict) and "text" in response:
            return response["text"]
        else:
            return str(response)
    except Exception:
        return str(response)