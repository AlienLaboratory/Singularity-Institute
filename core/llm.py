import httpx


class OllamaClient:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.last_response: dict | None = None

    def chat(self, messages: list[dict]) -> str:
        response = httpx.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
            },
            timeout=600,
        )

        response.raise_for_status()

        data = response.json()
        self.last_response = data
        return data["message"]["content"]
