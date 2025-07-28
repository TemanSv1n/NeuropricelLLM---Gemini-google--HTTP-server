import requests
import json


class GeminiClient:
    def __init__(self, base_url: str = "http://localhost:12345"):
        self.base_url = base_url

    def send_request(self, text: str, construct: str = "pricel", response_format: str = "short"):
        url = f"{self.base_url}/chat"
        payload = {
            "text": text,
            "construct": construct,
            "response_format": response_format
        }

        try:
            response = requests.post(
                url,
                json=payload,
                headers={'Accept': 'application/json; charset=utf-8'}
            )
            response.encoding = 'utf-8'
            return response.json()
        except Exception as e:
            return {"error": str(e)}


def print_response(response):
    print(json.dumps(response, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    client = GeminiClient()
    print("Priceling... (type 'exit' to quit)")

    while True:
        user_input = input("Enter request: ")
        if user_input.lower() in ('exit', 'quit'):
            break

        result = client.send_request(user_input)
        print_response(result["response"])