import google.generativeai as genai
import json
import re
import os
from google.generativeai.types import HarmCategory, HarmBlockThreshold


def load_token(filepath='token.json'):
    """Load API token from JSON file"""
    with open(filepath) as f:
        data = json.load(f)
    return data.get('api_key')


def load_text_file(filepath):
    """Load content from text file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: File not found - {filepath}")
        return ""


def clean_response(text):
    """Clean response text"""
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single
    return text.strip()


def configure_model(api_key, personality, response_format):
    """Configure Gemini model"""
    genai.configure(api_key=api_key)

    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    full_instruction = f"{personality}\n\n{response_format}"

    return genai.GenerativeModel(
        model_name='gemini-1.5-flash',
        safety_settings=safety_settings,
        system_instruction=full_instruction
    )


class GeminiAI:
    def __init__(self):
        # Create directories if they don't exist
        os.makedirs('constructs', exist_ok=True)
        os.makedirs('response_formats', exist_ok=True)

        self.api_key = load_token()
        self.personality = load_text_file('constructs/pricel.txt')
        self.response_format = load_text_file('response_formats/short.txt')
        self.model = configure_model(self.api_key, self.personality, self.response_format)
        self.chat = self.model.start_chat()

    def generate(self, prompt):
        """Generate and clean response"""
        try:
            response = self.chat.send_message(prompt)
            return clean_response(response.text)
        except Exception as e:
            return f"Error: {str(e)}"


if __name__ == "__main__":
    ai = GeminiAI()
    print("Система готова. Для выхода введите 'exit'")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            break

        response = ai.generate(user_input)
        print("AI:", response)