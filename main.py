import json
import re
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import uvicorn
from pydantic import BaseModel
from typing import Optional

# Initialize FastAPI app
app = FastAPI()

# Load configuration
try:
    with open('config.json') as f:
        config = json.load(f)
except FileNotFoundError:
    config = {"host": "0.0.0.0", "port": 8000}


# Request model without field shadowing
class ChatRequest(BaseModel):
    message_text: str
    personality_construct: Optional[str] = "pricel"
    answer_format: Optional[str] = "short"

    class Config:
        allow_population_by_field_name = True
        fields = {
            "message_text": "text",
            "personality_construct": "construct",
            "answer_format": "response_format"
        }


def load_token():
    """Load API token from token.json"""
    try:
        with open('token.json') as f:
            return json.load(f).get('api_key')
    except FileNotFoundError:
        raise Exception("token.json not found")


def load_text_file(filepath):
    """Load content from text file with UTF-8 encoding"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        raise Exception(f"File not found: {filepath}")


def clean_response(text):
    """Clean response text (spaces, etc.)"""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


class GeminiManager:
    def __init__(self):
        self.api_key = load_token()

    async def get_response(self, request: ChatRequest):
        """Get response from Gemini with proper encoding"""
        try:
            # Load requested construct and format
            construct_path = f"constructs/{request.personality_construct}.txt"
            format_path = f"response_formats/{request.answer_format}.txt"

            personality = load_text_file(construct_path)
            response_format = load_text_file(format_path)

            # Configure model
            genai.configure(api_key=self.api_key)
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }

            model = genai.GenerativeModel(
                model_name='gemini-1.5-flash',
                safety_settings=safety_settings,
                system_instruction=f"{personality}\n\n{response_format}"
            )

            # Get and clean response
            chat = model.start_chat()
            response = await chat.send_message_async(request.message_text)
            return clean_response(response.text)

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


# Initialize Gemini manager
gemini = GeminiManager()


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint with proper Cyrillic support"""
    try:
        response = await gemini.get_response(request)
        return JSONResponse(
            content=jsonable_encoder({"response": response}),
            media_type="application/json; charset=utf-8"
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}


if __name__ == "__main__":
    # Create required directories if missing
    os.makedirs("constructs", exist_ok=True)
    os.makedirs("response_formats", exist_ok=True)

    # Start server
    uvicorn.run(
        app,
        host=config.get("host", "0.0.0.0"),
        port=config.get("port", 8000),
        log_config=None
    )