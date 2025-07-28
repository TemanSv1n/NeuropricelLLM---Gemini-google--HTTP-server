import json
import re
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import uvicorn
from pydantic import BaseModel, Field
from typing import Optional

app = FastAPI()

# Load configuration
try:
    with open('config.json') as f:
        config = json.load(f)
except FileNotFoundError:
    config = {"host": "0.0.0.0", "port": 8000}


# Updated request model with proper field names and aliases
class ChatRequest(BaseModel):
    text: str = Field(..., alias="message_text")
    construct: str = Field("pricel", alias="personality_construct")
    response_format: str = Field("short", alias="answer_format")

    class Config:
        populate_by_name = True  # Updated for Pydantic v2


def load_token():
    with open('token.json') as f:
        return json.load(f).get('api_key')


def load_text_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def clean_response(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


class GeminiManager:
    def __init__(self):
        self.api_key = load_token()

    async def get_response(self, request: ChatRequest):
        try:
            construct_path = f"constructs/{request.construct}.txt"
            format_path = f"response_formats/{request.response_format}.txt"

            personality = load_text_file(construct_path)
            response_format = load_text_file(format_path)

            genai.configure(api_key=self.api_key)
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }

            model = genai.GenerativeModel(
                model_name='gemini-2.5-flash-lite',
                safety_settings=safety_settings,
                system_instruction=f"{personality}\n\n{response_format}"
            )

            chat = model.start_chat()
            response = await chat.send_message_async(request.text)
            return clean_response(response.text)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


gemini = GeminiManager()


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
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


if __name__ == "__main__":
    os.makedirs("constructs", exist_ok=True)
    os.makedirs("response_formats", exist_ok=True)
    uvicorn.run(
        app,
        host=config.get("host", "0.0.0.0"),
        port=config.get("port", 8000)
    )