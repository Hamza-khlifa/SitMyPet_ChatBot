from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow requests from Ionic app (localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For testing; use specific domains in production, e.g., ["http://localhost:8100"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
HUGGINGFACE_API_TOKEN = os.getenv("HF_API_TOKEN")
HUGGINGFACE_MODEL = "HuggingFaceH4/zephyr-7b-beta"

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(req: ChatRequest):
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
    data = {"inputs": f"You are a pet expert. Only answer pet-related questions. {req.message}"}

    response = requests.post(
        f"https://api-inference.huggingface.co/models/{HUGGINGFACE_MODEL}",
        json=data,
        headers=headers
    )

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.json())

    ai_output = response.json()
    return {"reply": ai_output[0]["generated_text"] if ai_output else "No response."}
