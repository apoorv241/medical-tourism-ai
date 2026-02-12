from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model


load_dotenv()

def create_llm(model_name: str = "google_genai:gemini-2.0-flash"):
    return init_chat_model(model_name)