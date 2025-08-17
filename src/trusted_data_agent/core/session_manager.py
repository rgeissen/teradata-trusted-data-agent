# src/trusted_data_agent/core/session_manager.py
import uuid
from datetime import datetime

# --- NEW: Import genai for type hinting and PROVIDER_SYSTEM_PROMPTS for provider-specific logic ---
import google.generativeai as genai
# --- FIX: Corrected the import to use the new prompts.py file ---
from trusted_data_agent.agent.prompts import PROVIDER_SYSTEM_PROMPTS

_SESSIONS = {}

# --- MODIFIED: Updated function signature and logic to use the correct provider-specific prompt ---
def create_session(provider: str, llm_instance: any, charting_intensity: str) -> str:
    session_id = str(uuid.uuid4())
    
    # Select the correct system prompt template based on the provider.
    system_prompt_template = PROVIDER_SYSTEM_PROMPTS.get(provider, PROVIDER_SYSTEM_PROMPTS["Google"])
    
    chat_object = None
    # Google provider requires starting the chat with a history to set the system prompt.
    # Other providers handle the system prompt per-call, so we just need an empty list.
    if provider == "Google":
        # NOTE: The actual system prompt isn't sent here anymore. 
        # We send a placeholder because the library requires it, but the real,
        # dynamically-built prompt will be sent with each user message.
        initial_history = [
            {"role": "user", "parts": [{"text": "You are a helpful assistant."}]},
            {"role": "model", "parts": [{"text": "Understood."}]}
        ]
        if isinstance(llm_instance, genai.GenerativeModel):
             chat_object = llm_instance.start_chat(history=initial_history)
    else:
        chat_object = []

    _SESSIONS[session_id] = {
        "system_prompt_template": system_prompt_template,
        "charting_intensity": charting_intensity,
        "generic_history": [],
        "chat_object": chat_object,
        "name": "New Chat",
        "created_at": datetime.now().isoformat(),
        "input_tokens": 0,
        "output_tokens": 0
    }
    return session_id

def get_session(session_id: str) -> dict | None:
    return _SESSIONS.get(session_id)

def get_all_sessions() -> list[dict]:
    session_summaries = [
        {"id": sid, "name": s_data["name"], "created_at": s_data["created_at"]}
        for sid, s_data in _SESSIONS.items()
    ]
    session_summaries.sort(key=lambda x: x["created_at"], reverse=True)
    return session_summaries

def add_to_history(session_id: str, role: str, content: str):
    if session_id in _SESSIONS:
        _SESSIONS[session_id]['generic_history'].append({'role': role, 'content': content})

def update_session_name(session_id: str, new_name: str):
    if session_id in _SESSIONS:
        _SESSIONS[session_id]['name'] = new_name

def get_session_history(session_id: str) -> list | None:
    if session_id in _SESSIONS:
        return _SESSIONS[session_id]['generic_history']
    return None

def update_token_count(session_id: str, input_tokens: int, output_tokens: int):
    """Updates the token counts for a given session."""
    if session_id in _SESSIONS:
        _SESSIONS[session_id]['input_tokens'] += input_tokens
        _SESSIONS[session_id]['output_tokens'] += output_tokens