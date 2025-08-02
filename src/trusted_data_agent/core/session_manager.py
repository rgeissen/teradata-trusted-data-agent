# src/trusted_data_agent/core/session_manager.py
import uuid
from datetime import datetime

_SESSIONS = {}

def create_session(system_prompt: str, chat_object: any) -> str:
    session_id = str(uuid.uuid4())
    _SESSIONS[session_id] = {
        "system_prompt": system_prompt,
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
