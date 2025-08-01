# src/trusted_data_agent/core/session_manager.py
import uuid
from datetime import datetime

_SESSIONS = {}

def create_session(system_prompt: str, chat_object: any) -> str:
    """
    Creates a new session with an initialized token counter.
    """
    session_id = str(uuid.uuid4())
    _SESSIONS[session_id] = {
        "system_prompt": system_prompt,
        "generic_history": [],
        "chat_object": chat_object,
        "name": "New Chat",
        "created_at": datetime.now().isoformat(),
        # NEW: Initialize token usage for the session
        "token_usage": {
            "input_tokens": 0,
            "output_tokens": 0
        }
    }
    return session_id

def get_session(session_id: str) -> dict | None:
    """
    Retrieves a session by its ID.
    """
    return _SESSIONS.get(session_id)

def get_all_sessions() -> list[dict]:
    """
    Returns a summary of all active sessions.
    """
    session_summaries = [
        {"id": sid, "name": s_data["name"], "created_at": s_data["created_at"]}
        for sid, s_data in _SESSIONS.items()
    ]
    session_summaries.sort(key=lambda x: x["created_at"], reverse=True)
    return session_summaries

def add_to_history(session_id: str, role: str, content: str):
    """
    Adds a message to a session's history.
    """
    if session_id in _SESSIONS:
        _SESSIONS[session_id]['generic_history'].append({'role': role, 'content': content})

def update_session_name(session_id: str, new_name: str):
    """
    Updates the display name of a session.
    """
    if session_id in _SESSIONS:
        _SESSIONS[session_id]['name'] = new_name

def get_session_history(session_id: str) -> list | None:
    """
    Retrieves the chat history for a specific session.
    """
    if session_id in _SESSIONS:
        return _SESSIONS[session_id]['generic_history']
    return None

# --- NEW: Token Management Functions ---

def update_token_usage(session_id: str, input_tokens: int = 0, output_tokens: int = 0):
    """
    Updates the token count for a specific session.

    Args:
        session_id: The ID of the session to update.
        input_tokens: The number of input tokens to add.
        output_tokens: The number of output tokens to add.
    """
    if session_id in _SESSIONS:
        session = _SESSIONS[session_id]
        if 'token_usage' not in session:
            # Defensive initialization in case of older session structures
            session['token_usage'] = {"input_tokens": 0, "output_tokens": 0}
        
        session['token_usage']['input_tokens'] += input_tokens
        session['token_usage']['output_tokens'] += output_tokens

def get_token_usage(session_id: str) -> dict:
    """
    Retrieves the token usage for a specific session.

    Args:
        session_id: The ID of the session.

    Returns:
        A dictionary containing the total input and output tokens.
    """
    if session_id in _SESSIONS and 'token_usage' in _SESSIONS[session_id]:
        return _SESSIONS[session_id]['token_usage']
    return {"input_tokens": 0, "output_tokens": 0}
# --- END NEW ---
