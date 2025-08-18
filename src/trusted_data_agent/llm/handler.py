# src/trusted_data_agent/llm/handler.py
import asyncio
import json
import logging
import httpx
import re
import random
import time

import google.generativeai as genai
from anthropic import APIError, AsyncAnthropic, InternalServerError, RateLimitError
from openai import AsyncOpenAI, APIError as OpenAI_APIError
import boto3

from trusted_data_agent.core.config import APP_CONFIG
from trusted_data_agent.core.session_manager import get_session, update_token_count
from trusted_data_agent.agent.prompts import CHARTING_INSTRUCTIONS, PROVIDER_SYSTEM_PROMPTS
from trusted_data_agent.core.config import (
    CERTIFIED_GOOGLE_MODELS, CERTIFIED_ANTHROPIC_MODELS,
    CERTIFIED_AMAZON_MODELS, CERTIFIED_AMAZON_PROFILES,
    CERTIFIED_OLLAMA_MODELS, CERTIFIED_OPENAI_MODELS
)

llm_logger = logging.getLogger("llm_conversation")
app_logger = logging.getLogger("quart.app")

class OllamaClient:
    """A simple async client for interacting with the Ollama API."""
    def __init__(self, host: str):
        self.host = host
        self.client = httpx.AsyncClient(base_url=self.host, timeout=120.0)

    async def list_models(self):
        try:
            response = await self.client.get("/api/tags")
            response.raise_for_status()
            return response.json().get("models", [])
        except httpx.RequestError as e:
            app_logger.error(f"Ollama API request error: {e}")
            raise RuntimeError("Could not connect to Ollama server.") from e

    async def chat(self, model: str, messages: list, system_prompt: str):
        try:
            payload = {
                "model": model,
                "messages": messages,
                "system": system_prompt,
                "stream": False
            }
            response = await self.client.post("/api/chat", json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            app_logger.error(f"Ollama API request error: {e}")
            raise RuntimeError("Error during chat completion with Ollama.") from e

def _sanitize_llm_output(text: str) -> str:
    """
    Strips invalid characters from LLM output.
    """
    sanitized_text = text.replace('\ufeff', '')
    sanitized_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', sanitized_text)
    return sanitized_text.strip()

def _extract_final_answer_from_json(text: str) -> str:
    """
    Detects if the LLM hallucinated and wrapped a FINAL_ANSWER inside a JSON object.
    If so, it extracts the FINAL_ANSWER string and returns it.
    This makes the agent more robust to common LLM formatting errors.
    """
    try:
        json_match = re.search(r"```json\s*\n(.*?)\n\s*```|(\{.*?\})", text, re.DOTALL)
        if not json_match:
            return text

        json_str = json_match.group(1) or json_match.group(2)
        if not json_str:
            return text
            
        data = json.loads(json_str.strip())

        def find_answer_in_values(d):
            if isinstance(d, dict):
                for value in d.values():
                    found = find_answer_in_values(value)
                    if found:
                        return found
            elif isinstance(d, list):
                for item in d:
                    found = find_answer_in_values(item)
                    if found:
                        return found
            elif isinstance(d, str) and "FINAL_ANSWER:" in d:
                return d
            return None

        final_answer_value = find_answer_in_values(data)

        if final_answer_value:
            app_logger.warning(f"LLM hallucination detected and corrected. Extracted FINAL_ANSWER from JSON.")
            return final_answer_value

    except (json.JSONDecodeError, AttributeError):
        return text
    
    return text

def _get_full_system_prompt(session_data: dict, dependencies: dict, system_prompt_override: str = None) -> str:
    """
    Constructs the final system prompt.
    """
    if system_prompt_override:
        return system_prompt_override

    if not session_data or not dependencies or 'STATE' not in dependencies:
        return "You are a helpful assistant."

    base_prompt_text = session_data.get("system_prompt_template", "You are a helpful assistant.")
    STATE = dependencies['STATE']

    charting_instructions_section = ""
    if APP_CONFIG.CHARTING_ENABLED:
        charting_intensity = session_data.get("charting_intensity", "medium")
        chart_instructions_detail = CHARTING_INSTRUCTIONS.get(charting_intensity, "")
        if chart_instructions_detail:
            charting_instructions_section = f"- **Charting Guidelines:** {chart_instructions_detail}"
    
    # --- MODIFIED: The concept of filtered contexts is removed. Always use the full context. ---
    tools_context = STATE.get('tools_context', '')
    prompts_context = STATE.get('prompts_context', '')

    final_system_prompt = base_prompt_text.replace(
        '{charting_instructions_section}', charting_instructions_section
    ).replace(
        '{tools_context}', tools_context
    ).replace(
        '{prompts_context}', prompts_context
    )
    
    return final_system_prompt

async def call_llm_api(llm_instance: any, prompt: str, session_id: str = None, chat_history=None, raise_on_error: bool = False, system_prompt_override: str = None, dependencies: dict = None, reason: str = "No reason provided.") -> tuple[str, int, int]:
    if not llm_instance:
        raise RuntimeError("LLM is not initialized.")
    
    full_log_message = ""
    response_text = ""
    input_tokens, output_tokens = 0, 0
    
    max_retries = APP_CONFIG.LLM_API_MAX_RETRIES
    base_delay = APP_CONFIG.LLM_API_BASE_DELAY
    
    session_data = get_session(session_id) if session_id else None
    system_prompt = _get_full_system_prompt(session_data, dependencies, system_prompt_override)

    history_for_log = []
    if session_data:
        if APP_CONFIG.CURRENT_PROVIDER == "Google" and hasattr(session_data.get('chat_object'), 'history'):
             history_for_log = [f"[{msg.role}]: {msg.parts[0].text}" for msg in session_data['chat_object'].history]
        elif 'chat_object' in session_data:
             history_for_log = [f"[{msg.get('role')}]: {msg.get('content')}" for msg in session_data.get('chat_object', [])]

    full_log_message = (
        f"--- FULL CONTEXT (Session: {session_id or 'one-off'}) ---\n"
        f"--- REASON FOR CALL ---\n{reason}\n\n"
        f"--- History ---\n{'\n'.join(history_for_log)}\n\n"
        f"--- Current User Prompt (with System Prompt) ---\n"
        f"SYSTEM PROMPT:\n{system_prompt}\n\n"
        f"USER PROMPT:\n{prompt}\n"
    )

    for attempt in range(max_retries):
        try:
            if APP_CONFIG.CURRENT_PROVIDER == "Google":
                is_session_call = session_data is not None and 'chat_object' in session_data
                
                if is_session_call:
                    chat_session = session_data['chat_object']
                    full_prompt_for_api = f"SYSTEM PROMPT:\n{system_prompt}\n\nUSER PROMPT:\n{prompt}"
                    response = await chat_session.send_message_async(full_prompt_for_api)
                else:
                    response = await llm_instance.generate_content_async(prompt)

                if not response or not hasattr(response, 'text'):
                    raise RuntimeError("Google LLM returned an empty or invalid response.")
                response_text = response.text.strip()
                
                if hasattr(response, 'usage_metadata'):
                    usage = response.usage_metadata
                    input_tokens = usage.prompt_token_count
                    output_tokens = usage.candidates_token_count
                
                break

            elif APP_CONFIG.CURRENT_PROVIDER in ["Anthropic", "OpenAI", "Ollama"]:
                history_source = chat_history if chat_history is not None else (session_data.get('chat_object', []) if session_id else [])
                messages_for_api = [{'role': 'assistant' if msg.get('role') == 'model' else msg.get('role'), 'content': msg.get('content')} for msg in history_source]
                messages_for_api.append({'role': 'user', 'content': prompt})
                
                if APP_CONFIG.CURRENT_PROVIDER == "Anthropic":
                    response = await llm_instance.messages.create(
                        model=APP_CONFIG.CURRENT_MODEL, system=system_prompt, messages=messages_for_api, max_tokens=4096, timeout=120.0
                    )
                    response_text = _sanitize_llm_output(response.content[0].text)
                    if hasattr(response, 'usage'):
                        input_tokens, output_tokens = response.usage.input_tokens, response.usage.output_tokens
                
                elif APP_CONFIG.CURRENT_PROVIDER == "OpenAI":
                    messages_for_api.insert(0, {'role': 'system', 'content': system_prompt})
                    response = await llm_instance.chat.completions.create(
                        model=APP_CONFIG.CURRENT_MODEL, messages=messages_for_api, max_tokens=4096, timeout=120.0
                    )
                    response_text = _sanitize_llm_output(response.choices[0].message.content)
                    if hasattr(response, 'usage'):
                        input_tokens, output_tokens = response.usage.prompt_tokens, response.usage.completion_tokens
                
                elif APP_CONFIG.CURRENT_PROVIDER == "Ollama":
                    response = await llm_instance.chat(
                        model=APP_CONFIG.CURRENT_MODEL, messages=messages_for_api, system_prompt=system_prompt
                    )
                    response_text = response["message"]["content"].strip()
                    input_tokens, output_tokens = response.get('prompt_eval_count', 0), response.get('eval_count', 0)
                
                break
            
            elif APP_CONFIG.CURRENT_PROVIDER == "Amazon":
                is_session_call = session_data is not None
                if is_session_call:
                    history = session_data.get('chat_object', [])
                else:
                    system_prompt = system_prompt_override or "You are a helpful assistant."
                    history = chat_history or []

                model_id_to_invoke = APP_CONFIG.CURRENT_MODEL
                body = ""

                if "anthropic" in model_id_to_invoke:
                    messages = [{'role': 'assistant' if msg.get('role') == 'model' else msg.get('role'), 'content': msg.get('content')} for msg in history]
                    messages.append({'role': 'user', 'content': prompt})
                    body = json.dumps({
                        "anthropic_version": "bedrock-2023-05-31", 
                        "max_tokens": 4096, 
                        "system": system_prompt, 
                        "messages": messages
                    })
                elif "amazon.nova" in model_id_to_invoke:
                    messages = [{'role': 'assistant' if msg.get('role') == 'model' else 'user', 'content': [{'text': msg.get('content')}]} for msg in history]
                    messages.append({"role": "user", "content": [{"text": prompt}]})
                    body_dict = {
                        "messages": messages, 
                        "inferenceConfig": {"maxTokens": 4096}
                    }
                    if system_prompt:
                        body_dict["system"] = [{"text": system_prompt}]
                    body = json.dumps(body_dict)
                else: 
                    text_prompt = f"{system_prompt}\n\n" + "".join([f"{msg['role']}: {msg['content']}\n\n" for msg in history]) + f"user: {prompt}\n\nassistant:"
                    body = json.dumps({
                        "inputText": text_prompt, 
                        "textGenerationConfig": {"maxTokenCount": 4096}
                    })
                
                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(None, lambda: llm_instance.invoke_model(body=body, modelId=model_id_to_invoke))
                response_body = json.loads(response.get('body').read())
                
                if "anthropic" in model_id_to_invoke:
                    response_text = response_body.get('content')[0].get('text')
                elif "amazon.nova" in model_id_to_invoke:
                    response_text = response_body.get('output', {}).get('message', {}).get('content', [{}])[0].get('text', '')
                else:
                    response_text = response_body.get('results')[0].get('outputText')
                
                break

            else:
                raise NotImplementedError(f"Provider '{APP_CONFIG.CURRENT_PROVIDER}' is not yet supported.")
        
        except (InternalServerError, RateLimitError, OpenAI_APIError) as e:
            if attempt < max_retries - 1:
                delay = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                app_logger.warning(f"API overloaded or rate limited. Retrying in {delay:.2f}s...")
                await asyncio.sleep(delay)
                continue
            else:
                raise e
        except Exception as e:
            app_logger.error(f"Error calling LLM API for provider {APP_CONFIG.CURRENT_PROVIDER}: {e}", exc_info=True)
            llm_logger.error(full_log_message)
            llm_logger.error(f"--- ERROR in LLM call ---\n{e}\n" + "-"*50 + "\n")
            raise e

    if not response_text and raise_on_error:
        raise RuntimeError(f"LLM call failed after {max_retries} retries.")

    response_text = _extract_final_answer_from_json(response_text)

    llm_logger.info(full_log_message)
    llm_logger.info(f"--- RESPONSE ---\n{response_text}\n" + "-"*50 + "\n")

    if session_id:
        update_token_count(session_id, input_tokens, output_tokens)

    return response_text, input_tokens, output_tokens


def _is_model_certified(model_name: str, certified_list: list[str]) -> bool:
    """
    Checks if a model is certified, supporting wildcards.
    """
    for pattern in certified_list:
        regex_pattern = re.escape(pattern).replace('\\*', '.*')
        if re.fullmatch(regex_pattern, model_name):
            return True
    return False

async def list_models(provider: str, credentials: dict) -> list[dict]:
    """
    Lists available models for a given provider and checks certification status.
    """
    certified_list = []
    model_names = []

    if provider == "Google":
        certified_list = CERTIFIED_GOOGLE_MODELS
        genai.configure(api_key=credentials.get("apiKey"))
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        model_names = [name.split('/')[-1] for name in models]
    
    elif provider == "Anthropic":
        certified_list = CERTIFIED_ANTHROPIC_MODELS
        client = AsyncAnthropic(api_key=credentials.get("apiKey"))
        models_page = await client.models.list()
        model_names = [model.id for model in models_page.data]

    elif provider == "OpenAI":
        certified_list = CERTIFIED_OPENAI_MODELS
        client = AsyncOpenAI(api_key=credentials.get("apiKey"))
        models_page = await client.models.list()
        model_names = [model.id for model in models_page.data if "gpt" in model.id]

    elif provider == "Amazon":
        bedrock_client = boto3.client(
            service_name='bedrock',
            aws_access_key_id=credentials.get("aws_access_key_id"),
            aws_secret_access_key=credentials.get("aws_secret_access_key"),
            region_name=credentials.get("aws_region")
        )
        loop = asyncio.get_running_loop()
        if credentials.get("listing_method") == "inference_profiles":
            certified_list = CERTIFIED_AMAZON_PROFILES
            response = await loop.run_in_executor(None, lambda: bedrock_client.list_inference_profiles())
            model_names = [p['inferenceProfileArn'] for p in response['inferenceProfileSummaries']]
        else:
            certified_list = CERTIFIED_AMAZON_MODELS
            response = await loop.run_in_executor(None, lambda: bedrock_client.list_foundation_models(byOutputModality='TEXT'))
            model_names = [m['modelId'] for m in response['modelSummaries']]
    
    elif provider == "Ollama":
        certified_list = CERTIFIED_OLLAMA_MODELS
        client = OllamaClient(host=credentials.get("host"))
        models_data = await client.list_models()
        model_names = [m.get("name") for m in models_data]

    return [
        {
            "name": name,
            "certified": APP_CONFIG.ALL_MODELS_UNLOCKED or _is_model_certified(name, certified_list)
        }
        for name in model_names
    ]
