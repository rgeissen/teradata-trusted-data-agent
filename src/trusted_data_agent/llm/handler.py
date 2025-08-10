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
from trusted_data_agent.agent.prompts import CHARTING_INSTRUCTIONS
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
            # For Ollama, the system prompt is a dedicated parameter
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
    Strips invalid characters from LLM output, specifically non-printable
    and zero-width characters that can cause parsing errors.
    """
    sanitized_text = text.replace('\ufeff', '')
    sanitized_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', sanitized_text)
    return sanitized_text.strip()

def _get_full_system_prompt(session_data: dict, dependencies: dict, system_prompt_override: str = None) -> str:
    """
    Constructs the final system prompt by injecting the latest context from the
    global application state into the session's prompt template.
    """
    if system_prompt_override:
        return system_prompt_override

    if not session_data or not dependencies or 'STATE' not in dependencies:
        return "You are a helpful assistant."

    base_prompt_text = session_data.get("system_prompt_template", "You are a helpful assistant.")
    charting_intensity = session_data.get("charting_intensity", "none")
    STATE = dependencies['STATE']

    chart_instructions = CHARTING_INSTRUCTIONS.get(charting_intensity, CHARTING_INSTRUCTIONS['none'])
    
    final_system_prompt = base_prompt_text
    final_system_prompt = final_system_prompt.replace("{charting_instructions}", chart_instructions)
    final_system_prompt = final_system_prompt.replace("{tools_context}", STATE.get('tools_context', ''))
    final_system_prompt = final_system_prompt.replace("{prompts_context}", STATE.get('prompts_context', ''))
    final_system_prompt = final_system_prompt.replace("{charts_context}", "")
    
    return final_system_prompt

async def call_llm_api(llm_instance: any, prompt: str, session_id: str = None, chat_history=None, raise_on_error: bool = False, system_prompt_override: str = None, dependencies: dict = None) -> tuple[str, int, int]:
    if not llm_instance:
        raise RuntimeError("LLM is not initialized.")
    
    full_log_message = ""
    response_text = ""
    input_tokens, output_tokens = 0, 0
    
    max_retries = APP_CONFIG.LLM_API_MAX_RETRIES
    base_delay = APP_CONFIG.LLM_API_BASE_DELAY
    
    for attempt in range(max_retries):
        try:
            session_data = get_session(session_id) if session_id else None
            system_prompt = _get_full_system_prompt(session_data, dependencies, system_prompt_override)

            if APP_CONFIG.CURRENT_PROVIDER == "Google":
                is_session_call = session_data is not None
                
                if is_session_call:
                    chat_session = session_data['chat_object']
                    full_prompt = f"SYSTEM PROMPT:\n{system_prompt}\n\nUSER PROMPT:\n{prompt}"

                    history_for_log = chat_session.history
                    if history_for_log:
                        formatted_lines = [f"[{msg.role}]: {msg.parts[0].text}" for msg in history_for_log]
                        full_log_message += f"--- FULL CONTEXT (Session: {session_id}) ---\n--- History ---\n" + "\n".join(formatted_lines) + "\n\n"
                    
                    full_log_message += f"--- Current User Prompt (with System Prompt) ---\n{full_prompt}\n"
                    response = await chat_session.send_message_async(full_prompt)
                else:
                    full_log_message += f"--- ONE-OFF CALL ---\n--- Prompt ---\n{prompt}\n"
                    response = await llm_instance.generate_content_async(prompt)

                if not response or not hasattr(response, 'text'):
                    raise RuntimeError("Google LLM returned an empty or invalid response.")
                response_text = response.text.strip()
                
                if hasattr(response, 'usage_metadata'):
                    usage = response.usage_metadata
                    input_tokens = usage.prompt_token_count
                    output_tokens = usage.candidates_token_count
                    if session_id:
                        update_token_count(session_id, input_tokens, output_tokens)
                
                llm_logger.info(full_log_message)
                llm_logger.info(f"--- RESPONSE ---\n{response_text}\n" + "-"*50 + "\n")
                return response_text, input_tokens, output_tokens

            elif APP_CONFIG.CURRENT_PROVIDER in ["Anthropic", "OpenAI", "Ollama", "Amazon"]:
                history_source = chat_history
                if history_source is None:
                    history_source = session_data.get('chat_object', []) if session_id else []

                messages_for_log = [{'role': 'assistant' if msg.get('role') == 'model' else msg.get('role'), 'content': msg.get('content')} for msg in history_source if msg.get('role') in ['user', 'model', 'assistant']]
                messages_for_log.append({'role': 'user', 'content': prompt})

                full_log_message += f"--- SYSTEM PROMPT ---\n{system_prompt}\n\n"
                full_log_message += f"--- FULL CONTEXT (Session: {session_id or 'one-off'}) ---\n"
                for msg in messages_for_log: full_log_message += f"[{msg['role']}]: {msg['content']}\n"
                
                if APP_CONFIG.CURRENT_PROVIDER == "Anthropic":
                    response = await llm_instance.messages.create(
                        model=APP_CONFIG.CURRENT_MODEL, system=system_prompt, messages=messages_for_log, max_tokens=4096, timeout=120.0
                    )
                    if not response or not response.content: raise RuntimeError("Anthropic LLM returned an empty or invalid response.")
                    response_text = _sanitize_llm_output(response.content[0].text)
                    if hasattr(response, 'usage'):
                        usage = response.usage
                        input_tokens = usage.input_tokens
                        output_tokens = usage.output_tokens
                
                elif APP_CONFIG.CURRENT_PROVIDER == "OpenAI":
                    messages_for_log.insert(0, {'role': 'system', 'content': system_prompt})
                    response = await llm_instance.chat.completions.create(
                        model=APP_CONFIG.CURRENT_MODEL, messages=messages_for_log, max_tokens=4096, timeout=120.0
                    )
                    if not response or not response.choices or not response.choices[0].message: raise RuntimeError("OpenAI LLM returned an empty or invalid response.")
                    response_text = _sanitize_llm_output(response.choices[0].message.content)
                    if hasattr(response, 'usage'):
                        usage = response.usage
                        input_tokens = usage.prompt_tokens
                        output_tokens = usage.completion_tokens
                
                elif APP_CONFIG.CURRENT_PROVIDER == "Ollama":
                    response = await llm_instance.chat(
                        model=APP_CONFIG.CURRENT_MODEL, messages=messages_for_log, system_prompt=system_prompt
                    )
                    if not response or "message" not in response or "content" not in response["message"]: raise RuntimeError("Ollama LLM returned an empty or invalid response.")
                    response_text = response["message"]["content"].strip()
                    if 'prompt_eval_count' in response and 'eval_count' in response:
                        input_tokens = response.get('prompt_eval_count', 0)
                        output_tokens = response.get('eval_count', 0)

                elif APP_CONFIG.CURRENT_PROVIDER == "Amazon":
                    model_id_to_invoke = APP_CONFIG.CURRENT_MODEL
                    if "amazon.titan" in model_id_to_invoke and not model_id_to_invoke.startswith("arn:aws:bedrock:") and APP_CONFIG.CURRENT_AWS_REGION:
                        region = APP_CONFIG.CURRENT_AWS_REGION
                        prefix = ""
                        if region.startswith("us-"): prefix = "us."
                        elif region.startswith("eu-"): prefix = "eu."
                        elif region.startswith("ap-"): prefix = "apac."
                        if prefix: model_id_to_invoke = f"{prefix}{model_id_to_invoke}"

                    body = ""
                    if "anthropic" in model_id_to_invoke:
                        body = json.dumps({"anthropic_version": "bedrock-2023-05-31", "max_tokens": 4096, "system": system_prompt, "messages": messages_for_log})
                    # --- FIX: Restore correct JSON Array format for Nova models ---
                    elif "amazon.nova" in model_id_to_invoke:
                        nova_messages = [{'role': 'assistant' if msg.get('role') == 'model' else 'user', 'content': [{'text': msg.get('content')}]} for msg in history_source]
                        nova_messages.append({"role": "user", "content": [{"text": prompt}]})
                        body_dict = {"messages": nova_messages, "inferenceConfig": {"maxTokens": 4096, "temperature": 0.7, "topP": 0.9}}
                        if system_prompt: body_dict["system"] = [{"text": system_prompt}]
                        body = json.dumps(body_dict)
                    else: # Legacy Titan
                        text_prompt = f"{system_prompt}\n\n" + "".join([f"{msg['role']}: {msg['content']}\n\n" for msg in history_source]) + f"user: {prompt}\n\nassistant:"
                        body = json.dumps({"inputText": text_prompt, "textGenerationConfig": {"maxTokenCount": 4096, "temperature": 0.7, "topP": 0.9}})
                    
                    loop = asyncio.get_running_loop()
                    response = await loop.run_in_executor(None, lambda: llm_instance.invoke_model(body=body, modelId=model_id_to_invoke))
                    response_body = json.loads(response.get('body').read())
                    
                    if "anthropic" in model_id_to_invoke:
                        response_text = response_body.get('content')[0].get('text')
                    elif "amazon.nova" in model_id_to_invoke:
                        response_text = response_body.get('output', {}).get('message', {}).get('content', [{}])[0].get('text', '')
                    else:
                        response_text = response_body.get('results')[0].get('outputText')
                
                if session_id:
                    update_token_count(session_id, input_tokens, output_tokens)

                llm_logger.info(full_log_message)
                llm_logger.info(f"--- RESPONSE ---\n{response_text}\n" + "-"*50 + "\n")
                return response_text, input_tokens, output_tokens

            else:
                raise NotImplementedError(f"Provider '{APP_CONFIG.CURRENT_PROVIDER}' is not yet supported.")
        
        except (InternalServerError, RateLimitError, OpenAI_APIError) as e:
            if (APP_CONFIG.CURRENT_PROVIDER in ["Anthropic", "OpenAI"]) and attempt < max_retries - 1:
                delay = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                app_logger.warning(f"{APP_CONFIG.CURRENT_PROVIDER} model overloaded or rate limited. Retrying in {delay:.2f} seconds... (Attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(delay)
                continue
            else:
                app_logger.error(f"{APP_CONFIG.CURRENT_PROVIDER} model still failing after {max_retries} retries. Aborting.")
                raise e
        except Exception as e:
            app_logger.error(f"Error calling LLM API for provider {APP_CONFIG.CURRENT_PROVIDER}: {e}", exc_info=True)
            llm_logger.error(full_log_message)
            llm_logger.error(f"--- ERROR in LLM call ---\n{e}\n" + "-"*50 + "\n")
            raise e
    
    error_message = f"FINAL_ANSWER: I'm sorry, but I encountered an error while communicating with the {APP_CONFIG.CURRENT_PROVIDER} model after {max_retries} retries."
    raise RuntimeError(error_message)


def _is_model_certified(model_name: str, certified_list: list[str]) -> bool:
    """
    Checks if a model is certified, supporting exact matches and wildcards
    anywhere in the string.
    """
    for pattern in certified_list:
        regex_pattern = re.escape(pattern).replace('\\*', '.*')
        if re.fullmatch(regex_pattern, model_name):
            return True
    return False

async def list_models(provider: str, credentials: dict) -> list[dict]:
    """
    Lists available models for a given provider and checks their certification status
    using a wildcard-aware function.
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
        try:
            client = AsyncAnthropic(api_key=credentials.get("apiKey"))
            models_page = await client.models.list()
            
            model_names = []
            for model in models_page.data:
                if hasattr(model, 'id'):
                    model_names.append(model.id)
                elif isinstance(model, tuple) and len(model) > 0 and hasattr(model[0], 'id'):
                    model_names.append(model[0].id)
                else:
                    app_logger.warning(f"Unexpected item structure in Anthropic models list: {type(model)}")
        except Exception as e:
            app_logger.error(f"Failed to fetch models from Anthropic: {e}")
            raise e

    elif provider == "OpenAI":
        certified_list = CERTIFIED_OPENAI_MODELS
        try:
            client = AsyncOpenAI(api_key=credentials.get("apiKey"))
            models_page = await client.models.list()
            model_names = [model.id for model in models_page.data if "gpt" in model.id]
        except Exception as e:
            app_logger.error(f"Failed to fetch models from OpenAI: {e}")
            raise e

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

    else:
        return []

    return [
        {
            "name": name,
            "certified": APP_CONFIG.ALL_MODELS_UNLOCKED or _is_model_certified(name, certified_list)
        }
        for name in model_names
    ]
