# src/trusted_data_agent/llm/handler.py
import asyncio
import json
import logging
import httpx

import google.generativeai as genai
from anthropic import APIError, AsyncAnthropic
import boto3

from trusted_data_agent.core.config import APP_CONFIG
from trusted_data_agent.core.session_manager import get_session, update_token_count
from trusted_data_agent.core.config import (
    CERTIFIED_GOOGLE_MODELS, CERTIFIED_ANTHROPIC_MODELS,
    CERTIFIED_AMAZON_MODELS, CERTIFIED_AMAZON_PROFILES,
    CERTIFIED_OLLAMA_MODELS
)

llm_logger = logging.getLogger("llm_conversation")
app_logger = logging.getLogger("quart.app")

class OllamaClient:
    """A simple async client for interacting with the Ollama API."""
    def __init__(self, host: str):
        self.host = host
        self.client = httpx.AsyncClient(base_url=self.host, timeout=60.0)

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

async def call_llm_api(llm_instance: any, prompt: str, session_id: str = None, chat_history=None, raise_on_error: bool = False, system_prompt_override: str = None) -> tuple[str, int, int]:
    if not llm_instance:
        raise RuntimeError("LLM is not initialized.")
    
    full_log_message = ""
    response_text = ""
    # --- MODIFIED: Initialize token counts ---
    input_tokens, output_tokens = 0, 0
    # --- END MODIFICATION ---

    try:
        session_data = get_session(session_id) if session_id else None

        if APP_CONFIG.CURRENT_PROVIDER == "Google":
            is_session_call = session_data is not None
            
            if is_session_call:
                chat_session = session_data['chat_object']
                history_for_log = chat_session.history
                if history_for_log:
                    formatted_lines = [f"[{msg.role}]: {msg.parts[0].text}" for msg in history_for_log]
                    full_log_message += f"--- FULL CONTEXT (Session: {session_id}) ---\n--- History ---\n" + "\n".join(formatted_lines) + "\n\n"
                
                full_log_message += f"--- Current User Prompt ---\n{prompt}\n"
                response = await chat_session.send_message_async(prompt)
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

        elif APP_CONFIG.CURRENT_PROVIDER == "Anthropic":
            system_prompt = system_prompt_override or (session_data['system_prompt'] if session_data else "")
            if not system_prompt:
                 raise ValueError("A session_id or system_prompt_override is required for Anthropic provider.")

            history_source = chat_history
            if history_source is None:
                history_source = session_data.get('chat_object', []) if session_data else []

            messages = [{'role': 'assistant' if msg.get('role') == 'model' else msg.get('role'), 'content': msg.get('content')} for msg in history_source if msg.get('role') in ['user', 'model', 'assistant']]
            messages.append({'role': 'user', 'content': prompt})

            full_log_message += f"--- SYSTEM PROMPT ---\n{system_prompt}\n\n"
            full_log_message += f"--- FULL CONTEXT (Session: {session_id or 'one-off'}) ---\n"
            for msg in messages: full_log_message += f"[{msg['role']}]: {msg['content']}\n"
            
            response = await llm_instance.messages.create(
                model=APP_CONFIG.CURRENT_MODEL,
                system=system_prompt,
                messages=messages,
                max_tokens=4096
            )
            if not response or not response.content:
                raise RuntimeError("Anthropic LLM returned an empty or invalid response.")
            response_text = response.content[0].text.strip()
            
            if hasattr(response, 'usage'):
                usage = response.usage
                input_tokens = usage.input_tokens
                output_tokens = usage.output_tokens
                if session_id:
                    update_token_count(session_id, input_tokens, output_tokens)
        
        elif APP_CONFIG.CURRENT_PROVIDER == "Amazon":
            is_session_call = session_data is not None

            if is_session_call:
                system_prompt = session_data['system_prompt']
                history = session_data['chat_object']
            else:
                system_prompt = system_prompt_override or "You are a helpful assistant."
                history = chat_history or []

            full_log_message += f"--- SYSTEM PROMPT ---\n{system_prompt}\n\n"
            full_log_message += f"--- FULL CONTEXT (Session: {session_id or 'one-off'}) ---\n"
            for msg in history: full_log_message += f"[{msg.get('role')}]: {msg.get('content')}\n"
            full_log_message += f"[user]: {prompt}\n"

            model_id_to_invoke = APP_CONFIG.CURRENT_MODEL
            
            if "amazon.nova" in model_id_to_invoke and not model_id_to_invoke.startswith("arn:aws:bedrock:") and APP_CONFIG.CURRENT_AWS_REGION:
                region = APP_CONFIG.CURRENT_AWS_REGION
                prefix = ""
                if region.startswith("us-"): prefix = "us."
                elif region.startswith("eu-"): prefix = "eu."
                elif region.startswith("ap-"): prefix = "apac."
                
                if prefix:
                    adjusted_id = f"{prefix}{model_id_to_invoke}"
                    app_logger.info(f"Adjusting Nova model ID from '{model_id_to_invoke}' to '{adjusted_id}' for region '{region}'.")
                    model_id_to_invoke = adjusted_id

            if "anthropic" in model_id_to_invoke:
                messages = [{'role': msg['role'], 'content': msg['content']} for msg in history]
                messages.append({'role': 'user', 'content': prompt})
                body = json.dumps({"anthropic_version": "bedrock-2023-05-31", "max_tokens": 4096, "system": system_prompt, "messages": messages})
            elif "amazon.nova" in model_id_to_invoke:
                messages = [{'role': 'assistant' if msg.get('role') == 'model' else 'user', 'content': [{'text': msg.get('content')}]} for msg in history]
                messages.append({"role": "user", "content": [{"text": prompt}]})
                body_dict = {"messages": messages, "inferenceConfig": {"maxTokens": 4096, "temperature": 0.7, "topP": 0.9}}
                if system_prompt: body_dict["system"] = [{"text": system_prompt}]
                body = json.dumps(body_dict)
            else:
                text_prompt = f"{system_prompt}\n\n" + "".join([f"{msg['role']}: {msg['content']}\n\n" for msg in history]) + f"user: {prompt}\n\nassistant:"
                body = json.dumps({"inputText": text_prompt, "textGenerationConfig": {"maxTokenCount": 4096, "temperature": 0.7, "topP": 0.9}})
            
            loop = asyncio.get_running_loop()

            app_logger.warning(f"Using non-streaming 'invoke_model' for model: {model_id_to_invoke}. Token usage not available for Amazon Bedrock.")
            response = await loop.run_in_executor(None, lambda: llm_instance.invoke_model(body=body, modelId=model_id_to_invoke))
            response_body = json.loads(response.get('body').read())

            if "anthropic" in model_id_to_invoke: response_text = response_body.get('content')[0].get('text')
            elif "amazon.nova" in model_id_to_invoke: response_text = response_body.get('output', {}).get('message', {}).get('content', [{}])[0].get('text', '')
            else: response_text = response_body.get('results')[0].get('outputText')

        elif APP_CONFIG.CURRENT_PROVIDER == "Ollama":
            system_prompt = system_prompt_override or (session_data['system_prompt'] if session_data else "")
            history_source = chat_history or (session_data.get('chat_object', []) if session_data else [])
            
            messages = [{'role': msg.get('role'), 'content': msg.get('content')} for msg in history_source]
            messages.append({'role': 'user', 'content': prompt})

            full_log_message += f"--- SYSTEM PROMPT ---\n{system_prompt}\n\n"
            full_log_message += f"--- FULL CONTEXT (Session: {session_id or 'one-off'}) ---\n"
            for msg in messages: full_log_message += f"[{msg['role']}]: {msg['content']}\n"

            response = await llm_instance.chat(
                model=APP_CONFIG.CURRENT_MODEL,
                messages=messages,
                system_prompt=system_prompt
            )
            if not response or "message" not in response or "content" not in response["message"]:
                raise RuntimeError("Ollama LLM returned an empty or invalid response.")
            response_text = response["message"]["content"].strip()
            
            if 'prompt_eval_count' in response and 'eval_count' in response:
                input_tokens = response.get('prompt_eval_count', 0)
                output_tokens = response.get('eval_count', 0)
                if session_id:
                    update_token_count(session_id, input_tokens, output_tokens)

        else:
            raise NotImplementedError(f"Provider '{APP_CONFIG.CURRENT_PROVIDER}' is not yet supported.")

        llm_logger.info(full_log_message)
        llm_logger.info(f"--- RESPONSE ---\n{response_text}\n" + "-"*50 + "\n")
        # --- MODIFIED: Return token counts along with text ---
        return response_text, input_tokens, output_tokens
        # --- END MODIFICATION ---

    except Exception as e:
        app_logger.error(f"Error calling LLM API for provider {APP_CONFIG.CURRENT_PROVIDER}: {e}", exc_info=True)
        llm_logger.error(full_log_message)
        llm_logger.error(f"--- ERROR in LLM call ---\n{e}\n" + "-"*50 + "\n")
        if raise_on_error:
            raise e
        # --- MODIFIED: Return zero tokens on error ---
        error_message = f"FINAL_ANSWER: I'm sorry, but I encountered an error while communicating with the language model: {str(e)}"
        return error_message, 0, 0
        # --- END MODIFICATION ---

async def list_models(provider: str, credentials: dict) -> list[dict]:
    if provider == "Google":
        genai.configure(api_key=credentials.get("apiKey"))
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        clean_models = [name.split('/')[-1] for name in models]
        return [{"name": name, "certified": APP_CONFIG.ALL_MODELS_UNLOCKED or name in CERTIFIED_GOOGLE_MODELS} for name in clean_models]
    
    elif provider == "Anthropic":
        return [{"name": name, "certified": APP_CONFIG.ALL_MODELS_UNLOCKED or name in CERTIFIED_ANTHROPIC_MODELS} for name in CERTIFIED_ANTHROPIC_MODELS]

    elif provider == "Amazon":
        bedrock_client = boto3.client(
            service_name='bedrock',
            aws_access_key_id=credentials.get("aws_access_key_id"),
            aws_secret_access_key=credentials.get("aws_secret_access_key"),
            region_name=credentials.get("aws_region")
        )
        loop = asyncio.get_running_loop()
        if credentials.get("listing_method") == "inference_profiles":
            response = await loop.run_in_executor(None, lambda: bedrock_client.list_inference_profiles())
            models = [p['inferenceProfileArn'] for p in response['inferenceProfileSummaries']]
            return [{"name": name, "certified": APP_CONFIG.ALL_MODELS_UNLOCKED or name in CERTIFIED_AMAZON_PROFILES} for name in models]
        else:
            response = await loop.run_in_executor(None, lambda: bedrock_client.list_foundation_models(byOutputModality='TEXT'))
            models = [m['modelId'] for m in response['modelSummaries']]
            return [{"name": name, "certified": APP_CONFIG.ALL_MODELS_UNLOCKED or name in CERTIFIED_AMAZON_MODELS} for name in models]
    
    elif provider == "Ollama":
        client = OllamaClient(host=credentials.get("host"))
        models_data = await client.list_models()
        model_names = [m.get("name") for m in models_data]
        return [{"name": name, "certified": APP_CONFIG.ALL_MODELS_UNLOCKED or name in CERTIFIED_OLLAMA_MODELS} for name in model_names]

    return []
