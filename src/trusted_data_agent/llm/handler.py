# src/trusted_data_agent/llm/handler.py
import asyncio
import json
import logging

import google.generativeai as genai
from anthropic import AsyncAnthropic
import boto3

from trusted_data_agent.core.config import APP_CONFIG
from trusted_data_agent.core.session_manager import get_session
from trusted_data_agent.core.config import (
    CERTIFIED_GOOGLE_MODELS, CERTIFIED_ANTHROPIC_MODELS,
    CERTIFIED_AMAZON_MODELS, CERTIFIED_AMAZON_PROFILES
)
from trusted_data_agent.core.utils import unwrap_exception

# Set up loggers
llm_logger = logging.getLogger("llm_conversation")
app_logger = logging.getLogger("quart.app")

async def call_llm_api(llm_instance: any, prompt: str, session_id: str = None, chat_history=None, raise_on_error: bool = False, system_prompt_override: str = None) -> str:
    """
    Calls the appropriate LLM API based on the current provider configuration.
    """
    if not llm_instance:
        raise RuntimeError("LLM is not initialized.")
    
    full_log_message = ""
    response_text = ""

    try:
        # --- Provider-Aware API Call Logic ---
        if APP_CONFIG.CURRENT_PROVIDER == "Google":
            session_data = get_session(session_id) if session_id else None
            is_session_call = session_data is not None
            
            if is_session_call:
                chat_session = session_data['chat_object']
                history_for_log = chat_session.history
                if history_for_log:
                    formatted_lines = [f"[{msg.role}]: {msg.parts[0].text}" for msg in history_for_log]
                    full_log_message += f"--- FULL CONTEXT (Session: {session_id}) ---\n--- History ---\n" + "\n".join(formatted_lines) + "\n\n"
                
                full_log_message += f"--- Current User Prompt ---\n{prompt}\n"
                llm_logger.info(full_log_message)
                response = await chat_session.send_message_async(prompt)
            else: # Session-less call
                full_log_message += f"--- ONE-OFF CALL ---\n--- Prompt ---\n{prompt}\n"
                llm_logger.info(full_log_message)
                response = await llm_instance.generate_content_async(prompt)

            if not response or not hasattr(response, 'text'):
                raise RuntimeError("Google LLM returned an empty or invalid response.")
            response_text = response.text.strip()

        elif APP_CONFIG.CURRENT_PROVIDER == "Anthropic":
            session_data = get_session(session_id) if session_id else None
            system_prompt = system_prompt_override or (session_data['system_prompt'] if session_data else "")
            if not system_prompt:
                 raise ValueError("A session_id or system_prompt_override is required for Anthropic provider.")

            history_source = chat_history if chat_history is not None else (session_data.get('chat_object', []))

            messages = [{'role': 'assistant' if msg.get('role') == 'model' else msg.get('role'), 'content': msg.get('content')} for msg in history_source if msg.get('role') in ['user', 'model', 'assistant']]
            messages.append({'role': 'user', 'content': prompt})

            full_log_message += f"--- SYSTEM PROMPT ---\n{system_prompt}\n\n"
            full_log_message += f"--- FULL CONTEXT (Session: {session_id or 'one-off'}) ---\n"
            for msg in messages: full_log_message += f"[{msg['role']}]: {msg['content']}\n"
            llm_logger.info(full_log_message)

            response = await llm_instance.messages.create(
                model=APP_CONFIG.CURRENT_MODEL,
                system=system_prompt,
                messages=messages,
                max_tokens=4096
            )
            if not response or not response.content:
                raise RuntimeError("Anthropic LLM returned an empty or invalid response.")
            response_text = response.content[0].text.strip()
        
        elif APP_CONFIG.CURRENT_PROVIDER == "Amazon":
            session_data = get_session(session_id) if session_id else None
            is_session_call = session_data is not None

            if is_session_call:
                system_prompt = session_data['system_prompt']
                history = session_data['chat_object']
            else: # One-off call
                system_prompt = system_prompt_override or "You are a helpful assistant."
                history = chat_history or []

            model_id_to_invoke = APP_CONFIG.CURRENT_MODEL
            # Adjust Nova model ID for regional endpoints
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

            # Determine payload structure based on model
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
            else: # Default to Amazon Titan format
                text_prompt = f"{system_prompt}\n\n" + "".join([f"{msg['role']}: {msg['content']}\n\n" for msg in history]) + f"user: {prompt}\n\nassistant:"
                body = json.dumps({"inputText": text_prompt, "textGenerationConfig": {"maxTokenCount": 4096, "temperature": 0.7, "topP": 0.9}})
            
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(None, lambda: llm_instance.invoke_model(body=body, modelId=model_id_to_invoke))
            response_body = json.loads(response.get('body').read())

            # Extract text based on model type
            if "anthropic" in model_id_to_invoke: response_text = response_body.get('content')[0].get('text')
            elif "amazon.nova" in model_id_to_invoke: response_text = response_body.get('output', {}).get('message', {}).get('content', [{}])[0].get('text', '')
            else: response_text = response_body.get('results')[0].get('outputText')

        else:
            raise NotImplementedError(f"Provider '{APP_CONFIG.CURRENT_PROVIDER}' is not yet supported.")

        llm_logger.info(f"--- RESPONSE ---\n{response_text}\n" + "-"*50 + "\n")
        return response_text

    except Exception as e:
        app_logger.error(f"Error calling LLM API for provider {APP_CONFIG.CURRENT_PROVIDER}: {e}", exc_info=True)
        llm_logger.error(f"--- ERROR in LLM call ---\n{e}\n" + "-"*50 + "\n")
        if raise_on_error:
            raise e
        return f"FINAL_ANSWER: I'm sorry, but I encountered an error while communicating with the language model: {str(e)}"

async def list_models(provider: str, credentials: dict) -> list[dict]:
    """Fetches a list of available models from the specified provider."""
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
        else: # Foundation models
            response = await loop.run_in_executor(None, lambda: bedrock_client.list_foundation_models(byOutputModality='TEXT'))
            models = [m['modelId'] for m in response['modelSummaries']]
            return [{"name": name, "certified": APP_CONFIG.ALL_MODELS_UNLOCKED or name in CERTIFIED_AMAZON_MODELS} for name in models]
    
    # This final return should only be hit if the provider is unknown
    return []