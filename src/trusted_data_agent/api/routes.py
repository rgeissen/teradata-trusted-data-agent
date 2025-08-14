# src/trusted_data_agent/api/routes.py
import json
import os
import logging
import asyncio
import sys
import copy

from quart import Blueprint, request, jsonify, render_template, Response
from google.api_core import exceptions as google_exceptions
from anthropic import APIError, AsyncAnthropic
from openai import AsyncOpenAI, APIError as OpenAI_APIError
from botocore.exceptions import ClientError
import google.generativeai as genai
import boto3
from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp.shared.exceptions import McpError

from trusted_data_agent.core.config import APP_CONFIG
from trusted_data_agent.core import session_manager
from trusted_data_agent.agent.prompts import PROVIDER_SYSTEM_PROMPTS, CHARTING_INSTRUCTIONS
from trusted_data_agent.agent.executor import PlanExecutor, _format_sse
from trusted_data_agent.llm import handler as llm_handler
from trusted_data_agent.mcp import adapter as mcp_adapter

api_bp = Blueprint('api', __name__)
app_logger = logging.getLogger("quart.app")

# --- NEW: Define local dummy classes to avoid invalid imports ---
# This uses duck typing to create objects that are structurally
# compatible with what the MCP library expects, without relying on
# the library's internal, non-public class structure.

class _DummyContent:
    """A duck-typed object to stand in for the MCP Content class."""
    def __init__(self, text=""):
        self.text = text

class _DummyMessage:
    """A duck-typed object to stand in for the MCP Message class."""
    def __init__(self, role="user", content=None):
        self.role = role
        self.content = content if content is not None else _DummyContent()


def unwrap_exception(e: BaseException) -> BaseException:
    """Recursively unwraps ExceptionGroups to find the root cause."""
    if isinstance(e, ExceptionGroup) and e.exceptions:
        return unwrap_exception(e.exceptions[0])
    return e

def set_dependencies(app_state):
    """Injects the global application state into this blueprint."""
    global STATE
    STATE = app_state

def _regenerate_contexts():
    """
    Updates all capability contexts ('tools_context', 'prompts_context', etc.)
    in the global STATE based on the current disabled lists and prints the
    current status to the console for debugging.
    """
    print("\n--- Regenerating Agent Capability Contexts ---")
    
    # Regenerate Tool Contexts
    if STATE.get('mcp_tools'):
        all_tools = list(STATE['mcp_tools'].values())
        disabled_tools_list = STATE.get("disabled_tools", [])
        enabled_tools = [t for t in all_tools if t.name not in disabled_tools_list]
        
        print(f"\n[ Tools Status ]")
        print(f"  - Active: {len(enabled_tools)}")
        for tool in enabled_tools:
            print(f"    - {tool.name}")
        print(f"  - Inactive: {len(disabled_tools_list)}")
        for tool_name in disabled_tools_list:
            print(f"    - {tool_name}")

        tool_details_list = []
        for tool in enabled_tools:
            tool_str = f"- `{tool.name}`: {tool.description}"
            args_dict = tool.args if isinstance(tool.args, dict) else {}
            if args_dict:
                tool_str += "\n  - Arguments:"
                for arg_name, arg_details in args_dict.items():
                    arg_type = arg_details.get('type', 'any')
                    is_required = arg_details.get('required', False)
                    req_str = "required" if is_required else "optional"
                    arg_desc = arg_details.get('description', 'No description.')
                    tool_str += f"\n    - `{arg_name}` ({arg_type}, {req_str}): {arg_desc}"
            tool_details_list.append(tool_str)
        
        STATE['tools_context'] = "--- Available Tools ---\n" + "\n".join(tool_details_list)
        app_logger.info(f"Regenerated LLM tool context. {len(enabled_tools)} tools are active.")

        if STATE.get('structured_tools'):
            for category, tool_list in STATE['structured_tools'].items():
                for tool_info in tool_list:
                    tool_info['disabled'] = tool_info['name'] in disabled_tools_list
            app_logger.info("Updated 'disabled' status in structured tools for the UI.")

    # Regenerate Prompt Contexts
    if STATE.get('mcp_prompts'):
        all_prompts = list(STATE['mcp_prompts'].values())
        disabled_prompts_list = STATE.get("disabled_prompts", [])
        enabled_prompts = [p for p in all_prompts if p.name not in disabled_prompts_list]
        
        print(f"\n[ Prompts Status ]")
        print(f"  - Active: {len(enabled_prompts)}")
        for prompt in enabled_prompts:
            print(f"    - {prompt.name}")
        print(f"  - Inactive: {len(disabled_prompts_list)}")
        for prompt_name in disabled_prompts_list:
            print(f"    - {prompt_name}")

        if enabled_prompts:
            STATE['prompts_context'] = "--- Available Prompts ---\n" + "\n".join([f"- `{p.name}`: {p.description or 'No description available.'}" for p in enabled_prompts])
        else:
            STATE['prompts_context'] = "--- No Prompts Available ---"
        
        app_logger.info(f"Regenerated LLM prompt context. {len(enabled_prompts)} prompts are active.")

        if STATE.get('structured_prompts'):
            for category, prompt_list in STATE['structured_prompts'].items():
                for prompt_info in prompt_list:
                    prompt_info['disabled'] = prompt_info['name'] in disabled_prompts_list
            app_logger.info("Updated 'disabled' status in structured prompts for the UI.")
    
    print("\n" + "-"*44)


@api_bp.route("/")
async def index():
    """Serves the main HTML page."""
    return await render_template("index.html")

@api_bp.route("/simple_chat", methods=["POST"])
async def simple_chat():
    """
    Handles direct, tool-less chat with the configured LLM.
    This is used by the 'Chat' modal in the UI.
    """
    if not STATE.get('llm'):
        return jsonify({"error": "LLM not configured."}), 400

    data = await request.get_json()
    message = data.get("message")
    history = data.get("history", [])
    
    if not message:
        return jsonify({"error": "No message provided."}), 400

    try:
        response_text, _, _ = await llm_handler.call_llm_api(
            llm_instance=STATE.get('llm'),
            prompt=message,
            chat_history=history,
            system_prompt_override="You are a helpful assistant.",
            dependencies={'STATE': STATE},
            reason="Simple, tool-less chat."
        )
        
        final_response = response_text.replace("FINAL_ANSWER:", "").strip()

        return jsonify({"response": final_response})

    except Exception as e:
        app_logger.error(f"Error in simple_chat: {e}", exc_info=True)
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@api_bp.route("/app-config")
async def get_app_config():
    """Returns the startup configuration flags."""
    return jsonify({
        "all_models_unlocked": APP_CONFIG.ALL_MODELS_UNLOCKED,
        "charting_enabled": APP_CONFIG.CHARTING_ENABLED
    })

@api_bp.route("/api_key/<provider>")
async def get_api_key(provider):
    """Retrieves API keys from environment variables for pre-population."""
    key = None
    provider_lower = provider.lower()
    
    if provider_lower == 'google':
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        return jsonify({"apiKey": key or ""})
    elif provider_lower == 'anthropic':
        key = os.environ.get("ANTHROPIC_API_KEY")
        return jsonify({"apiKey": key or ""})
    elif provider_lower == 'openai':
        key = os.environ.get("OPENAI_API_KEY")
        return jsonify({"apiKey": key or ""})
    elif provider_lower == 'amazon':
        keys = {
            "aws_access_key_id": os.environ.get("AWS_ACCESS_KEY_ID"),
            "aws_secret_access_key": os.environ.get("AWS_SECRET_ACCESS_KEY"),
            "aws_region": os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        }
        return jsonify(keys)
    elif provider_lower == 'ollama':
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        return jsonify({"host": host})
        
    return jsonify({"error": "Unknown provider"}), 404

@api_bp.route("/tools")
async def get_tools():
    """Returns the categorized list of MCP tools."""
    if not STATE.get("mcp_client"): return jsonify({"error": "Not configured"}), 400
    return jsonify(STATE.get("structured_tools", {}))

@api_bp.route("/prompts")
async def get_prompts():
    """
    Returns the categorized list of MCP prompts with metadata only.
    The UI will fetch the full content on demand.
    """
    if not STATE.get("mcp_client"):
        return jsonify({"error": "Not configured"}), 400
    return jsonify(STATE.get("structured_prompts", {}))

@api_bp.route("/tool/toggle_status", methods=["POST"])
async def toggle_tool_status():
    """
    Enables or disables a tool by adding/removing it from the runtime
    'disabled_tools' list and immediately regenerates the agent's context.
    """
    data = await request.get_json()
    tool_name = data.get("name")
    is_disabled = data.get("disabled")

    if not tool_name or is_disabled is None:
        return jsonify({"status": "error", "message": "Missing 'name' or 'disabled' field."}), 400

    disabled_tools_set = set(STATE.get("disabled_tools", []))

    if is_disabled:
        disabled_tools_set.add(tool_name)
        app_logger.info(f"Disabling tool '{tool_name}' for agent use.")
    else:
        disabled_tools_set.discard(tool_name)
        app_logger.info(f"Enabling tool '{tool_name}' for agent use.")
    
    STATE["disabled_tools"] = list(disabled_tools_set)
    
    _regenerate_contexts()

    return jsonify({"status": "success", "message": f"Tool '{tool_name}' status updated."})

@api_bp.route("/prompt/toggle_status", methods=["POST"])
async def toggle_prompt_status():
    """
    Enables or disables a prompt by adding/removing it from the runtime
    'disabled_prompts' list and immediately regenerates the agent's context.
    """
    data = await request.get_json()
    prompt_name = data.get("name")
    is_disabled = data.get("disabled")

    if not prompt_name or is_disabled is None:
        return jsonify({"status": "error", "message": "Missing 'name' or 'disabled' field."}), 400

    disabled_prompts_set = set(STATE.get("disabled_prompts", []))

    if is_disabled:
        disabled_prompts_set.add(prompt_name)
        app_logger.info(f"Disabling prompt '{prompt_name}' for agent use.")
    else:
        disabled_prompts_set.discard(prompt_name)
        app_logger.info(f"Enabling prompt '{prompt_name}' for agent use.")
    
    STATE["disabled_prompts"] = list(disabled_prompts_set)
    
    _regenerate_contexts()

    return jsonify({"status": "success", "message": f"Prompt '{prompt_name}' status updated."})

@api_bp.route("/prompt/<prompt_name>", methods=["GET"])
async def get_prompt_content(prompt_name):
    """
    Retrieves the raw content of a specific MCP prompt, applying shims
    and handling different prompt object structures.
    """
    mcp_client = STATE.get("mcp_client")
    if not mcp_client:
        return jsonify({"error": "MCP client not configured."}), 400
    
    try:
        # 1. Fetch the FULL prompt object from the server.
        async with mcp_client.session("teradata_mcp_server") as temp_session:
            prompt_obj = await temp_session.get_prompt(name=prompt_name)
        
        if not prompt_obj:
            return jsonify({"error": f"Prompt '{prompt_name}' not found."}), 404
        
        # 2. Robustly extract the text content.
        prompt_text = "Prompt content is not available."
        if (hasattr(prompt_obj, 'messages') and 
            isinstance(prompt_obj.messages, list) and 
            len(prompt_obj.messages) > 0 and
            hasattr(prompt_obj.messages[0], 'content') and
            hasattr(prompt_obj.messages[0].content, 'text')):
            prompt_text = prompt_obj.messages[0].content.text
        elif hasattr(prompt_obj, 'text') and isinstance(prompt_obj.text, str):
            prompt_text = prompt_obj.text

        return jsonify({"name": prompt_name, "content": prompt_text})
    
    except Exception as e:
        # Unwrap the exception to find the root cause, especially for ExceptionGroups
        root_exception = unwrap_exception(e)
        
        # Check if the root cause is the specific McpError we want to handle
        if isinstance(root_exception, McpError) and "Missing required arguments" in str(root_exception):
            app_logger.warning(f"Handled dynamic prompt error for '{prompt_name}': {root_exception}")
            return jsonify({
                "error": "dynamic_prompt_error",
                "message": "This is a dynamic prompt. Its content is generated at runtime and cannot be previewed."
            }), 400
        
        # Handle all other errors generically
        app_logger.error(f"Error fetching prompt content for '{prompt_name}': {e}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred while fetching the prompt."}), 500


@api_bp.route("/resources")
async def get_resources_route():
    """Returns the categorized list of MCP resources."""
    if not STATE.get("mcp_client"): return jsonify({"error": "Not configured"}), 400
    return jsonify(STATE.get("structured_resources", {}))

@api_bp.route("/charts")
async def get_charts():
    """Returns the categorized list of chart tools."""
    return jsonify({})

@api_bp.route("/sessions", methods=["GET"])
async def get_sessions():
    """Returns a list of all active chat sessions."""
    return jsonify(session_manager.get_all_sessions())

@api_bp.route("/session/<session_id>", methods=["GET"])
async def get_session_history(session_id):
    """Retrieves the chat history and token counts for a specific session."""
    session_data = session_manager.get_session(session_id)
    if session_data:
        response_data = {
            "history": session_data.get("generic_history", []),
            "input_tokens": session_data.get("input_tokens", 0),
            "output_tokens": session_data.get("output_tokens", 0)
        }
        return jsonify(response_data)
    return jsonify({"error": "Session not found"}), 404

@api_bp.route("/session", methods=["POST"])
async def new_session():
    """Creates a new chat session."""
    if not STATE.get('llm') or not APP_CONFIG.TERADATA_MCP_CONNECTED:
        return jsonify({"error": "Application not configured. Please set MCP and LLM details in Config."}), 400
    
    data = await request.get_json()
    system_prompt_template = data.get("system_prompt")
    charting_intensity = data.get("charting_intensity", "medium") if APP_CONFIG.CHARTING_ENABLED else "none"

    try:
        session_id = session_manager.create_session(
            system_prompt_template=system_prompt_template,
            charting_intensity=charting_intensity,
            provider=APP_CONFIG.CURRENT_PROVIDER,
            llm_instance=STATE.get('llm')
        )
        app_logger.info(f"Created new session: {session_id} for provider {APP_CONFIG.CURRENT_PROVIDER}.")
        return jsonify({"session_id": session_id, "name": "New Chat"})
    except Exception as e:
        app_logger.error(f"Failed to create new session: {e}", exc_info=True)
        return jsonify({"error": f"Failed to initialize a new chat session: {e}"}), 500

@api_bp.route("/models", methods=["POST"])
async def get_models():
    """Fetches the list of available models from the selected provider."""
    try:
        data = await request.get_json()
        provider = data.get("provider")
        credentials = { "listing_method": data.get("listing_method", "foundation_models") }
        if provider == 'Amazon':
            credentials.update({
                "aws_access_key_id": data.get("aws_access_key_id"),
                "aws_secret_access_key": data.get("aws_secret_access_key"),
                "aws_region": data.get("aws_region")
            })
        elif provider == 'Ollama':
            credentials["host"] = data.get("host")
        else:
            credentials["apiKey"] = data.get("apiKey")

        models = await llm_handler.list_models(provider, credentials)
        return jsonify({"status": "success", "models": models})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@api_bp.route("/system_prompt/<provider>/<model_name>", methods=["GET"])
async def get_default_system_prompt(provider, model_name):
    """Gets the default system prompt for a given model."""
    base_prompt_template = PROVIDER_SYSTEM_PROMPTS.get(provider, PROVIDER_SYSTEM_PROMPTS["Google"])
    return jsonify({"status": "success", "system_prompt": base_prompt_template})

@api_bp.route("/configure", methods=["POST"])
async def configure_services():
    """
    Configures and validates the core LLM and MCP services.
    """
    data = await request.get_json()
    provider = data.get("provider")
    model = data.get("model")
    
    temp_llm_instance = None
    temp_mcp_client = None
    
    try:
        app_logger.info(f"Validating credentials for provider: {provider}")
        if provider == "Google":
            genai.configure(api_key=data.get("apiKey"))
            temp_llm_instance = genai.GenerativeModel(model)
            await temp_llm_instance.generate_content_async("test", generation_config={"max_output_tokens": 1})
        elif provider == "Anthropic":
            temp_llm_instance = AsyncAnthropic(api_key=data.get("apiKey"))
            await temp_llm_instance.models.list()
        elif provider == "OpenAI":
            temp_llm_instance = AsyncOpenAI(api_key=data.get("apiKey"))
            await temp_llm_instance.models.list()
        elif provider == "Amazon":
            aws_region = data.get("aws_region")
            temp_llm_instance = boto3.client(
                service_name='bedrock-runtime',
                aws_access_key_id=data.get("aws_access_key_id"),
                aws_secret_access_key=data.get("aws_secret_access_key"),
                region_name=aws_region
            )
            app_logger.info("Boto3 client for Bedrock created. Skipping pre-flight model invocation.")
        elif provider == "Ollama":
            host = data.get("ollama_host")
            if not host:
                raise ValueError("Ollama host is required.")
            temp_llm_instance = llm_handler.OllamaClient(host=host)
            await temp_llm_instance.list_models()
        else:
            raise NotImplementedError(f"Provider '{provider}' is not yet supported.")
        app_logger.info("LLM credentials/connection validated successfully.")

        mcp_server_url = f"http://{data.get('host')}:{data.get('port')}{data.get('path')}"
        temp_server_configs = {'teradata_mcp_server': {"url": mcp_server_url, "transport": "streamable_http"}}
        temp_mcp_client = MultiServerMCPClient(temp_server_configs)
        async with temp_mcp_client.session("teradata_mcp_server") as temp_session:
            await temp_session.list_tools()
        app_logger.info("MCP server connection validated successfully.")

        app_logger.info("All validations passed. Committing configuration to application state.")
        
        APP_CONFIG.CURRENT_PROVIDER = provider
        APP_CONFIG.CURRENT_MODEL = model
        APP_CONFIG.CURRENT_AWS_REGION = data.get("aws_region") if provider == "Amazon" else None
        APP_CONFIG.CURRENT_MODEL_PROVIDER_IN_PROFILE = None
        
        STATE['llm'] = temp_llm_instance
        STATE['mcp_client'] = temp_mcp_client
        STATE['server_configs'] = temp_server_configs

        if provider == "Amazon" and model.startswith("arn:aws:bedrock:"):
            profile_part = model.split('/')[-1]
            APP_CONFIG.CURRENT_MODEL_PROVIDER_IN_PROFILE = profile_part.split('.')[1]
        
        await mcp_adapter.load_and_categorize_teradata_resources(STATE)
        APP_CONFIG.TERADATA_MCP_CONNECTED = True
        
        APP_CONFIG.CHART_MCP_CONNECTED = True

        _regenerate_contexts()

        return jsonify({"status": "success", "message": "Teradata MCP and LLM configured successfully."})

    except (APIError, OpenAI_APIError, google_exceptions.PermissionDenied, ClientError, RuntimeError, Exception) as e:
        app_logger.error(f"Configuration failed during validation: {e}", exc_info=True)
        STATE['llm'] = None
        STATE['mcp_client'] = None
        APP_CONFIG.TERADATA_MCP_CONNECTED = False
        APP_CONFIG.CHART_MCP_CONNECTED = False
        
        root_exception = unwrap_exception(e)
        error_message = getattr(root_exception, 'message', str(root_exception))
        
        if isinstance(root_exception, (google_exceptions.PermissionDenied, ClientError)):
             if 'AccessDeniedException' in str(e):
                 error_message = "Access denied. Please check your AWS IAM permissions for the selected model."
             else:
                error_message = "Authentication failed. Please check your API keys or credentials."
        elif isinstance(root_exception, (APIError, OpenAI_APIError)) and "authentication_error" in str(e).lower():
             error_message = f"Authentication failed. Please check your {provider} API key."

        return jsonify({"status": "error", "message": f"Configuration failed: {error_message}"}), 500

@api_bp.route("/ask_stream", methods=["POST"])
async def ask_stream():
    """Handles the main chat conversation stream."""
    data = await request.get_json()
    user_input = data.get("message")
    session_id = data.get("session_id")
    
    async def stream_generator():
        if not all([user_input, session_id]) or not session_manager.get_session(session_id):
            yield _format_sse({"error": "Missing 'message' or invalid 'session_id'"}, "error")
            return

        try:
            session_manager.add_to_history(session_id, 'user', user_input)
            
            session_data = session_manager.get_session(session_id)
            if session_data['name'] == 'New Chat':
                new_name = user_input[:40] + '...' if len(user_input) > 40 else user_input
                session_manager.update_session_name(session_id, new_name)
                yield _format_sse({"session_name_update": {"id": session_id, "name": new_name}}, "session_update")

            if user_input.lower().strip() in ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]:
                greeting_response = "Hello! How can I assist you with your Teradata database queries or analysis today?"
                yield _format_sse({"final_answer": greeting_response}, "final_answer")
                session_manager.add_to_history(session_id, 'assistant', greeting_response)
                return

            yield _format_sse({"step": "Assistant is thinking...", "details": "Analyzing request and selecting best action."})
            
            yield _format_sse({"step": "Calling LLM", "details": "Analyzing user query to determine the first action."})

            llm_reasoning_and_command, statement_input_tokens, statement_output_tokens = await llm_handler.call_llm_api(
                STATE['llm'], user_input, session_id, dependencies={'STATE': STATE},
                reason="Analyzing user query to determine the first action."
            )
            
            updated_session = session_manager.get_session(session_id)
            if updated_session:
                token_data = {
                    "statement_input": statement_input_tokens,
                    "statement_output": statement_output_tokens,
                    "total_input": updated_session.get("input_tokens", 0),
                    "total_output": updated_session.get("output_tokens", 0)
                }
                yield _format_sse(token_data, "token_update")
            
            if STATE['llm'] and APP_CONFIG.CURRENT_PROVIDER in ["Anthropic", "Amazon", "Ollama", "OpenAI"]:
                session_data['chat_object'].append({'role': 'user', 'content': user_input})
                session_data['chat_object'].append({'role': 'assistant', 'content': llm_reasoning_and_command})

            executor = PlanExecutor(session_id=session_id, initial_instruction=llm_reasoning_and_command, original_user_input=user_input, dependencies={'STATE': STATE})
            async for event in executor.run():
                yield event

        except Exception as e:
            app_logger.error(f"An unhandled error occurred in /ask_stream: {e}", exc_info=True)
            yield _format_sse({"error": "An unexpected server error occurred.", "details": str(e)}, "error")

    return Response(stream_generator(), mimetype="text/event-stream")

@api_bp.route("/invoke_prompt_stream", methods=["POST"])
async def invoke_prompt_stream():
    """Handles the direct invocation of a prompt from the UI."""
    data = await request.get_json()
    session_id = data.get("session_id")
    prompt_name = data.get("prompt_name")
    arguments = data.get("arguments", {})
    
    async def stream_generator():
        user_input = f"Manual execution of prompt: {prompt_name}"
        session_manager.add_to_history(session_id, 'user', user_input)
        
        session_data = session_manager.get_session(session_id)
        if session_data['name'] == 'New Chat':
            new_name = user_input[:40] + '...' if len(user_input) > 40 else user_input
            session_manager.update_session_name(session_id, new_name)
            yield _format_sse({"session_name_update": {"id": session_id, "name": new_name}}, "session_update")

        initial_instruction = f"""
        Thought: The user has manually selected the prompt `{prompt_name}`. I will execute it directly.
        ```json
        {{
            "prompt_name": "{prompt_name}",
            "arguments": {json.dumps(arguments)}
        }}
        ```
        """
        try:
            if APP_CONFIG.CURRENT_PROVIDER in ["Anthropic", "Amazon", "Ollama", "OpenAI"]:
                session_data['chat_object'].append({'role': 'user', 'content': user_input})
            
            executor = PlanExecutor(session_id=session_id, initial_instruction=initial_instruction, original_user_input=user_input, dependencies={'STATE': STATE})
            async for event in executor.run():
                yield event
        except Exception as e:
            app_logger.error(f"An unhandled error occurred in /invoke_prompt_stream: {e}", exc_info=True)
            yield _format_sse({"error": "An unexpected server error occurred during prompt invocation.", "details": str(e)}, "error")

    return Response(stream_generator(), mimetype="text/event-stream")