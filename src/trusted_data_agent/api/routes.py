# src/trusted_data_agent/api/routes.py
import json
import os
import logging

from quart import Blueprint, request, jsonify, render_template, Response
from google.api_core import exceptions as google_exceptions
from anthropic import APIError, AsyncAnthropic
from botocore.exceptions import ClientError
import google.generativeai as genai
import boto3
from langchain_mcp_adapters.client import MultiServerMCPClient

from trusted_data_agent.core.config import APP_CONFIG
from trusted_data_agent.core import session_manager
from trusted_data_agent.agent.prompts import PROVIDER_SYSTEM_PROMPTS, CHARTING_INSTRUCTIONS
from trusted_data_agent.agent.executor import PlanExecutor, _format_sse
from trusted_data_agent.llm import handler as llm_handler
from trusted_data_agent.mcp import adapter as mcp_adapter

api_bp = Blueprint('api', __name__)
app_logger = logging.getLogger("quart.app")

def unwrap_exception(e: BaseException) -> BaseException:
    """Recursively unwraps ExceptionGroups to find the root cause."""
    if isinstance(e, ExceptionGroup) and e.exceptions:
        return unwrap_exception(e.exceptions[0])
    return e

def set_dependencies(app_state):
    """Injects the global application state into this blueprint."""
    global STATE
    STATE = app_state

@api_bp.route("/")
async def index():
    """Serves the main HTML page."""
    return await render_template("index.html")

# --- NEW: Simple Chat Endpoint ---
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
        system_prompt = "You are a helpful assistant."
        
        response_text, _, _ = await llm_handler.call_llm_api(
            llm_instance=STATE.get('llm'),
            prompt=message,
            chat_history=history,
            system_prompt_override=system_prompt
        )
        
        final_response = response_text.replace("FINAL_ANSWER:", "").strip()

        return jsonify({"response": final_response})

    except Exception as e:
        app_logger.error(f"Error in simple_chat: {e}", exc_info=True)
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500
# --- END NEW ---

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
    if provider.lower() == 'google':
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        return jsonify({"apiKey": key or ""})
    elif provider.lower() == 'anthropic':
        key = os.environ.get("ANTHROPIC_API_KEY")
        return jsonify({"apiKey": key or ""})
    elif provider.lower() == 'amazon':
        keys = {
            "aws_access_key_id": os.environ.get("AWS_ACCESS_KEY_ID"),
            "aws_secret_access_key": os.environ.get("AWS_SECRET_ACCESS_KEY"),
            "aws_region": os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        }
        return jsonify(keys)
    elif provider.lower() == 'ollama':
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
    """Returns the categorized list of MCP prompts."""
    if not STATE.get("mcp_client"): return jsonify({"error": "Not configured"}), 400
    return jsonify(STATE.get("structured_prompts", {}))

@api_bp.route("/resources")
async def get_resources_route():
    """Returns the categorized list of MCP resources."""
    if not STATE.get("mcp_client"): return jsonify({"error": "Not configured"}), 400
    return jsonify(STATE.get("structured_resources", {}))

@api_bp.route("/charts")
async def get_charts():
    """Returns the categorized list of chart tools."""
    if not APP_CONFIG.CHART_MCP_CONNECTED: return jsonify({"error": "Chart server not connected"}), 400
    return jsonify(STATE.get("structured_charts", {}))

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

def get_full_system_prompt(base_prompt_text, charting_intensity_val):
    """Constructs the final system prompt by injecting context."""
    chart_instructions = CHARTING_INSTRUCTIONS.get(charting_intensity_val, CHARTING_INSTRUCTIONS['none'])
    final_charts_context = STATE.get('charts_context') if APP_CONFIG.CHART_MCP_CONNECTED else CHARTING_INSTRUCTIONS['none']
    
    final_system_prompt = base_prompt_text
    final_system_prompt = final_system_prompt.replace("{charting_instructions}", chart_instructions)
    final_system_prompt = final_system_prompt.replace("{tools_context}", STATE.get('tools_context'))
    final_system_prompt = final_system_prompt.replace("{prompts_context}", STATE.get('prompts_context'))
    final_system_prompt = final_system_prompt.replace("{charts_context}", final_charts_context)
    return final_system_prompt

@api_bp.route("/session", methods=["POST"])
async def new_session():
    """Creates a new chat session."""
    try:
        log_file_path = os.path.join("logs", "llm_conversations.log")
        if os.path.exists(log_file_path):
            with open(log_file_path, 'w'):
                pass
            app_logger.info(f"Cleared LLM conversation log: {log_file_path}")
    except Exception as e:
        app_logger.error(f"Could not clear log file: {e}")

    if not STATE.get('llm') or not APP_CONFIG.TERADATA_MCP_CONNECTED:
        return jsonify({"error": "Application not configured. Please set MCP and LLM details in Config."}), 400
    
    data = await request.get_json()
    system_prompt_from_client = data.get("system_prompt")
    
    try:
        charting_intensity = "medium" if APP_CONFIG.CHARTING_ENABLED else "none"
        final_system_prompt = get_full_system_prompt(system_prompt_from_client, charting_intensity)
        
        chat_object = None
        if APP_CONFIG.CURRENT_PROVIDER == "Google":
            initial_history = [
                {"role": "user", "parts": [{"text": final_system_prompt}]},
                {"role": "model", "parts": [{"text": "Understood. I will follow all instructions."}]}
            ]
            chat_object = STATE.get('llm').start_chat(history=initial_history)
        elif APP_CONFIG.CURRENT_PROVIDER in ["Anthropic", "Amazon", "Ollama"]:
             chat_object = []

        session_id = session_manager.create_session(final_system_prompt, chat_object)
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
    This function now performs a pre-flight check to validate credentials
    before committing any changes to the application's state.
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
        
        return jsonify({"status": "success", "message": "Teradata MCP and LLM configured successfully."})

    except (APIError, google_exceptions.PermissionDenied, ClientError, RuntimeError, Exception) as e:
        app_logger.error(f"Configuration failed during validation: {e}", exc_info=True)
        STATE['llm'] = None
        STATE['mcp_client'] = None
        APP_CONFIG.TERADATA_MCP_CONNECTED = False
        
        root_exception = unwrap_exception(e)
        error_message = getattr(root_exception, 'message', str(root_exception))
        
        if isinstance(root_exception, (google_exceptions.PermissionDenied, ClientError)):
             if 'AccessDeniedException' in str(e):
                 error_message = "Access denied. Please check your AWS IAM permissions for the selected model."
             else:
                error_message = "Authentication failed. Please check your API keys or credentials."
        elif isinstance(root_exception, APIError) and "authentication_error" in str(e):
             error_message = "Authentication failed. Please check your Anthropic API key."

        return jsonify({"status": "error", "message": f"Configuration failed: {error_message}"}), 500

@api_bp.route("/configure_chart", methods=["POST"])
async def configure_chart_service():
    """Configures the optional charting engine service."""
    if not APP_CONFIG.TERADATA_MCP_CONNECTED:
        return jsonify({"status": "error", "message": "Main MCP client not configured."}), 400
    
    data = await request.get_json()
    try:
        chart_server_url = f"http://{data.get('chart_host')}:{data.get('chart_port')}{data.get('chart_path')}"
        STATE['server_configs']['chart_mcp_server'] = {"url": chart_server_url, "transport": "sse"}
        
        STATE['mcp_client'] = MultiServerMCPClient(STATE['server_configs'])
        APP_CONFIG.CHART_MCP_CONNECTED = True
        
        return jsonify({"status": "success", "message": "Chart MCP server configured successfully."})
    except Exception as e:
        if 'chart_mcp_server' in STATE['server_configs']:
            del STATE['server_configs']['chart_mcp_server']
        STATE['mcp_client'] = MultiServerMCPClient(STATE['server_configs'])
        APP_CONFIG.CHART_MCP_CONNECTED = False
        app_logger.error(f"Chart configuration failed: {e}", exc_info=True)
        return jsonify({"status": "error", "message": f"Chart server connection failed: {e}"}), 500

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

            yield _format_sse({"step": "Assistant is thinking...", "details": "Analyzing request and selecting best action."})
            
            # --- MODIFICATION: Add a dedicated event before the LLM call ---
            yield _format_sse({"step": "Calling LLM"})
            # --- END MODIFICATION ---

            llm_reasoning_and_command, statement_input_tokens, statement_output_tokens = await llm_handler.call_llm_api(STATE['llm'], user_input, session_id)
            
            updated_session = session_manager.get_session(session_id)
            if updated_session:
                token_data = {
                    "statement_input": statement_input_tokens,
                    "statement_output": statement_output_tokens,
                    "total_input": updated_session.get("input_tokens", 0),
                    "total_output": updated_session.get("output_tokens", 0)
                }
                yield _format_sse(token_data, "token_update")
            
            if STATE['llm'] and APP_CONFIG.CURRENT_PROVIDER in ["Anthropic", "Amazon", "Ollama"]:
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
            if APP_CONFIG.CURRENT_PROVIDER in ["Anthropic", "Amazon", "Ollama"]:
                session_data['chat_object'].append({'role': 'user', 'content': user_input})
            
            executor = PlanExecutor(session_id=session_id, initial_instruction=initial_instruction, original_user_input=user_input, dependencies={'STATE': STATE})
            async for event in executor.run():
                yield event
        except Exception as e:
            app_logger.error(f"An unhandled error occurred in /invoke_prompt_stream: {e}", exc_info=True)
            yield _format_sse({"error": "An unexpected server error occurred during prompt invocation.", "details": str(e)}, "error")

    return Response(stream_generator(), mimetype="text/event-stream")
