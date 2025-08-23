# src/trusted_data_agent/main.py
import asyncio
import os
import sys
import logging
import shutil
import argparse

from quart import Quart
from quart_cors import cors
import hypercorn.asyncio
from hypercorn.config import Config

os.environ["LANGCHAIN_TRACING_V2"] = "false"

from trusted_data_agent.core.config import APP_CONFIG
from trusted_data_agent.api.routes import api_bp, set_dependencies

APP_STATE = {
    "llm": None, "mcp_client": None, "server_configs": {},
    "mcp_tools": {}, "mcp_prompts": {}, "mcp_charts": {},
    "structured_tools": {}, "structured_prompts": {}, "structured_resources": {}, "structured_charts": {},
    "tool_scopes": {},
    "tools_context": "--- No Tools Available ---", "prompts_context": "--- No Prompts Available ---", "charts_context": "--- No Charts Available ---",
    "disabled_prompts": list(APP_CONFIG.INITIALLY_DISABLED_PROMPTS),
    "disabled_tools": list(APP_CONFIG.INITIALLY_DISABLED_TOOLS)
}

# --- Custom log filter to suppress benign SSE connection warnings ---
class SseConnectionFilter(logging.Filter):
    """
    Filters out benign validation warnings from the MCP client related to
    the initial SSE connection handshake message from the chart server.
    """
    def filter(self, record):
        # The specific message to filter out
        is_validation_error = "Failed to validate notification" in record.getMessage()
        is_sse_connection_method = "sse/connection" in record.getMessage()
        # Return False to suppress the log record if both conditions are met
        return not (is_validation_error and is_sse_connection_method)

def create_app():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    template_folder = os.path.join(project_root, 'templates')
    static_folder = os.path.join(project_root, 'static') # Add this line
    
    app = Quart(__name__, template_folder=template_folder, static_folder=static_folder) # Add static_folder here
    app = cors(app, allow_origin="*")

    set_dependencies(APP_STATE)
    app.register_blueprint(api_bp)

    # --- MODIFICATION START: Add Content Security Policy headers ---
    @app.after_request
    async def add_security_headers(response):
        """
        Adds Content Security Policy headers to every response to prevent
        common cross-site scripting attacks and to allow necessary
        external resources like Google APIs and CDNs.
        """
        csp_policy = [
            "default-src 'self'",
            "script-src 'self' https://cdn.tailwindcss.com https://unpkg.com",
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
            "font-src 'self' https://fonts.gstatic.com",
            "connect-src 'self' *.googleapis.com https://*.withgoogle.com",
            "worker-src 'self' blob:",
            "img-src 'self' data:"
        ]
        response.headers['Content-Security-Policy'] = "; ".join(csp_policy)
        return response
    # --- MODIFICATION END ---
    
    return app

app = create_app()

async def main():
    LOG_DIR = "logs"
    if os.path.exists(LOG_DIR): shutil.rmtree(LOG_DIR)
    os.makedirs(LOG_DIR)
    
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    # Apply the custom filter to the handler
    handler.addFilter(SseConnectionFilter())
    
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

    # The app's logger will propagate to the root, so we just set its level.
    app.logger.setLevel(logging.INFO)

    # --- MODIFICATION START: Silence noisy third-party libraries ---
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("mcp.client.streamable_http").setLevel(logging.WARNING)
    # --- MODIFICATION END ---
    
    # Prevent Hypercorn's loggers from propagating to the root logger
    logging.getLogger("hypercorn.access").propagate = False
    logging.getLogger("hypercorn.error").propagate = False

    # Configure the separate logger for LLM conversations
    llm_log_handler = logging.FileHandler(os.path.join(LOG_DIR, "llm_conversations.log"))
    llm_log_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    llm_logger = logging.getLogger("llm_conversation")
    llm_logger.setLevel(logging.INFO)
    llm_logger.addHandler(llm_log_handler)
    llm_logger.propagate = False
    
    print("\n--- Starting Hypercorn Server for Quart App ---")
    print("Web client initialized and ready. Navigate to http://127.0.0.1:5000")
    config = Config()
    config.bind = ["127.0.0.1:5000"]
    config.accesslog = None
    config.errorlog = None 
    await hypercorn.asyncio.serve(app, config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Trusted Data Agent web client.")
    parser.add_argument("--all-models", action="store_true", help="Allow selection of all available models.")
    args = parser.parse_args()

    if args.all_models:
        APP_CONFIG.ALL_MODELS_UNLOCKED = True
        print("\n--- DEV MODE: All models will be selectable. ---")
    
    # Charting is now enabled by default in config.py
    print("\n--- CHARTING ENABLED: Charting configuration is active. ---")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer shut down.")