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
}

def create_app():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    template_folder = os.path.join(project_root, 'templates')
    
    app = Quart(__name__, template_folder=template_folder)
    app = cors(app, allow_origin="*")

    set_dependencies(APP_STATE)
    app.register_blueprint(api_bp)
    return app

app = create_app()

async def main():
    LOG_DIR = "logs"
    if os.path.exists(LOG_DIR): shutil.rmtree(LOG_DIR)
    os.makedirs(LOG_DIR)
    
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    app.logger.addHandler(handler)
    app.logger.setLevel(logging.INFO)

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
    config.accesslog = "-"
    config.errorlog = "-"
    await hypercorn.asyncio.serve(app, config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Trusted Data Agent web client.")
    parser.add_argument("--all-models", action="store_true", help="Allow selection of all available models.")
    parser.add_argument("--charting", action="store_true", help="Enable the charting engine.")
    args = parser.parse_args()

    if args.all_models:
        APP_CONFIG.ALL_MODELS_UNLOCKED = True
        print("\n--- DEV MODE: All models will be selectable. ---")
    if args.charting:
        APP_CONFIG.CHARTING_ENABLED = True
        print("\n--- CHARTING ENABLED: Charting configuration is active. ---")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer shut down.")
