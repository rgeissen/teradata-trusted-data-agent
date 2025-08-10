# src/trusted_data_agent/core/config.py
from dotenv import load_dotenv

load_dotenv()

class AppConfig:
    ALL_MODELS_UNLOCKED = False
    CHARTING_ENABLED = True
    TERADATA_MCP_CONNECTED = False
    CHART_MCP_CONNECTED = False
    CURRENT_PROVIDER = None
    CURRENT_MODEL = None
    CURRENT_AWS_REGION = None
    CURRENT_MODEL_PROVIDER_IN_PROFILE = None
    LLM_API_MAX_RETRIES = 5
    LLM_API_BASE_DELAY = 2 # The base delay in seconds for exponential backoff
    INITIALLY_DISABLED_PROMPTS = ["base_tableBusinessDesc", "base_databaseBusinessDesc"]
    # --- NEW: Add a list of tools to be disabled at startup ---
    INITIALLY_DISABLED_TOOLS = []

APP_CONFIG = AppConfig()

CERTIFIED_GOOGLE_MODELS = ["*gemini-1.5-flash-latest"]
CERTIFIED_ANTHROPIC_MODELS = ["*claude-sonnet-4*"]
CERTIFIED_AMAZON_MODELS = ["*amazon.nova-pro-v1*"]
CERTIFIED_AMAZON_PROFILES = ["*amazon.nova-pro-v1*"]
CERTIFIED_OLLAMA_MODELS = ["llama2"] 
CERTIFIED_OPENAI_MODELS = ["*gpt-4.1"]
