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

APP_CONFIG = AppConfig()

CERTIFIED_GOOGLE_MODELS = ["gemini-1.5-flash-latest"]
CERTIFIED_ANTHROPIC_MODELS = ["claude-3-sonnet-20240229"]
CERTIFIED_AMAZON_MODELS = ["amazon.titan-text-express-v1"]
CERTIFIED_AMAZON_PROFILES = ["arn:aws:bedrock:us-east-1::inference-profile/amazon.titan-text-express-v1"]
CERTIFIED_OLLAMA_MODELS = ["llama2"] 
# --- NEW: Add a list for certified OpenAI models ---
CERTIFIED_OPENAI_MODELS = ["gpt-4.1"]