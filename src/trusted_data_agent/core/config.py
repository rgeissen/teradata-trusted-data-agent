# trusted_data_agent/core/config.py
from dotenv import load_dotenv

# Load environment variables from .env file at the earliest point
load_dotenv()

class AppConfig:
    """Holds the dynamic configuration state of the application."""
    ALL_MODELS_UNLOCKED = False
    CHARTING_ENABLED = False
    TERADATA_MCP_CONNECTED = False
    CHART_MCP_CONNECTED = False
    CURRENT_PROVIDER = None
    CURRENT_MODEL = None
    CURRENT_AWS_REGION = None
    CURRENT_MODEL_PROVIDER_IN_PROFILE = None

# Global instance of the app configuration
APP_CONFIG = AppConfig()

# Certified model lists
CERTIFIED_GOOGLE_MODELS = [
    "gemini-1.5-flash-latest"
]
CERTIFIED_ANTHROPIC_MODELS = [
    "claude-3-5-sonnet-20240620"
]
CERTIFIED_AMAZON_MODELS = [
    "amazon.nova-pro-v1:0",
]
CERTIFIED_AMAZON_PROFILES = ["arn:aws:bedrock:eu-central-1:960887920495:inference-profile/eu.amazon.nova-pro-v1:0"]