# src/trusted_data_agent/core/config.py
from dotenv import load_dotenv

load_dotenv()

class AppConfig:
    ALL_MODELS_UNLOCKED = False
    CHARTING_ENABLED = True
    # --- MODIFICATION START ---
    DEFAULT_CHARTING_INTENSITY = "medium" # Options: "none", "medium", "heavy"
    # --- MODIFICATION END ---
    TERADATA_MCP_CONNECTED = False
    CHART_MCP_CONNECTED = False
    CURRENT_PROVIDER = None
    CURRENT_MODEL = None
    CURRENT_AWS_REGION = None
    CURRENT_MODEL_PROVIDER_IN_PROFILE = None
    LLM_API_MAX_RETRIES = 5
    LLM_API_BASE_DELAY = 2 # The base delay in seconds for exponential backoff
    # When True, allows the agent to answer questions by synthesizing from conversation history
    # without re-running tools. When False, it will force a re-plan if this scenario is detected.
    ALLOW_SYNTHESIS_FROM_HISTORY = False
    
    # Configuration for context distillation
    CONTEXT_DISTILLATION_MAX_ROWS = 500
    CONTEXT_DISTILLATION_MAX_CHARS = 10000

    INITIALLY_DISABLED_PROMPTS = ["cust_promptExample","qlty_databaseQuality","dba_tableArchive","dba_databaseLineage", "dba_tableDropImpact", "dba_databaseHealthAssessment", "dba_userActivityAnalysis", "dba_systemVoice", "base_databaseBusinessDesc", "sales_prompt", "test_evsTools", "test_secTools", "test_dbaTools", "test_ragTools", "test_qltyTools", "test_fsTools", "test_baseTools", "rag_guidelines" ]
    INITIALLY_DISABLED_TOOLS = []

APP_CONFIG = AppConfig()

CERTIFIED_GOOGLE_MODELS = ["*gemini-2.0-flash"]
CERTIFIED_ANTHROPIC_MODELS = ["*claude-3-7-sonnet*"]
CERTIFIED_AMAZON_MODELS = ["*amazon.nova-pro-v1*"]
CERTIFIED_AMAZON_PROFILES = ["*amazon.nova-pro-v1*"]
CERTIFIED_OLLAMA_MODELS = ["llama2"] 
CERTIFIED_OPENAI_MODELS = ["*gpt-4.1-mini-2025*"]