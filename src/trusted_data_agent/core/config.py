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
    #INITIALLY_DISABLED_PROMPTS = ["base_databaseBusinessDesc"]
    INITIALLY_DISABLED_PROMPTS = ["cust_promptExample","qlty_databaseQuality","dba_tableArchive","dba_databaseLineage", "dba_tableDropImpact", "dba_databaseHealthAssessment", "dba_userActivityAnalysis", "dba_systemVoice", "base_databaseBusinessDesc", "sales_prompt", "test_evsTools", "test_secTools", "test_dbaTools", "test_ragTools", "test_qltyTools", "test_fsTools", "test_baseTools", "rag_guidelines" ]
    # --- MODIFICATION START: Disable most tools by default to simplify the context for local models ---
    INITIALLY_DISABLED_TOOLS = [
        "sales_customer_profile", "qlty_distinctCategories", "get_cube_sales_cube",
        "qlty_missingValues", "qlty_negativeValues", "qlty_columnSummary",
        "qlty_rowsWithMissingValues", "dba_tableSpace", "sales_top_customers",
        "qlty_standardDeviation", "qlty_univariateStatistics", "rag_executeWorkflow",
        "rag_executeWorkflow_ivsm", "sec_rolePermissions", "sec_userDbPermissions",
        "sec_userRoles", "evs_similarity_search", "base_tableAffinity",
        "base_tablePreview", "base_tableUsage", "tmpl_nameOfTool", "dba_databaseSpace",
        "dba_resusageSummary", "dba_tableSqlList", "dba_tableUsageImpact",
        "dba_userSqlList", "cust_activeUsers", "cust_td_serverInfo",
        "get_cube_cust_cube_db_space_metrics", "dba_flowControl", "dba_featureUsage",
        "dba_userDelay", "dba_sessionInfo", "dba_systemSpace", "viz_createChart",
        "util_getCurrentDate", "CoreLLMTask"
    ]
    # --- MODIFICATION END ---

APP_CONFIG = AppConfig()

CERTIFIED_GOOGLE_MODELS = ["*gemini-2.0-flash"]
CERTIFIED_ANTHROPIC_MODELS = ["*claude-3-7-sonnet*"]
CERTIFIED_AMAZON_MODELS = ["*amazon.nova-pro-v1*"]
CERTIFIED_AMAZON_PROFILES = ["*amazon.nova-pro-v1*"]
CERTIFIED_OLLAMA_MODELS = ["llama2"] 
CERTIFIED_OPENAI_MODELS = ["*gpt-4.1-mini-2025*"]
