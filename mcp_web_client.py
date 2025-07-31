# web_client.py
import asyncio
import json
import os
import sys
import re
import uuid
import logging
import shutil
import argparse
from quart import Quart, request, jsonify, render_template, Response
from quart_cors import cors
import hypercorn.asyncio
from hypercorn.config import Config
from enum import Enum, auto
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# This environment variable MUST be set to "false" before any LangChain
# modules are imported to programmatically disable the problematic tracer.
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.resources import load_mcp_resources

# Import provider-specific libraries
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from anthropic import Anthropic, AsyncAnthropic, APIError
import boto3
from botocore.exceptions import ClientError


# --- Globals for Web App ---
app = Quart(__name__)
app = cors(app, allow_origin="*") # Enable CORS for all origins

# --- App Configuration ---
class AppConfig:
    ALL_MODELS_UNLOCKED = False
    CHARTING_ENABLED = False
    TERADATA_MCP_CONNECTED = False
    CHART_MCP_CONNECTED = False
    # New state to hold the current provider and model name
    CURRENT_PROVIDER = None
    CURRENT_MODEL = None
    CURRENT_AWS_REGION = None # Store the current AWS region
    # Store the underlying provider for Amazon Bedrock Inference Profiles
    CURRENT_MODEL_PROVIDER_IN_PROFILE = None


APP_CONFIG = AppConfig()
CERTIFIED_MODEL = "gemini-1.5-flash-latest"
CERTIFIED_ANTHROPIC_MODELS = [
    "claude-3-5-sonnet-20240620"
]
CERTIFIED_AMAZON_MODELS = [
    "amazon.nova-lite-v1:0",
]
CERTIFIED_AMAZON_PROFILES = ["amazon.titan-text-express-v1"]


# --- System Prompt Templates ---
PROVIDER_SYSTEM_PROMPTS = {
    "Google": (
        "You are a specialized assistant for interacting with a Teradata database. Your primary goal is to fulfill user requests by selecting the best tool or prompt.\n\n"
        "--- **CRITICAL RESPONSE PROTOCOL** ---\n"
        "Your primary task is to select a single capability to fulfill the user's request. You have two lists of capabilities available: `--- Available Prompts ---` and `--- Available Tools ---`.\n\n"
        "1.  **CHOOSE ONE CAPABILITY:** First, review both lists and select the single best capability (either a prompt or a tool) that can fulfill the user's request. If a prompt can solve the entire request, you MUST choose the prompt.\n\n"
        "2.  **IDENTIFY THE SOURCE:** Determine which list the chosen capability came from.\n\n"
        "3.  **GENERATE RESPONSE JSON:** Your response MUST be a single JSON object. The key you use in this JSON object depends entirely on the source list of your chosen capability:\n"
        "    -   If your chosen capability is from the `--- Available Prompts ---` list, you **MUST** use the key `\"prompt_name\"`.\n"
        "    -   If your chosen capability is from the `--- Available Tools ---` list, you **MUST** use the key `\"tool_name\"`.\n\n"
        "**This is not a suggestion. It is a strict rule. Using `tool_name` for a prompt, or `prompt_name` for a tool, will cause a critical system failure.**\n\n"
        "**Example for a Prompt:**\n"
        "```json\n"
        "{{\n"
        "  \"prompt_name\": \"base_tableBusinessDesc\",\n"
        "  \"arguments\": {{\"db_name\": \"some_db\", \"table_name\": \"some_table\"}}\n"
        "}}\n"
        "```\n\n"
        "**Example for a Tool:**\n"
        "```json\n"
        "{{\n"
        "  \"tool_name\": \"base_tableList\",\n"
        "  \"arguments\": {{\"db_name\": \"some_db\"}}\n"
        "}}\n"
        "```\n\n"
        "--- **CRITICAL RULE: CONTEXT and PARAMETER INFERENCE** ---\n"
        "You **MUST** remember and reuse information from previous turns.\n"
        "**Example of CORRECT Inference:**\n"
        "    -   USER (Turn 1): \"what is the business description for the `equipment` table in database `DEMO_Customer360_db`?\"\n"
        "    -   ASSISTANT (Turn 1): (Executes the request)\n"
        "    -   USER (Turn 2): \"ok now what is the quality of that table?\"\n"
        "    -   YOUR CORRECT REASONING (Turn 2): \"The user is asking about 'that table'. The previous turn mentioned the `equipment` table in the `DEMO_Customer360_db` database. I will reuse these parameters.\"\n"
        "    -   YOUR CORRECT ACTION (Turn 2): `json ...` for `qlty_columnSummary` with `db_name`: `DEMO_Customer360_db` and `table_name`: `equipment`.\n\n"
        "**Another Example of CORRECT Inference:**\n"
        "    -   USER (Turn 1): \"what's in DEMO_Customer360_db?\"\n"
        "    -   ASSISTANT (Turn 1): (Responds with a list of tables, including `Equipment`)\n"
        "    -   USER (Turn 2): \"what is the quality of Equipment?\"\n"
        "    -   YOUR CORRECT REASONING (Turn 2): \"The user is asking about the 'Equipment' table. The previous turns established the context of the `DEMO_Customer360_db` database. I must reuse this database name.\"\n"
        "    -   YOUR CORRECT ACTION (Turn 2): `json ...` for `qlty_columnSummary` with `db_name`: `DEMO_Customer360_db` and `table_name`: `Equipment`.\n\n"
        "--- **CRITICAL RULE: TOOL ARGUMENT ADHERENCE** ---\n"
        "You **MUST** use the exact parameter names provided in the tool definitions. Do not invent or guess parameter names.\n\n"
        "--- **CRITICAL RULE: SQL GENERATION** ---\n"
        "When using the `base_readQuery` tool, if you know the database name, you **MUST** use fully qualified table names in your SQL query (e.g., `SELECT ... FROM my_database.my_table`).\n\n"
        "--- **CRITICAL RULE: HANDLING TIME-SENSITIVE QUERIES** ---\n"
        "If the user asks a question involving a relative date (e.g., 'today', 'yesterday', 'this week'), you do not know this information. Your first step **MUST** be to find the current date before proceeding.\n\n"
        "**Example of CORRECT Multi-Step Plan:**\n"
        "    -   USER: \"what is the system utilization in number of queries for today?\"\n"
        "    -   YOUR CORRECT REASONING (Step 1): \"The user is asking about 'today'. I do not know the current date. My first step must be to get the current date from the database.\"\n"
        "    -   YOUR CORRECT ACTION (Step 1):\n"
        "        ```json\n"
        "        {{\n"
        "          \"tool_name\": \"base_readQuery\",\n"
        "          \"arguments\": {{ \"sql\": \"SELECT CURRENT_DATE\" }}\n"
        "        }}\n"
        "        ```\n"
        "    -   TOOL RESPONSE (Step 1): `{{\"results\": [{{\"Date\": \"2025-07-29\"}}]}}`\n"
        "    -   YOUR CORRECT REASONING (Step 2): \"The database returned the current date as 2025-07-29. Now I can use this date to answer the user's original question about system utilization.\"\n"
        "    -   YOUR CORRECT ACTION (Step 2):\n"
        "        ```json\n"
        "        {{\n"
        "          \"tool_name\": \"dba_resusageSummary\",\n"
        "          \"arguments\": {{ \"date\": \"2025-07-29\" }}\n"
        "        }}\n"
        "        ```\n\n"
        "--- **CRITICAL RULE: TOOL FAILURE AND RECOVERY** ---\n"
        "If a tool call fails with an error message, you **MUST** attempt to recover. Your recovery process is as follows:\n"
        "1.  **Analyze the Error:** Read the error message carefully. If it indicates an invalid column, parameter, or dimension (e.g., 'Column not found'), identify the specific argument that caused the failure.\n"
        "2.  **Consult Tool Docs:** Review the documentation for the failed tool that is provided in this system prompt.\n"
        "3.  **Formulate a New Plan:** Your next thought process should explain the error and propose a corrected tool call. Typically, this means re-issuing the tool call *without* the single failing parameter.\n"
        "4.  **Retry the Tool:** Execute the corrected tool call.\n"
        "5.  **Ask for Help:** Only if the corrected tool call also fails should you give up and ask the user for clarification.\n\n"
        "{charting_instructions}\n\n"
        "{tools_context}\n\n"
        "{prompts_context}\n\n"
        "{charts_context}\n\n"
    ),
    "Anthropic": (
        "You are a specialized assistant for interacting with a Teradata database. Your primary goal is to fulfill user requests by selecting the best tool or prompt.\n\n"
        "--- **CRITICAL RESPONSE PROTOCOL** ---\n"
        "Your primary task is to select a single capability to fulfill the user's request. You have two lists of capabilities available: `--- Available Prompts ---` and `--- Available Tools ---`.\n\n"
        "1.  **CHOOSE ONE CAPABILITY:** First, review both lists and select the single best capability (either a prompt or a tool) that can fulfill the user's request. If a prompt can solve the entire request, you MUST choose the prompt.\n\n"
        "2.  **IDENTIFY THE SOURCE:** Determine which list the chosen capability came from.\n\n"
        "3.  **GENERATE RESPONSE JSON:** Your response MUST be a single JSON object. The key you use in this JSON object depends entirely on the source list of your chosen capability:\n"
        "    -   If your chosen capability is from the `--- Available Prompts ---` list, you **MUST** use the key `\"prompt_name\"`.\n"
        "    -   If your chosen capability is from the `--- Available Tools ---` list, you **MUST** use the key `\"tool_name\"`.\n\n"
        "**This is not a suggestion. It is a strict rule. Using `tool_name` for a prompt, or `prompt_name` for a tool, will cause a critical system failure.**\n\n"
        "**Example for a Prompt:**\n"
        "```json\n"
        "{{\n"
        "  \"prompt_name\": \"base_tableBusinessDesc\",\n"
        "  \"arguments\": {{\"db_name\": \"some_db\", \"table_name\": \"some_table\"}}\n"
        "}}\n"
        "```\n\n"
        "**Example for a Tool:**\n"
        "```json\n"
        "{{\n"
        "  \"tool_name\": \"base_tableList\",\n"
        "  \"arguments\": {{\"db_name\": \"some_db\"}}\n"
        "}}\n"
        "```\n\n"
        "--- **CRITICAL RULE: CONTEXT and PARAMETER INFERENCE** ---\n"
        "You **MUST** remember and reuse information from previous turns.\n"
        "**Example of CORRECT Inference:**\n"
        "    -   USER (Turn 1): \"what is the business description for the `equipment` table in database `DEMO_Customer360_db`?\"\n"
        "    -   ASSISTANT (Turn 1): (Executes the request)\n"
        "    -   USER (Turn 2): \"ok now what is the quality of that table?\"\n"
        "    -   YOUR CORRECT REASONING (Turn 2): \"The user is asking about 'that table'. The previous turn mentioned the `equipment` table in the `DEMO_Customer360_db` database. I will reuse these parameters.\"\n"
        "    -   YOUR CORRECT ACTION (Turn 2): `json ...` for `qlty_columnSummary` with `db_name`: `DEMO_Customer360_db` and `table_name`: `equipment`.\n\n"
        "**Another Example of CORRECT Inference:**\n"
        "    -   USER (Turn 1): \"what's in DEMO_Customer360_db?\"\n"
        "    -   ASSISTANT (Turn 1): (Responds with a list of tables, including `Equipment`)\n"
        "    -   USER (Turn 2): \"what is the quality of Equipment?\"\n"
        "    -   YOUR CORRECT REASONING (Turn 2): \"The user is asking about the 'Equipment' table. The previous turns established the context of the `DEMO_Customer360_db` database. I must reuse this database name.\"\n"
        "    -   YOUR CORRECT ACTION (Turn 2): `json ...` for `qlty_columnSummary` with `db_name`: `DEMO_Customer360_db` and `table_name`: `Equipment`.\n\n"
        "--- **CRITICAL RULE: TOOL ARGUMENT ADHERENCE** ---\n"
        "You **MUST** use the exact parameter names provided in the tool definitions. Do not invent or guess parameter names.\n\n"
        "--- **CRITICAL RULE: SQL GENERATION** ---\n"
        "When using the `base_readQuery` tool, if you know the database name, you **MUST** use fully qualified table names in your SQL query (e.g., `SELECT ... FROM my_database.my_table`).\n\n"
        "--- **CRITICAL RULE: HANDLING TIME-SENSITIVE QUERIES** ---\n"
        "If the user asks a question involving a relative date (e.g., 'today', 'yesterday', 'this week'), you do not know this information. Your first step **MUST** be to find the current date before proceeding.\n\n"
        "**Example of CORRECT Multi-Step Plan:**\n"
        "    -   USER: \"what is the system utilization in number of queries for today?\"\n"
        "    -   YOUR CORRECT REASONING (Step 1): \"The user is asking about 'today'. I do not know the current date. My first step must be to get the current date from the database.\"\n"
        "    -   YOUR CORRECT ACTION (Step 1):\n"
        "        ```json\n"
        "        {{\n"
        "          \"tool_name\": \"base_readQuery\",\n"
        "          \"arguments\": {{ \"sql\": \"SELECT CURRENT_DATE\" }}\n"
        "        }}\n"
        "        ```\n"
        "    -   TOOL RESPONSE (Step 1): `{{\"results\": [{{\"Date\": \"2025-07-29\"}}]}}`\n"
        "    -   YOUR CORRECT REASONING (Step 2): \"The database returned the current date as 2025-07-29. Now I can use this date to answer the user's original question about system utilization.\"\n"
        "    -   YOUR CORRECT ACTION (Step 2):\n"
        "        ```json\n"
        "        {{\n"
        "          \"tool_name\": \"dba_resusageSummary\",\n"
        "          \"arguments\": {{ \"date\": \"2025-07-29\" }}\n"
        "        }}\n"
        "        ```\n\n"
        "--- **CRITICAL RULE: TOOL FAILURE AND RECOVERY** ---\n"
        "If a tool call fails with an error message, you **MUST** attempt to recover. Your recovery process is as follows:\n"
        "1.  **Analyze the Error:** Read the error message carefully. If it indicates an invalid column, parameter, or dimension (e.g., 'Column not found'), identify the specific argument that caused the failure.\n"
        "2.  **Consult Tool Docs:** Review the documentation for the failed tool that is provided in this system prompt.\n"
        "3.  **Formulate a New Plan:** Your next thought process should explain the error and propose a corrected tool call. Typically, this means re-issuing the tool call *without* the single failing parameter.\n"
        "4.  **Retry the Tool:** Execute the corrected tool call.\n"
        "5.  **Ask for Help:** Only if the corrected tool call also fails should you give up and ask the user for clarification.\n\n"
        "{charting_instructions}\n\n"
        "{tools_context}\n\n"
        "{prompts_context}\n\n"
        "{charts_context}\n\n"
    ),
     "Amazon": (
        "You are a specialized assistant for interacting with a Teradata database. Your primary goal is to fulfill user requests by selecting the best tool or prompt.\n\n"
        "--- **CRITICAL RESPONSE PROTOCOL** ---\n"
        "Your primary task is to select a single capability to fulfill the user's request. You have two lists of capabilities available: `--- Available Prompts ---` and `--- Available Tools ---`.\n\n"
        "1.  **CHOOSE ONE CAPABILITY:** First, review both lists and select the single best capability (either a prompt or a tool) that can fulfill the user's request. If a prompt can solve the entire request, you MUST choose the prompt.\n\n"
        "2.  **IDENTIFY THE SOURCE:** Determine which list the chosen capability came from.\n\n"
        "3.  **GENERATE RESPONSE JSON:** Your response MUST be a single JSON object. The key you use in this JSON object depends entirely on the source list of your chosen capability:\n"
        "    -   If your chosen capability is from the `--- Available Prompts ---` list, you **MUST** use the key `\"prompt_name\"`.\n"
        "    -   If your chosen capability is from the `--- Available Tools ---` list, you **MUST** use the key `\"tool_name\"`.\n\n"
        "**This is not a suggestion. It is a strict rule. Using `tool_name` for a prompt, or `prompt_name` for a tool, will cause a critical system failure.**\n\n"
        "**Example for a Prompt:**\n"
        "```json\n"
        "{{\n"
        "  \"prompt_name\": \"base_tableBusinessDesc\",\n"
        "  \"arguments\": {{\"db_name\": \"some_db\", \"table_name\": \"some_table\"}}\n"
        "}}\n"
        "```\n\n"
        "**Example for a Tool:**\n"
        "```json\n"
        "{{\n"
        "  \"tool_name\": \"base_tableList\",\n"
        "  \"arguments\": {{\"db_name\": \"some_db\"}}\n"
        "}}\n"
        "```\n\n"
        "--- **CRITICAL RULE: CONTEXT and PARAMETER INFERENCE** ---\n"
        "You **MUST** remember and reuse information from previous turns.\n"
        "**Example of CORRECT Inference:**\n"
        "    -   USER (Turn 1): \"what is the business description for the `equipment` table in database `DEMO_Customer360_db`?\"\n"
        "    -   ASSISTANT (Turn 1): (Executes the request)\n"
        "    -   USER (Turn 2): \"ok now what is the quality of that table?\"\n"
        "    -   YOUR CORRECT REASONING (Turn 2): \"The user is asking about 'that table'. The previous turn mentioned the `equipment` table in the `DEMO_Customer360_db` database. I will reuse these parameters.\"\n"
        "    -   YOUR CORRECT ACTION (Turn 2): `json ...` for `qlty_columnSummary` with `db_name`: `DEMO_Customer360_db` and `table_name`: `equipment`.\n\n"
        "**Another Example of CORRECT Inference:**\n"
        "    -   USER (Turn 1): \"what's in DEMO_Customer360_db?\"\n"
        "    -   ASSISTANT (Turn 1): (Responds with a list of tables, including `Equipment`)\n"
        "    -   USER (Turn 2): \"what is the quality of Equipment?\"\n"
        "    -   YOUR CORRECT REASONING (Turn 2): \"The user is asking about the 'Equipment' table. The previous turns established the context of the `DEMO_Customer360_db` database. I must reuse this database name.\"\n"
        "    -   YOUR CORRECT ACTION (Turn 2): `json ...` for `qlty_columnSummary` with `db_name`: `DEMO_Customer360_db` and `table_name`: `Equipment`.\n\n"
        "--- **CRITICAL RULE: TOOL ARGUMENT ADHERENCE** ---\n"
        "You **MUST** use the exact parameter names provided in the tool definitions. Do not invent or guess parameter names.\n\n"
        "--- **CRITICAL RULE: SQL GENERATION** ---\n"
        "When using the `base_readQuery` tool, if you know the database name, you **MUST** use fully qualified table names in your SQL query (e.g., `SELECT ... FROM my_database.my_table`).\n\n"
        "--- **CRITICAL RULE: HANDLING TIME-SENSITIVE QUERIES** ---\n"
        "If the user asks a question involving a relative date (e.g., 'today', 'yesterday', 'this week'), you do not know this information. Your first step **MUST** be to find the current date before proceeding.\n\n"
        "**Example of CORRECT Multi-Step Plan:**\n"
        "    -   USER: \"what is the system utilization in number of queries for today?\"\n"
        "    -   YOUR CORRECT REASONING (Step 1): \"The user is asking about 'today'. I do not know the current date. My first step must be to get the current date from the database.\"\n"
        "    -   YOUR CORRECT ACTION (Step 1):\n"
        "        ```json\n"
        "        {{\n"
        "          \"tool_name\": \"base_readQuery\",\n"
        "          \"arguments\": {{ \"sql\": \"SELECT CURRENT_DATE\" }}\n"
        "        }}\n"
        "        ```\n"
        "    -   TOOL RESPONSE (Step 1): `{{\"results\": [{{\"Date\": \"2025-07-29\"}}]}}`\n"
        "    -   YOUR CORRECT REASONING (Step 2): \"The database returned the current date as 2025-07-29. Now I can use this date to answer the user's original question about system utilization.\"\n"
        "    -   YOUR CORRECT ACTION (Step 2):\n"
        "        ```json\n"
        "        {{\n"
        "          \"tool_name\": \"dba_resusageSummary\",\n"
        "          \"arguments\": {{ \"date\": \"2025-07-29\" }}\n"
        "        }}\n"
        "        ```\n\n"
        "--- **CRITICAL RULE: TOOL FAILURE AND RECOVERY** ---\n"
        "If a tool call fails with an error message, you **MUST** attempt to recover. Your recovery process is as follows:\n"
        "1.  **Analyze the Error:** Read the error message carefully. If it indicates an invalid column, parameter, or dimension (e.g., 'Column not found'), identify the specific argument that caused the failure.\n"
        "2.  **Consult Tool Docs:** Review the documentation for the failed tool that is provided in this system prompt.\n"
        "3.  **Formulate a New Plan:** Your next thought process should explain the error and propose a corrected tool call. Typically, this means re-issuing the tool call *without* the single failing parameter.\n"
        "4.  **Retry the Tool:** Execute the corrected tool call.\n"
        "5.  **Ask for Help:** Only if the corrected tool call also fails should you give up and ask the user for clarification.\n\n"
        "{charting_instructions}\n\n"
        "{tools_context}\n\n"
        "{prompts_context}\n\n"
        "{charts_context}\n\n"
    ),
    "OpenAI": "Placeholder prompt for OpenAI models."
}


CHARTING_INSTRUCTIONS = {
    "none": "--- **Charting Rules** ---\n- Charting is disabled. Do NOT use any charting tools.",
    "medium": (
        "--- **Charting Rules** ---\n"
        "- After successfully gathering data with Teradata tools, consider if a visualization would enhance the answer.\n"
        "- Use a chart tool if it provides a clear summary (e.g., bar chart for space usage, pie chart for distributions).\n"
        "- Do not generate charts for simple data retrievals that are easily readable in a table.\n"
        "- When you use a chart tool, tell the user in your final answer what the chart represents."
    ),
    "heavy": (
        "--- **Charting Rules** ---\n"
        "- You should actively look for opportunities to visualize data.\n"
        "- After nearly every successful data-gathering operation, your next step should be to call an appropriate chart tool to visualize the results.\n"
        "- Prefer visual answers over text-based tables whenever possible.\n"
        "- When you use a chart tool, tell the user in your final answer what the chart represents."
    )
}

# --- Globals ---
tools_context = "--- No Tools Available ---"
prompts_context = "--- No Prompts Available ---"
charts_context = "--- No Charts Available ---"

llm = None
mcp_client = None
# This will store the configurations for all connected MCP servers
SERVER_CONFIGS = {}

mcp_tools = {}
mcp_prompts = {}
mcp_charts = {}

structured_tools = {}
structured_prompts = {}
structured_resources = {}
structured_charts = {}

tool_scopes = {}

SESSIONS = {}

# --- Helper for Server-Sent Events ---
def _format_sse(data: dict, event: str = None) -> str:
    """Formats a dictionary into a server-sent event string."""
    msg = f"data: {json.dumps(data)}\n"
    if event is not None:
        msg += f"event: {event}\n"
    return f"{msg}\n"

def _unwrap_exception(e: BaseException) -> BaseException:
    """Recursively unwraps ExceptionGroups to find the root cause."""
    if isinstance(e, ExceptionGroup) and e.exceptions:
        return _unwrap_exception(e.exceptions[0])
    return e


# --- Core Logic ---

class OutputFormatter:
    """
    Parses raw LLM output and structured tool data to generate professional,
    failure-safe HTML for the UI.
    """
    def __init__(self, llm_summary_text: str, collected_data: list):
        self.raw_summary = llm_summary_text
        self.collected_data = collected_data
        self.processed_data_indices = set()

    def _sanitize_summary(self) -> str:
        # This regex now also removes markdown tables
        markdown_table_pattern = re.compile(r"\|.*\|[\n\r]*\|[-| :]*\|[\n\r]*(?:\|.*\|[\n\r]*)*", re.MULTILINE)
        clean_summary = re.sub(markdown_table_pattern, "\n(Data table is shown below)\n", self.raw_summary)

        sql_ddl_pattern = re.compile(r"```sql\s*CREATE MULTISET TABLE.*?;?\s*```|CREATE MULTISET TABLE.*?;", re.DOTALL | re.IGNORECASE)
        clean_summary = re.sub(sql_ddl_pattern, "\n(Formatted DDL shown below)\n", self.raw_summary)
        
        lines = clean_summary.strip().split('\n')
        html_output = ""
        in_list = False

        def process_line_markdown(line):
            line = re.sub(r'\*{2,3}(.*?):\*{1,3}', r'<strong>\1:</strong>', line)
            line = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', line)
            line = re.sub(r'`(.*?)`', r'<code class="bg-gray-900/70 text-teradata-orange rounded-md px-1.5 py-0.5 font-mono text-sm">\1</code>', line)
            return line

        for line in lines:
            line = line.strip()
            if not line:
                if in_list:
                    html_output += '</ul>'
                    in_list = False
                continue

            if line.startswith(('* ', '- ')):
                if not in_list:
                    html_output += '<ul class="list-disc list-inside space-y-2 text-gray-300 mb-4">'
                    in_list = True
                content = line[2:]
                processed_content = process_line_markdown(content)
                html_output += f'<li>{processed_content}</li>'
            elif line.startswith('# '):
                if in_list: html_output += '</ul>'; in_list = False
                content = line[2:]
                html_output += f'<h3 class="text-xl font-bold text-white mb-3 border-b border-gray-700 pb-2">{content}</h3>'
            elif line.startswith('## '):
                if in_list: html_output += '</ul>'; in_list = False
                content = line[3:]
                html_output += f'<h4 class="text-lg font-semibold text-white mt-4 mb-2">{content}</h4>'
            else:
                if in_list:
                    html_output += '</ul>'
                    in_list = False
                processed_line = process_line_markdown(line)
                html_output += f'<p class="text-gray-300 mb-4">{processed_line}</p>'
        
        if in_list:
            html_output += '</ul>'
            
        return html_output

    def _render_ddl(self, tool_result: dict, index: int) -> str:
        if not isinstance(tool_result, dict) or "results" not in tool_result: return ""
        results = tool_result.get("results")
        if not isinstance(results, list) or not results: return ""
        ddl_text = results[0].get('Request Text', 'DDL not available.')
        ddl_text_sanitized = ddl_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        metadata = tool_result.get("metadata", {})
        table_name = metadata.get("table", "DDL")
        self.processed_data_indices.add(index)
        return f"""
        <div class="response-card">
            <div class="sql-code-block">
                <div class="sql-header">
                    <span>SQL DDL: {table_name}</span>
                    <button class="copy-button" onclick="copyToClipboard(this)">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M4 1.5H3a2 2 0 0 0-2 2V14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V3.5a2 2 0 0 0-2-2h-1v1h1a1 1 0 0 1 1 1V14a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1V3.5a1 1 0 0 1 1-1h1v-1z"/><path d="M9.5 1a.5.5 0 0 1 .5.5v1a.5.5 0 0 1-.5.5h-3a.5.5 0 0 1-.5-.5v-1a.5.5 0 0 1 .5-.5h3zM-1 7a.5.5 0 0 1 .5-.5h15a.5.5 0 0 1 0 1H-.5A.5.5 0 0 1-1 7z"/></svg>
                        Copy
                    </button>
                </div>
                <pre><code class="language-sql">{ddl_text_sanitized}</code></pre>
            </div>
        </div>
        """

    def _render_table(self, tool_result: dict, index: int, default_title: str) -> str:
        if not isinstance(tool_result, dict) or "results" not in tool_result: return ""
        results = tool_result.get("results")
        if not isinstance(results, list) or not results or not all(isinstance(item, dict) for item in results): return ""
        metadata = tool_result.get("metadata", {})
        title = metadata.get("table_name", default_title)
        headers = results[0].keys()
        html = f"""
        <div class="response-card">
            <h4 class="text-lg font-semibold text-white mb-2">Data: <code>{title}</code></h4>
            <div class='table-container'>
                <table class='assistant-table'>
                    <thead><tr>{''.join(f'<th>{h}</th>' for h in headers)}</tr></thead>
                    <tbody>
        """
        for row in results:
            html += "<tr>"
            for header in headers:
                cell_data = str(row.get(header, ''))
                sanitized_cell = cell_data.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                html += f"<td>{sanitized_cell}</td>"
            html += "</tr>"
        html += "</tbody></table></div></div>"
        self.processed_data_indices.add(index)
        return html
        
    def _render_chart(self, chart_data: dict, index: int) -> str:
        chart_id = f"chart-render-target-{uuid.uuid4()}"
        # The data-spec attribute will hold the JSON string for the chart
        chart_spec_json = json.dumps(chart_data.get("spec", {}))
        self.processed_data_indices.add(index)
        return f"""
        <div class="response-card">
            <div id="{chart_id}" class="chart-render-target" data-spec='{chart_spec_json}'></div>
        </div>
        """

    def render(self) -> str:
        final_html = ""
        clean_summary_html = self._sanitize_summary()
        if clean_summary_html:
            final_html += f'<div class="response-card summary-card">{clean_summary_html}</div>'

        for i, tool_result in enumerate(self.collected_data):
            if i in self.processed_data_indices or not isinstance(tool_result, dict):
                continue
            
            # Check for chart type first
            if tool_result.get("type") == "chart":
                final_html += self._render_chart(tool_result, i)
                continue

            metadata = tool_result.get("metadata", {})
            tool_name = metadata.get("tool_name")

            if tool_name == 'base_tableDDL':
                final_html += self._render_ddl(tool_result, i)
            elif tool_name and "results" in tool_result:
                 final_html += self._render_table(tool_result, i, f"Result for {tool_name}")

        if not final_html.strip():
            return "<p>The agent completed its work but did not produce a visible output.</p>"
        return final_html

async def call_llm_api(prompt: str, session_id: str = None, chat_history=None, raise_on_error: bool = False, system_prompt_override: str = None) -> str:
    if not llm: raise RuntimeError("LLM is not initialized.")
    
    llm_logger = logging.getLogger("llm_conversation")
    full_log_message = ""
    response_text = ""

    try:
        # --- Provider-Aware API Call Logic ---
        if APP_CONFIG.CURRENT_PROVIDER == "Google":
            is_session_call = session_id is not None and session_id in SESSIONS
            
            if is_session_call:
                chat_session = SESSIONS[session_id]['chat_object']
                history_for_log = chat_session.history
                if history_for_log:
                    formatted_lines = [f"[{msg.role}]: {msg.parts[0].text}" for msg in history_for_log]
                    full_log_message += f"--- FULL CONTEXT (Session: {session_id}) ---\n--- History ---\n" + "\n".join(formatted_lines) + "\n\n"
                
                full_log_message += f"--- Current User Prompt ---\n{prompt}\n"
                llm_logger.info(full_log_message)
                response = await chat_session.send_message_async(prompt)
            else: # This is a session-less, one-off call (like for categorization)
                full_log_message += f"--- ONE-OFF CALL ---\n--- Prompt ---\n{prompt}\n"
                llm_logger.info(full_log_message)
                response = await llm.generate_content_async(prompt)

            if not response or not hasattr(response, 'text'):
                raise RuntimeError("Google LLM returned an empty or invalid response.")
            response_text = response.text.strip()

        elif APP_CONFIG.CURRENT_PROVIDER == "Anthropic":
            if system_prompt_override:
                system_prompt = system_prompt_override
            elif session_id and session_id in SESSIONS:
                system_prompt = SESSIONS[session_id]['system_prompt']
            else:
                # This case should not be hit for session-less calls if override is provided
                raise ValueError("A session_id or system_prompt_override is required for Anthropic provider.")

            history_source = chat_history if chat_history is not None else (SESSIONS.get(session_id, {}).get('chat_object', []))

            messages = []
            for msg in history_source:
                role = msg.get('role')
                if role == 'model': role = 'assistant'
                if role in ['user', 'assistant']:
                    content = msg.get('content')
                    messages.append({'role': role, 'content': content})
            
            messages.append({'role': 'user', 'content': prompt})

            full_log_message += f"--- SYSTEM PROMPT ---\n{system_prompt}\n\n"
            full_log_message += f"--- FULL CONTEXT (Session: {session_id or 'one-off'}) ---\n--- History ---\n"
            for msg in messages:
                full_log_message += f"[{msg['role']}]: {msg['content']}\n"
            full_log_message += "\n"
            llm_logger.info(full_log_message)

            response = await llm.messages.create(
                model=APP_CONFIG.CURRENT_MODEL,
                system=system_prompt,
                messages=messages,
                max_tokens=4096
            )
            if not response or not response.content:
                raise RuntimeError("Anthropic LLM returned an empty or invalid response.")
            response_text = response.content[0].text.strip()
        
        elif APP_CONFIG.CURRENT_PROVIDER == "Amazon":
            is_session_call = session_id is not None and session_id in SESSIONS
            
            if is_session_call:
                system_prompt = SESSIONS[session_id]['system_prompt']
                history = SESSIONS[session_id]['chat_object']
            else: # One-off call
                system_prompt = system_prompt_override or "You are a helpful assistant."
                history = chat_history or []

            # --- START: ENHANCED PAYLOAD AND MODEL ID LOGIC ---
            model_id_to_invoke = APP_CONFIG.CURRENT_MODEL
            # Adjust model_id for Nova models if they are foundation models (not ARNs)
            if "amazon.nova" in model_id_to_invoke and not model_id_to_invoke.startswith("arn:aws:bedrock:") and APP_CONFIG.CURRENT_AWS_REGION:
                region = APP_CONFIG.CURRENT_AWS_REGION
                if region.startswith("us-"):
                    prefix = "us."
                elif region.startswith("eu-"):
                    prefix = "eu."
                elif region.startswith("ap-"):
                    prefix = "apac."
                else:
                    prefix = ""
                
                if prefix:
                    adjusted_id = f"{prefix}{model_id_to_invoke}"
                    app.logger.info(f"Adjusting Nova model ID from '{model_id_to_invoke}' to '{adjusted_id}' for region '{region}'.")
                    model_id_to_invoke = adjusted_id

            # Determine the payload structure based on the model ID string
            if "anthropic" in model_id_to_invoke:
                messages = []
                for msg in history:
                    messages.append({'role': msg['role'], 'content': msg['content']})
                messages.append({'role': 'user', 'content': prompt})
                body = json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 4096,
                    "system": system_prompt,
                    "messages": messages
                })
            elif "amazon.nova" in model_id_to_invoke:
                messages = []
                # Add history messages
                for msg in history:
                    role = 'assistant' if msg.get('role') == 'assistant' or msg.get('role') == 'model' else 'user'
                    messages.append({'role': role, 'content': [{'text': msg.get('content')}]})
                # Add the current user prompt
                messages.append({"role": "user", "content": [{"text": prompt}]})

                body_dict = {
                    "messages": messages,
                    "inferenceConfig": { "maxTokens": 4096, "temperature": 0.7, "topP": 0.9 }
                }
                # Add the system prompt if it exists
                if system_prompt:
                    body_dict["system"] = [{"text": system_prompt}]
                
                body = json.dumps(body_dict)
            elif "meta" in model_id_to_invoke:
                 body = json.dumps({
                    "prompt": prompt,
                    "max_gen_len": 2048,
                    "temperature": 0.5,
                })
            else: # Default to Amazon Titan format
                text_prompt = f"{system_prompt}\n\n"
                for msg in history:
                    text_prompt += f"{msg['role']}: {msg['content']}\n\n"
                text_prompt += f"user: {prompt}\n\nassistant:"
                body = json.dumps({
                    "inputText": text_prompt,
                    "textGenerationConfig": { "maxTokenCount": 4096, "temperature": 0.7, "topP": 0.9 }
                })
            
            loop = asyncio.get_running_loop()
            
            # Boto3's bedrock-runtime client is synchronous, so we need to run it in an executor.
            response = await loop.run_in_executor(
                None, 
                lambda: llm.invoke_model(body=body, modelId=model_id_to_invoke)
            )

            response_body = json.loads(response.get('body').read())

            # Extract the text based on model type
            if "anthropic" in model_id_to_invoke:
                response_text = response_body.get('content')[0].get('text')
            elif "amazon.nova" in model_id_to_invoke:
                response_text = response_body.get('output', {}).get('message', {}).get('content', [{}])[0].get('text', '')
            elif "meta" in model_id_to_invoke:
                response_text = response_body.get('generation')
            else: # Titan
                response_text = response_body.get('results')[0].get('outputText')
            # --- END: ENHANCED PAYLOAD AND MODEL ID LOGIC ---

        else:
            raise NotImplementedError(f"Provider '{APP_CONFIG.CURRENT_PROVIDER}' is not yet supported.")

        llm_logger.info(f"--- RESPONSE ---\n{response_text}\n" + "-"*50 + "\n")
        return response_text

    except Exception as e:
        app.logger.error(f"Error calling LLM API for provider {APP_CONFIG.CURRENT_PROVIDER}: {e}", exc_info=True)
        llm_logger.error(f"--- ERROR in LLM call ---\n{e}\n" + "-"*50 + "\n")
        if raise_on_error:
            raise e
        return f"FINAL_ANSWER: I'm sorry, but I encountered an error while communicating with the language model: {str(e)}"

async def validate_and_correct_parameters(command: dict) -> dict:
    """
    Validates LLM-generated parameters against the tool spec and attempts correction.
    Returns a corrected command or a command indicating failure.
    """
    tool_name = command.get("tool_name")
    if not tool_name or tool_name not in mcp_tools:
        return command

    args = command.get("arguments", {})

    # --- START: Programmatic Shim for Legacy Quality Tools ---
    LEGACY_QUALITY_TOOLS = [
        "qlty_missingValues", "qlty_negativeValues", "qlty_distinctCategories",
        "qlty_standardDeviation", "qlty_columnSummary", "qlty_univariateStatistics",
        "qlty_rowsWithMissingValues"
    ]
    if tool_name in LEGACY_QUALITY_TOOLS:
        db_name = args.get("db_name")
        table_name = args.get("table_name")
        if db_name and table_name and '.' not in table_name:
            args["table_name"] = f"{db_name}.{table_name}"
            del args["db_name"]
            app.logger.info(f"Applied shim for '{tool_name}': Combined db_name and table_name.")
    # --- END: Programmatic Shim ---

    llm_arg_names = set(args.keys())
    tool_spec = mcp_tools[tool_name]
    spec_arg_names = set(tool_spec.args.keys())
    
    required_params = {name for name, field in tool_spec.args.items() if field.get("required", False)}

    # --- START: Refined Validation Logic ---
    # A call is valid if all required parameters are present.
    # It's okay if optional parameters are missing.
    if required_params.issubset(llm_arg_names):
        return command
    # --- END: Refined Validation Logic ---

    app.logger.info(f"Parameter mismatch for tool '{tool_name}'. Attempting correction with LLM.")
    correction_prompt = f"""
        You are a parameter-mapping specialist. Your task is to map the 'LLM-Generated Parameters' to the 'Official Tool Parameters'.
        The user wants to call the tool '{tool_name}', which is described as: '{tool_spec.description}'.

        Official Tool Parameters: {list(spec_arg_names)}
        LLM-Generated Parameters: {list(llm_arg_names)}

        Respond with a single JSON object that maps each generated parameter name to its correct official name.
        If a generated parameter does not sensibly map to any official parameter, use `null` as the value.
        Example response: {{"database": "db_name", "table": "table_name", "extra_param": null}}
    """
    
    correction_response_text = await call_llm_api(prompt=correction_prompt, chat_history=[])
    
    try:
        json_match = re.search(r"```json\s*\n(.*?)\n\s*```", correction_response_text, re.DOTALL)
        if not json_match:
            json_match = re.search(r'\{.*\}', correction_response_text, re.DOTALL)
        
        if not json_match:
             raise ValueError("LLM did not return a valid JSON object for parameter mapping.")

        name_mapping = json.loads(json_match.group(0).strip())

        if any(v is None for v in name_mapping.values()):
            raise ValueError("LLM could not confidently map all parameters.")

        corrected_args = {}
        for llm_name, spec_name in name_mapping.items():
            if llm_name in args and spec_name in spec_arg_names:
                corrected_args[spec_name] = args[llm_name]
        
        if not required_params.issubset(set(corrected_args.keys())):
             raise ValueError(f"Corrected parameters are still missing required arguments. Missing: {required_params - set(corrected_args.keys())}")

        app.logger.info(f"Successfully corrected parameters for tool '{tool_name}'. New args: {corrected_args}")
        command['arguments'] = corrected_args
        return command

    except (ValueError, json.JSONDecodeError, AttributeError) as e:
        app.logger.warning(f"Parameter correction failed for '{tool_name}': {e}. Requesting user input.")
        spec_arguments = list(tool_spec.args.values())
        return {
            "error": "parameter_mismatch",
            "tool_name": tool_name,
            "message": "The agent could not determine the correct parameters for the tool. Please provide them below.",
            "specification": {
                "name": tool_name,
                "description": tool_spec.description,
                "arguments": spec_arguments
            }
        }

async def invoke_mcp_tool(command: dict) -> any:
    if command.get("tool_name") not in mcp_charts:
        validated_command = await validate_and_correct_parameters(command)
        if "error" in validated_command:
            return validated_command
    else:
        validated_command = command

    global mcp_client
    if not mcp_client:
        return {"error": "MCP client is not connected."}

    tool_name = validated_command.get("tool_name")
    args = validated_command.get("arguments", validated_command.get("parameters", {}))

    if tool_name in mcp_charts:
        app.logger.info(f"Locally handling chart generation for tool: {tool_name}")
        try:
            is_bar_chart = "generate_bar_chart" in tool_name
            
            data = args.get("data", [])
            
            # --- START: MODIFIED CHART LOGIC ---
            # Prioritize explicit axis arguments from the LLM
            x_field = args.get("x_axis") or args.get("x_field")
            y_field = args.get("y_axis") or args.get("y_field")
            angle_field = args.get("angle_field")
            color_field = args.get("color_field")

            # Fallback to inference only if explicit fields are missing
            if not x_field or not y_field:
                app.logger.info("Axis fields not specified by LLM, inferring from data types.")
                if data:
                    first_row = data[0]
                    # Find first string-like column for x-axis
                    x_field = next((k for k, v in first_row.items() if isinstance(v, str)), None)
                    # Find first numeric-like column for y-axis (handles strings of numbers)
                    y_field = next((k for k, v in first_row.items() if isinstance(v, (int, float)) or (isinstance(v, str) and v.replace('.', '', 1).isdigit())), None)

            # Ensure the data for the y-axis is numeric
            if y_field and data:
                for row in data:
                    try:
                        row[y_field] = float(row[y_field])
                    except (ValueError, TypeError):
                        app.logger.warning(f"Could not convert value '{row.get(y_field)}' to float for y-axis '{y_field}'.")
                        row[y_field] = 0 # Default to 0 if conversion fails

            spec_options = {
                "data": data,
                "xField": y_field if is_bar_chart else x_field, # Bar charts in G2Plot swap x/y
                "yField": x_field if is_bar_chart else y_field,
                "angleField": angle_field,
                "colorField": color_field,
                "seriesField": args.get("series_field", args.get("series")),
                "title": { "visible": True, "text": args.get("title", "Generated Chart") }
            }
            # --- END: MODIFIED CHART LOGIC ---

            chart_type_mapping = {
                "generate_bar_chart": "Bar", "generate_column_chart": "Column",
                "generate_pie_chart": "Pie", "generate_line_chart": "Line",
                "generate_area_chart": "Area", "generate_scatter_chart": "Scatter",
                "generate_histogram_chart": "Histogram", "generate_boxplot_chart": "Box",
                "generate_dual_axes_chart": "DualAxes",
            }
            plot_type = next((v for k, v in chart_type_mapping.items() if k in tool_name), "Column")
            final_spec_options = {k: v for k, v in spec_options.items() if v is not None}
            chart_spec = { "type": plot_type, "options": final_spec_options }
            return {"type": "chart", "spec": chart_spec, "metadata": {"tool_name": tool_name}}
        except Exception as e:
            app.logger.error(f"Error during local chart generation: {e}", exc_info=True)
            return {"error": f"Failed to generate chart spec locally: {e}"}

    server_name = "teradata_mcp_server"
    if tool_name not in mcp_tools:
        return {"error": f"Tool '{tool_name}' not found in any connected server."}

    # General parameter name normalization
    if 'database_name' in args: args['db_name'] = args.pop('database_name')
    if 'database' in args: args['db_name'] = args.pop('database')
    if 'table' in args: args['table_name'] = args.pop('table')
    if 'column_name' in args and 'col_name' not in args:
        args['col_name'] = args.pop('column_name')
    
    try:
        app.logger.debug(f"Creating temporary session on '{server_name}' to invoke tool '{tool_name}' with args: {args}")
        async with mcp_client.session(server_name) as temp_session:
            call_tool_result = await temp_session.call_tool(tool_name, args)
            app.logger.debug(f"Successfully invoked tool '{tool_name}'. Raw response: {call_tool_result}")
            
            if hasattr(call_tool_result, 'content') and isinstance(call_tool_result.content, list) and len(call_tool_result.content) > 0:
                text_content = call_tool_result.content[0]
                if hasattr(text_content, 'text') and isinstance(text_content.text, str):
                    try:
                        parsed_json = json.loads(text_content.text)
                        return parsed_json
                    except json.JSONDecodeError:
                        app.logger.error(f"Tool '{tool_name}' returned a string that is not valid JSON: {text_content.text}")
                        return {"error": "Tool returned non-JSON string", "data": text_content.text}
            
            raise RuntimeError(f"Unexpected tool result format for '{tool_name}': {call_tool_result}")
    except Exception as e:
        app.logger.error(f"Error during tool invocation for '{tool_name}': {e}", exc_info=True)
        return {"error": f"An exception occurred while invoking tool '{tool_name}'."}

def _evaluate_inline_math(json_str: str) -> str:
    """Finds and evaluates simple inline math expressions in a JSON-like string."""
    # This regex finds simple numeric expressions like '279+20' or '113 + 32'
    math_expr_pattern = re.compile(r'\b(\d+\.?\d*)\s*([+\-*/])\s*(\d+\.?\d*)\b')
    
    # Use a loop to handle multiple expressions until none are left
    while True:
        match = math_expr_pattern.search(json_str)
        if not match:
            break
        
        num1_str, op, num2_str = match.groups()
        original_expr = match.group(0)
        
        try:
            num1 = float(num1_str)
            num2 = float(num2_str)
            result = 0
            if op == '+': result = num1 + num2
            elif op == '-': result = num1 - num2
            elif op == '*': result = num1 * num2
            elif op == '/': result = num1 / num2
            
            # Replace only the first occurrence to avoid infinite loops on tricky strings
            json_str = json_str.replace(original_expr, str(result), 1)
        except (ValueError, ZeroDivisionError):
            # If something goes wrong, just leave the original expression and break
            break
            
    return json_str

class AgentState(Enum):
    DECIDING = auto()
    EXECUTING_TOOL = auto()
    SUMMARIZING = auto()
    DONE = auto()
    ERROR = auto()

class PlanExecutor:
    def __init__(self, session_id: str, initial_instruction: str, original_user_input: str):
        self.session_id = session_id
        self.original_user_input = original_user_input
        self.state = AgentState.DECIDING
        self.next_action_str = initial_instruction
        self.collected_data = []
        self.max_steps = 40 
        self.active_prompt_plan = None
        self.active_prompt_name = None
        self.current_command = None
        self.iteration_context = None

    async def run(self):
        for i in range(self.max_steps):
            if self.state in [AgentState.DONE, AgentState.ERROR]: break
            
            try:
                if self.state == AgentState.DECIDING:
                    yield _format_sse({"step": "Assistant has decided on an action", "details": self.next_action_str}, "llm_thought")
                    async for event in self._handle_deciding(): yield event
                
                elif self.state == AgentState.EXECUTING_TOOL:
                    tool_name = self.current_command.get("tool_name")
                    if tool_scopes.get(tool_name) == 'column':
                        async for event in self._execute_column_iteration(): yield event
                    else:
                        async for event in self._execute_standard_tool(): yield event
                
                elif self.state == AgentState.SUMMARIZING:
                    async for event in self._handle_summarizing(): yield event

            except Exception as e:
                app.logger.error(f"Error in state {self.state.name}: {e}", exc_info=True)
                self.state = AgentState.ERROR
                yield _format_sse({"error": "An error occurred during execution.", "details": str(e)}, "error")
        
        if self.state not in [AgentState.DONE, AgentState.ERROR]:
            app.logger.warning("Plan execution finished due to max steps.")
            async for event in self._handle_summarizing(): yield event

    async def _handle_deciding(self):
        if re.search(r'FINAL_ANSWER:', self.next_action_str, re.IGNORECASE):
            self.state = AgentState.SUMMARIZING
            return

        json_match = re.search(r"```json\s*\n(.*?)\n\s*```", self.next_action_str, re.DOTALL)
        if not json_match:
            if self.iteration_context:
                ctx = self.iteration_context
                current_item_name = ctx["items"][ctx["item_index"]]
                ctx["results_per_item"][current_item_name].append(self.next_action_str)
                ctx["action_count_for_item"] += 1
                await self._get_next_action_from_llm()
                return

            app.logger.warning(f"LLM response not a tool command or FINAL_ANSWER. Summarizing. Response: {self.next_action_str}")
            self.state = AgentState.SUMMARIZING
            return

        command_str = json_match.group(1).strip()
        
        # --- START: MODIFIED LOGIC ---
        # First, parse the JSON to identify the tool being called
        try:
            temp_command = json.loads(command_str)
            tool_name = temp_command.get("tool_name")

            # Only apply the math evaluation if it's a chart tool
            if tool_name in mcp_charts:
                corrected_command_str = _evaluate_inline_math(command_str)
                command = json.loads(corrected_command_str)
                if command_str != corrected_command_str:
                    app.logger.info(f"Corrected inline math in chart JSON. Corrected string: {corrected_command_str}")
            else:
                command = temp_command # Use the already parsed command
        
        except json.JSONDecodeError as e:
            app.logger.error(f"JSON parsing failed. Error: {e}. Original string was: {command_str}")
            # Re-raise or handle the error appropriately
            raise e
        # --- END: MODIFIED LOGIC ---
            
        self.current_command = command
        
        if "prompt_name" in command:
            prompt_name = command.get("prompt_name")
            self.active_prompt_name = prompt_name
            arguments = command.get("arguments", command.get("parameters", {}))
            
            # Normalize db_name to database_name for prompts
            if 'db_name' in arguments and 'database_name' not in arguments:
                arguments['database_name'] = arguments.pop('db_name')

            if not mcp_client:
                raise RuntimeError("MCP client is not connected.")

            try:
                get_prompt_result = None
                async with mcp_client.session("teradata_mcp_server") as temp_session:
                    # Call get_prompt with only the name, not the arguments
                    get_prompt_result = await temp_session.get_prompt(name=prompt_name)
                
                if get_prompt_result is None:
                    raise ValueError("Prompt retrieval from MCP server returned None.")

                # Now, manually render the prompt with the arguments on the client side
                prompt_text = get_prompt_result.content.text if hasattr(get_prompt_result, 'content') and hasattr(get_prompt_result.content, 'text') else str(get_prompt_result)
                self.active_prompt_plan = prompt_text.format(**arguments)

                yield _format_sse({"step": f"Executing Prompt: {prompt_name}", "details": self.active_prompt_plan, "prompt_name": prompt_name}, "prompt_selected")

                await self._get_next_action_from_llm()
            except Exception as e:
                app.logger.error(f"Failed to get or process prompt '{prompt_name}': {e}", exc_info=True)
                raise RuntimeError(f"Could not retrieve the plan for prompt '{prompt_name}'.") from e

        elif "tool_name" in command:
            self.state = AgentState.EXECUTING_TOOL
        else:
            self.state = AgentState.SUMMARIZING

    async def _execute_standard_tool(self):
        yield _format_sse({"step": "Tool Execution Intent", "details": self.current_command}, "tool_result")
        tool_result = await invoke_mcp_tool(self.current_command)
        
        # --- START: MODIFIED LOGIC ---
        tool_result_str = ""
        if isinstance(tool_result, dict) and "error" in tool_result:
            # If the tool returns an error, format it for the LLM's context
            error_details = tool_result.get("data", tool_result.get("error"))
            tool_result_str = json.dumps({
                "tool_name": self.current_command.get("tool_name"),
                "tool_output": {
                    "status": "error",
                    "error_message": error_details
                }
            })
            # Don't add raw error objects to collected_data, let the summary handle it
        else:
            # On success, proceed as before
            tool_result_str = json.dumps({"tool_name": self.current_command.get("tool_name"), "tool_output": tool_result})
            if isinstance(tool_result, dict) and tool_result.get("type") == "chart":
                if self.collected_data:
                    app.logger.info("Chart generated. Removing previous data source from collected data to avoid duplicate display.")
                    self.collected_data.pop()
            self.collected_data.append(tool_result)
        # --- END: MODIFIED LOGIC ---

        if isinstance(tool_result, dict) and tool_result.get("error") == "parameter_mismatch":
            yield _format_sse({"details": tool_result}, "request_user_input")
            self.state = AgentState.ERROR
            return

        yield _format_sse({"step": "Tool Execution Result", "details": tool_result, "tool_name": self.current_command.get("tool_name")}, "tool_result")

        if self.active_prompt_plan and not self.iteration_context:
            plan_text = self.active_prompt_plan.lower()
            is_iterative_plan = any(keyword in plan_text for keyword in ["cycle through", "for each", "iterate"])
            
            if is_iterative_plan and self.current_command.get("tool_name") == "base_tableList" and isinstance(tool_result, dict) and tool_result.get("status") == "success":
                items_to_iterate = [res.get("TableName") for res in tool_result.get("results", []) if res.get("TableName")]
                if items_to_iterate:
                    self.iteration_context = {
                        "items": items_to_iterate, "item_index": 0, "action_count_for_item": 0,
                        "results_per_item": {item: [] for item in items_to_iterate}
                    }
                    yield _format_sse({"step": "Starting Multi-Step Iteration", "details": f"Plan requires processing {len(items_to_iterate)} items."})
        
        yield _format_sse({"step": "Thinking about the next action...", "details": "The agent is reasoning based on the current context."})
        await self._get_next_action_from_llm(tool_result_str=tool_result_str)

    async def _execute_column_iteration(self):
        base_command = self.current_command
        tool_name = base_command.get("tool_name")
        base_args = base_command.get("arguments", base_command.get("parameters", {}))
        db_name = base_args.get("db_name")
        table_name = base_args.get("table_name")

        specific_column = base_args.get("col_name") or base_args.get("column_name")
        if specific_column:
            yield _format_sse({"step": "Tool Execution Intent", "details": base_command}, "tool_result")
            col_result = await invoke_mcp_tool(base_command)
            
            if isinstance(col_result, dict) and col_result.get("error") == "parameter_mismatch":
                yield _format_sse({"details": col_result}, "request_user_input")
                self.state = AgentState.ERROR
                return

            yield _format_sse({"step": f"Tool Execution Result for column: {specific_column}", "details": col_result, "tool_name": tool_name}, "tool_result")
            self.collected_data.append(col_result)
            tool_result_str = json.dumps({"tool_name": tool_name, "tool_output": col_result})
            yield _format_sse({"step": "Thinking about the next action...", "details": "Single column execution complete. Resuming main plan."})
            await self._get_next_action_from_llm(tool_result_str=tool_result_str)
            return

        yield _format_sse({"step": f"Column tool detected: {tool_name}", "details": "Fetching column list to begin iteration."})
        cols_command = {"tool_name": "base_columnDescription", "arguments": {"db_name": db_name, "obj_name": table_name}}
        cols_result = await invoke_mcp_tool(cols_command)

        if not (cols_result and isinstance(cols_result, dict) and cols_result.get('status') == 'success' and cols_result.get('results')):
            raise ValueError(f"Failed to retrieve column list for iteration. Response: {cols_result}")
        
        all_columns = cols_result.get('results', [])
        columns_to_iterate = all_columns
        
        all_column_results = []
        for column_info in columns_to_iterate:
            col_name = column_info.get("ColumnName")
            iter_args = base_args.copy()
            iter_args['col_name'] = col_name
            
            if db_name and table_name and '.' not in table_name:
                iter_args["table_name"] = f"{db_name}.{table_name}"
                if 'db_name' in iter_args: del iter_args["db_name"]

            iter_command = {"tool_name": tool_name, "arguments": iter_args}

            yield _format_sse({"step": "Tool Execution Intent", "details": iter_command}, "tool_result")
            col_result = await invoke_mcp_tool(iter_command)
            
            if isinstance(col_result, dict) and col_result.get("error") == "parameter_mismatch":
                yield _format_sse({"details": col_result}, "request_user_input")
                self.state = AgentState.ERROR
                return # Stop the entire iteration if one column fails validation

            yield _format_sse({"step": f"Tool Execution Result for column: {col_name}", "details": col_result, "tool_name": tool_name}, "tool_result")
            all_column_results.append(col_result)

        if self.iteration_context:
            ctx = self.iteration_context
            current_item_name = ctx["items"][ctx["item_index"]]
            ctx["results_per_item"][current_item_name].append(all_column_results)
            ctx["action_count_for_item"] += 1
        else:
            self.collected_data.append(all_column_results)
        
        tool_result_str = json.dumps({"tool_name": tool_name, "tool_output": all_column_results})

        yield _format_sse({"step": "Thinking about the next action...", "details": "Column iteration complete. Resuming main plan."})
        await self._get_next_action_from_llm(tool_result_str=tool_result_str)


    async def _get_next_action_from_llm(self, tool_result_str: str | None = None):
        prompt_for_next_step = "" 
        
        if self.active_prompt_plan:
            app.logger.info("Applying generic plan-aware reasoning for next step.")
            last_tool_name = self.current_command.get("tool_name") if self.current_command else "N/A"
            prompt_for_next_step = (
                "You are executing a multi-step plan. Your goal is to follow it precisely.\n\n"
                f"--- ORIGINAL PLAN ---\n{self.active_prompt_plan}\n\n"
                "--- CURRENT STATE ---\n"
                f"- The last action was the execution of the tool `{last_tool_name}`.\n"
                "- The result of this tool call is now in the conversation history.\n\n"
                "--- YOUR TASK ---\n"
                "1. **Analyze the ORIGINAL PLAN.** Determine which phase of the plan you have just completed.\n"
                "2. **Determine the NEXT STEP.** Based on the plan, what is the very next action you must take?\n"
                "   - If the plan says the next step is to call another tool, provide the JSON for that tool call.\n"
                "   - If the plan says the next step is to analyze the previous results and provide a final answer, your response **MUST** start with `FINAL_ANSWER:`. Do not call any more tools.\n"
            )
        elif self.iteration_context:
            ctx = self.iteration_context
            current_item_name = ctx["items"][ctx["item_index"]]
            last_tool_failed = tool_result_str and '"error":' in tool_result_str.lower()
            
            if last_tool_failed:
                prompt_for_next_step = (
                    "The last tool call failed. Analyze the error message in the history.\n"
                    f"You are still working on item: **`{current_item_name}`**.\n"
                    "Based on the error, is the tool incompatible with the parameters (e.g., wrong data type)?\n"
                    "- If it is an incompatibility issue, acknowledge the error and **skip this step**. Determine the next logical step in the **ORIGINAL PLAN** for the current item.\n"
                    "- If it is a different kind of error, try to correct it. If you cannot, ask for help.\n"
                    f"--- ORIGINAL PLAN ---\n{self.active_prompt_plan}\n\n"
                )
            else:
                last_tool_table = self.current_command.get("arguments", {}).get("table_name", "")
                
                if last_tool_table and last_tool_table != current_item_name:
                    try:
                        new_index = ctx["items"].index(last_tool_table)
                        ctx["item_index"] = new_index
                        ctx["action_count_for_item"] = 1
                        current_item_name = ctx["items"][ctx["item_index"]]
                    except ValueError:
                        pass 

                if ctx["action_count_for_item"] >= 4:
                    ctx["item_index"] += 1
                    ctx["action_count_for_item"] = 0
                    if ctx["item_index"] >= len(ctx["items"]):
                        self.iteration_context = None
                        prompt_for_next_step = (
                            "You have successfully completed all steps for all items in the iterative phase of the plan. "
                            "All results are now in the conversation history. Your next and final task is to proceed to the next major phase of the **ORIGINAL PLAN** (Phase 3). "
                            "This final phase requires you to synthesize all the information you have gathered into a comprehensive dashboard or report. "
                            "This is a text generation task. Do not call any more tools. "
                            "Your response **MUST** start with `FINAL_ANSWER:`."
                        )
                    else:
                         current_item_name = ctx["items"][ctx["item_index"]]
                         prompt_for_next_step = (
                            f"You have finished all steps for the previous item. Now, begin Phase 2 for the **next item**: `{current_item_name}`. "
                            "According to the original plan, what is the first step for this new item?"
                        )
                else:
                    prompt_for_next_step = (
                        "You are executing a multi-step, iterative plan.\n\n"
                        f"--- ORIGINAL PLAN ---\n{self.active_prompt_plan}\n\n"
                        f"--- CURRENT FOCUS ---\n"
                        f"- You are working on item: **`{current_item_name}`**.\n"
                        f"- You have taken {ctx['action_count_for_item']} action(s) for this item so far.\n"
                        "- The result of the last action is in the conversation history.\n\n"
                        "--- YOUR NEXT TASK ---\n"
                        "1. Look at the **ORIGINAL PLAN** to see what the next step in the sequence is for the current item (`{current_item_name}`).\n"
                        "2. Execute the correct next step. Provide a tool call in a `json` block or perform the required text generation."
                    )
        else:
            prompt_for_next_step = (
                "You have just received data from a tool call. Review the data and your instructions to decide the next step.\n\n"
                "1.  **Consider a Chart:** Review the `--- Charting Rules ---` in your system prompt. Based on the data you just received, would a chart be an appropriate and helpful way to visualize the information for the user?\n\n"
                "2.  **Choose Your Action:**\n"
                "    -   If a chart is appropriate, your next action is to call the correct chart-generation tool. Respond with only a `Thought:` and a ```json...``` block for that tool.\n"
                "    -   If you still need more information from other tools, call the next appropriate tool by responding with a `Thought:` and a ```json...``` block.\n"
                "    -   If a chart is **not** appropriate and you have all the information needed to answer the user's request, you **MUST** provide the final answer. Your response **MUST** be plain text that starts with `FINAL_ANSWER:`. **DO NOT** use a JSON block for the final answer."
            )
        
        if tool_result_str:
            final_prompt_to_llm = f"{prompt_for_next_step}\n\nThe last tool execution returned the following result. Use this to inform your next action:\n\n{tool_result_str}"
        else:
            final_prompt_to_llm = prompt_for_next_step

        self.next_action_str = await call_llm_api(prompt=final_prompt_to_llm, session_id=self.session_id)
        
        if not self.next_action_str: raise ValueError("LLM failed to provide a response.")
        self.state = AgentState.DECIDING


    async def _handle_summarizing(self):
        llm_response = self.next_action_str
        summary_text = ""

        final_answer_match = re.search(r'FINAL_ANSWER:(.*)', llm_response, re.DOTALL | re.IGNORECASE)

        if final_answer_match:
            summary_text = final_answer_match.group(1).strip()
            thought_process = llm_response.split(final_answer_match.group(0))[0]
            yield _format_sse({"step": "Agent finished execution", "details": thought_process}, "llm_thought")
        else:
            yield _format_sse({"step": "Plan finished, generating final summary...", "details": "The agent is synthesizing all collected data."})
            final_prompt = (
                "You have executed a multi-step plan. All results are in the history. "
                f"Your final task is to synthesize this information into a comprehensive, natural language answer for the user's original request: '{self.original_user_input}'. "
                "Your response MUST start with `FINAL_ANSWER:`.\n\n"
                "**CRITICAL INSTRUCTIONS:**\n"
                "1. Provide a concise, user-focused summary in plain text or simple markdown.\n"
                "2. **DO NOT** include raw data, SQL code, or complex tables in your summary. The system will format and append this data automatically.\n"
                "3. Do not describe your internal thought process."
            )
            final_llm_response = await call_llm_api(prompt=final_prompt, session_id=self.session_id)
            final_answer_match_inner = re.search(r'FINAL_ANSWER:(.*)', final_llm_response or "", re.DOTALL | re.IGNORECASE)
            if final_answer_match_inner:
                summary_text = final_answer_match_inner.group(1).strip()
            else:
                summary_text = final_llm_response or "The agent finished its plan but did not provide a final summary."

        formatter = OutputFormatter(llm_summary_text=summary_text, collected_data=self.collected_data)
        final_html = formatter.render()

        # Append assistant's final HTML response to the generic history list
        SESSIONS[self.session_id]['generic_history'].append({'role': 'assistant', 'content': final_html})
        yield _format_sse({"final_answer": final_html}, "final_answer")
        self.state = AgentState.DONE

# --- Web Server Routes ---

@app.route("/")
async def index():
    return await render_template("index.html")

@app.route("/app-config")
async def get_app_config():
    return jsonify({
        "all_models_unlocked": APP_CONFIG.ALL_MODELS_UNLOCKED,
        "charting_enabled": APP_CONFIG.CHARTING_ENABLED
    })

@app.route("/api_key/<provider>")
async def get_api_key(provider):
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
    
    return jsonify({"error": "Unknown provider"}), 404


@app.route("/tools")
async def get_tools():
    if not mcp_client: return jsonify({"error": "Not configured"}), 400
    return jsonify(structured_tools)

@app.route("/prompts")
async def get_prompts():
    if not mcp_client: return jsonify({"error": "Not configured"}), 400
    return jsonify(structured_prompts)

@app.route("/resources")
async def get_resources_route():
    if not mcp_client: return jsonify({"error": "Not configured"}), 400
    return jsonify(structured_resources)

@app.route("/charts")
async def get_charts():
    if not APP_CONFIG.CHART_MCP_CONNECTED: return jsonify({"error": "Chart server not connected"}), 400
    return jsonify(structured_charts)


@app.route("/sessions", methods=["GET"])
async def get_sessions():
    session_summaries = [
        {"id": sid, "name": s_data["name"], "created_at": s_data["created_at"]}
        for sid, s_data in SESSIONS.items()
    ]
    session_summaries.sort(key=lambda x: x["created_at"], reverse=True)
    return jsonify(session_summaries)

@app.route("/session/<session_id>", methods=["GET"])
async def get_session_history(session_id):
    if session_id in SESSIONS:
        # The history now contains generic dicts, safe to send for any provider
        return jsonify(SESSIONS[session_id]["generic_history"])
    return jsonify({"error": "Session not found"}), 404

def get_full_system_prompt(base_prompt_text, charting_intensity_val):
    global tools_context, prompts_context, charts_context
    chart_instructions = CHARTING_INSTRUCTIONS.get(charting_intensity_val, CHARTING_INSTRUCTIONS['none'])
    
    final_charts_context = charts_context if APP_CONFIG.CHART_MCP_CONNECTED else CHARTING_INSTRUCTIONS['none']

    # Safely replace placeholders if they exist in the prompt from the client
    final_system_prompt = base_prompt_text
    if "{charting_instructions}" in final_system_prompt:
        final_system_prompt = final_system_prompt.replace("{charting_instructions}", chart_instructions)
    if "{tools_context}" in final_system_prompt:
        final_system_prompt = final_system_prompt.replace("{tools_context}", tools_context)
    if "{prompts_context}" in final_system_prompt:
        final_system_prompt = final_system_prompt.replace("{prompts_context}", prompts_context)
    if "{charts_context}" in final_system_prompt:
        final_system_prompt = final_system_prompt.replace("{charts_context}", final_charts_context)
    return final_system_prompt

@app.route("/session", methods=["POST"])
async def new_session():
    global llm
    if not llm or not APP_CONFIG.TERADATA_MCP_CONNECTED:
        return jsonify({"error": "Application not configured. Please set MCP and LLM details in Config."}), 400
    
    data = await request.get_json()
    system_prompt_from_client = data.get("system_prompt")
    
    try:
        session_id = str(uuid.uuid4())
        
        charting_intensity = "medium" if APP_CONFIG.CHARTING_ENABLED else "none"
        final_system_prompt = get_full_system_prompt(system_prompt_from_client, charting_intensity)
        
        SESSIONS[session_id] = {
            "system_prompt": final_system_prompt,
            "generic_history": [], # This stores the UI-facing history
            "name": "New Chat",
            "created_at": datetime.now().isoformat()
        }
        
        if APP_CONFIG.CURRENT_PROVIDER == "Google":
            initial_history = [
                {"role": "user", "parts": [{"text": final_system_prompt}]},
                {"role": "model", "parts": [{"text": "Understood. I will follow all instructions."}]}
            ]
            SESSIONS[session_id]['chat_object'] = llm.start_chat(history=initial_history)
        elif APP_CONFIG.CURRENT_PROVIDER in ["Anthropic", "Amazon"]:
             SESSIONS[session_id]['chat_object'] = [] # For these, the chat object is just the history list

        app.logger.info(f"Created new session: {session_id} for provider {APP_CONFIG.CURRENT_PROVIDER}.")
        app.logger.debug(f"Final prompt for session {session_id}:\n{final_system_prompt}")
        return jsonify({"session_id": session_id, "name": SESSIONS[session_id]["name"]})
    except Exception as e:
        app.logger.error(f"Failed to create new session: {e}", exc_info=True)
        return jsonify({"error": f"Failed to initialize a new chat session on the server: {e}"}), 500

@app.route("/ask_stream", methods=["POST"])
async def ask_stream():
    data = await request.get_json()
    user_input = data.get("message")
    session_id = data.get("session_id")
    
    async def stream_generator(user_input, session_id):
        if not all([user_input, session_id]):
            yield _format_sse({"error": "Missing 'message' or 'session_id'"}, "error")
            return
        if session_id not in SESSIONS:
            yield _format_sse({"error": "Invalid or expired session ID"}, "error")
            return

        try:
            SESSIONS[session_id]['generic_history'].append({'role': 'user', 'content': user_input})
            
            if SESSIONS[session_id]['name'] == 'New Chat':
                SESSIONS[session_id]['name'] = user_input[:40] + '...' if len(user_input) > 40 else user_input
                yield _format_sse({"session_name_update": {"id": session_id, "name": SESSIONS[session_id]['name']}}, "session_update")

            yield _format_sse({"step": "Assistant is thinking...", "details": "Analyzing request and selecting best action."})
            
            llm_reasoning_and_command = await call_llm_api(user_input, session_id)
            
            # Update history for stateless providers
            if APP_CONFIG.CURRENT_PROVIDER in ["Anthropic", "Amazon"] and llm_reasoning_and_command:
                SESSIONS[session_id]['chat_object'].append({'role': 'user', 'content': user_input})
                SESSIONS[session_id]['chat_object'].append({'role': 'assistant', 'content': llm_reasoning_and_command})

            executor = PlanExecutor(session_id=session_id, initial_instruction=llm_reasoning_and_command, original_user_input=user_input)
            async for event in executor.run():
                yield event

        except Exception as e:
            app.logger.error(f"An unhandled error occurred in /ask_stream: {e}", exc_info=True)
            yield _format_sse({"error": "An unexpected server error occurred.", "details": str(e)}, "error")

    return Response(stream_generator(user_input, session_id), mimetype="text/event-stream")

@app.route("/invoke_prompt_stream", methods=["POST"])
async def invoke_prompt_stream():
    data = await request.get_json()
    session_id = data.get("session_id")
    prompt_name = data.get("prompt_name")
    arguments = data.get("arguments", {})
    
    async def stream_generator():
        user_input = f"Manual execution of prompt: {prompt_name}"
        SESSIONS[session_id]['generic_history'].append({'role': 'user', 'content': user_input})
        if SESSIONS[session_id]['name'] == 'New Chat':
            SESSIONS[session_id]['name'] = user_input[:40] + '...' if len(user_input) > 40 else user_input
            yield _format_sse({"session_name_update": {"id": session_id, "name": SESSIONS[session_id]['name']}}, "session_update")

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
            # Manually add user message to history for the LLM call
            if APP_CONFIG.CURRENT_PROVIDER == "Google":
                 await SESSIONS[session_id]['chat_object'].send_message_async(user_input)
            elif APP_CONFIG.CURRENT_PROVIDER in ["Anthropic", "Amazon"]:
                SESSIONS[session_id]['chat_object'].append({'role': 'user', 'content': user_input})
            
            executor = PlanExecutor(session_id=session_id, initial_instruction=initial_instruction, original_user_input=user_input)
            async for event in executor.run():
                yield event
        except Exception as e:
            app.logger.error(f"An unhandled error occurred in /invoke_prompt_stream: {e}", exc_info=True)
            yield _format_sse({"error": "An unexpected server error occurred during prompt invocation.", "details": str(e)}, "error")

    return Response(stream_generator(), mimetype="text/event-stream")

def classify_tool_scopes(tools: list) -> dict:
    scopes = {}
    for tool in tools:
        arg_names = set(tool.args.keys())
        if 'col_name' in arg_names or 'column_name' in arg_names: scopes[tool.name] = 'column'
        elif 'table_name' in arg_names or 'obj_name' in arg_names: scopes[tool.name] = 'table'
        else: scopes[tool.name] = 'database'
    return scopes

def classify_prompt_scopes(prompts: list) -> dict:
    scopes = {}
    for prompt in prompts:
        arg_names = {arg.name for arg in prompt.arguments}
        if 'table_name' in arg_names: scopes[prompt.name] = 'table'
        elif 'database_name' in arg_names: scopes[prompt.name] = 'database'
        else: scopes[prompt.name] = 'general'
    return scopes


async def load_and_categorize_teradata_resources():
    global tools_context, structured_tools, structured_prompts, prompts_context, mcp_tools, mcp_prompts, tool_scopes, structured_resources
    
    if not mcp_client:
        raise Exception("MCP Client not initialized.")

    async with mcp_client.session("teradata_mcp_server") as temp_session:
        app.logger.info("--- Loading Teradata tools, prompts, and resources... ---")
        
        # Load Tools
        loaded_tools = await load_mcp_tools(temp_session)
        mcp_tools = {tool.name: tool for tool in loaded_tools}
        tool_scopes = classify_tool_scopes(loaded_tools)
        tools_context = "--- Available Tools ---\n" + "\n".join([f"- `{tool.name}`: {tool.description}" for tool in loaded_tools])
        
        app.logger.info(f"[DEBUG] Loaded {len(loaded_tools)} tools from Teradata MCP Server.")
        
        # Categorize Tools for UI
        tool_list_for_prompt = "\n".join([f"- {tool.name}: {tool.description}" for tool in loaded_tools])
        categorization_prompt = (
            "You are a helpful assistant that organizes lists of technical tools for a **Teradata database system** into logical categories for a user interface. "
            "Your response MUST be a single, valid JSON object. The keys should be the category names, "
            "and the values should be an array of tool names belonging to that category.\n\n"
            f"--- Tool List ---\n{tool_list_for_prompt}"
        )
        # This call must now raise an error on failure to stop the configuration process
        categorization_system_prompt = "You are a helpful assistant that organizes lists into JSON format."
        categorized_tools_str = await call_llm_api(
            categorization_prompt, 
            chat_history=[], 
            raise_on_error=True, 
            system_prompt_override=categorization_system_prompt
        )
        app.logger.info(f"[DEBUG] LLM Tool Categorization Raw Response:\n{categorized_tools_str}")
        
        # No more try/except here; if the above call fails, the /configure endpoint will catch it.
        cleaned_str = re.search(r'\{.*\}', categorized_tools_str, re.DOTALL).group(0)
        categorized_tools = json.loads(cleaned_str)
        structured_tools = {category: [{"name": name, "description": mcp_tools[name].description} for name in tool_names if name in mcp_tools] for category, tool_names in categorized_tools.items()}

        # Load and Process Prompts
        loaded_prompts = []
        try:
            app.logger.info("[DEBUG] >>> Attempting to call temp_session.list_prompts()")
            list_prompts_result = await temp_session.list_prompts()
            app.logger.info(f"[DEBUG] >>> temp_session.list_prompts() returned object of type: {type(list_prompts_result)}")
            app.logger.info(f"[DEBUG] >>> Full return object: {list_prompts_result}")
            
            if hasattr(list_prompts_result, 'prompts'):
                loaded_prompts = list_prompts_result.prompts
                app.logger.info(f"[DEBUG] >>> Extracted .prompts attribute. Count: {len(loaded_prompts)}")
            else:
                app.logger.warning("[DEBUG] >>> The result from list_prompts() does NOT have a .prompts attribute.")

        except Exception as e:
            app.logger.error(f"CRITICAL ERROR while loading prompts: {e}", exc_info=True)

        if loaded_prompts:
            mcp_prompts = {prompt.name: prompt for prompt in loaded_prompts}
            prompts_context = "--- Available Prompts ---\n" + "\n".join([f"- `{p.name}`: {p.description}" for p in loaded_prompts])
            
            # Categorize Prompts for UI
            serializable_prompts = [{"name": p.name, "description": p.description, "arguments": [arg.model_dump() for arg in p.arguments]} for p in loaded_prompts]
            prompt_list_for_prompt = "\n".join([f"- {p['name']}: {p['description']}" for p in serializable_prompts])
            categorization_prompt_for_prompts = (
                "You are a helpful assistant that organizes lists of technical prompts for a **Teradata database system** into logical categories for a user interface. "
                "Your response MUST be a single, valid JSON object. The keys should be the category names, "
                "and the values should be an array of prompt names belonging to that category.\n\n"
                f"--- Prompt List ---\n{prompt_list_for_prompt}"
            )
            # This call must also raise an error on failure
            categorized_prompts_str = await call_llm_api(
                categorization_prompt_for_prompts, 
                chat_history=[], 
                raise_on_error=True,
                system_prompt_override=categorization_system_prompt
            )
            app.logger.info(f"[DEBUG] LLM Prompt Categorization Raw Response:\n{categorized_prompts_str}")
            
            cleaned_str_prompts = re.search(r'\{.*\}', categorized_prompts_str, re.DOTALL).group(0)
            categorized_prompts = json.loads(cleaned_str_prompts)
            structured_prompts = {category: [p for p in serializable_prompts if p['name'] in prompt_names] for category, prompt_names in categorized_prompts.items()}
        else:
            prompts_context = "--- No Prompts Available ---"
            structured_prompts = {}


async def load_and_categorize_chart_resources():
    global charts_context, structured_charts, mcp_charts
    if not mcp_client or not APP_CONFIG.CHART_MCP_CONNECTED:
        raise Exception("Chart MCP Client not initialized or connected.")

    async with mcp_client.session("chart_mcp_server") as temp_session:
        app.logger.info("--- Loading Chart tools... ---")
        
        loaded_charts = await load_mcp_tools(temp_session)
        
        app.logger.info(f"[DEBUG] Loaded {len(loaded_charts)} charts from Chart MCP Server.")

        if not loaded_charts:
            app.logger.warning("The chart server returned 0 tools. Please ensure it is configured correctly.")
            mcp_charts = {}
            structured_charts = {}
            charts_context = "--- No Charts Available ---"
            return

        mcp_charts = {tool.name: tool for tool in loaded_charts}
        charts_context = "--- Available Chart Tools ---\n" + "\n".join([f"- `{tool.name}`: {tool.description}" for tool in loaded_charts])
        
        chart_list_for_prompt = "\n".join([f"- {tool.name}: {tool.description}" for tool in loaded_charts])
        categorization_prompt = (
            "You are a helpful assistant that organizes lists of charting tools into logical categories for a user interface. "
            "Your response MUST be a single, valid JSON object where each key is a category name and its value is an object containing a 'description' and a 'tools' array of tool names.\n\n"
            f"--- Chart Tool List ---\n{chart_list_for_prompt}"
        )
        categorization_system_prompt = "You are a helpful assistant that organizes lists into JSON format."
        categorized_charts_str = await call_llm_api(
            categorization_prompt, 
            chat_history=[], 
            raise_on_error=True,
            system_prompt_override=categorization_system_prompt
        )
        app.logger.info(f"[DEBUG] LLM Chart Categorization Raw Response:\n{categorized_charts_str}")
        
        cleaned_str = re.search(r'\{.*\}', categorized_charts_str, re.DOTALL).group(0)
        categorized_charts = json.loads(cleaned_str)
        
        structured_charts = {}
        for category, details in categorized_charts.items():
            tool_names = details.get("tools", [])
            structured_charts[category] = [
                {"name": name, "description": mcp_charts[name].description}
                for name in tool_names if name in mcp_charts
            ]

@app.route("/system_prompt/<provider>/<model_name>", methods=["GET"])
async def get_default_system_prompt(provider, model_name):
    # This endpoint now returns the raw template. The client is responsible for
    # storing it, and the /session endpoint is responsible for formatting it
    # with the live context just before starting a session.
    base_prompt_template = PROVIDER_SYSTEM_PROMPTS.get(provider, PROVIDER_SYSTEM_PROMPTS["Google"])
    return jsonify({"status": "success", "system_prompt": base_prompt_template})

@app.route("/models", methods=["POST"])
async def get_models():
    try:
        data = await request.get_json()
        provider = data.get("provider")
        listing_method = data.get("listing_method", "foundation_models") # Default to foundation models

        if provider == "Google":
            api_key = data.get("apiKey")
            genai.configure(api_key=api_key)
            models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            clean_models = [name.split('/')[-1] for name in models]
            structured_model_list = [{"name": name, "certified": APP_CONFIG.ALL_MODELS_UNLOCKED or name == CERTIFIED_MODEL} for name in clean_models]
            return jsonify({"status": "success", "models": structured_model_list})
        
        elif provider == "Anthropic":
            # Anthropic SDK doesn't have a model list function, so we use a curated list.
            structured_model_list = [{"name": name, "certified": APP_CONFIG.ALL_MODELS_UNLOCKED or name in CERTIFIED_ANTHROPIC_MODELS} for name in CERTIFIED_ANTHROPIC_MODELS]
            return jsonify({"status": "success", "models": structured_model_list})

        elif provider == "Amazon":
            bedrock_client = boto3.client(
                service_name='bedrock',
                aws_access_key_id=data.get("aws_access_key_id"),
                aws_secret_access_key=data.get("aws_secret_access_key"),
                region_name=data.get("aws_region")
            )
            loop = asyncio.get_running_loop()
            if listing_method == "inference_profiles":
                response = await loop.run_in_executor(
                    None,
                    lambda: bedrock_client.list_inference_profiles()
                )
                models = [p['inferenceProfileArn'] for p in response['inferenceProfileSummaries']]
                structured_model_list = [{"name": name, "certified": APP_CONFIG.ALL_MODELS_UNLOCKED or name in CERTIFIED_AMAZON_PROFILES} for name in models]
            else: # Default to foundation models
                response = await loop.run_in_executor(
                    None,
                    lambda: bedrock_client.list_foundation_models(byOutputModality='TEXT')
                )
                models = [m['modelId'] for m in response['modelSummaries']]
                structured_model_list = [{"name": name, "certified": APP_CONFIG.ALL_MODELS_UNLOCKED or name in CERTIFIED_AMAZON_MODELS} for name in models]
            
            return jsonify({"status": "success", "models": structured_model_list})

        else:
            return jsonify({"status": "success", "models": []})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/configure", methods=["POST"])
async def configure_services():
    global mcp_client, llm, SERVER_CONFIGS
    data = await request.get_json()
    
    try:
        provider = data.get("provider")
        model = data.get("model")

        APP_CONFIG.CURRENT_PROVIDER = provider
        APP_CONFIG.CURRENT_MODEL = model
        APP_CONFIG.CURRENT_MODEL_PROVIDER_IN_PROFILE = None # Reset on each config

        if provider == "Google":
            genai.configure(api_key=data.get("apiKey"))
            llm = genai.GenerativeModel(model)
        elif provider == "Anthropic":
            llm = AsyncAnthropic(api_key=data.get("apiKey"))
        elif provider == "Amazon":
            aws_region = data.get("aws_region")
            APP_CONFIG.CURRENT_AWS_REGION = aws_region # Store region
            llm = boto3.client(
                service_name='bedrock-runtime',
                aws_access_key_id=data.get("aws_access_key_id"),
                aws_secret_access_key=data.get("aws_secret_access_key"),
                region_name=aws_region
            )
            if model.startswith("arn:aws:bedrock:"):
                # Parse the provider from the ARN instead of making a separate API call
                # e.g., ...:inference-profile/eu.anthropic.claude-3-sonnet... -> anthropic
                try:
                    profile_part = model.split('/')[-1]
                    provider_name = profile_part.split('.')[1]
                    APP_CONFIG.CURRENT_MODEL_PROVIDER_IN_PROFILE = provider_name
                except IndexError:
                    raise ValueError(f"Could not parse underlying provider from Inference Profile ARN: {model}")

        else:
            raise NotImplementedError(f"Provider '{provider}' is not yet supported.")

        mcp_server_url = f"http://{data.get('host')}:{data.get('port')}{data.get('path')}"
        SERVER_CONFIGS = {'teradata_mcp_server': {"url": mcp_server_url, "transport": "streamable_http"}}
        
        mcp_client = MultiServerMCPClient(SERVER_CONFIGS)
        
        await load_and_categorize_teradata_resources()
        APP_CONFIG.TERADATA_MCP_CONNECTED = True
        return jsonify({"status": "success", "message": "Teradata MCP and LLM configured successfully."})
    except (APIError, google_exceptions.PermissionDenied, ClientError, ExceptionGroup) as e:
        llm = None
        mcp_client = None
        SERVER_CONFIGS = {}
        APP_CONFIG.TERADATA_MCP_CONNECTED = False
        
        # Recursively find the root cause of the exception
        root_exception = _unwrap_exception(e)
        
        error_message = ""
        if isinstance(root_exception, APIError):
            error_message = root_exception.body.get('error', {}).get('message', 'Unknown Anthropic API error.')
        elif isinstance(root_exception, google_exceptions.PermissionDenied):
            error_message = root_exception.message
        elif isinstance(root_exception, ClientError):
            error_message = root_exception.response.get("Error", {}).get("Message", "Unknown AWS error.")
        else:
            error_message = str(root_exception)
            
        app.logger.error(f"Configuration failed: {error_message}", exc_info=True)
        return jsonify({"status": "error", "message": f"Configuration failed: {error_message}"}), 500


@app.route("/configure_chart", methods=["POST"])
async def configure_chart_service():
    global mcp_client, SERVER_CONFIGS
    if not APP_CONFIG.TERADATA_MCP_CONNECTED:
        return jsonify({"status": "error", "message": "Main MCP client not configured. Please connect to Teradata & LLM first."}), 400
    
    data = await request.get_json()
    try:
        chart_server_url = f"http://{data.get('chart_host')}:{data.get('chart_port')}{data.get('chart_path')}"
        SERVER_CONFIGS['chart_mcp_server'] = {"url": chart_server_url, "transport": "sse"}
        
        mcp_client = MultiServerMCPClient(SERVER_CONFIGS)
        
        APP_CONFIG.CHART_MCP_CONNECTED = True
        
        await load_and_categorize_chart_resources()
        
        return jsonify({"status": "success", "message": "Chart MCP server configured successfully."})
    except Exception as e:
        if 'chart_mcp_server' in SERVER_CONFIGS:
            del SERVER_CONFIGS['chart_mcp_server']
        mcp_client = MultiServerMCPClient(SERVER_CONFIGS)
        
        APP_CONFIG.CHART_MCP_CONNECTED = False
        app.logger.error(f"Chart configuration failed: {e}", exc_info=True)
        return jsonify({"status": "error", "message": f"Chart server connection failed: {e}"}), 500

@app.after_serving
async def shutdown():
    app.logger.info("Server shutting down.")

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
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Allow selection of all available Google models, not just the certified one."
    )
    parser.add_argument(
        "--charting",
        action="store_true",
        help="Enable the charting engine configuration and capabilities."
    )
    # Deprecated but kept for backward compatibility
    parser.add_argument("--unlock-models", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.all_models or args.unlock_models:
        APP_CONFIG.ALL_MODELS_UNLOCKED = True
        print("\n--- DEV MODE: All Google models will be selectable. ---\"")
    
    if args.charting:
        APP_CONFIG.CHARTING_ENABLED = True
        print("\n--- CHARTING ENABLED: Charting configuration and features are active. ---")


    script_dir = os.path.dirname(os.path.abspath(__file__))
    templates_dir = os.path.join(script_dir, 'templates')
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer shut down.")
