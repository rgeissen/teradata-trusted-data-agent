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

# This environment variable MUST be set to "false" before any LangChain
# modules are imported to programmatically disable the problematic tracer.
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.resources import load_mcp_resources

# Using the base google.generativeai library for stateful chat
import google.generativeai as genai

# --- Globals for Web App ---
app = Quart(__name__)
app = cors(app, allow_origin="*") # Enable CORS for all origins

# --- App Configuration ---
class AppConfig:
    MODELS_UNLOCKED = False
    TERADATA_MCP_CONNECTED = False
    CHART_MCP_CONNECTED = False

APP_CONFIG = AppConfig()
CERTIFIED_MODEL = "gemini-1.5-flash"

# --- System Prompt Templates ---
BASE_SYSTEM_PROMPT = (
    "You are a specialized assistant for interacting with a Teradata database. Your primary goal is to fulfill user requests by selecting the best tool, prompt, or sequence of tools.\n\n"
    "--- **Core Reasoning Hierarchy** ---\n"
    "1.  **Check for a Perfect Prompt:** First, analyze the user's request and see if there is a single, pre-defined **prompt** that exactly matches the user's intent and scope.\n"
    "2.  **Synthesize a Plan from Tools:** If no single prompt is a perfect match, you must become a **planner**. Create a logical sequence of steps to solve the user's request.\n"
    "3.  **Execute the First Step:** Your response will be the JSON for the *first tool* in your plan.\n\n"
    "--- **CRITICAL RULE: CONTEXT and PARAMETER INFERENCE** ---\n"
    "You **MUST** remember and reuse information from previous turns.\n"
    "**Example of CORRECT Inference:**\n"
    "    -   USER (Turn 1): \"what is the business description for the `equipment` table in database `DEMO_Customer360_db`?\"\n"
    "    -   ASSISTANT (Turn 1): (Executes the request)\n"
    "    -   USER (Turn 2): \"ok now what is the quality of that table?\"\n"
    "    -   YOUR CORRECT REASONING (Turn 2): \"The user is asking about 'that table'. The previous turn mentioned the `equipment` table in the `DEMO_Customer360_db` database. I will reuse these parameters.\"\n"
    "    -   YOUR CORRECT ACTION (Turn 2): `json ...` for `qlty_columnSummary` with `db_name`: `DEMO_Customer360_db` and `table_name`: `equipment`.\n\n"
    "--- **CRITICAL RULE: TOOL ARGUMENT ADHERENCE** ---\n"
    "You **MUST** use the exact parameter names provided in the tool definitions. Do not invent or guess parameter names.\n\n"
    "--- **CRITICAL RULE: SQL GENERATION** ---\n"
    "When using the `base_readQuery` tool, if you know the database name, you **MUST** use fully qualified table names in your SQL query (e.g., `SELECT ... FROM my_database.my_table`).\n\n"
    "{charting_instructions}\n\n"
    "--- **Response Formatting** ---\n"
    "-   **To execute a tool:** Respond with 'Thought:' explaining your choice, followed by a ```json ... ``` block with the `tool_name` and `arguments`.\n"
    "-   **To execute a prompt:** Respond with 'Thought:' explaining your choice, followed by a ```json ... ``` block with the `prompt_name` and `arguments`.\n"
    "-   **Clarifying Question:** Only ask if information is truly missing.\n\n"
    "{tools_context}\n\n"
    "{prompts_context}\n\n"
    "{charts_context}\n\n"
)

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

async def call_llm_api(prompt: str, session_id: str = None, chat_history=None) -> str:
    if not llm: raise RuntimeError("LLM is not initialized.")
    
    llm_logger = logging.getLogger("llm_conversation")
    try:
        full_log_message = ""
        history_for_llm = SESSIONS[session_id]['chat'].history if session_id and session_id in SESSIONS else chat_history
        if history_for_llm:
            formatted_lines = [f"[{msg.role}]: {msg.parts[0].text}" for msg in history_for_llm]
            formatted_history = "\n".join(formatted_lines)
            full_log_message += f"--- FULL CONTEXT (Session: {session_id or 'one-off'}) ---\n--- History ---\n{formatted_history}\n\n"
        
        full_log_message += f"--- Current User Prompt ---\n{prompt}\n"
        llm_logger.info(full_log_message)

        chat_session = SESSIONS[session_id]['chat'] if session_id else llm.start_chat(history=chat_history)
        response = await chat_session.send_message_async(prompt)

        if not response or not hasattr(response, 'text'):
            raise RuntimeError("LLM returned an empty or invalid response.")

        response_text = response.text.strip()
        llm_logger.info(f"--- RESPONSE ---\n{response_text}\n" + "-"*50 + "\n")
        return response_text
    except Exception as e:
        app.logger.error(f"Error calling LLM API: {e}", exc_info=True)
        llm_logger.error(f"--- ERROR in LLM call ---\n{e}\n" + "-"*50 + "\n")
        return None

async def invoke_mcp_tool(command: dict) -> any:
    global mcp_client
    if not mcp_client:
        return {"error": "MCP client is not connected."}

    tool_name = command.get("tool_name")
    args = command.get("arguments", command.get("parameters", {}))

    # Determine which server to call
    server_name = "teradata_mcp_server"
    if tool_name in mcp_charts:
        server_name = "chart_mcp_server"
    elif tool_name not in mcp_tools:
        return {"error": f"Tool '{tool_name}' not found in any connected server."}

    # Shim for legacy Teradata tools
    if server_name == "teradata_mcp_server":
        # Argument name normalization
        if 'database_name' in args: args['db_name'] = args.pop('database_name')
        if 'database' in args: args['db_name'] = args.pop('database')
        if 'table' in args: args['table_name'] = args.pop('table')

        # FIX: Shim for col_name vs column_name
        LEGACY_COL_NAME_TOOLS = [
            "qlty_distinctCategories", "qlty_standardDeviation",
            "qlty_univariateStatistics", "qlty_rowsWithMissingValues"
        ]
        if tool_name in LEGACY_COL_NAME_TOOLS:
            if 'column_name' in args and 'col_name' not in args:
                args['col_name'] = args.pop('column_name')

        # Table name concatenation shim
        LEGACY_TOOLS_MISSING_DB_PARAM = [
            "qlty_missingValues", "qlty_negativeValues", "qlty_distinctCategories",
            "qlty_standardDeviation", "qlty_columnSummary", "qlty_univariateStatistics",
            "qlty_rowsWithMissingValues"
        ]
        if tool_name in LEGACY_TOOLS_MISSING_DB_PARAM:
            db_name = args.get("db_name")
            table_name = args.get("table_name")
            if db_name and table_name and '.' not in table_name:
                args["table_name"] = f"{db_name}.{table_name}"
                if 'db_name' in args: del args["db_name"]
    
    try:
        app.logger.info(f"Creating temporary session on '{server_name}' to invoke tool '{tool_name}'")
        async with mcp_client.session(server_name) as temp_session:
            call_tool_result = await temp_session.call_tool(tool_name, args)
            app.logger.info(f"Successfully invoked tool '{tool_name}'. Raw response: {call_tool_result}")
            
            if hasattr(call_tool_result, 'content') and isinstance(call_tool_result.content, list) and len(call_tool_result.content) > 0:
                text_content = call_tool_result.content[0]
                if hasattr(text_content, 'text') and isinstance(text_content.text, str):
                    try:
                        parsed_json = json.loads(text_content.text)
                        # If the tool call was to the chart server, wrap the result
                        if server_name == "chart_mcp_server":
                            return {"type": "chart", "spec": parsed_json, "metadata": {"tool_name": tool_name}}
                        return parsed_json
                    except json.JSONDecodeError:
                        app.logger.error(f"Tool '{tool_name}' returned a string that is not valid JSON: {text_content.text}")
                        return {"error": "Tool returned non-JSON string", "data": text_content.text}
            
            raise RuntimeError(f"Unexpected tool result format for '{tool_name}': {call_tool_result}")
    except Exception as e:
        app.logger.error(f"Error during tool invocation for '{tool_name}': {e}", exc_info=True)
        return {"error": f"An exception occurred while invoking tool '{tool_name}'."}

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

        command = json.loads(json_match.group(1).strip())
        self.current_command = command
        
        if "prompt_name" in command:
            prompt_name = command.get("prompt_name")
            self.active_prompt_name = prompt_name
            arguments = command.get("arguments", command.get("parameters", {}))
            if 'db_name' in arguments and 'database_name' not in arguments: arguments['database_name'] = arguments.pop('db_name')

            if not mcp_client:
                raise RuntimeError("MCP client is not connected.")

            async with mcp_client.session("teradata_mcp_server") as temp_session:
                get_prompt_result = await temp_session.get_prompt(name=prompt_name, arguments=arguments)
            
            self.active_prompt_plan = get_prompt_result.content.text if hasattr(get_prompt_result, 'content') else str(get_prompt_result)

            yield _format_sse({"step": f"Executing Prompt: {prompt_name}", "details": self.active_prompt_plan, "prompt_name": prompt_name}, "prompt_selected")

            await self._get_next_action_from_llm()

        elif "tool_name" in command:
            self.state = AgentState.EXECUTING_TOOL
        else:
            self.state = AgentState.SUMMARIZING

    async def _execute_standard_tool(self):
        tool_name = self.current_command.get("tool_name")
        yield _format_sse({"step": f"Calling tool: {tool_name}", "details": self.current_command}, "tool_result")
        tool_result = await invoke_mcp_tool(self.current_command)
        
        if self.iteration_context:
            ctx = self.iteration_context
            current_item_name = ctx["items"][ctx["item_index"]]
            ctx["results_per_item"][current_item_name].append(tool_result)
            ctx["action_count_for_item"] += 1
        else:
            self.collected_data.append(tool_result)

        yield _format_sse({"step": "Tool execution finished", "details": tool_result, "tool_name": tool_name}, "tool_result")

        if self.active_prompt_plan and not self.iteration_context:
            plan_text = self.active_prompt_plan.lower()
            is_iterative_plan = any(keyword in plan_text for keyword in ["cycle through", "for each", "iterate"])
            
            if is_iterative_plan and tool_name == "base_tableList" and isinstance(tool_result, dict) and tool_result.get("status") == "success":
                items_to_iterate = [res.get("TableName") for res in tool_result.get("results", []) if res.get("TableName")]
                if items_to_iterate:
                    self.iteration_context = {
                        "items": items_to_iterate, "item_index": 0, "action_count_for_item": 0,
                        "results_per_item": {item: [] for item in items_to_iterate}
                    }
                    yield _format_sse({"step": "Starting Multi-Step Iteration", "details": f"Plan requires processing {len(items_to_iterate)} items."})
        
        tool_result_str = json.dumps({"tool_name": tool_name, "tool_output": tool_result})
        
        yield _format_sse({"step": "Thinking about the next action...", "details": "The agent is reasoning based on the current context."})
        await self._get_next_action_from_llm(tool_result_str=tool_result_str)

    async def _execute_column_iteration(self):
        base_command = self.current_command
        tool_name = base_command.get("tool_name")
        base_args = base_command.get("arguments", base_command.get("parameters", {}))
        db_name = base_args.get("db_name")
        table_name = base_args.get("table_name")

        # FIX: Check if a specific column was already provided by the LLM
        specific_column = base_args.get("col_name") or base_args.get("column_name")
        if specific_column:
            yield _format_sse({"step": f"Executing for specific column: {specific_column}", "details": base_command}, "tool_result")
            col_result = await invoke_mcp_tool(base_command)
            yield _format_sse({"step": f"Result for column: {specific_column}", "details": col_result, "tool_name": tool_name}, "tool_result")
            self.collected_data.append(col_result)
            tool_result_str = json.dumps({"tool_name": tool_name, "tool_output": col_result})
            yield _format_sse({"step": "Thinking about the next action...", "details": "Single column execution complete. Resuming main plan."})
            await self._get_next_action_from_llm(tool_result_str=tool_result_str)
            return # Exit the function after single execution

        # If no specific column, proceed with iteration logic
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

            yield _format_sse({"step": f"Executing for column: {col_name}", "details": iter_command}, "tool_result")
            col_result = await invoke_mcp_tool(iter_command)
            yield _format_sse({"step": f"Result for column: {col_name}", "details": col_result, "tool_name": tool_name}, "tool_result")
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
                "Based on the history, what is the next action to complete the user's request? "
                "If you have enough information, your response **MUST** start with `FINAL_ANSWER:`. "
                "Otherwise, provide the JSON for the next tool call."
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

        SESSIONS[self.session_id]['history'].append({'role': 'assistant', 'content': final_html})
        yield _format_sse({"final_answer": final_html}, "final_answer")
        self.state = AgentState.DONE

# --- Web Server Routes ---

@app.route("/")
async def index():
    return await render_template("index.html")

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
        return jsonify(SESSIONS[session_id]["history"])
    return jsonify({"error": "Session not found"}), 404

def get_full_system_prompt(base_prompt_text, charting_intensity_val):
    global tools_context, prompts_context, charts_context
    chart_instructions = CHARTING_INSTRUCTIONS.get(charting_intensity_val, CHARTING_INSTRUCTIONS['none'])
    
    final_charts_context = charts_context if APP_CONFIG.CHART_MCP_CONNECTED else CHARTING_INSTRUCTIONS['none']

    return base_prompt_text.format(
        charting_instructions=chart_instructions,
        tools_context=tools_context,
        prompts_context=prompts_context,
        charts_context=final_charts_context
    )

@app.route("/session", methods=["POST"])
async def new_session():
    global llm
    if not llm or not APP_CONFIG.TERADATA_MCP_CONNECTED:
        return jsonify({"error": "Application not configured. Please set MCP and LLM details in Config."}), 400
    
    data = await request.get_json()
    system_prompt_from_client = data.get("system_prompt")
    charting_intensity = data.get("charting_intensity", "none")
    
    try:
        session_id = str(uuid.uuid4())
        final_system_prompt = get_full_system_prompt(system_prompt_from_client, charting_intensity)
        
        initial_history = [
            {"role": "user", "parts": [{"text": final_system_prompt}]},
            {"role": "model", "parts": [{"text": "Understood. I will follow all instructions."}]}
        ]
        SESSIONS[session_id] = {
            "chat": llm.start_chat(history=initial_history),
            "history": [],
            "name": "New Chat",
            "created_at": datetime.now().isoformat()
        }
        app.logger.info(f"Created new session: {session_id} with charting: {charting_intensity}")
        return jsonify({"session_id": session_id, "name": SESSIONS[session_id]["name"]})
    except Exception as e:
        app.logger.error(f"Failed to create new session: {e}", exc_info=True)
        return jsonify({"error": "Failed to initialize a new chat session on the server."}), 500

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
            SESSIONS[session_id]['history'].append({'role': 'user', 'content': user_input})
            
            if SESSIONS[session_id]['name'] == 'New Chat':
                SESSIONS[session_id]['name'] = user_input[:40] + '...' if len(user_input) > 40 else user_input
                yield _format_sse({"session_name_update": {"id": session_id, "name": SESSIONS[session_id]['name']}}, "session_update")

            yield _format_sse({"step": "Assistant is thinking...", "details": "Analyzing request and selecting best action."})
            
            llm_reasoning_and_command = await call_llm_api(user_input, session_id)
            
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
        SESSIONS[session_id]['history'].append({'role': 'user', 'content': user_input})
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
            await SESSIONS[session_id]['chat'].send_message_async(user_input)
            
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


# --- MODIFICATION START: Replaced the entire function with the logic from the working file ---
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
        
        # Categorize Tools for UI
        tool_list_for_prompt = "\n".join([f"- {tool.name}: {tool.description}" for tool in loaded_tools])
        categorization_prompt = (
            "You are a helpful assistant that organizes lists of technical tools for a **Teradata database system** into logical categories for a user interface. "
            "Your response MUST be a single, valid JSON object. The keys should be the category names, "
            "and the values should be an array of tool names belonging to that category.\n\n"
            f"--- Tool List ---\n{tool_list_for_prompt}"
        )
        categorized_tools_str = await call_llm_api(categorization_prompt)
        try:
            cleaned_str = re.search(r'\{.*\}', categorized_tools_str, re.DOTALL).group(0)
            categorized_tools = json.loads(cleaned_str)
            structured_tools = {category: [{"name": name, "description": mcp_tools[name].description} for name in tool_names if name in mcp_tools] for category, tool_names in categorized_tools.items()}
        except Exception as e:
            app.logger.warning(f"Could not categorize tools for UI. Falling back. Error: {e}")
            structured_tools = {"All Tools": [{"name": tool.name, "description": tool.description} for tool in loaded_tools]}

        # Load Resources
        loaded_resources = await load_mcp_resources(temp_session)
        if loaded_resources:
            mcp_resources = {res.name: res for res in loaded_resources}
            resource_list_for_prompt = "\n".join([f"- {res.name}: {res.description}" for res in loaded_resources])
            categorization_prompt_for_resources = (
                "You are a helpful assistant that organizes a list of Teradata database resources (like databases or tables) into logical categories. "
                "Your response MUST be a single, valid JSON object.\n\n"
                f"--- Resource List ---\n{resource_list_for_prompt}"
            )
            categorized_resources_str = await call_llm_api(categorization_prompt_for_resources)
            try:
                cleaned_str = re.search(r'\{.*\}', categorized_resources_str, re.DOTALL).group(0)
                categorized_resources = json.loads(cleaned_str)
                structured_resources = {category: [{"name": name, "description": mcp_resources[name].description} for name in res_names if name in mcp_resources] for category, res_names in categorized_resources.items()}
            except Exception as e:
                app.logger.warning(f"Could not categorize resources. Falling back. Error: {e}")
                structured_resources = {"All Resources": [{"name": res.name, "description": res.description} for res in loaded_resources]}

        # Load and Process Prompts
        loaded_prompts = []
        try:
            list_prompts_result = await temp_session.list_prompts()
            loaded_prompts = list_prompts_result.prompts
            app.logger.info(f"Successfully loaded {len(loaded_prompts)} prompts.")
        except Exception as e:
            app.logger.warning(f"WARNING: Could not load prompts from MCP server. Error: {e}")

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
            categorized_prompts_str = await call_llm_api(categorization_prompt_for_prompts)
            try:
                cleaned_str = re.search(r'\{.*\}', categorized_prompts_str, re.DOTALL).group(0)
                categorized_prompts = json.loads(cleaned_str)
                structured_prompts = {category: [p for p in serializable_prompts if p['name'] in prompt_names] for category, prompt_names in categorized_prompts.items()}
            except Exception as e:
                app.logger.warning(f"Could not categorize prompts. Falling back. Error: {e}")
                structured_prompts = {"All Prompts": serializable_prompts}
        else:
            prompts_context = "--- No Prompts Available ---"
            structured_prompts = {}
# --- MODIFICATION END ---


async def load_and_categorize_chart_resources():
    global charts_context, structured_charts, mcp_charts
    if not mcp_client or not APP_CONFIG.CHART_MCP_CONNECTED:
        raise Exception("Chart MCP Client not initialized or connected.")

    async with mcp_client.session("chart_mcp_server") as temp_session:
        app.logger.info("--- Loading Chart tools... ---")
        loaded_charts = await load_mcp_tools(temp_session)
        # Add diagnostic logging
        app.logger.info(f"Found {len(loaded_charts)} chart tools from the chart server.")
        if not loaded_charts:
            app.logger.warning("The chart server returned 0 tools. Please ensure it is configured correctly.")

        mcp_charts = {tool.name: tool for tool in loaded_charts}
        charts_context = "--- Available Chart Tools ---\n" + "\n".join([f"- `{tool.name}`: {tool.description}" for tool in loaded_charts])
        
        if not loaded_charts:
            structured_charts = {}
            return

        chart_list_for_prompt = "\n".join([f"- {tool.name}: {tool.description}" for tool in loaded_charts])
        categorization_prompt = (
            "You are a helpful assistant that organizes lists of charting tools into logical categories for a user interface. "
            "Your response MUST be a single, valid JSON object.\n\n"
            f"--- Chart Tool List ---\n{chart_list_for_prompt}"
        )
        categorized_charts_str = await call_llm_api(categorization_prompt)
        try:
            cleaned_str = re.search(r'\{.*\}', categorized_charts_str, re.DOTALL).group(0)
            categorized_charts = json.loads(cleaned_str)
            structured_charts = {category: [{"name": name, "description": mcp_charts[name].description} for name in tool_names if name in mcp_charts] for category, tool_names in categorized_charts.items()}
        except Exception as e:
            app.logger.warning(f"Could not categorize charts for UI. Error: {e}")
            structured_charts = {"All Charts": [{"name": tool.name, "description": tool.description} for tool in loaded_charts]}

@app.route("/system_prompt/<model_name>", methods=["GET"])
async def get_default_system_prompt(model_name):
    charting_intensity_val = request.args.get("charting_intensity", "medium")
    final_prompt = get_full_system_prompt(BASE_SYSTEM_PROMPT, charting_intensity_val)
    return jsonify({"status": "success", "system_prompt": final_prompt})

@app.route("/models", methods=["POST"])
async def get_models():
    try:
        data = await request.get_json()
        api_key = data.get("apiKey")
        genai.configure(api_key=api_key)
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        clean_models = [name.split('/')[-1] for name in models]
        structured_model_list = [{"name": name, "certified": APP_CONFIG.MODELS_UNLOCKED or name == CERTIFIED_MODEL} for name in clean_models]
        return jsonify({"status": "success", "models": structured_model_list})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/configure", methods=["POST"])
async def configure_services():
    global mcp_client, llm, SERVER_CONFIGS
    data = await request.get_json()
    
    try:
        # LLM Config
        genai.configure(api_key=data.get("apiKey"))
        llm = genai.GenerativeModel(data.get("model"))
        await llm.generate_content_async("test")
        
        # Teradata MCP Config
        mcp_server_url = f"http://{data.get('host')}:{data.get('port')}{data.get('path')}"
        SERVER_CONFIGS = {'teradata_mcp_server': {"url": mcp_server_url, "transport": "streamable_http"}}
        
        mcp_client = MultiServerMCPClient(SERVER_CONFIGS)
        
        await load_and_categorize_teradata_resources()
        APP_CONFIG.TERADATA_MCP_CONNECTED = True
        return jsonify({"status": "success", "message": "Teradata MCP and LLM configured successfully."})
    except Exception as e:
        llm = None
        mcp_client = None
        SERVER_CONFIGS = {}
        APP_CONFIG.TERADATA_MCP_CONNECTED = False
        app.logger.error(f"Configuration failed: {e}", exc_info=True)
        return jsonify({"status": "error", "message": f"Configuration failed: {e}"}), 500

@app.route("/configure_chart", methods=["POST"])
async def configure_chart_service():
    global mcp_client, SERVER_CONFIGS
    if not APP_CONFIG.TERADATA_MCP_CONNECTED:
        return jsonify({"status": "error", "message": "Main MCP client not configured. Please connect to Teradata & LLM first."}), 400
    
    data = await request.get_json()
    try:
        chart_server_url = f"http://{data.get('chart_host')}:{data.get('chart_port')}{data.get('chart_path')}"
        SERVER_CONFIGS['chart_mcp_server'] = {"url": chart_server_url, "transport": "streamable_http"}
        
        # Re-initialize the client with both server configurations
        mcp_client = MultiServerMCPClient(SERVER_CONFIGS)
        
        APP_CONFIG.CHART_MCP_CONNECTED = True
        
        # Test connection by loading resources from the chart server
        await load_and_categorize_chart_resources()
        
        return jsonify({"status": "success", "message": "Chart MCP server configured successfully."})
    except Exception as e:
        # If chart connection fails, remove it from configs and re-initialize client with just Teradata
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
        "--unlock-models",
        action="store_true",
        help="Allow selection of all available models, not just the certified one."
    )
    args = parser.parse_args()

    if args.unlock_models:
        APP_CONFIG.MODELS_UNLOCKED = True
        print("\n--- DEVELOPMENT MODE: All models will be selectable. ---")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    templates_dir = os.path.join(script_dir, 'templates')
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer shut down.")
