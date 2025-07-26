# web_client.py
import asyncio
import json
import os
import sys
import re
import uuid
import logging
import shutil
import argparse # Added for command-line arguments
from quart import Quart, request, jsonify, render_template, Response
from quart_cors import cors
import hypercorn.asyncio
from hypercorn.config import Config
from dotenv import load_dotenv
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

APP_CONFIG = AppConfig()
CERTIFIED_MODEL = "gemini-1.5-flash"

tools_context = None
prompts_context = None
llm = None
mcp_client = None
mcp_tools = {}
mcp_prompts = {}
structured_tools = {}
structured_prompts = {}
structured_resources = {}
tool_scopes = {}

TOOL_COLUMN_TYPE_REQUIREMENTS = {
    "qlty_univariateStatistics": ["INTEGER", "SMALLINT", "BIGINT", "DECIMAL", "FLOAT", "NUMBER", "BYTEINT"],
}

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

    def render(self) -> str:
        final_html = ""
        clean_summary_html = self._sanitize_summary()
        if clean_summary_html:
            final_html += f'<div class="response-card summary-card">{clean_summary_html}</div>'

        for i, tool_result in enumerate(self.collected_data):
            if i in self.processed_data_indices or not isinstance(tool_result, dict):
                continue
            metadata = tool_result.get("metadata", {})
            tool_name = metadata.get("tool_name")
            if tool_name == 'base_tableDDL':
                final_html += self._render_ddl(tool_result, i)
            elif tool_name in ['base_tablePreview', 'qlty_columnSummary', 'base_tableList', 'base_columnDescription', 'read_query_sqlalchemy']:
                final_html += self._render_table(tool_result, i, f"Query Result")
            elif "results" in tool_result:
                 final_html += self._render_table(tool_result, i, f"Result for {tool_name}")

        if not final_html.strip():
            return "<p>The agent completed its work but did not produce a visible output.</p>"
        return final_html

async def call_llm_api(prompt: str, session_id: str = None, chat_history=None) -> str:
    if not llm: raise RuntimeError("LLM is not initialized. Please configure the LLM provider in the settings.")
    
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
            app.logger.error("LLM returned an empty or invalid response.")
            llm_logger.error("--- RESPONSE (Empty or Invalid) ---\n" + "-"*50 + "\n")
            return "Error: The language model returned an empty response."

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
        return {"error": "MCP client is not connected. Please configure the connection first."}

    tool_name = command.get("tool_name")
    args = command.get("arguments", command.get("parameters", {}))

    if 'database_name' in args: args['db_name'] = args.pop('database_name')
    if 'database' in args: args['db_name'] = args.pop('database')
    if 'table' in args: args['table_name'] = args.pop('table')

    LEGACY_TOOLS_MISSING_DB_PARAM = [
        "qlty_missingValues", "qlty_negativeValues", "qlty_distinctCategories",
        "qlty_standardDeviation", "qlty_columnSummary", "qlty_univariateStatistics",
        "qlty_rowsWithMissingValues"
    ]

    if tool_name in LEGACY_TOOLS_MISSING_DB_PARAM:
        app.logger.warning(f"Applying legacy tool shim for '{tool_name}'.")
        db_name = args.get("db_name")
        table_name = args.get("table_name")
        if db_name and table_name and '.' not in table_name:
            args["table_name"] = f"{db_name}.{table_name}"
            app.logger.info(f"Shim modified 'table_name' to: '{args['table_name']}'")
            if 'db_name' in args: del args["db_name"]
    
    try:
        app.logger.info(f"Creating temporary session to invoke tool '{tool_name}'")
        async with mcp_client.session("mcp_server") as temp_session:
            call_tool_result = await temp_session.call_tool(tool_name, args)
            app.logger.info(f"Successfully invoked tool. Raw response: {call_tool_result}")
            
            if hasattr(call_tool_result, 'content') and isinstance(call_tool_result.content, list) and len(call_tool_result.content) > 0:
                text_content = call_tool_result.content[0]
                if hasattr(text_content, 'text') and isinstance(text_content.text, str):
                    try:
                        return json.loads(text_content.text)
                    except json.JSONDecodeError:
                        app.logger.error(f"Tool '{tool_name}' returned a string that is not valid JSON: {text_content.text}")
                        return {"error": "Tool returned non-JSON string", "data": text_content.text}
            
            app.logger.error(f"Unexpected tool result format for '{tool_name}': {call_tool_result}")
            return {"error": "Unexpected tool result format from MCP server."}
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
                yield _format_sse({"error": "MCP client is not connected."}, "error")
                self.state = AgentState.ERROR
                return

            async with mcp_client.session("mcp_server") as temp_session:
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
            
            if is_iterative_plan and tool_name == "base_tableList" and tool_result.get("status") == "success":
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

        yield _format_sse({"step": f"Column tool detected: {tool_name}", "details": "Fetching column list to begin iteration."})

        cols_command = {"tool_name": "base_columnDescription", "arguments": {"db_name": db_name, "obj_name": table_name}}
        cols_result = await invoke_mcp_tool(cols_command)

        if not (cols_result and cols_result.get('status') == 'success' and cols_result.get('results')):
            raise ValueError(f"Failed to retrieve column list for iteration. Response: {cols_result}")
        
        all_columns = cols_result.get('results', [])
        
        columns_to_iterate = all_columns
        required_types = TOOL_COLUMN_TYPE_REQUIREMENTS.get(tool_name)
        if required_types:
            yield _format_sse({"step": "Filtering columns", "details": f"Tool {tool_name} requires types: {required_types}. Applying filter."})
            columns_to_iterate = [col for col in all_columns if col.get("CType") in required_types]
            yield _format_sse({"step": "Column list filtered", "details": f"Found {len(columns_to_iterate)} compatible columns to process."})
        else:
            yield _format_sse({"step": "Column list retrieved", "details": f"Found {len(all_columns)} columns to process."})

        all_column_results = []
        for column_info in columns_to_iterate:
            col_name = column_info.get("ColumnName")
            iter_args = base_args.copy()
            iter_args['col_name'] = col_name
            
            LEGACY_TOOLS_MISSING_DB_PARAM = [
                "qlty_missingValues", "qlty_negativeValues", "qlty_distinctCategories",
                "qlty_standardDeviation", "qlty_columnSummary", "qlty_univariateStatistics",
                "qlty_rowsWithMissingValues"
            ]
            if tool_name in LEGACY_TOOLS_MISSING_DB_PARAM:
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

@app.route("/session", methods=["POST"])
async def new_session():
    global llm, tools_context, prompts_context
    if not llm or not mcp_client:
        return jsonify({"error": "Application not configured. Please set MCP and LLM details in Config."}), 400
    try:
        session_id = str(uuid.uuid4())
        system_prompt = (
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
            "You **MUST** use the exact parameter names provided in the tool definitions. Do not invent or guess parameter names. For example, if a tool is defined to accept a parameter named `sql`, you MUST use `\"sql\": \"...\"` in your JSON. Using a guess like `\"query\": \"...\"` or `\"sql_query\": \"...\"` will fail.\n\n"
            "--- **CRITICAL RULE: SQL GENERATION** ---\n"
            "When using the `base_readQuery` tool, if you know the database name, you **MUST** use fully qualified table names in your SQL query (e.g., `SELECT ... FROM my_database.my_table`). Do **NOT** pass the database name as a separate `db_name` argument to `base_readQuery`.\n\n"
            "--- **Response Formatting** ---\n"
            "-   **To execute a tool:** Respond with 'Thought:' explaining your choice, followed by a ```json ... ``` block with the `tool_name` and `arguments`.\n"
            "-   **To execute a prompt:** Respond with 'Thought:' explaining your choice, followed by a ```json ... ``` block with the `prompt_name` and `arguments`.\n"
            "-   **Clarifying Question:** Only ask if information is truly missing.\n\n"
            f"{tools_context}\n\n"
            f"{prompts_context}\n\n"
        )
        initial_history = [
            {"role": "user", "parts": [{"text": system_prompt}]},
            {"role": "model", "parts": [{"text": "Understood. I will follow all instructions, paying special attention to context, parameter inference, tool arguments, and SQL generation rules."}]}
        ]
        SESSIONS[session_id] = {
            "chat": llm.start_chat(history=initial_history),
            "history": [],
            "name": "New Chat",
            "created_at": datetime.now().isoformat()
        }
        app.logger.info(f"Created new session: {session_id}")
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

        app.logger.info(f"Received stream request for session {session_id}: {user_input}")

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


async def load_and_categorize_resources():
    global tools_context, structured_tools, structured_prompts, prompts_context, mcp_tools, mcp_prompts, tool_scopes
    
    if not mcp_client:
        raise Exception("MCP Client not initialized.")

    async with mcp_client.session("mcp_server") as temp_session:
        app.logger.info("--- MCP CLIENT SESSION ACTIVE (for loading) ---")
        
        app.logger.info("--- Loading tools, prompts, and resources from MCP server... ---")
        
        loaded_tools = await load_mcp_tools(temp_session)
        app.logger.info(f"Successfully loaded {len(loaded_tools)} tools.")
        if not loaded_tools:
            raise Exception("No tools were loaded from the server.")

        mcp_tools = {tool.name: tool for tool in loaded_tools}
        
        app.logger.info("\n--- Pre-classifying tool scopes using deterministic logic ---")
        tool_scopes = classify_tool_scopes(loaded_tools)
        app.logger.info("Successfully classified tool scopes.")
        
        scoped_tools_categorized = {"database": [], "table": [], "column": []}
        for tool in loaded_tools:
            scope = tool_scopes.get(tool.name, "table")
            scoped_tools_categorized[scope].append(f"- `{tool.name}`: {tool.description}")

        tools_context = "--- Available Tools by Scope ---\n\n"
        if scoped_tools_categorized["database"]: tools_context += "## Database Level Tools\n" + "\n".join(scoped_tools_categorized["database"]) + "\n\n"
        if scoped_tools_categorized["table"]: tools_context += "## Table Level Tools\n" + "\n".join(scoped_tools_categorized["table"]) + "\n\n"
        if scoped_tools_categorized["column"]: tools_context += "## Column Level Tools\n" + "\n".join(scoped_tools_categorized["column"]) + "\n\n"

        loaded_prompts = []
        try:
            list_prompts_result = await temp_session.list_prompts()
            loaded_prompts = list_prompts_result.prompts
            app.logger.info(f"Successfully loaded {len(loaded_prompts)} prompts.")
        except Exception as e:
            app.logger.warning(f"WARNING: Could not load prompts. Error: {e}")

        if loaded_prompts:
            mcp_prompts = {prompt.name: prompt for prompt in loaded_prompts}
            prompt_scopes = classify_prompt_scopes(loaded_prompts)
            scoped_prompts_categorized = {"database": [], "table": [], "general": []}
            for prompt in loaded_prompts:
                scope = prompt_scopes.get(prompt.name, "general")
                scoped_prompts_categorized[scope].append(f"- `{prompt.name}`: {prompt.description}")
            
            prompts_context = "--- Available Prompts by Scope ---\n\n"
            if scoped_prompts_categorized["database"]: prompts_context += "## Database Level Prompts\n" + "\n".join(scoped_prompts_categorized["database"]) + "\n\n"
            if scoped_prompts_categorized["table"]: prompts_context += "## Table Level Prompts\n" + "\n".join(scoped_prompts_categorized["table"]) + "\n\n"
            if scoped_prompts_categorized["general"]: prompts_context += "## General Prompts\n" + "\n".join(scoped_prompts_categorized["general"]) + "\n\n"
        else:
            prompts_context = "--- No Prompts Available ---"

        app.logger.info("\n--- Categorizing tools for UI using the LLM ---")
        tool_list_for_prompt = "\n".join([f"- {tool.name}: {tool.description}" for tool in loaded_tools])
        
        categorization_prompt = (
            "You are a helpful assistant that organizes lists of technical tools for a **Teradata database system** into logical categories for a user interface. "
            "Your response MUST be a single, valid JSON object. The keys should be the category names, "
            "and the values should be an array of tool names belonging to that category.\n\n"
            "Example Format:\n"
            "{\n  \"Database & Schema\": [\"tool_db_create\"],\n  \"Data Quality\": [\"qlty_columnSummary\"]\n}\n\n"
            "--- Tool List ---\n"
            f"{tool_list_for_prompt}"
        )
        
        categorized_tools_str = await call_llm_api(categorization_prompt)
        try:
            cleaned_str = re.search(r'\{.*\}', categorized_tools_str, re.DOTALL).group(0)
            categorized_tools = json.loads(cleaned_str)
            
            structured_tools = {category: [{"name": name, "description": mcp_tools[name].description} for name in tool_names if name in mcp_tools] for category, tool_names in categorized_tools.items()}
            app.logger.info("Successfully categorized tools for UI.")
        except Exception as e:
            app.logger.warning(f"Warning: Could not categorize tools for UI. Error: {e}")
            structured_tools = {"All Tools": [{"name": tool.name, "description": tool.description} for tool in loaded_tools]}
        
        if loaded_prompts:
            app.logger.info("\n--- Categorizing prompts using the LLM ---")
            serializable_prompts = [{"name": p.name, "description": p.description, "arguments": [arg.model_dump() for arg in p.arguments]} for p in loaded_prompts]
            prompt_list_for_prompt = "\n".join([f"- {p['name']}: {p['description']}" for p in serializable_prompts])
            
            categorization_prompt_for_prompts = (
                "You are a helpful assistant that organizes lists of technical prompts for a **Teradata database system** into logical categories for a user interface. "
                "Your response MUST be a single, valid JSON object. The keys should be the category names, "
                "and the values should be an array of prompt names belonging to that category.\n\n"
                "Example Format:\n"
                "{\n  \"Database Analysis\": [\"dba_databaseLineage\"],\n  \"Impact Analysis\": [\"dba_tableDropImpact\"]\n}\n\n"
                "--- Prompt List ---\n"
                f"{prompt_list_for_prompt}"
            )
            categorized_prompts_str = await call_llm_api(categorization_prompt_for_prompts)
            try:
                cleaned_str = re.search(r'\{.*\}', categorized_prompts_str, re.DOTALL).group(0)
                categorized_prompts = json.loads(cleaned_str)
                structured_prompts = {category: [p for p in serializable_prompts if p['name'] in prompt_names] for category, prompt_names in categorized_prompts.items()}
                app.logger.info("Successfully categorized prompts.")
            except Exception as e:
                app.logger.warning(f"Warning: Could not categorize prompts. Error: {e}")
                structured_prompts = { "All Prompts": serializable_prompts }


# --- NEW AND MODIFIED ENDPOINTS ---

async def get_models_for_provider(provider: str, api_key: str):
    """Fetches a list of available models for a given provider."""
    if provider == "Google":
        if not api_key:
            raise ValueError("API Key is required to fetch Google models.")
        try:
            genai.configure(api_key=api_key)
            models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            clean_models = [name.split('/')[-1] for name in models]
            
            # Structure the response based on the lock status
            structured_model_list = []
            for model_name in clean_models:
                is_certified = APP_CONFIG.MODELS_UNLOCKED or model_name == CERTIFIED_MODEL
                structured_model_list.append({"name": model_name, "certified": is_certified})
            
            return structured_model_list
        except Exception as e:
            app.logger.error(f"Failed to fetch Google models: {e}")
            if "API key not valid" in str(e):
                raise ValueError("The provided Google API Key is not valid.")
            raise ConnectionError("Could not connect to Google API to fetch models.")
    else:
        raise NotImplementedError(f"Provider '{provider}' is not yet supported.")

@app.route("/models", methods=["POST"])
async def get_models():
    """Endpoint to dynamically fetch models for a provider."""
    try:
        data = await request.get_json()
        provider = data.get("provider")
        api_key = data.get("apiKey")
        models = await get_models_for_provider(provider, api_key)
        return jsonify({"status": "success", "models": models})
    except (ValueError, NotImplementedError, ConnectionError) as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        app.logger.error(f"Unexpected error fetching models: {e}", exc_info=True)
        return jsonify({"status": "error", "message": "An unexpected server error occurred."}), 500


async def initialize_llm(provider: str, model: str, api_key: str):
    """Initializes the global LLM client based on provider and credentials."""
    global llm
    if provider == "Google":
        if not api_key:
            raise ValueError("API Key is required for Google provider.")
        try:
            genai.configure(api_key=api_key)
            temp_llm = genai.GenerativeModel(model)
            # Test the connection with a simple, non-costly call
            await temp_llm.generate_content_async("test", generation_config={"max_output_tokens": 5})
            llm = temp_llm # Assign to global only after successful test
            app.logger.info(f"Successfully initialized and tested Google LLM model: {model}")
            return True, "LLM connection successful."
        except Exception as e:
            llm = None
            app.logger.error(f"Failed to initialize Google LLM: {e}", exc_info=True)
            # Try to extract a more user-friendly error message
            error_str = str(e)
            if "API key not valid" in error_str:
                raise ValueError("The provided Google API Key is not valid. Please check and try again.")
            raise ValueError(f"LLM connection failed: {e}")
    else:
        raise NotImplementedError(f"Provider '{provider}' is not yet supported.")

@app.route("/configure", methods=["POST"])
async def configure_services():
    """A unified endpoint to configure both MCP and LLM services."""
    global mcp_client, llm
    data = await request.get_json()
    
    # --- LLM Configuration ---
    llm_provider = data.get("provider")
    llm_model = data.get("model")
    llm_api_key = data.get("apiKey")
    
    try:
        await initialize_llm(llm_provider, llm_model, llm_api_key)
    except (ValueError, NotImplementedError) as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": f"An unexpected error occurred during LLM initialization: {e}"}), 500

    # --- MCP Configuration ---
    mcp_host = data.get("host", "127.0.0.1")
    mcp_port = data.get("port", "8001")
    mcp_path = data.get("path", "/mcp/")
    
    mcp_server_url = f"http://{mcp_host}:{mcp_port}{mcp_path}"
    mcp_client = MultiServerMCPClient({"mcp_server": {"url": mcp_server_url, "transport": "streamable_http"}})
    
    try:
        await load_and_categorize_resources()
        return jsonify({"status": "success", "message": "MCP and LLM configured successfully."})
    except Exception as e:
        app.logger.error(f"Failed to load MCP resources after LLM init: {e}", exc_info=True)
        # Reset LLM since the full config failed
        llm = None
        mcp_client = None
        return jsonify({"status": "error", "message": f"LLM connection succeeded, but failed to load MCP resources: {e}"}), 500


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
    # --- Command-Line Argument Parser ---
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
