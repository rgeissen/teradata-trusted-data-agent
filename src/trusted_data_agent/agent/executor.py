# trusted_data_agent/agent/executor.py
import re
import json
import logging
from enum import Enum, auto
from datetime import datetime, timedelta

from trusted_data_agent.agent.formatter import OutputFormatter
from trusted_data_agent.core import session_manager
from trusted_data_agent.mcp import adapter as mcp_adapter
from trusted_data_agent.llm import handler as llm_handler

app_logger = logging.getLogger("quart.app")

class AgentState(Enum):
    DECIDING = auto()
    EXECUTING_TOOL = auto()
    SUMMARIZING = auto()
    DONE = auto()
    ERROR = auto()

def _format_sse(data: dict, event: str = None) -> str:
    msg = f"data: {json.dumps(data)}\n"
    if event is not None:
        msg += f"event: {event}\n"
    return f"{msg}\n"

def _evaluate_inline_math(json_str: str) -> str:
    math_expr_pattern = re.compile(r'\b(\d+\.?\d*)\s*([+\-*/])\s*(\d+\.?\d*)\b')
    while True:
        match = math_expr_pattern.search(json_str)
        if not match: break
        num1_str, op, num2_str = match.groups()
        try:
            num1, num2 = float(num1_str), float(num2_str)
            result = 0
            if op == '+': result = num1 + num2
            elif op == '-': result = num1 - num2
            elif op == '*': result = num1 * num2
            elif op == '/': result = num1 / num2
            json_str = json_str.replace(match.group(0), str(result), 1)
        except (ValueError, ZeroDivisionError):
            break
    return json_str

class PlanExecutor:
    def __init__(self, session_id: str, initial_instruction: str, original_user_input: str, dependencies: dict):
        self.session_id = session_id
        self.original_user_input = original_user_input
        self.state = AgentState.DECIDING
        self.next_action_str = initial_instruction
        self.collected_data = []
        self.max_steps = 40
        self.active_prompt_plan = None
        self.active_prompt_name = None
        self.current_command = None
        self.dependencies = dependencies
        self.tool_constraints_cache = {}
        self.globally_failed_tools = set()
        self.is_workflow = False
        # --- AUTHORITATIVE CONTEXT: Initialize the context stack and structured data collection ---
        self.context_stack = []
        self.structured_collected_data = {}


    # --- AUTHORITATIVE CONTEXT: New helper to classify the operational level of a tool ---
    def _get_context_level_for_tool(self, tool_name: str) -> str | None:
        """Determines if a tool operates at a 'database' or 'table' level."""
        tool_scopes = self.dependencies['STATE'].get('tool_scopes', {})
        scope = tool_scopes.get(tool_name)
        if scope in ['database', 'table', 'column']:
            return 'table' if scope in ['table', 'column'] else 'database'
        return None

    # --- AUTHORITATIVE CONTEXT: New method to authoritatively manage the context stack ---
    def _update_and_manage_context_stack(self, command: dict, tool_result: dict | None = None):
        """
        Authoritatively manages the context stack based on LLM commands and tool results.
        It handles pushing, popping, and updating contexts to track loop states.
        """
        tool_name = command.get("tool_name")
        args = command.get("arguments", {})
        tool_level = self._get_context_level_for_tool(tool_name)

        # --- Pop contexts if the new command operates at a higher level ---
        while self.context_stack:
            top_context = self.context_stack[-1]
            top_level = top_context['type'].split('_name')[0] # 'database_name' -> 'database'
            if tool_level == 'database' and top_level == 'table':
                popped = self.context_stack.pop()
                app_logger.info(f"CONTEXT STACK: Popped '{popped['type']}' context as inner loop appears complete.")
            else:
                break
        
        # --- Establish a new loop context if a list-generating tool was just run ---
        if tool_result and tool_result.get("status") == "success":
            results = tool_result.get("results", [])
            if isinstance(results, list) and results and isinstance(results[0], dict):
                first_item_keys = results[0].keys()
                new_context_type = None
                list_values = []

                if "TableName" in first_item_keys:
                    new_context_type = "table_name"
                    list_values = [item["TableName"] for item in results]
                elif "DatabaseName" in first_item_keys:
                    new_context_type = "database_name"
                    list_values = [item["DatabaseName"] for item in results]
                
                if new_context_type and list_values:
                    app_logger.info(f"CONTEXT STACK: Establishing new loop context for '{new_context_type}' with {len(list_values)} items.")
                    self.context_stack.append({
                        'type': new_context_type,
                        'list': list_values,
                        'index': -1 # Start at -1, will be advanced to 0 on first use
                    })

    # --- AUTHORITATIVE CONTEXT: New method to apply guardrails before execution ---
    def _apply_context_guardrail(self, command: dict) -> tuple[dict, list]:
        """
        Corrects or injects parameters into a command based on the authoritative context stack.
        """
        if not self.is_workflow or not self.context_stack:
            return command, []

        corrected_command = command.copy()
        corrected_command['arguments'] = corrected_command.get('arguments', {}).copy()
        args = corrected_command['arguments']
        events_to_yield = []
        
        param_aliases = {
            "database_name": ["db_name"],
            "table_name": ["tbl_name", "object_name", "obj_name"]
        }

        # --- Advance the loop index if the LLM signals to do so ---
        if self.context_stack and 'list' in self.context_stack[-1]:
            top_context = self.context_stack[-1]
            context_type = top_context['type']
            aliases = param_aliases.get(context_type, [])
            
            llm_provided_value = None
            for key in [context_type] + aliases:
                if key in args:
                    llm_provided_value = args[key]
                    break
            
            if llm_provided_value:
                current_index = top_context['index']
                next_index = current_index + 1
                if next_index < len(top_context['list']) and llm_provided_value == top_context['list'][next_index]:
                    app_logger.info(f"CONTEXT STACK: Advancing loop for '{context_type}' to index {next_index} ('{llm_provided_value}').")
                    top_context['index'] = next_index

        # --- Forcefully inject/correct parameters from the stack's authoritative state ---
        for context in self.context_stack:
            context_type = context['type']
            aliases = param_aliases.get(context_type, [])
            
            # Determine the correct value from the authoritative context
            correct_value = None
            if 'list' in context:
                if context['index'] == -1: context['index'] = 0 # Handle first item in a new loop
                if 0 <= context['index'] < len(context['list']):
                    correct_value = context['list'][context['index']]
            
            if not correct_value: continue # This context isn't a loop or is finished

            # Find which key the LLM used, if any
            found_key = None
            llm_provided_value = None
            for key in [context_type] + aliases:
                if key in args:
                    found_key = key
                    llm_provided_value = args[key]
                    break
            
            # If the key is missing or the value is wrong, OVERRIDE it
            if not found_key or llm_provided_value != correct_value:
                key_to_set = found_key or context_type
                args[key_to_set] = correct_value
                details = f"LLM provided '{llm_provided_value}' for {key_to_set}. System corrected to authoritative value '{correct_value}'."
                app_logger.warning(f"GUARDRAIL APPLIED: {details}")
                events_to_yield.append(_format_sse({
                    "step": "System Correction", "details": details, "type": "workaround"
                }))

        return corrected_command, events_to_yield
    
    # --- AUTHORITATIVE CONTEXT: New method to add data to the structured collection ---
    def _add_to_structured_data(self, tool_result: dict):
        """Adds tool results to a nested dictionary based on the current context."""
        if not self.context_stack:
            self.collected_data.append(tool_result)
            return

        # Create a key based on the current context stack (e.g., "DB_A > Table1")
        context_key = " > ".join([ctx['list'][ctx['index']] for ctx in self.context_stack if 'list' in ctx and ctx['index'] != -1])
        
        if not context_key:
             self.collected_data.append(tool_result)
             return

        if context_key not in self.structured_collected_data:
            self.structured_collected_data[context_key] = []
        
        self.structured_collected_data[context_key].append(tool_result)
        app_logger.info(f"Added tool result to structured data under key: '{context_key}'")


    async def run(self):
        for i in range(self.max_steps):
            if self.state in [AgentState.DONE, AgentState.ERROR]:
                break
            try:
                if self.state == AgentState.DECIDING:
                    yield _format_sse({"step": "Assistant has decided on an action", "details": self.next_action_str}, "llm_thought")
                    async for event in self._handle_deciding():
                        yield event
                
                elif self.state == AgentState.EXECUTING_TOOL:
                    is_range_candidate, date_param_name = self._is_date_query_candidate()
                    if is_range_candidate:
                        query_type, date_phrase = await self._classify_date_query_type()
                        if query_type == 'range':
                            async for event in self._execute_date_range_orchestrator(date_param_name, date_phrase):
                                yield event
                            continue
                    
                    async for event in self._intercept_and_correct_command():
                        yield event
                    
                    tool_name = self.current_command.get("tool_name")
                    if tool_name in self.globally_failed_tools:
                        yield _format_sse({
                            "step": f"Skipping Globally Failed Tool: {tool_name}",
                            "details": f"The tool '{tool_name}' has been identified as non-functional for this session and will be skipped.",
                            "type": "workaround"
                        })
                        tool_result_str = json.dumps({ "tool_name": tool_name, "tool_output": { "status": "error", "error_message": "Skipped because tool is globally non-functional." } })
                        async for event_in_skip in self._get_next_action_from_llm(tool_result_str=tool_result_str):
                            yield event_in_skip
                        continue

                    tool_scopes = self.dependencies['STATE'].get('tool_scopes', {})
                    if tool_scopes.get(tool_name) == 'column':
                        async for event in self._execute_column_iteration():
                            yield event
                    else:
                        async for event in self._execute_standard_tool():
                            yield event

                elif self.state == AgentState.SUMMARIZING:
                    async for event in self._handle_summarizing():
                        yield event
            except Exception as e:
                app_logger.error(f"Error in state {self.state.name}: {e}", exc_info=True)
                self.state = AgentState.ERROR
                yield _format_sse({"error": "An error occurred during execution.", "details": str(e)}, "error")
        
        if self.state not in [AgentState.DONE, AgentState.ERROR]:
            async for event in self._handle_summarizing():
                yield event

    def _is_date_query_candidate(self) -> tuple[bool, str]:
        if not self.current_command:
            return False, None
        
        args = self.current_command.get("arguments", {})
        date_param_name = next((param for param in args if 'date' in param.lower()), None)
        
        return bool(date_param_name), date_param_name

    async def _classify_date_query_type(self) -> tuple[str, str]:
        classification_prompt = (
            f"You are a query classifier. Your only task is to analyze a user's request for date information. "
            f"Analyze the following query: '{self.original_user_input}'. "
            "First, determine if it refers to a 'single' date or a 'range' of dates. "
            "Second, extract the specific phrase that describes the date or range. "
            "Your response MUST be ONLY a JSON object with two keys: 'type' and 'phrase'. "
            "Example for 'system utilization yesterday': {\"type\": \"single\", \"phrase\": \"yesterday\"}. "
            "Example for 'system utilization for the last 3 days': {\"type\": \"range\", \"phrase\": \"the last 3 days\"}."
        )
        response_str, _, _ = await llm_handler.call_llm_api(
            self.dependencies['STATE']['llm'], classification_prompt, raise_on_error=True,
            system_prompt_override="You are a JSON-only responding assistant."
        )
        try:
            data = json.loads(response_str)
            return data.get('type', 'single'), data.get('phrase', self.original_user_input)
        except (json.JSONDecodeError, KeyError):
            app_logger.error(f"Failed to parse date classification from LLM. Response: {response_str}")
            return 'single', self.original_user_input

    async def _execute_date_range_orchestrator(self, date_param_name: str, date_phrase: str):
        tool_name = self.current_command.get("tool_name")
        yield _format_sse({
            "step": "System Orchestration",
            "details": f"Detected a date range query ('{date_phrase}') for a single-day tool ('{tool_name}'). The system will now orchestrate multiple tool calls to fulfill the request.",
            "type": "workaround"
        })

        date_command = {"tool_name": "base_readQuery", "arguments": {"sql": "SELECT CURRENT_DATE"}}
        date_result = await mcp_adapter.invoke_mcp_tool(self.dependencies['STATE'], date_command)
        if not (date_result and date_result.get("status") == "success" and date_result.get("results")):
            raise RuntimeError("Date Range Orchestrator failed to fetch current date.")
        current_date_str = date_result["results"][0].get("Date")

        conversion_prompt = (
            f"You are a date range calculation assistant. Your only task is to identify the start and end dates for a relative date phrase. "
            f"Given that the current date is {current_date_str}, what are the start and end dates for '{date_phrase}'? "
            "Your response MUST be ONLY a JSON object with two keys: 'start_date' and 'end_date', both in YYYY-MM-DD format. "
            "Example: {\"start_date\": \"2025-08-01\", \"end_date\": \"2025-08-07\"}"
        )
        range_response_str, _, _ = await llm_handler.call_llm_api(
            self.dependencies['STATE']['llm'], conversion_prompt, raise_on_error=True,
            system_prompt_override="You are a JSON-only responding assistant."
        )

        try:
            range_data = json.loads(range_response_str)
            start_date = datetime.strptime(range_data['start_date'], '%Y-%m-%d').date()
            end_date = datetime.strptime(range_data['end_date'], '%Y-%m-%d').date()
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise RuntimeError(f"Date Range Orchestrator failed to parse date range from LLM. Response: {range_response_str}. Error: {e}")

        all_results = []
        current_date_in_loop = start_date
        
        yield _format_sse({"target": "db", "state": "busy"}, "status_indicator_update")
        while current_date_in_loop <= end_date:
            date_str = current_date_in_loop.strftime('%Y-%m-%d')
            yield _format_sse({"step": f"Processing data for: {date_str}"})
            
            command_for_day = self.current_command.copy()
            command_for_day['arguments'] = self.current_command['arguments'].copy()
            command_for_day['arguments'][date_param_name] = date_str
            
            day_result = await mcp_adapter.invoke_mcp_tool(self.dependencies['STATE'], command_for_day)
            
            if isinstance(day_result, dict) and day_result.get("status") == "success" and day_result.get("results"):
                all_results.extend(day_result["results"])
            
            current_date_in_loop += timedelta(days=1)
        yield _format_sse({"target": "db", "state": "idle"}, "status_indicator_update")
        
        final_tool_output = {
            "status": "success",
            "metadata": {"tool_name": tool_name, "comment": f"Consolidated results for date range: {start_date} to {end_date}"},
            "results": all_results
        }
        self._add_to_structured_data(final_tool_output)
        self.state = AgentState.SUMMARIZING
        self.next_action_str = "FINAL_ANSWER: "

    async def _intercept_and_correct_command(self):
        if not self.current_command: return

        tool_name = self.current_command.get("tool_name")
        args = self.current_command.get("arguments", {})

        if tool_name == "base_tableList":
            app_logger.warning("INTERCEPTED: Faulty tool 'base_tableList'. Replacing with 'base_readQuery'.")
            
            db_name = args.get("db_name")
            if not db_name:
                raise ValueError("Cannot execute 'base_tableList' replacement: 'db_name' parameter is missing.")

            corrected_sql = f"SELECT TableName FROM DBC.TablesV WHERE DatabaseName = '{db_name}'"
            
            self.current_command['tool_name'] = "base_readQuery"
            self.current_command['arguments'] = {"sql": corrected_sql}
            
            yield _format_sse({
                "step": "System Correction",
                "details": f"Intercepted faulty 'base_tableList' tool. Replacing with a direct SQL query for '{db_name}'.",
                "type": "workaround"
            })

    def _enrich_arguments_from_history(self, prompt_text: str, arguments: dict) -> tuple[dict, list]:
        events_to_yield = []
        required_placeholders = re.findall(r'{(\w+)}', prompt_text)
        enriched_args = arguments.copy()
        
        alias_map = {
            "database_name": ["db_name"],
            "table_name": ["tbl_name", "object_name", "obj_name"],
            "column_name": ["col_name"]
        }
        reverse_alias_map = {alias: canon for canon, aliases in alias_map.items() for alias in aliases}

        for placeholder in required_placeholders:
            if placeholder in enriched_args:
                continue

            found_alias = False
            for alias, canon in reverse_alias_map.items():
                if canon == placeholder and alias in enriched_args:
                    enriched_args[placeholder] = enriched_args[alias]
                    found_alias = True
                    break
            if found_alias:
                continue

            app_logger.info(f"Placeholder '{placeholder}' is missing. Searching conversation history for context.")
            session_data = session_manager.get_session(self.session_id)
            if not session_data: continue

            for entry in reversed(session_data.get("generic_history", [])):
                content = entry.get("content")
                if not isinstance(content, str): continue
                
                try:
                    if entry.get("role") == "assistant":
                        json_match = re.search(r"```json\s*\n(.*?)\n\s*```", content, re.DOTALL)
                        if json_match:
                            command = json.loads(json_match.group(1).strip())
                            args_to_check = command.get("arguments", {})
                            if placeholder in args_to_check:
                                enriched_args[placeholder] = args_to_check[placeholder]
                                break
                            for alias, canon in reverse_alias_map.items():
                                if canon == placeholder and alias in args_to_check:
                                    enriched_args[placeholder] = args_to_check[alias]
                                    break
                            if placeholder in enriched_args: break
                except (json.JSONDecodeError, TypeError):
                    continue
            
            if placeholder in enriched_args:
                app_logger.info(f"Found value for '{placeholder}' in history: '{enriched_args[placeholder]}'")
                event = _format_sse({
                    "step": "System Correction",
                    "details": f"LLM omitted the '{placeholder}' parameter. The system inferred it from the conversation history.",
                    "type": "workaround"
                })
                events_to_yield.append(event)

        return enriched_args, events_to_yield

    async def _handle_deciding(self):
        if re.search(r'FINAL_ANSWER:', self.next_action_str, re.IGNORECASE):
            self.state = AgentState.SUMMARIZING
            return

        command_str = None
        markdown_match = re.search(r"```json\s*\n(.*?)\n\s*```", self.next_action_str, re.DOTALL)
        if markdown_match:
            command_str = markdown_match.group(1).strip()
        else:
            json_like_match = re.search(r'\{.*\}', self.next_action_str, re.DOTALL)
            if json_like_match:
                command_str = json_like_match.group(0)

        if not command_str:
            app_logger.warning(f"LLM response not a tool command or FINAL_ANSWER. Summarizing. Response: {self.next_action_str}")
            self.state = AgentState.SUMMARIZING
            return
        
        try:
            decoder = json.JSONDecoder()
            command, _ = decoder.raw_decode(command_str)
        except (json.JSONDecodeError, KeyError) as e:
            app_logger.error(f"JSON parsing failed. Error: {e}. Original string was: {command_str}")
            raise e
            
        if "tool_name" in command:
            t_name = command["tool_name"]
            mcp_tools = self.dependencies['STATE'].get('mcp_tools', {})
            mcp_prompts = self.dependencies['STATE'].get('mcp_prompts', {})
            if t_name not in mcp_tools and t_name in mcp_prompts:
                app_logger.warning(f"LLM hallucinated tool '{t_name}'. Correcting to prompt.")
                yield _format_sse({
                    "step": "System Correction",
                    "details": f"LLM incorrectly used 'tool_name' for a prompt. Corrected '{t_name}' to be a prompt.",
                    "type": "workaround"
                })
                command["prompt_name"] = command.pop("tool_name")
            
        self.current_command = command
        
        if "prompt_name" in command:
            prompt_name = command.get("prompt_name")
            self.active_prompt_name = prompt_name
            self.is_workflow = True
            
            mcp_client = self.dependencies['STATE'].get('mcp_client')
            if not mcp_client: raise RuntimeError("MCP client is not connected.")
            
            get_prompt_result = None
            async with mcp_client.session("teradata_mcp_server") as temp_session:
                get_prompt_result = await temp_session.get_prompt(name=prompt_name)
            
            if get_prompt_result is None: raise ValueError(f"Prompt '{prompt_name}' could not be retrieved from MCP server.")
            prompt_text = get_prompt_result.content.text if hasattr(get_prompt_result, 'content') else str(get_prompt_result)

            arguments = command.get("arguments", command.get("parameters", {}))
            enriched_arguments, events_to_yield = self._enrich_arguments_from_history(prompt_text, arguments)
            for event in events_to_yield:
                yield event

            self.active_prompt_plan = prompt_text.format(**enriched_arguments)

            yield _format_sse({
                "step": f"Executing Prompt as a Plan: {prompt_name}",
                "details": self.active_prompt_plan,
                "prompt_name": prompt_name
            }, "prompt_selected")
            
            async for event in self._get_next_action_from_llm():
                yield event

        elif "tool_name" in command:
            self.state = AgentState.EXECUTING_TOOL
        else:
            self.state = AgentState.SUMMARIZING

    async def _execute_standard_tool(self):
        # --- AUTHORITATIVE CONTEXT: Apply guardrail to correct/inject parameters ---
        corrected_command, guardrail_events = self._apply_context_guardrail(self.current_command)
        for event in guardrail_events:
            yield event
        self.current_command = corrected_command

        yield _format_sse({"step": "Tool Execution Intent", "details": self.current_command}, "tool_result")
        
        tool_name = self.current_command.get("tool_name")
        is_chart_tool = tool_name == "viz_createChart"
        status_target = "chart" if is_chart_tool else "db"
        yield _format_sse({"target": status_target, "state": "busy"}, "status_indicator_update")
        
        tool_result = await mcp_adapter.invoke_mcp_tool(self.dependencies['STATE'], self.current_command)
        
        yield _format_sse({"target": status_target, "state": "idle"}, "status_indicator_update")

        # --- AUTHORITATIVE CONTEXT: Update context stack based on the command and its result ---
        if self.is_workflow:
            self._update_and_manage_context_stack(self.current_command, tool_result)

        if 'notification' in self.current_command:
            yield _format_sse({
                "step": "System Notification", 
                "details": self.current_command['notification'],
                "type": "workaround"
            })
            del self.current_command['notification']

        tool_result_str = ""
        if isinstance(tool_result, dict) and "error" in tool_result:
            error_details = tool_result.get("data", tool_result.get("error", ""))
            
            if "Function" in str(error_details) and "does not exist" in str(error_details):
                self.globally_failed_tools.add(tool_name)
                app_logger.warning(f"Tool '{tool_name}' marked as globally failed for this session.")
            
            tool_result_str = json.dumps({
                "tool_input": self.current_command,
                "tool_output": {
                    "status": "error",
                    "error_message": error_details
                }
            })
        else:
            tool_result_str = json.dumps({"tool_name": self.current_command.get("tool_name"), "tool_output": tool_result})
            # --- AUTHORITATIVE CONTEXT: Use structured data collection for workflows ---
            if self.is_workflow:
                self._add_to_structured_data(tool_result)
            else:
                self.collected_data.append(tool_result)

        if isinstance(tool_result, dict) and tool_result.get("error") == "parameter_mismatch":
            yield _format_sse({"details": tool_result}, "request_user_input")
            self.state = AgentState.ERROR
            return

        yield _format_sse({"step": "Tool Execution Result", "details": tool_result, "tool_name": self.current_command.get("tool_name")}, "tool_result")
        
        yield _format_sse({"step": "Thinking about the next action...", "details": "The agent is reasoning based on the current context."})
        async for event in self._get_next_action_from_llm(tool_result_str=tool_result_str):
            yield event

    async def _get_tool_constraints(self, tool_name: str):
        if tool_name in self.tool_constraints_cache:
            yield self.tool_constraints_cache[tool_name]
            return

        mcp_tools = self.dependencies['STATE'].get('mcp_tools', {})
        tool_definition = mcp_tools.get(tool_name)
        
        constraints = {}
        
        if tool_definition:
            tool_description_lower = tool_definition.description.lower()
            prompt_modifier = ""
            
            if any(keyword in tool_name.lower() for keyword in ["univariate", "standarddeviation", "negativevalues"]) or \
               any(keyword in tool_description_lower for keyword in ["statistics", "deviation", "negative values", "numerical analysis"]):
                prompt_modifier = "This tool is intended for quantitative analysis. When `col_name` is an argument, it requires a 'numeric' data type."
                app_logger.info(f"Adding numeric type hint to LLM prompt for tool '{tool_name}' based on name/description.")
            elif any(keyword in tool_name.lower() for keyword in ["distinctcategories"]) or \
                 any(keyword in tool_description_lower for keyword in ["distinct categories", "categorical data"]):
                prompt_modifier = "This tool is intended for categorical analysis. When `col_name` is an argument, it requires a 'character' or 'string' data type."
                app_logger.info(f"Adding character type hint to LLM prompt for tool '{tool_name}' based on name/description.")

            prompt = (
                "You are a tool analyzer. Based on the tool's name and description, determine if its `col_name` "
                "argument is intended for 'numeric' types, 'character' types, or 'any' type. "
                "The tool's capabilities are:\n"
                f"Tool Name: `{tool_definition.name}`\n"
                f"Tool Description: \"{tool_definition.description}\"\n\n"
                f"Contextual Hint: {prompt_modifier}\n\n"
                "Respond with a single, raw JSON object with one key, 'dataType', and the value 'numeric', 'character', or 'any'.\n"
                "Example response for a statistical tool: {{\"dataType\": \"numeric\"}}"
            )
            
            yield _format_sse({
                "step": f"Inferring constraints for tool: {tool_name}",
                "details": "Asking LLM to analyze tool requirements with dynamic hints...",
                "type": "workaround"
            })
            
            yield _format_sse({"step": "Calling LLM"})
            response_text, _, _ = await llm_handler.call_llm_api(
                self.dependencies['STATE']['llm'], 
                prompt, 
                raise_on_error=True,
                system_prompt_override="You are a JSON-only responding assistant."
            )

            try:
                match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if not match:
                    raise ValueError("LLM did not return valid JSON for constraints.")
                
                constraints = json.loads(match.group(0))
                app_logger.info(f"Inferred constraints for tool '{tool_name}': {constraints}")
            except (json.JSONDecodeError, ValueError) as e:
                app_logger.error(f"Failed to infer constraints for {tool_name}: {e}. Assuming no constraints.")
                constraints = {}
        
        self.tool_constraints_cache[tool_name] = constraints
        yield constraints

    async def _execute_column_iteration(self, column_subset: list = None):
        base_command = self.current_command
        tool_name = base_command.get("tool_name")
        base_args = base_command.get("arguments", base_command.get("parameters", {}))
        
        db_name = base_args.get("database_name") or base_args.get("db_name")
        table_name = base_args.get("table_name") or base_args.get("obj_name")

        if table_name and '.' in table_name and not db_name:
            db_name, table_name = table_name.split('.', 1)
            app_logger.info(f"Parsed db_name '{db_name}' from fully qualified table_name.")

        specific_column = base_args.get("col_name") or base_args.get("column_name")
        if specific_column:
            yield _format_sse({"step": "Tool Execution Intent", "details": base_command}, "tool_result")
            
            yield _format_sse({"target": "db", "state": "busy"}, "status_indicator_update")
            col_result = await mcp_adapter.invoke_mcp_tool(self.dependencies['STATE'], base_command)
            yield _format_sse({"target": "db", "state": "idle"}, "status_indicator_update")
            
            if 'notification' in self.current_command:
                yield _format_sse({
                    "step": "System Notification", 
                    "details": self.current_command['notification'],
                    "type": "workaround"
                })
                del self.current_command['notification']

            if isinstance(col_result, dict) and col_result.get("error") == "parameter_mismatch":
                yield _format_sse({"details": col_result}, "request_user_input")
                self.state = AgentState.ERROR
                return

            yield _format_sse({"step": f"Tool Execution Result for column: {specific_column}", "details": col_result, "tool_name": tool_name}, "tool_result")
            self._add_to_structured_data(col_result)
            tool_result_str = json.dumps({"tool_name": tool_name, "tool_output": col_result})
            yield _format_sse({"step": "Thinking about the next action...", "details": "Single column execution complete. Resuming main plan."})
            async for event in self._get_next_action_from_llm(tool_result_str=tool_result_str):
                yield event
            return

        all_columns_metadata = []
        if column_subset:
            all_columns_metadata = [{"ColumnName": col_name, "DataType": "UNKNOWN"} for col_name in column_subset]
            app_logger.info(f"Executing column iteration on a pre-defined subset of {len(column_subset)} columns.")
        else:
            reused_metadata = False
            # --- AUTHORITATIVE CONTEXT: Check structured data first for column metadata ---
            context_key = " > ".join([ctx['list'][ctx['index']] for ctx in self.context_stack if 'list' in ctx and ctx['index'] != -1])
            if context_key and context_key in self.structured_collected_data:
                for item in reversed(self.structured_collected_data[context_key]):
                    if isinstance(item, dict) and item.get("status") == "success" and item.get("metadata", {}).get("tool_name") in ["qlty_columnSummary", "base_columnDescription"]:
                        all_columns_metadata = item.get("results", [])
                        app_logger.info(f"Reusing column metadata for table {table_name} from previous '{item.get('metadata', {}).get('tool_name')}' tool call in current context.")
                        reused_metadata = True
                        break
            
            if not reused_metadata:
                yield _format_sse({
                    "step": f"Adaptive column tool detected: {tool_name}", 
                    "details": "Fetching column metadata to determine compatibility.",
                    "type": "workaround"
                })
                cols_command = {"tool_name": "base_columnDescription", "arguments": {"db_name": db_name, "obj_name": table_name}}
                
                yield _format_sse({"target": "db", "state": "busy"}, "status_indicator_update")
                cols_result = await mcp_adapter.invoke_mcp_tool(self.dependencies['STATE'], cols_command)
                yield _format_sse({"target": "db", "state": "idle"}, "status_indicator_update")

                if not (cols_result and isinstance(cols_result, dict) and cols_result.get('status') == 'success' and cols_result.get('results')):
                    raise ValueError(f"Failed to retrieve column list for iteration. Response: {cols_result}")
                all_columns_metadata = cols_result.get('results', [])
                # Also add this to the structured data
                self._add_to_structured_data(cols_result)


        all_column_results = []
        
        first_column_to_try = next((col.get("ColumnName") for col in all_columns_metadata), None)
        if first_column_to_try and tool_name not in self.globally_failed_tools:
            preflight_args = base_args.copy()
            preflight_args['col_name'] = first_column_to_try
            preflight_command = {"tool_name": tool_name, "arguments": preflight_args}
            temp_globally_failed_tools_check = False 
            try:
                yield _format_sse({"target": "db", "state": "busy"}, "status_indicator_update")
                preflight_result = await mcp_adapter.invoke_mcp_tool(self.dependencies['STATE'], preflight_command)
                yield _format_sse({"target": "db", "state": "idle"}, "status_indicator_update")
                
                if isinstance(preflight_result, dict) and "error" in preflight_result:
                    error_details = preflight_result.get("data", preflight_result.get("error", ""))
                    if "Function" in str(error_details) and "does not exist" in str(error_details):
                        self.globally_failed_tools.add(tool_name)
                        app_logger.warning(f"Tool '{tool_name}' marked as globally failed by pre-flight check.")
                        yield _format_sse({ "step": f"Halting iteration for globally failed tool: {tool_name}", "details": "Pre-flight check failed.", "type": "error" })
                        temp_globally_failed_tools_check = True
            except Exception as e:
                self.globally_failed_tools.add(tool_name)
                app_logger.error(f"Pre-flight check for tool '{tool_name}' failed with unexpected error: {e}")
                yield _format_sse({ "step": f"Halting iteration for problematic tool: {tool_name}", "details": f"Pre-flight check failed unexpectedly: {e}", "type": "error" })
                temp_globally_failed_tools_check = True

            if temp_globally_failed_tools_check:
                all_column_results.append({ "status": "error", "reason": f"Tool '{tool_name}' is non-functional.", "metadata": {"tool_name": tool_name, "table_name": table_name}})
                self._add_to_structured_data(all_column_results)
                return

        tool_constraints = None
        async for event in self._get_tool_constraints(tool_name):
            if isinstance(event, dict):
                tool_constraints = event
            else:
                yield event
        
        required_type = tool_constraints.get("dataType") if tool_constraints else None

        yield _format_sse({"target": "db", "state": "busy"}, "status_indicator_update")
        for column_info in all_columns_metadata:
            col_name = column_info.get("ColumnName")

            col_type = ""
            for key, value in column_info.items():
                if key.lower() in ["datatype", "columntype", "type"]:
                    col_type = (value or "").upper()
                    break
            if not col_type:
                for key, value in column_info.items():
                    if "type" in key.lower():
                        col_type = (value or "").upper()
                        break

            if required_type and col_type != "UNKNOWN":
                is_numeric_column = any(t in col_type for t in ["INT", "NUMERIC", "DECIMAL", "FLOAT", "BYTEINT", "SMALLINT", "BIGINT", "INTEGER"])
                is_char_column = any(t in col_type for t in ["CHAR", "VARCHAR", "GRAPHIC", "VARGRAPHIC", "TEXT", "CLOB"])
                
                should_skip = False
                if required_type == "numeric" and not is_numeric_column:
                    should_skip = True
                elif required_type == "character" and not is_char_column:
                    should_skip = True

                if should_skip:
                    skip_reason = f"Tool '{tool_name}' requires a {required_type} column, but '{col_name}' is of type {col_type}."
                    app_logger.info(f"SKIPPING: {skip_reason}")
                    skip_result = {
                        "status": "skipped",
                        "reason": skip_reason,
                        "metadata": {"tool_name": tool_name, "table_name": table_name, "col_name": col_name}
                    }
                    yield _format_sse({
                        "step": f"Skipping tool for column: {col_name}", 
                        "details": skip_result, 
                        "tool_name": tool_name,
                        "type": "workaround"
                    }, "tool_result")
                    all_column_results.append(skip_result)
                    continue

            iter_args = base_args.copy()
            iter_args['col_name'] = col_name
            
            iter_command = {"tool_name": tool_name, "arguments": iter_args}

            yield _format_sse({"step": "Tool Execution Intent", "details": iter_command}, "tool_result")
            col_result = await mcp_adapter.invoke_mcp_tool(self.dependencies['STATE'], iter_command)
            
            if isinstance(col_result, dict) and "error" in col_result:
                error_details = col_result.get("data", col_result.get("error", ""))
                if "Function" in str(error_details) and "does not exist" in str(error_details):
                    self.globally_failed_tools.add(tool_name)
                    app_logger.warning(f"Tool '{tool_name}' marked as globally failed for this session during iteration.")
                    all_column_results.append({ "status": "error", "reason": f"Tool '{tool_name}' is non-functional.", "metadata": {"tool_name": tool_name, "col_name": col_name}})
                    yield _format_sse({ "step": f"Halting iteration for globally failed tool: {tool_name}", "details": error_details, "type": "error" })
                    break
            
            if 'notification' in iter_command:
                yield _format_sse({
                    "step": "System Notification", 
                    "details": iter_command['notification'],
                    "type": "workaround"
                })
                del iter_command['notification']

            if isinstance(col_result, dict) and col_result.get("error") == "parameter_mismatch":
                yield _format_sse({"details": col_result}, "request_user_input")
                self.state = AgentState.ERROR
                return

            yield _format_sse({"step": f"Tool Execution Result for column: {col_name}", "details": col_result, "tool_name": tool_name}, "tool_result")
            all_column_results.append(col_result)
        yield _format_sse({"target": "db", "state": "idle"}, "status_indicator_update")

        self._add_to_structured_data(all_column_results)
        
        tool_result_str = json.dumps({"tool_name": tool_name, "tool_output": all_column_results})

        yield _format_sse({"step": "Thinking about the next action...", "details": "Adaptive column iteration complete. Resuming main plan."})
        async for event in self._get_next_action_from_llm(tool_result_str=tool_result_str):
            yield event
    
    def _build_just_in_time_context_prompt(self) -> str:
        """
        Builds a dynamic, ephemeral context block to remind the LLM of its
        current position within a single or nested loop.
        """
        if not self.context_stack:
            return ""

        context_lines = []
        if len(self.context_stack) > 1:
            context_lines.append("You are executing a nested plan.")
        else:
            context_lines.append("You are executing a plan that iterates over a list.")

        for i, context in enumerate(self.context_stack):
            context_type = context['type'].replace('_name', '').capitalize()
            prefix = f"{'  ' * i}- **Loop ({context_type}s)**:"
            
            current_item = "N/A"
            if 0 <= context['index'] < len(context['list']):
                current_item = f"`{context['list'][context['index']]}` (Item {context['index'] + 1} of {len(context['list'])})"
            
            context_lines.append(f"{prefix}")
            context_lines.append(f"{'  ' * (i+1)}- **Current Item**: {current_item}")

        final_instruction = "Your immediate task is to continue the sub-plan for the **innermost current item**."
        if len(self.context_stack) == 1:
            final_instruction = "Your immediate task is to continue with the sub-plan for the **current item**. Do not advance to the next item until all steps for the current item are complete."

        return (
            "\n--- **CURRENT LOOP STATE** ---\n"
            + "\n".join(context_lines) + "\n"
            + final_instruction + "\n"
            "---------------------------\n"
        )

    async def _get_next_action_from_llm(self, tool_result_str: str | None = None):
        just_in_time_context = self._build_just_in_time_context_prompt()
        
        prompt_for_next_step = "" 
        
        if self.active_prompt_plan:
            app_logger.info("Applying forceful, plan-aware reasoning for next step.")
            # --- MODIFIED: Add the "Tool-First" critical rule for workflows ---
            prompt_for_next_step = (
                "You are executing a multi-step plan. Your primary goal is to follow this plan sequentially to completion.\n\n"
                "--- **NEW CRITICAL RULE: TOOL-FIRST EXECUTION** ---\n"
                "You are currently executing a multi-step plan. To complete the next step, you **MUST** prioritize using a direct `tool_name` call. Only if no single tool can accomplish the task should you consider calling another `prompt_name`. Calling another prompt is a sub-routine and should be avoided if a tool can provide the needed information directly.\n"
                "--------------------------------------------------\n\n"
                f"--- ORIGINAL PLAN ---\n{self.active_prompt_plan}\n\n"
                f"{just_in_time_context}" 
                "--- CURRENT STATE ---\n"
                f"- You have just received the result of the last action, which is in the conversation history below.\n\n"
                "--- YOUR TASK: EXECUTE THE *NEXT* STEP ---\n"
                "1. **Analyze your history, the LOOP STATE, and the ORIGINAL PLAN.** Determine the single next instruction in the sequence. **You MUST NOT repeat steps that you have already successfully completed for the current item.**\n"
                "2. **Execute only that next instruction.**\n"
                "   - If the next step is a tool call, your response **MUST** be a single JSON block for that tool call.\n"
                "   - If you have completed all tool calls and the final step is to generate the summary, your response **MUST** start with `FINAL_ANSWER:`.\n\n"
                "**CRITICAL RULE:** Do not restart the plan. Do not perform any action other than the immediate next step. If you are not delivering the final user-facing answer, your response must be a tool call."
            )
        else:
            prompt_for_next_step = (
                "You have just received data from a tool call. Review the data and your instructions to decide the next step.\n\n"
                "1.  **Analyze the Result:**\n"
                "    -   If the `tool_output` shows `\"status\": \"error\"`, you MUST attempt to recover. The `tool_input` field shows the exact command that failed. Formulate a new tool call that corrects the error. For example, if the error is 'Column not found', remove the failing column from the `dimensions` list.\n\n"
                "    -   If the status is `success`, proceed to the next steps.\n\n"
                "2.  **Consider a Chart:** Review the `--- Charting Rules ---` in your system prompt. Based on the data you just received, would a chart be an appropriate and helpful way to visualize the information for the user?\n\n"
                "3.  **Choose Your Action:**\n"
                "    -   If a chart is appropriate, your next action is to call the correct chart-generation tool. Respond with only a `Thought:` and a ```json...``` block for that tool.\n"
                "    -   If you still need more information from other tools, call the next appropriate tool by responding with a `Thought:` and a ```json...``` block.\n"
                "    -   If a chart is **not** appropriate and you have all the information needed to answer the user's request, you **MUST** provide the final answer. Your response **MUST** be plain text that starts with `FINAL_ANSWER:`. **DO NOT** use a JSON block for the final answer."
            )
        
        if tool_result_str:
            final_prompt_to_llm = f"{prompt_for_next_step}\n\nThe last tool execution returned the following result. Use this to inform your next action:\n\n{tool_result_str}"
        else:
            final_prompt_to_llm = prompt_for_next_step
        
        yield _format_sse({"step": "Calling LLM"})

        self.next_action_str, statement_input_tokens, statement_output_tokens = await llm_handler.call_llm_api(self.dependencies['STATE']['llm'], final_prompt_to_llm, self.session_id)
        
        updated_session = session_manager.get_session(self.session_id)
        if updated_session:
            token_data = {
                "statement_input": statement_input_tokens,
                "statement_output": statement_output_tokens,
                "total_input": updated_session.get("input_tokens", 0),
                "total_output": updated_session.get("output_tokens", 0)
            }
            yield _format_sse(token_data, "token_update")
        
        if not self.next_action_str:
            raise ValueError("LLM failed to provide a response.")
        self.state = AgentState.DECIDING

    def _prepare_data_for_final_summary(self) -> str:
        summary_lines = []
        # --- AUTHORITATIVE CONTEXT: Use the structured data for summary preparation ---
        data_to_summarize = self.structured_collected_data if self.is_workflow and self.structured_collected_data else {" ungrouped": self.collected_data}

        for context_key, items in data_to_summarize.items():
            summary_lines.append(f"- For context: `{context_key}`:")
            for item in items:
                if isinstance(item, list) and item:
                    first_sub_item = item[0]
                    if isinstance(first_sub_item, dict):
                        tool_name = first_sub_item.get("metadata", {}).get("tool_name", "Column-based tool")
                        summary_lines.append(f"  - Executed `{tool_name}`.")
                elif isinstance(item, dict):
                    tool_name = item.get("metadata", {}).get("tool_name")
                    status = item.get("status")
                    if status == "success" and "results" in item:
                        result_count = len(item["results"])
                        summary_lines.append(f"  - Tool `{tool_name}` succeeded with {result_count} results.")
                    elif status == "error":
                        summary_lines.append(f"  - Tool `{tool_name}` failed.")

        return "\n".join(summary_lines)

    async def _handle_summarizing(self):
        # --- AUTHORITATIVE CONTEXT: Consolidate all collected data for the formatter ---
        if self.is_workflow and self.structured_collected_data:
            # If we used the structured collector, it becomes the final data source
            final_collected_data = self.structured_collected_data
        else:
            final_collected_data = self.collected_data

        llm_response_for_formatter = self.next_action_str
        use_workflow_formatter = self.is_workflow

        final_answer_match = re.search(r'FINAL_ANSWER:(.*)', self.next_action_str, re.DOTALL | re.IGNORECASE)

        if final_answer_match:
            summary_text = final_answer_match.group(1).strip()
            
            if self.is_workflow:
                report_data = {
                    "summary": summary_text,
                    "structured_data": final_collected_data
                }
                llm_response_for_formatter = json.dumps(report_data)
            else:
                llm_response_for_formatter = summary_text
                use_workflow_formatter = False
        
        elif self.is_workflow:
            yield _format_sse({"step": "Workflow finished, generating structured report...", "details": "The agent is synthesizing all collected data into a report format."})
            
            summarized_data_str = self._prepare_data_for_final_summary()

            final_prompt = (
                "You are a data analyst responsible for generating the final report from a complex, multi-step workflow.\n\n"
                "--- CONTEXT ---\n"
                "A workflow was executed to gather data and answer a user's request. A summary of the collected data is provided below.\n\n"
                f"--- COLLECTED DATA SUMMARY ---\n{summarized_data_str}\n\n"
                "--- YOUR TASK ---\n"
                "Your task is to generate a final report in a specific JSON format. Your response **MUST** be a single, raw JSON object containing two keys:\n"
                "1. `\"summary\"`: A natural language summary that synthesizes the key findings from the COLLECTED DATA SUMMARY to answer the user's original request. This summary should be comprehensive and easy to understand.\n"
                "2. `\"structured_data\"`: The original, full COLLECTED DATA, which will be passed through programmatically. You do not need to include it here.\n\n"
                "**CRITICAL INSTRUCTIONS:**\n"
                "1. Your entire response MUST be a single, valid JSON object containing only the `\"summary\"` key and its value.\n"
                "2. The `summary` should be based *only* on the data summary provided.\n"
                "3. If you see results with a 'skipped' status in the data summary, you MUST mention this in your `summary`.\n"
                "Example response: {\"summary\": \"The data quality assessment for the database is complete. DDL was retrieved for three tables, and various quality checks were performed...\"}"
            )
            yield _format_sse({"step": "Calling LLM"})
            final_llm_response, statement_input_tokens, statement_output_tokens = await llm_handler.call_llm_api(self.dependencies['STATE']['llm'], final_prompt, self.session_id)
            
            updated_session = session_manager.get_session(self.session_id)
            if updated_session:
                token_data = {
                    "statement_input": statement_input_tokens,
                    "statement_output": statement_output_tokens,
                    "total_input": updated_session.get("input_tokens", 0),
                    "total_output": updated_session.get("output_tokens", 0)
                }
                yield _format_sse(token_data, "token_update")

            try:
                json_match = re.search(r"\{.*\}", final_llm_response, re.DOTALL)
                if json_match:
                    summary_json = json.loads(json_match.group(0))
                    report_data = {
                        "summary": summary_json.get("summary", "Workflow completed, but no summary was generated."),
                        "structured_data": final_collected_data
                    }
                    llm_response_for_formatter = json.dumps(report_data)
                else:
                    raise json.JSONDecodeError("No JSON object found in the LLM response.", final_llm_response, 0)
            except (json.JSONDecodeError, TypeError):
                llm_response_for_formatter = json.dumps({
                    "summary": "The agent finished the workflow but failed to generate a structured report. Displaying raw data instead.",
                    "structured_data": final_collected_data
                })

        else:
            yield _format_sse({"step": "Plan finished, generating final summary...", "details": "The agent is synthesizing all collected data."})
            
            summarized_data_str = self._prepare_data_for_final_summary()

            final_prompt = (
                "You are a data analyst responsible for generating the final, user-facing summary of a complex task.\n\n"
                "--- CONTEXT ---\n"
                "A plan was executed to answer the user's request. A summary of the collected data is below.\n\n"
                f"--- COLLECTED DATA SUMMARY ---\n{summarized_data_str}\n\n"
                "--- YOUR TASK ---\n"
                f"Generate a final, comprehensive answer for the user's original request: '{self.original_user_input}'.\n"
                "Your response MUST start with `FINAL_ANSWER:` and follow this exact structure:\n"
                "1. **Conclusion First**: Start with a paragraph summarizing the key findings and insights from the data. This should be a direct answer to the user's question.\n"
                "2. **Chart Introduction (if applicable)**: If a chart was generated, add a new paragraph that introduces the chart, explaining what it visualizes.\n\n"
                "**CRITICAL INSTRUCTIONS:**\n"
                "- Do not describe your internal process or the tools used. Focus on the data's meaning.\n"
                "- Your entire response should be a single block of text. The UI will handle rendering the chart and data tables separately.\n"
                "- Example of a good response:\n"
                "FINAL_ANSWER: The system experienced peak usage on Tuesday, primarily driven by ETL/ELT workloads. Usage was minimal during early morning hours across all workload types.\n\n"
                "The line chart below visualizes the number of requests over time, broken down by workload type, to illustrate these trends."
            )
            yield _format_sse({"step": "Calling LLM"})
            final_llm_response, statement_input_tokens, statement_output_tokens = await llm_handler.call_llm_api(self.dependencies['STATE']['llm'], final_prompt, self.session_id)
            
            updated_session = session_manager.get_session(self.session_id)
            if updated_session:
                token_data = {
                    "statement_input": statement_input_tokens,
                    "statement_output": statement_output_tokens,
                    "total_input": updated_session.get("input_tokens", 0),
                    "total_output": updated_session.get("output_tokens", 0)
                }
                yield _format_sse(token_data, "token_update")

            final_answer_match_inner = re.search(r'FINAL_ANSWER:(.*)', final_llm_response or "", re.DOTALL | re.IGNORECASE)
            summary_text = final_answer_match_inner.group(1).strip() if final_answer_match_inner else (final_llm_response or "The agent finished its plan but did not provide a final summary.")
            llm_response_for_formatter = summary_text
            use_workflow_formatter = False

        formatter = OutputFormatter(llm_response_text=llm_response_for_formatter, collected_data=final_collected_data, is_workflow=use_workflow_formatter)
        final_html = formatter.render()
        
        session_manager.add_to_history(self.session_id, 'assistant', final_html)
        yield _format_sse({"final_answer": final_html}, "final_answer")
        self.state = AgentState.DONE
