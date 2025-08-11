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

class WorkflowManager:
    """
    Parses and manages the state of a multi-step workflow prompt.
    """
    def __init__(self, prompt_text: str, arguments: dict):
        self.steps = self._parse_steps(prompt_text)
        self.current_step_index = 0 if self.steps and "Phase 0" in self.steps[0].get("title", "") else -1
        self.arguments = arguments
        app_logger.info(f"WorkflowManager initialized with {len(self.steps)} steps. Starting at index {self.current_step_index + 1}.")

    def _parse_steps(self, prompt_text: str) -> list[dict]:
        """
        Parses the prompt text into a list of steps based on '## Phase X' headers.
        """
        step_pattern = re.compile(r"^##\s+(?:Phase|Step)\s+\d+\s*[:\-]\s*(.*)", re.MULTILINE)
        steps = []
        matches = list(step_pattern.finditer(prompt_text))
        
        for i, match in enumerate(matches):
            step_title = match.group(1).strip()
            start_pos = match.end()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(prompt_text)
            step_content = prompt_text[start_pos:end_pos].strip()
            
            steps.append({
                "title": f"Phase {i} - {step_title}", # Standardize title for checks
                "content": step_content
            })
        return steps

    def _find_tool_in_step(self, step_content: str) -> str | None:
        """
        Finds an explicitly mentioned tool in the step's content.
        This version makes the backticks around the tool name optional to be more robust.
        """
        tool_match = re.search(r"Use the `?(\w+)`?\sfunction", step_content)
        return tool_match.group(1) if tool_match else None

    def get_next_action(self) -> dict | None:
        """
        Determines the next action to take in the workflow.
        """
        self.current_step_index += 1
        if self.current_step_index >= len(self.steps):
            return None

        next_step = self.steps[self.current_step_index]
        app_logger.info(f"WorkflowManager advancing to step {self.current_step_index}: {next_step['title']}")
        tool_name = self._find_tool_in_step(next_step['content'])

        if tool_name:
            return {
                "type": "tool_call",
                "command": {
                    "tool_name": tool_name,
                    "arguments": self.arguments
                }
            }
        else:
            return {
                "type": "llm_prompt",
                "prompt_content": next_step['content']
            }

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
        self.collected_data = [] # Retained for non-workflow specific data (charts, etc.)
        self.max_steps = 40
        self.active_prompt_plan = None
        self.active_prompt_name = None
        self.current_command = None
        self.dependencies = dependencies
        self.tool_constraints_cache = {}
        self.globally_failed_tools = set()
        self.is_workflow = False
        self.workflow_manager = None
        self.last_tool_result_str = None
        self.context_stack = []
        self.structured_collected_data = {} # For workflow-specific, grouped data
        self.last_command_str = None

    def _get_context_level_for_tool(self, tool_name: str) -> str | None:
        tool_scopes = self.dependencies['STATE'].get('tool_scopes', {})
        scope = tool_scopes.get(tool_name)
        if scope in ['database', 'table', 'column']:
            return 'table' if scope in ['table', 'column'] else 'database'
        return None

    def _apply_context_guardrail(self, command: dict) -> tuple[dict, list]:
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

        if self.context_stack and 'list' in self.context_stack[-1]:
            top_context = self.context_stack[-1]
            context_type = top_context['type']
            
            llm_provided_value = None
            for key in [context_type] + param_aliases.get(context_type, []):
                if key in args:
                    llm_provided_value = args[key]
                    break
            
            if llm_provided_value:
                current_index = top_context['index']
                next_index = current_index + 1
                if next_index < len(top_context['list']) and llm_provided_value == top_context['list'][next_index]:
                    app_logger.info(f"CONTEXT STACK: Advancing loop for '{context_type}' to index {next_index} ('{llm_provided_value}').")
                    top_context['index'] = next_index

        for context in self.context_stack:
            context_type = context['type']
            correct_value = None
            if 'list' in context and 0 <= context.get('index', -1) < len(context['list']):
                correct_value = context['list'][context['index']]
            
            if not correct_value: continue 

            found_key, llm_provided_value = None, None
            for key in [context_type] + param_aliases.get(context_type, []):
                if key in args:
                    found_key, llm_provided_value = key, args[key]
                    break
            
            if not found_key or llm_provided_value != correct_value:
                key_to_set = found_key or context_type
                args[key_to_set] = correct_value
                details = f"LLM provided '{llm_provided_value}' for {key_to_set}. System corrected to authoritative value '{correct_value}'."
                app_logger.warning(f"GUARDRAIL APPLIED: {details}")
                events_to_yield.append(_format_sse({
                    "step": "System Correction", "details": details, "type": "workaround"
                }))

        return corrected_command, events_to_yield

    def _update_and_manage_context_stack(self, command: dict, tool_result: dict | None = None):
        tool_name = command.get("tool_name")
        tool_level = self._get_context_level_for_tool(tool_name)

        while self.context_stack:
            top_context = self.context_stack[-1]
            top_level = top_context['type'].split('_name')[0]
            if tool_level == 'database' and top_level == 'table':
                popped = self.context_stack.pop()
                app_logger.info(f"CONTEXT STACK: Popped '{popped['type']}' context as inner loop appears complete.")
            else:
                break
        
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
                        'index': -1 
                    })

    def _add_to_structured_data(self, tool_result: dict):
        # --- MODIFIED: Always add to structured_collected_data if it's a workflow ---
        if self.is_workflow:
            context_key = " > ".join([ctx['list'][ctx['index']] for ctx in self.context_stack if 'list' in ctx and ctx['index'] != -1])
            
            if not context_key:
                # If no specific context, use a generic key for overall workflow results
                context_key = "Overall Workflow Results" 
            
            if context_key not in self.structured_collected_data:
                self.structured_collected_data[context_key] = []
            
            self.structured_collected_data[context_key].append(tool_result)
            app_logger.info(f"Added tool result to structured data under key: '{context_key}' for workflow.")
        else:
            # For non-workflow execution, continue to use collected_data as before
            self.collected_data.append(tool_result)
            app_logger.info(f"Added tool result to collected data for non-workflow execution.")

    async def run(self):
        for i in range(self.max_steps):
            if self.state in [AgentState.DONE, AgentState.ERROR]:
                break
            try:
                if self.state == AgentState.DECIDING:
                    yield _format_sse({"step": "LLM has decided on an action", "details": self.next_action_str}, "llm_thought")
                    async for event in self._handle_deciding():
                        yield event
                
                elif self.state == AgentState.EXECUTING_TOOL:
                    is_range_candidate, date_param_name = self._is_date_query_candidate()
                    if is_range_candidate:
                        yield _format_sse({"step": "Calling LLM", "details": "Classifying date query."})
                        query_type, date_phrase = await self._classify_date_query_type()
                        if query_type == 'range':
                            async for event in self._execute_date_range_orchestrator(date_param_name, date_phrase):
                                yield event
                            continue
                    
                    tool_name = self.current_command.get("tool_name")
                    if tool_name in self.globally_failed_tools:
                        yield _format_sse({
                            "step": f"Skipping Globally Failed Tool: {tool_name}",
                            "details": f"The tool '{tool_name}' has been identified as non-functional for this session and will be skipped.",
                            "type": "workaround"
                        })
                        self.last_tool_result_str = json.dumps({ "tool_name": tool_name, "tool_output": { "status": "error", "error_message": "Skipped because tool is globally non-functional." } })
                        self.state = AgentState.DECIDING
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
            "Your response MUST be ONLY a JSON object with two keys: 'type' and 'phrase'."
        )
        response_str, _, _ = await llm_handler.call_llm_api(
            self.dependencies['STATE']['llm'], classification_prompt, raise_on_error=True,
            system_prompt_override="You are a JSON-only responding assistant.",
            dependencies=self.dependencies,
            reason="Classifying date query."
        )
        try:
            return json.loads(response_str).get('type', 'single'), json.loads(response_str).get('phrase', self.original_user_input)
        except (json.JSONDecodeError, KeyError):
            return 'single', self.original_user_input

    async def _execute_date_range_orchestrator(self, date_param_name: str, date_phrase: str):
        tool_name = self.current_command.get("tool_name")
        yield _format_sse({
            "step": "System Orchestration",
            "details": f"Detected a date range query ('{date_phrase}') for a single-day tool ('{tool_name}').",
            "type": "workaround"
        })

        date_command = {"tool_name": "base_readQuery", "arguments": {"sql": "SELECT CURRENT_DATE"}}
        date_result = await mcp_adapter.invoke_mcp_tool(self.dependencies['STATE'], date_command)
        if not (date_result and date_result.get("status") == "success" and date_result.get("results")):
            raise RuntimeError("Date Range Orchestrator failed to fetch current date.")
        current_date_str = date_result["results"][0].get("Date")

        conversion_prompt = (
            f"You are a date range calculation assistant. Given that the current date is {current_date_str}, "
            f"what are the start and end dates for '{date_phrase}'? "
            "Your response MUST be ONLY a JSON object with 'start_date' and 'end_date' in YYYY-MM-DD format."
        )
        yield _format_sse({"step": "Calling LLM", "details": "Calculating date range."})
        range_response_str, _, _ = await llm_handler.call_llm_api(
            self.dependencies['STATE']['llm'], conversion_prompt, raise_on_error=True,
            system_prompt_override="You are a JSON-only responding assistant.",
            dependencies=self.dependencies,
            reason="Calculating date range."
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
            
            command_for_day = {**self.current_command, 'arguments': {**self.current_command['arguments'], date_param_name: date_str}}
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
            if placeholder in enriched_args: continue

            found_alias = False
            for alias, canon in reverse_alias_map.items():
                if canon == placeholder and alias in enriched_args:
                    enriched_args[placeholder] = enriched_args.pop(alias)
                    found_alias = True
                    break
            if found_alias: continue

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
                app_logger.info(f"Inferred '{placeholder}' from history: '{enriched_args[placeholder]}'")
                events_to_yield.append(_format_sse({
                    "step": "System Correction",
                    "details": f"LLM omitted '{placeholder}'. System inferred it from history.",
                    "type": "workaround"
                }))

        return enriched_args, events_to_yield

    async def _handle_deciding(self):
        if self.is_workflow and self.workflow_manager:
            next_action = self.workflow_manager.get_next_action()
            if next_action:
                if next_action['type'] == 'tool_call':
                    self.current_command = next_action['command']
                    self.state = AgentState.EXECUTING_TOOL
                elif next_action['type'] == 'llm_prompt':
                    async for event in self._get_next_action_from_llm(scoped_prompt_content=next_action['prompt_content'], reason="Executing workflow step."):
                        yield event
            else:
                app_logger.info("WorkflowManager reports all steps are complete.")
                self.state = AgentState.SUMMARIZING
            return

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
            self.state = AgentState.SUMMARIZING
            return
        
        if command_str == self.last_command_str:
            app_logger.warning(f"LOOP DETECTED: The LLM is trying to repeat the exact same command. Command: {command_str}")
            error_message = f"Repetitive action detected. The agent tried to call the same tool with the same arguments twice in a row. This usually indicates the tool is not behaving as expected or the agent's plan is flawed."
            tool_result_str = json.dumps({
                "tool_input": json.loads(command_str), 
                "tool_output": {"status": "error", "error_message": error_message}
            })
            yield _format_sse({"step": "System Error", "details": error_message, "type": "error"}, "tool_result")
            
            self.last_command_str = None 
            
            async for event in self._get_next_action_from_llm(tool_result_str=tool_result_str, reason="Recovering from repetitive action error."):
                yield event
            return
        
        self.last_command_str = command_str
        
        try:
            command = json.loads(command_str)
        except (json.JSONDecodeError, KeyError) as e:
            app_logger.error(f"JSON parsing failed. Error: {e}. Original string was: {command_str}")
            raise e
            
        if "tool_name" in command:
            t_name = command["tool_name"]
            if t_name not in self.dependencies['STATE'].get('mcp_tools', {}) and t_name in self.dependencies['STATE'].get('mcp_prompts', {}):
                command["prompt_name"] = command.pop("tool_name")
            
        self.current_command = command
        
        if "prompt_name" in command:
            prompt_name = command.get("prompt_name")
            self.active_prompt_name = prompt_name
            
            mcp_client = self.dependencies['STATE'].get('mcp_client')
            if not mcp_client: raise RuntimeError("MCP client is not connected.")
            
            async with mcp_client.session("teradata_mcp_server") as temp_session:
                get_prompt_result = await temp_session.get_prompt(name=prompt_name)
            
            if get_prompt_result is None: raise ValueError(f"Prompt '{prompt_name}' could not be retrieved.")
            
            prompt_text = ""
            if hasattr(get_prompt_result, 'messages') and get_prompt_result.messages:
                first_message = get_prompt_result.messages[0]
                if hasattr(first_message, 'content') and hasattr(first_message.content, 'text'):
                    prompt_text = first_message.content.text
            
            if not prompt_text:
                raise ValueError(f"Could not extract text content from prompt '{prompt_name}'.")

            arguments = command.get("arguments", {})
            enriched_arguments, events_to_yield = self._enrich_arguments_from_history(prompt_text, arguments)
            for event in events_to_yield: yield event
            
            required_placeholders = set(re.findall(r'{(\w+)}', prompt_text))
            provided_keys = set(enriched_arguments.keys())
            missing_args = required_placeholders - provided_keys

            if missing_args:
                app_logger.error(f"LLM failed to provide required arguments for prompt '{prompt_name}'. Missing: {missing_args}")
                error_message = f"Prompt execution failed. The following required arguments were not provided: {list(missing_args)}"
                tool_result_str = json.dumps({
                    "tool_input": command, 
                    "tool_output": {"status": "error", "error_message": error_message}
                })
                yield _format_sse({"step": "System Error", "details": error_message, "type": "error"}, "tool_result")
                
                async for event in self._get_next_action_from_llm(tool_result_str=tool_result_str, reason="Recovering from missing arguments."):
                    yield event
                return 
            
            self.workflow_manager = WorkflowManager(prompt_text, enriched_arguments)
            self.is_workflow = True
            self.active_prompt_plan = prompt_text.format(**enriched_arguments)

            yield _format_sse({
                "step": f"Executing Prompt as a Workflow: {prompt_name}",
                "details": self.active_prompt_plan,
                "prompt_name": prompt_name
            }, "prompt_selected")
            
            async for event in self._handle_deciding():
                yield event

        elif "tool_name" in command:
            self.state = AgentState.EXECUTING_TOOL
        else:
            self.state = AgentState.SUMMARIZING

    async def _execute_standard_tool(self):
        corrected_command, guardrail_events = self._apply_context_guardrail(self.current_command)
        for event in guardrail_events: yield event
        self.current_command = corrected_command

        yield _format_sse({"step": "Tool Execution Intent", "details": self.current_command}, "tool_result")
        
        tool_name = self.current_command.get("tool_name")
        status_target = "chart" if tool_name == "viz_createChart" else "db"
        yield _format_sse({"target": status_target, "state": "busy"}, "status_indicator_update")
        
        tool_result = await mcp_adapter.invoke_mcp_tool(self.dependencies['STATE'], self.current_command)
        
        yield _format_sse({"target": status_target, "state": "idle"}, "status_indicator_update")

        if self.is_workflow:
            self._update_and_manage_context_stack(self.current_command, tool_result)

        if 'notification' in self.current_command:
            yield _format_sse({"step": "System Notification", "details": self.current_command['notification'], "type": "workaround"})
            del self.current_command['notification']

        tool_result_str = ""
        if isinstance(tool_result, dict) and "error" in tool_result:
            error_details = tool_result.get("data", tool_result.get("error", ""))
            if "Function" in str(error_details) and "does not exist" in str(error_details):
                self.globally_failed_tools.add(tool_name)
            tool_result_str = json.dumps({"tool_input": self.current_command, "tool_output": {"status": "error", "error_message": error_details}})
        else:
            tool_result_str = json.dumps({"tool_name": tool_name, "tool_output": tool_result})
            # --- MODIFIED: Ensure _add_to_structured_data is always called for workflows ---
            self._add_to_structured_data(tool_result)

        if isinstance(tool_result, dict) and tool_result.get("error") == "parameter_mismatch":
            yield _format_sse({"details": tool_result}, "request_user_input")
            self.state = AgentState.ERROR
            return

        yield _format_sse({"step": "Tool Execution Result", "details": tool_result, "tool_name": tool_name}, "tool_result")
        
        if self.is_workflow:
            self.last_tool_result_str = tool_result_str
            self.state = AgentState.DECIDING
        else:
            yield _format_sse({"step": "Thinking about the next action...", "details": f"Analyzing result from the `{tool_name}` tool to decide the next step."})
            async for event in self._get_next_action_from_llm(tool_result_str=tool_result_str, reason="Deciding next action based on tool result."):
                yield event

    async def _get_tool_constraints(self, tool_name: str):
        if tool_name in self.tool_constraints_cache:
            yield self.tool_constraints_cache[tool_name]
            return

        tool_definition = self.dependencies['STATE'].get('mcp_tools', {}).get(tool_name)
        constraints = {}
        
        if tool_definition:
            prompt_modifier = ""
            if any(k in tool_name.lower() for k in ["univariate", "standarddeviation", "negativevalues"]):
                prompt_modifier = "This tool is for quantitative analysis and requires a 'numeric' data type for `col_name`."
            elif any(k in tool_name.lower() for k in ["distinctcategories"]):
                prompt_modifier = "This tool is for categorical analysis and requires a 'character' data type for `col_name`."

            prompt = (
                f"Analyze the tool to determine if its `col_name` argument is for 'numeric', 'character', or 'any' type.\n"
                f"Tool: `{tool_definition.name}`\nDescription: \"{tool_definition.description}\"\nHint: {prompt_modifier}\n"
                "Respond with a single JSON object: {\"dataType\": \"numeric\" | \"character\" | \"any\"}"
            )
            
            yield _format_sse({"step": f"Inferring constraints for tool: {tool_name}", "type": "workaround"})
            yield _format_sse({"step": "Calling LLM", "details": "Determining tool constraints."})
            response_text, _, _ = await llm_handler.call_llm_api(
                self.dependencies['STATE']['llm'], prompt, raise_on_error=True,
                system_prompt_override="You are a JSON-only responding assistant.",
                dependencies=self.dependencies,
                reason="Determining tool constraints."
            )

            try:
                constraints = json.loads(re.search(r'\{.*\}', response_text, re.DOTALL).group(0))
            except (json.JSONDecodeError, AttributeError):
                constraints = {}
        
        self.tool_constraints_cache[tool_name] = constraints
        yield constraints

    async def _execute_column_iteration(self, column_subset: list = None):
        base_command = self.current_command
        tool_name = base_command.get("tool_name")
        base_args = base_command.get("arguments", {})
        
        db_name, table_name = base_args.get("db_name"), base_args.get("obj_name")
        if table_name and '.' in table_name and not db_name:
            db_name, table_name = table_name.split('.', 1)

        specific_column = base_args.get("col_name")
        if specific_column:
            yield _format_sse({"step": "Tool Execution Intent", "details": base_command}, "tool_result")
            yield _format_sse({"target": "db", "state": "busy"}, "status_indicator_update")
            col_result = await mcp_adapter.invoke_mcp_tool(self.dependencies['STATE'], base_command)
            yield _format_sse({"target": "db", "state": "idle"}, "status_indicator_update")
            
            if 'notification' in self.current_command:
                yield _format_sse({"step": "System Notification", "details": self.current_command['notification'], "type": "workaround"})

            if isinstance(col_result, dict) and col_result.get("error") == "parameter_mismatch":
                yield _format_sse({"details": col_result}, "request_user_input")
                self.state = AgentState.ERROR
                return

            yield _format_sse({"step": f"Tool Execution Result for column: {specific_column}", "details": col_result, "tool_name": tool_name}, "tool_result")
            self._add_to_structured_data(col_result)
            self.last_tool_result_str = json.dumps({"tool_name": tool_name, "tool_output": col_result})
            self.state = AgentState.DECIDING
            return

        all_columns_metadata = []
        if column_subset:
            all_columns_metadata = [{"ColumnName": col_name, "DataType": "UNKNOWN"} for col_name in column_subset]
        else:
            context_key = " > ".join([ctx['list'][ctx['index']] for ctx in self.context_stack if 'list' in ctx and ctx['index'] != -1])
            reused_metadata = False
            if context_key and context_key in self.structured_collected_data:
                for item in reversed(self.structured_collected_data[context_key]):
                    if isinstance(item, dict) and item.get("status") == "success" and item.get("metadata", {}).get("tool_name") in ["qlty_columnSummary", "base_columnDescription"]:
                        all_columns_metadata = item.get("results", [])
                        reused_metadata = True
                        break
            
            if not reused_metadata:
                cols_command = {"tool_name": "base_columnDescription", "arguments": {"db_name": db_name, "obj_name": table_name}}
                yield _format_sse({"target": "db", "state": "busy"}, "status_indicator_update")
                cols_result = await mcp_adapter.invoke_mcp_tool(self.dependencies['STATE'], cols_command)
                yield _format_sse({"target": "db", "state": "idle"}, "status_indicator_update")
                if not (cols_result and isinstance(cols_result, dict) and cols_result.get('status') == 'success' and cols_result.get('results')):
                    raise ValueError(f"Failed to retrieve column list for iteration. Response: {cols_result}")
                all_columns_metadata = cols_result.get('results', [])
                self._add_to_structured_data(cols_result)

        all_column_results = []
        
        first_column_to_try = next((col.get("ColumnName") for col in all_columns_metadata), None)
        if first_column_to_try and tool_name not in self.globally_failed_tools:
            preflight_command = {"tool_name": tool_name, "arguments": {**base_args, 'col_name': first_column_to_try}}
            try:
                yield _format_sse({"target": "db", "state": "busy"}, "status_indicator_update")
                preflight_result = await mcp_adapter.invoke_mcp_tool(self.dependencies['STATE'], preflight_command)
                yield _format_sse({"target": "db", "state": "idle"}, "status_indicator_update")
                if isinstance(preflight_result, dict) and "error" in preflight_result and "Function" in str(preflight_result.get("data", "")):
                    self.globally_failed_tools.add(tool_name)
                    yield _format_sse({ "step": f"Halting iteration for globally failed tool: {tool_name}", "type": "error" })
            except Exception as e:
                self.globally_failed_tools.add(tool_name)
                yield _format_sse({ "step": f"Halting iteration for problematic tool: {tool_name}", "details": str(e), "type": "error" })

        tool_constraints = None
        async for event in self._get_tool_constraints(tool_name):
            if isinstance(event, dict): tool_constraints = event
            else: yield event
        required_type = tool_constraints.get("dataType") if tool_constraints else None

        yield _format_sse({"target": "db", "state": "busy"}, "status_indicator_update")
        for column_info in all_columns_metadata:
            col_name = column_info.get("ColumnName")
            col_type = next((v for k, v in column_info.items() if "type" in k.lower()), "").upper()

            if required_type and col_type != "UNKNOWN":
                is_numeric = any(t in col_type for t in ["INT", "NUMERIC", "DECIMAL", "FLOAT"])
                is_char = any(t in col_type for t in ["CHAR", "VARCHAR", "TEXT"])
                if (required_type == "numeric" and not is_numeric) or (required_type == "character" and not is_char):
                    all_column_results.append({"status": "skipped", "reason": f"Tool requires {required_type}, but '{col_name}' is {col_type}."})
                    continue

            iter_command = {"tool_name": tool_name, "arguments": {**base_args, 'col_name': col_name}}
            col_result = await mcp_adapter.invoke_mcp_tool(self.dependencies['STATE'], iter_command)
            
            if isinstance(col_result, dict) and "error" in col_result and "Function" in str(col_result.get("data", "")):
                self.globally_failed_tools.add(tool_name)
                all_column_results.append({ "status": "error", "reason": f"Tool '{tool_name}' is non-functional."})
                break
            
            all_column_results.append(col_result)
        yield _format_sse({"target": "db", "state": "idle"}, "status_indicator_update")

        self._add_to_structured_data(all_column_results)
        self.last_tool_result_str = json.dumps({"tool_name": tool_name, "tool_output": all_column_results})
        self.state = AgentState.DECIDING

    def _build_just_in_time_context_prompt(self) -> str:
        if not self.context_stack: return ""
        context_lines = ["You are executing a nested plan." if len(self.context_stack) > 1 else "You are executing a plan that iterates over a list."]
        for i, context in enumerate(self.context_stack):
            context_type = context['type'].replace('_name', '').capitalize()
            current_item = f"`{context['list'][context['index']]}` (Item {context['index'] + 1} of {len(context['list'])})" if 0 <= context['index'] < len(context['list']) else "N/A"
            context_lines.append(f"{'  ' * i}- **Loop ({context_type}s)**: - **Current Item**: {current_item}")
        final_instruction = "Your immediate task is to continue the sub-plan for the **innermost current item**."
        return "\n--- **CURRENT LOOP STATE** ---\n" + "\n".join(context_lines) + "\n" + final_instruction + "\n---------------------------\n"

    async def _get_next_action_from_llm(self, tool_result_str: str | None = None, scoped_prompt_content: str | None = None, reason: str = "No reason provided."):
        prompt_for_next_step = "" 
        
        if self.is_workflow:
            app_logger.info("Applying deterministic, workflow-driven reasoning for next step.")
            current_step_info = self.workflow_manager.steps[self.workflow_manager.current_step_index]
            prompt_for_next_step = (
                "You are an expert assistant executing one specific step of a larger plan.\n\n"
                f"--- CURRENT TASK: {current_step_info['title']} ---\n"
                f"{scoped_prompt_content or ''}\n\n"
                "--- CONTEXT FROM PREVIOUS STEP ---\n"
                "The last action returned the following data. Use this information to complete your current task.\n"
                f"```json\n{self.last_tool_result_str or 'No previous tool result.'}\n```\n\n"
                "--- YOUR INSTRUCTIONS ---\n"
                "1.  **Analyze the provided data** and the task description above.\n"
                "2.  **Fulfill the task.** If the task is to describe or analyze something, provide that analysis directly.\n"
                "3.  Your response **MUST** start with `FINAL_ANSWER:`."
            )
            final_prompt_to_llm = prompt_for_next_step
        else:
            prompt_for_next_step = (
                f"You are an assistant that has just received data from a tool call. Your task is to decide if this data is enough to answer the user's original question, or if another step is needed.\n\n"
                f"--- User's Original Question ---\n"
                f"'{self.original_user_input}'\n\n"
                f"--- Data from Last Tool Call ---\n"
                f"{tool_result_str}\n\n"
                "--- Your Decision Process ---\n"
                "1.  **Analyze the Data:** Does the data above directly and completely answer the user's original question?\n"
                "2.  **Choose Your Action:**\n"
                "    -   If the data IS sufficient, your response **MUST** be a final answer to the user. Start your response with `FINAL_ANSWER:` and provide a brief summary.\n"
                "    -   If the data is NOT sufficient and you need more information, call another tool or prompt by providing the appropriate JSON block.\n"
                "    -   If the last tool call resulted in an error, you MUST attempt to recover.\n"
            )
            final_prompt_to_llm = prompt_for_next_step
        
        yield _format_sse({"step": "Calling LLM", "details": reason})

        self.next_action_str, statement_input_tokens, statement_output_tokens = await llm_handler.call_llm_api(
            self.dependencies['STATE']['llm'], final_prompt_to_llm, self.session_id, dependencies=self.dependencies, reason=reason
        )
        
        updated_session = session_manager.get_session(self.session_id)
        if updated_session:
            yield _format_sse({
                "statement_input": statement_input_tokens, "statement_output": statement_output_tokens,
                "total_input": updated_session.get("input_tokens", 0), "total_output": updated_session.get("output_tokens", 0)
            }, "token_update")
        
        if not self.next_action_str: raise ValueError("LLM failed to provide a response.")
        
        self.state = AgentState.DECIDING

    def _prepare_data_for_final_summary(self) -> str:
        summary_lines = []
        # --- MODIFIED: Ensure data_to_summarize correctly reflects workflow structure ---
        # If it's a workflow, we expect structured_collected_data to hold the relevant items.
        # If structured_collected_data is empty or not a dict, default to an empty dict for safe iteration.
        data_to_summarize = self.structured_collected_data if self.is_workflow and isinstance(self.structured_collected_data, dict) else {}

        # Add a default key if the structured data is empty but collected_data has items
        if not data_to_summarize and self.collected_data:
            data_to_summarize = {"Overall Workflow Results": self.collected_data}


        for context_key, items in data_to_summarize.items():
            summary_lines.append(f"- For context: `{context_key}`:")
            for item in items:
                if isinstance(item, list) and item and isinstance(item[0], dict):
                    tool_name = item[0].get("metadata", {}).get("tool_name", "Column-based tool")
                    summary_lines.append(f"  - Executed `{tool_name}`.")
                elif isinstance(item, dict):
                    tool_name = item.get("metadata", {}).get("tool_name")
                    status = item.get("status")
                    if status == "success" and "results" in item:
                        summary_lines.append(f"  - Tool `{tool_name}` succeeded with {len(item['results'])} results.")
                    elif status == "error":
                        summary_lines.append(f"  - Tool `{tool_name}` failed.")
        return "\n".join(summary_lines)

    async def _handle_summarizing(self):
        # --- MODIFIED: Ensure final_collected_data uses structured_collected_data for workflows ---
        final_collected_data = self.structured_collected_data if self.is_workflow else self.collected_data
        
        final_summary_text = ""
        # --- MODIFIED: Prioritize plain text FINAL_ANSWER, then JSON, then raw text ---
        # 1. Try to extract plain text FINAL_ANSWER:
        final_answer_match = re.search(r'FINAL_ANSWER:(.*)', self.next_action_str, re.DOTALL | re.IGNORECASE)
        if final_answer_match:
            final_summary_text = final_answer_match.group(1).strip()
        else:
            # 2. If not found, try to parse a JSON object with "FINAL_ANSWER" key
            json_match = re.search(r"\{.*\}", self.next_action_str, re.DOTALL)
            if json_match:
                try:
                    parsed_json = json.loads(json_match.group(0))
                    if "FINAL_ANSWER" in parsed_json:
                        final_summary_text = parsed_json["FINAL_ANSWER"].strip()
                except (json.JSONDecodeError, TypeError):
                    pass # Not a valid JSON or doesn't contain FINAL_ANSWER key
        
        # 3. If still no final_summary_text, use the raw next_action_str (stripped of "FINAL_ANSWER:")
        if not final_summary_text:
            final_summary_text = self.next_action_str.replace("FINAL_ANSWER:", "").strip()

        # --- END MODIFIED ---
        
        if not final_summary_text and self.is_workflow:
            yield _format_sse({"step": "Workflow finished, generating final summary..."})
            summarized_data_str = self._prepare_data_for_final_summary()
            
            final_prompt = (
                "You are a data analyst generating the final, user-facing summary of a complex task.\n\n"
                f"--- COLLECTED DATA SUMMARY ---\n{summarized_data_str}\n\n"
                "--- YOUR TASK ---\n"
                f"Generate a final, comprehensive answer for the user's original request: '{self.original_user_input}'.\n"
                "Your response MUST start with `FINAL_ANSWER:` and should not contain any other formatting or conversational text."
            )
            yield _format_sse({"step": "Calling LLM", "details": "Generating final workflow summary."})
            final_llm_response, _, _ = await llm_handler.call_llm_api(
                self.dependencies['STATE']['llm'], final_prompt, self.session_id, dependencies=self.dependencies,
                reason="Generating final workflow summary."
            )
            
            final_answer_match = re.search(r'FINAL_ANSWER:(.*)', final_llm_response, re.DOTALL | re.IGNORECASE)
            if final_answer_match:
                final_summary_text = final_answer_match.group(1).strip()
            else:
                final_summary_text = final_llm_response or "The agent finished its plan but did not provide a final summary."

        elif not final_summary_text:
            final_summary_text = self.next_action_str.replace("FINAL_ANSWER:", "").strip()

        formatter = OutputFormatter(
            llm_response_text=final_summary_text, 
            collected_data=final_collected_data, 
            is_workflow=self.is_workflow
        )
        final_html = formatter.render()
        
        session_manager.add_to_history(self.session_id, 'assistant', final_html)
        yield _format_sse({"final_answer": final_html}, "final_answer")
        self.state = AgentState.DONE

