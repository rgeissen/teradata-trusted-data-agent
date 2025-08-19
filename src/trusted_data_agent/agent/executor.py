# trusted_data_agent/agent/executor.py
import re
import json
import logging
import os
from enum import Enum, auto
from datetime import datetime, timedelta

from langchain_mcp_adapters.prompts import load_mcp_prompt

from trusted_data_agent.agent.formatter import OutputFormatter
from trusted_data_agent.core import session_manager
from trusted_data_agent.mcp import adapter as mcp_adapter
from trusted_data_agent.llm import handler as llm_handler
from trusted_data_agent.agent.workflow_executor import WorkflowExecutor
from trusted_data_agent.agent.prompts import ERROR_RECOVERY_PROMPT

app_logger = logging.getLogger("quart.app")

def get_prompt_text_content(prompt_obj):
    """
    Extracts the text content from a loaded prompt object, handling different
    potential formats returned by the MCP adapter.
    """
    if isinstance(prompt_obj, str):
        return prompt_obj
    if (isinstance(prompt_obj, list) and
        len(prompt_obj) > 0 and
        hasattr(prompt_obj[0], 'content') and
        isinstance(prompt_obj[0].content, str)):
        return prompt_obj[0].content
    elif (isinstance(prompt_obj, dict) and 
        'messages' in prompt_obj and
        isinstance(prompt_obj['messages'], list) and 
        len(prompt_obj['messages']) > 0 and
        'content' in prompt_obj['messages'][0] and
        isinstance(prompt_obj['messages'][0]['content'], dict) and
        'text' in prompt_obj['messages'][0]['content']):
        return prompt_obj['messages'][0]['content']['text']
    
    return ""

class AgentState(Enum):
    DECIDING = auto()
    EXECUTING_TOOL = auto()
    SUMMARIZING = auto()
    DONE = auto()
    ERROR = auto()

def unwrap_exception(e: BaseException) -> BaseException:
    """Recursively unwraps ExceptionGroups to find the root cause."""
    if isinstance(e, ExceptionGroup) and e.exceptions:
        return unwrap_exception(e.exceptions[0])
    return e

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
    AgentState = AgentState

    def __init__(self, session_id: str, initial_instruction: str, original_user_input: str, dependencies: dict):
        self.session_id = session_id
        self.original_user_input = original_user_input
        self.state = self.AgentState.DECIDING
        self.next_action_str = initial_instruction
        self.collected_data = []
        self.max_steps = 40
        self.current_command = None
        self.dependencies = dependencies
        self.tool_constraints_cache = {}
        self.globally_skipped_tools = set()
        self.last_command_str = None
        self.charting_intent_detected = self._detect_charting_intent(original_user_input)
        self.last_tool_output = None
        self.temp_data_holder = None
        self.llm_debug_history = []
        
        self.is_workflow = False
        self.active_prompt_name = None
        self.workflow_goal_prompt = ""
        self.structured_collected_data = {}

    @staticmethod
    def _format_sse(data: dict, event: str = None) -> str:
        msg = f"data: {json.dumps(data)}\n"
        if event is not None:
            msg += f"event: {event}\n"
        return f"{msg}\n"

    async def _call_llm_and_update_tokens(self, prompt: str, reason: str, system_prompt_override: str = None, raise_on_error: bool = False) -> tuple[str, int, int]:
        """
        A centralized wrapper for calling the LLM that handles token updates.
        This now returns the response and token counts directly.
        """
        response_text, statement_input_tokens, statement_output_tokens = await llm_handler.call_llm_api(
            self.dependencies['STATE']['llm'],
            prompt,
            self.session_id,
            dependencies=self.dependencies,
            reason=reason,
            system_prompt_override=system_prompt_override,
            raise_on_error=raise_on_error
        )
        self.llm_debug_history.append({"reason": reason, "response": response_text})
        app_logger.info(f"LLM RESPONSE (DEBUG): Reason='{reason}', Response='{response_text}'")
        return response_text, statement_input_tokens, statement_output_tokens

    def _detect_charting_intent(self, user_input: str) -> bool:
        """
        Detects if the user's original query explicitly asks for a chart or graphical representation.
        """
        chart_keywords = ["chart", "graph", "plot", "visualize", "diagram", "representation", "picture"]
        return any(keyword in user_input.lower() for keyword in chart_keywords)

    def _add_to_structured_data(self, tool_result: dict):
        context_key = f"Workflow: {self.active_prompt_name}" if self.active_prompt_name else "Workflow Results"
        if context_key not in self.structured_collected_data:
            self.structured_collected_data[context_key] = []
        
        if (self.is_workflow and 
            isinstance(tool_result, dict) and
            tool_result.get("status") == "success" and
            tool_result.get("results") and 
            isinstance(tool_result["results"], list) and
            len(tool_result["results"]) > 0 and
            isinstance(tool_result["results"][0], dict) and
            "response" in tool_result["results"][0]):
            
            task_name = "CoreLLMTask Result"
            
            modified_tool_result = tool_result.copy()
            modified_tool_result.setdefault("metadata", {})["tool_name"] = task_name
            self.structured_collected_data[context_key].append(modified_tool_result)
            app_logger.info(f"Added modified CoreLLMTask result to structured data under key: '{context_key}' for workflow.")
        elif isinstance(tool_result, dict) and "results" in tool_result:
            self.structured_collected_data[context_key].append(tool_result)
            app_logger.info(f"Added tool result to structured data under key: '{context_key}' for workflow.")
        elif isinstance(tool_result, list):
             self.structured_collected_data[context_key].extend(tool_result)

    async def run(self):
        for i in range(self.max_steps):
            if self.state in [self.AgentState.SUMMARIZING, self.AgentState.DONE, self.AgentState.ERROR]:
                break
            try:
                if self.is_workflow:
                    workflow_executor = WorkflowExecutor(parent_executor=self)
                    async for event in workflow_executor.run():
                        yield event
                    if self.state == self.AgentState.SUMMARIZING:
                        break
                elif self.state == self.AgentState.DECIDING:
                    async for event in self._handle_deciding():
                        yield event
                elif self.state == self.AgentState.EXECUTING_TOOL:
                    async for event in self._execute_tool_with_orchestrators():
                        yield event

            except Exception as e:
                root_exception = unwrap_exception(e)
                app_logger.error(f"Error in state {self.state.name}: {root_exception}", exc_info=True)
                async for event in self._recover_with_llm(f"The plan failed with this error: {root_exception}"):
                    yield event

        if self.state == self.AgentState.SUMMARIZING:
            async for event in self._generate_final_summary():
                yield event
        elif self.state == self.AgentState.ERROR:
             yield self._format_sse({"error": "Execution stopped due to an unrecoverable workflow error.", "details": "The agent entered an error state and could not complete the multi-step plan."}, "error")


    async def _recover_with_llm(self, error_message: str):
        failed_tool_name = self.current_command.get("tool_name") if self.current_command else "N/A"
        self.globally_skipped_tools.add(failed_tool_name)
        
        recovery_prompt = ERROR_RECOVERY_PROMPT.format(
            user_question=self.original_user_input,
            error_message=error_message,
            failed_tool_name=failed_tool_name,
            all_collected_data=self._prepare_data_for_prompt(),
            workflow_goal_and_plan=self.workflow_goal_prompt
        )
        reason = "Recovering from error."
        yield self._format_sse({"step": "Calling LLM for Recovery", "details": reason})
        response_text, input_tokens, output_tokens = await self._call_llm_and_update_tokens(
            prompt=recovery_prompt, 
            reason=reason,
            raise_on_error=True
        )
        
        updated_session = session_manager.get_session(self.session_id)
        if updated_session:
            yield self._format_sse({"statement_input": input_tokens, "statement_output": output_tokens, "total_input": updated_session.get("input_tokens", 0), "total_output": updated_session.get("output_tokens", 0)}, "token_update")

        self.next_action_str = response_text
        self.is_workflow = False
        self.state = self.AgentState.DECIDING

    async def _execute_tool_with_orchestrators(self):
        is_range_candidate, date_param_name, tool_supports_range = self._is_date_query_candidate()
        if is_range_candidate and not tool_supports_range:
            async for event in self._classify_date_query_type(): yield event
            if self.temp_data_holder and self.temp_data_holder.get('type') == 'range':
                async for event in self._execute_date_range_orchestrator(date_param_name, self.temp_data_holder.get('phrase')): yield event
                return

        tool_name = self.current_command.get("tool_name")
        if self.dependencies['STATE'].get('tool_scopes', {}).get(tool_name) == 'column' and not self.current_command.get("arguments", {}).get("column_name"):
            async for event in self._execute_column_iteration(): yield event
        else:
            async for event in self._execute_standard_tool():
                yield event

    def _is_date_query_candidate(self) -> tuple[bool, str, bool]:
        if not self.current_command:
            return False, None, False

        tool_name = self.current_command.get("tool_name")
        tool_spec = self.dependencies['STATE'].get('mcp_tools', {}).get(tool_name)
        if not tool_spec or not hasattr(tool_spec, 'args') or not isinstance(tool_spec.args, dict):
            return False, None, False

        tool_arg_names = set(tool_spec.args.keys())
        tool_supports_range = 'start_date' in tool_arg_names and 'end_date' in tool_arg_names
        
        args = self.current_command.get("arguments", {})
        date_param_name = next((param for param in args if 'date' in param.lower()), None)
        
        return bool(date_param_name), date_param_name, tool_supports_range

    async def _classify_date_query_type(self):
        classification_prompt = (
            f"You are a query classifier. Your only task is to analyze a user's request for date information. "
            f"Analyze the following query: '{self.original_user_input}'. "
            "First, determine if it refers to a 'single' date or a 'range' of dates. "
            "Second, extract the specific phrase that describes the date or range. "
            "Your response MUST be ONLY a JSON object with two keys: 'type' and 'phrase'."
        )
        
        reason="Classifying date query."
        yield self._format_sse({"step": "Calling LLM", "details": reason})
        response_str, input_tokens, output_tokens = await self._call_llm_and_update_tokens(
            prompt=classification_prompt,
            reason=reason,
            system_prompt_override="You are a JSON-only responding assistant.",
            raise_on_error=True
        )

        updated_session = session_manager.get_session(self.session_id)
        if updated_session:
            yield self._format_sse({ "statement_input": input_tokens, "statement_output": output_tokens, "total_input": updated_session.get("input_tokens", 0), "total_output": updated_session.get("output_tokens", 0) }, "token_update")
        
        try:
            self.temp_data_holder = json.loads(response_str)
        except (json.JSONDecodeError, KeyError):
            self.temp_data_holder = {'type': 'single', 'phrase': self.original_user_input}


    async def _execute_date_range_orchestrator(self, date_param_name: str, date_phrase: str):
        tool_name = self.current_command.get("tool_name")
        yield self._format_sse({
            "step": "System Orchestration",
            "details": f"Detected a date range query ('{date_phrase}') for a single-day tool ('{tool_name}').",
            "type": "workaround"
        })

        date_command = {"tool_name": "util_getCurrentDate"}
        date_result = await mcp_adapter.invoke_mcp_tool(self.dependencies['STATE'], date_command)
        if not (date_result and date_result.get("status") == "success" and date_result.get("results")):
            raise RuntimeError("Date Range Orchestrator failed to fetch current date.")
        current_date_str = date_result["results"][0].get("current_date")

        conversion_prompt = (
            f"You are a date range calculation assistant. Given that the current date is {current_date_str}, "
            f"what are the start and end dates for '{date_phrase}'? "
            "Your response MUST be ONLY a JSON object with 'start_date' and 'end_date' in YYYY-MM-DD format."
        )

        reason = "Calculating date range."
        yield self._format_sse({"step": "Calling LLM", "details": reason})
        range_response_str, input_tokens, output_tokens = await self._call_llm_and_update_tokens(
            prompt=conversion_prompt,
            reason=reason,
            system_prompt_override="You are a JSON-only responding assistant.",
            raise_on_error=True
        )
        updated_session = session_manager.get_session(self.session_id)
        if updated_session:
            yield self._format_sse({ "statement_input": input_tokens, "statement_output": output_tokens, "total_input": updated_session.get("input_tokens", 0), "total_output": updated_session.get("output_tokens", 0) }, "token_update")
        
        try:
            range_data = json.loads(range_response_str)
            start_date = datetime.strptime(range_data['start_date'], '%Y-%m-%d').date()
            end_date = datetime.strptime(range_data['end_date'], '%Y-%m-%d').date()
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise RuntimeError(f"Date Range Orchestrator failed to parse date range from LLM. Response: {range_response_str}. Error: {e}")

        all_results = []
        current_date_in_loop = start_date
        
        yield self._format_sse({"target": "db", "state": "busy"}, "status_indicator_update")
        while current_date_in_loop <= end_date:
            date_str = current_date_in_loop.strftime('%Y-%m-%d')
            yield self._format_sse({"step": f"Processing data for: {date_str}"})
            
            command_for_day = {**self.current_command, 'arguments': {**self.current_command['arguments'], date_param_name: date_str}}
            day_result = await mcp_adapter.invoke_mcp_tool(self.dependencies['STATE'], command_for_day)
            
            if isinstance(day_result, dict) and day_result.get("status") == "success" and day_result.get("results"):
                all_results.extend(day_result["results"])
            
            current_date_in_loop += timedelta(days=1)
        yield self._format_sse({"target": "db", "state": "idle"}, "status_indicator_update")
        
        final_tool_output = {
            "status": "success",
            "metadata": {"tool_name": tool_name, "comment": f"Consolidated results for date range: {start_date} to {end_date}"},
            "results": all_results
        }
        self._add_to_structured_data(final_tool_output)
        self.state = self.AgentState.SUMMARIZING
        self.next_action_str = "FINAL_ANSWER: "

    def _enrich_arguments_from_history(self, prompt_name: str, arguments: dict) -> tuple[dict, list]:
        events_to_yield = []
        enriched_args = arguments.copy()
        
        prompt_definition = self.dependencies['STATE'].get('mcp_prompts', {}).get(prompt_name)
        if not prompt_definition or not hasattr(prompt_definition, 'arguments'):
            return enriched_args, events_to_yield

        required_arg_names = {arg.name for arg in prompt_definition.arguments} if prompt_definition.arguments else set()
        
        for arg_name in required_arg_names:
            if arg_name in enriched_args: continue

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
                            if arg_name in args_to_check:
                                enriched_args[arg_name] = args_to_check[arg_name]
                                break
                except (json.JSONDecodeError, TypeError):
                    continue
            
            if arg_name in enriched_args:
                app_logger.info(f"Inferred '{arg_name}' from history: '{enriched_args[arg_name]}'")
                events_to_yield.append(self._format_sse({
                    "step": "System Correction",
                    "details": f"LLM omitted '{arg_name}'. System inferred it from history.",
                    "type": "workaround"
                }))

        return enriched_args, events_to_yield

    async def _handle_deciding(self):
        is_final_answer = "FINAL_ANSWER:" in self.next_action_str.upper() or "SYSTEM_ACTION_COMPLETE" in self.next_action_str
        
        if not is_final_answer:
            yield self._format_sse({"step": "LLM has decided on an action", "details": self.next_action_str}, "llm_thought")

        if "SYSTEM_ACTION_COMPLETE" in self.next_action_str:
            self.state = self.AgentState.SUMMARIZING
            return

        if re.search(r'FINAL_ANSWER:', self.next_action_str, re.IGNORECASE):
            self.state = self.AgentState.SUMMARIZING
            return

        command_str = None
        json_match = re.search(r"```json\s*\n(.*?)\n\s*```|(\{.*\})", self.next_action_str, re.DOTALL)
        if json_match:
            command_str = json_match.group(1) or json_match.group(2)
        
        if not command_str:
            self.state = self.AgentState.SUMMARIZING
            return
        
        command = json.loads(command_str.strip())

        # --- MODIFIED: Normalize common key name hallucinations ---
        if "tool" in command and "tool_name" not in command:
            command["tool_name"] = command.pop("tool")
        if "toolInput" in command and "arguments" not in command:
            command["arguments"] = command.pop("toolInput")
        if "prompt" in command and "prompt_name" not in command:
            command["prompt_name"] = command.pop("prompt")

        if "tool_name" in command:
            potential_prompt_name = command.get("tool_name")
            all_prompts = self.dependencies['STATE'].get('mcp_prompts', {})
            if potential_prompt_name in all_prompts:
                yield self._format_sse({
                    "step": "System Correction",
                    "details": f"LLM incorrectly used 'tool_name' for a prompt. Correcting to 'prompt_name' for '{potential_prompt_name}'.",
                    "type": "workaround"
                })
                command["prompt_name"] = command.pop("tool_name")

        if "prompt_name" in command:
            self.is_workflow = True
            self.active_prompt_name = command.get("prompt_name")
            
            mcp_client = self.dependencies['STATE'].get('mcp_client')
            if not mcp_client: raise RuntimeError("MCP client is not connected.")
            
            final_args_for_fetch, enrich_events = self._enrich_arguments_from_history(
                self.active_prompt_name, command.get("arguments", {})
            )
            for event in enrich_events:
                yield event
            
            async with mcp_client.session("teradata_mcp_server") as temp_session:
                prompt_obj = await load_mcp_prompt(
                    temp_session, 
                    name=self.active_prompt_name, 
                    arguments=final_args_for_fetch
                )

            if not prompt_obj: raise ValueError(f"Prompt '{self.active_prompt_name}' could not be loaded or rendered.")
            
            prompt_text = get_prompt_text_content(prompt_obj)
            if not prompt_text:
                raise ValueError(f"Could not extract text content from rendered prompt '{self.active_prompt_name}'.")

            self.workflow_goal_prompt = prompt_text
            
            yield self._format_sse({
                "step": f"Executing Prompt as a Workflow: {self.active_prompt_name}",
                "details": self.workflow_goal_prompt,
                "prompt_name": self.active_prompt_name
            }, "prompt_selected")
            
            return

        if not self.is_workflow and command_str == self.last_command_str:
            app_logger.warning(f"LOOP DETECTED: The LLM is trying to repeat the exact same command. Command: {command_str}")
            error_message = f"Repetitive action detected."
            tool_result_str = json.dumps({"tool_input": json.loads(command_str), "tool_output": {"status": "error", "error_message": error_message}})
            yield self._format_sse({"step": "System Error", "details": error_message, "type": "error"}, "tool_result")
            self.last_command_str = None 
            async for event in self._get_next_action_from_llm(tool_result_str=tool_result_str, reason="Recovering from repetitive action error."):
                yield event
            return
        
        self.last_command_str = command_str
        self.current_command = command
            
        if "tool_name" in command:
            self.state = self.AgentState.EXECUTING_TOOL
        else:
            self.state = self.AgentState.SUMMARIZING

    async def _execute_standard_tool(self):
        tool_name = self.current_command.get("tool_name")
        yield self._format_sse({"step": "Tool Execution Intent", "details": self.current_command}, "tool_result")
        
        status_target = "chart" if tool_name == "viz_createChart" else "db"
        yield self._format_sse({"target": status_target, "state": "busy"}, "status_indicator_update")
        
        tool_result = await mcp_adapter.invoke_mcp_tool(self.dependencies['STATE'], self.current_command)
        
        yield self._format_sse({"target": status_target, "state": "idle"}, "status_indicator_update")

        if 'notification' in self.current_command:
            yield self._format_sse({"step": "System Notification", "details": self.current_command['notification'], "type": "workaround"})
            del self.current_command['notification']

        self.last_tool_output = tool_result

        if isinstance(tool_result, dict) and tool_result.get("status") == "error":
            error_details = tool_result.get("data", tool_result.get("error", ""))
            self.globally_skipped_tools.add(tool_name)
            tool_result_str = json.dumps({"tool_input": self.current_command, "tool_output": {"status": "error", "error_message": error_details}})
            yield self._format_sse({"details": tool_result, "tool_name": tool_name}, "tool_error")
            async for event in self._recover_with_llm(tool_result_str):
                yield event
            return
        
        tool_result_str = json.dumps({"tool_name": tool_name, "tool_output": tool_result})
        
        self.last_tool_result_str = tool_result_str 
        if self.is_workflow:
            self._add_to_structured_data(tool_result)
        else:
            self.collected_data.append(tool_result)

        if isinstance(tool_result, dict) and tool_result.get("error") == "parameter_mismatch":
            yield self._format_sse({"details": tool_result}, "request_user_input")
            self.state = self.AgentState.ERROR
            return

        yield self._format_sse({"step": "Tool Execution Result", "details": tool_result, "tool_name": tool_name}, "tool_result")
        
        if not self.is_workflow:
            async for event in self._get_next_action_from_llm(
                tool_result_str=tool_result_str, 
                reason="Deciding next action based on tool result."
            ):
                yield event
        else:
            self.state = self.AgentState.DECIDING

    async def _get_tool_constraints(self, tool_name: str) -> dict:
        if tool_name in self.tool_constraints_cache:
            return self.tool_constraints_cache[tool_name]

        tool_definition = self.dependencies['STATE'].get('mcp_tools', {}).get(tool_name)
        constraints = {}
        
        if tool_definition:
            prompt_modifier = ""
            if any(k in tool_name.lower() for k in ["univariate", "standarddeviation", "negativevalues"]):
                prompt_modifier = "This tool is for quantitative analysis and requires a 'numeric' data type for `column_name`."
            elif any(k in tool_name.lower() for k in ["distinctcategories"]):
                prompt_modifier = "This tool is for categorical analysis and requires a 'character' data type for `column_name`."

            prompt = (
                f"Analyze the tool to determine if its `column_name` argument is for 'numeric', 'character', or 'any' type.\n"
                f"Tool: `{tool_definition.name}`\nDescription: \"{tool_definition.description}\"\nHint: {prompt_modifier}\n"
                "Respond with a single JSON object: {\"dataType\": \"numeric\" | \"character\" | \"any\"}"
            )
            
            reason="Determining tool constraints."
            response_text, _, _ = await self._call_llm_and_update_tokens(
                prompt=prompt,
                reason=reason,
                system_prompt_override="You are a JSON-only responding assistant.",
                raise_on_error=True
            )

            try:
                constraints = json.loads(re.search(r'\{.*\}', response_text, re.DOTALL).group(0))
            except (json.JSONDecodeError, AttributeError):
                constraints = {}
        
        self.tool_constraints_cache[tool_name] = constraints
        return constraints

    async def _execute_column_iteration(self):
        base_command = self.current_command
        tool_name = base_command.get("tool_name")
        base_args = base_command.get("arguments", {})
        
        db_name, table_name = base_args.get("db_name"), base_args.get("obj_name")
        if table_name and '.' in table_name and not db_name:
            db_name, table_name = table_name.split('.', 1)

        specific_column = base_args.get("column_name")
        if specific_column:
            yield self._format_sse({"step": "Tool Execution Intent", "details": base_command}, "tool_result")
            yield self._format_sse({"target": "db", "state": "busy"}, "status_indicator_update")
            col_result = await mcp_adapter.invoke_mcp_tool(self.dependencies['STATE'], base_command)
            yield self._format_sse({"target": "db", "state": "idle"}, "status_indicator_update")
            
            if 'notification' in self.current_command:
                yield self._format_sse({"step": "System Notification", "details": self.current_command['notification'], "type": "workaround"})

            if isinstance(col_result, dict) and col_result.get("error") == "parameter_mismatch":
                yield self._format_sse({"details": col_result}, "request_user_input")
                self.state = self.AgentState.ERROR
                return

            yield self._format_sse({"step": f"Tool Execution Result for column: {specific_column}", "details": col_result, "tool_name": tool_name}, "tool_result")
            self._add_to_structured_data(col_result)
            self.last_tool_result_str = json.dumps({"tool_name": tool_name, "tool_output": col_result})
            self.last_tool_output = col_result
            self.state = self.AgentState.DECIDING
            return

        cols_command = {"tool_name": "base_columnDescription", "arguments": {"db_name": db_name, "obj_name": table_name}}
        yield self._format_sse({"target": "db", "state": "busy"}, "status_indicator_update")
        cols_result = await mcp_adapter.invoke_mcp_tool(self.dependencies['STATE'], cols_command)
        yield self._format_sse({"target": "db", "state": "idle"}, "status_indicator_update")
        if not (cols_result and isinstance(cols_result, dict) and cols_result.get('status') == 'success' and cols_result.get('results')):
            raise ValueError(f"Failed to retrieve column list for iteration. Response: {cols_result}")
        all_columns_metadata = cols_result.get('results', [])
        self._add_to_structured_data(cols_result)

        all_column_results = []
        
        reason="Determining tool constraints."
        yield self._format_sse({"step": "Calling LLM", "details": reason})
        tool_constraints = await self._get_tool_constraints(tool_name)
        required_type = tool_constraints.get("dataType") if tool_constraints else None

        yield self._format_sse({"target": "db", "state": "busy"}, "status_indicator_update")
        for column_info in all_columns_metadata:
            column_name = column_info.get("ColumnName")
            col_type = next((v for k, v in column_info.items() if "type" in k.lower()), "").upper()

            if required_type and col_type != "UNKNOWN":
                is_numeric = any(t in col_type for t in ["INT", "NUMERIC", "DECIMAL", "FLOAT"])
                is_char = any(t in col_type for t in ["CHAR", "VARCHAR", "TEXT"])
                if (required_type == "numeric" and not is_numeric) or (required_type == "character" and not is_char):
                    all_column_results.append({"status": "skipped", "reason": f"Tool requires {required_type}, but '{column_name}' is {col_type}."})
                    continue

            iter_command = {"tool_name": tool_name, "arguments": {**base_args, 'column_name': column_name}}
            col_result = await mcp_adapter.invoke_mcp_tool(self.dependencies['STATE'], iter_command)
            all_column_results.append(col_result)
        yield self._format_sse({"target": "db", "state": "idle"}, "status_indicator_update")

        self._add_to_structured_data(all_column_results)
        self.last_tool_result_str = json.dumps({"tool_name": tool_name, "tool_output": all_column_results})
        self.last_tool_output = {"metadata": {"tool_name": tool_name}, "results": all_column_results, "status": "success"}
        self.state = self.AgentState.DECIDING

    def _prepare_data_for_prompt(self) -> str:
        """
        Gathers all successful tool results and formats them into a concise, readable string.
        """
        successful_results = []
        for item in self.collected_data:
            if isinstance(item, dict) and item.get("status") == "success" and "results" in item:
                if item.get("type") != "chart":
                    successful_results.append(item)
        
        if not successful_results:
            return "No data has been successfully collected yet."
            
        summary_list = []
        for result in successful_results:
            tool_name = result.get("metadata", {}).get("tool_name", "Unknown Tool")
            result_count = len(result.get("results", []))
            summary_list.append(f"• Tool `{tool_name}` returned {result_count} rows of data.")
            
        return "\n".join(summary_list)

    async def _get_next_action_from_llm(self, tool_result_str: str | None = None, reason: str = "No reason provided."):
        all_collected_data_str = self._prepare_data_for_prompt()

        charting_guidance = ""
        if self.charting_intent_detected:
            is_data_tool_success = False
            tool_name_from_result = None
            if tool_result_str:
                try:
                    tool_output = json.loads(tool_result_str).get("tool_output", {})
                    if tool_output.get("status") == "success" and tool_output.get("results"):
                        tool_name_from_result = tool_output.get("metadata", {}).get("tool_name")
                        data_gathering_tools = ["qlty_distinctCategories", "base_databaseList", "base_readQuery"]
                        if tool_name_from_result in data_gathering_tools:
                            is_data_tool_success = True
                except json.JSONDecodeError:
                    pass 

                if is_data_tool_success:
                    charting_guidance = (
                        "**CRITICAL CHARTING DIRECTIVE**: The user explicitly requested a chart, and you have successfully gathered relevant data from the previous tool call using the `{tool_name_from_result}` tool. Your **NEXT ACTION MUST BE TO CALL `viz_createChart`**. Do NOT re-call data gathering tools. Use the `results` array from the 'Data from Last Tool Call' directly as the `data` argument for `viz_createChart`. Focus solely on creating the requested visualization.\n"
                    ).format(tool_name_from_result=tool_name_from_result or "data-gathering")
                else:
                    charting_guidance = (
                        "**CRITICAL CHARTING DIRECTIVE**: The user explicitly requested a chart. If the 'Data from Last Tool Call' is suitable for a chart, your **next action MUST be to call `viz_createChart`**. Do NOT re-call data gathering tools. Focus on creating the requested visualization.\n"
                    )

        failed_tools_list = list(self.globally_skipped_tools)
        failed_tools_context = f"--- FAILED TOOLS (DO NOT RE-CALL) ---\n{', '.join(failed_tools_list)}" if failed_tools_list else ""
        
        prompt_for_next_step = (
            "You are an assistant responsible for coordinating a data gathering plan. Your task is to decide if enough data has been collected to answer the user's question.\n\n"
            f"--- User's Original Question ---\n"
            f"'{self.original_user_input}'\n\n"
            f"--- All Data Collected So Far ---\n"
            f"{all_collected_data_str}\n\n"
            f"--- Data from Last Tool Call ---\n"
            f"{tool_result_str}\n\n"
            f"{charting_guidance}"
            f"{failed_tools_context}"
            "--- Your Decision Process ---\n"
            "1.  **Analyze the situation:** Your primary goal is to gather all necessary data to comprehensively answer the user's question. You may need to call several tools in sequence.\n"
            "2.  **Choose Your Action:**\n"
            "    -   If you need more information to answer the question, call another tool or prompt by providing the appropriate JSON block.\n"
            "    -   If you are confident that you have now gathered **all** the necessary data from this and any previous tool calls, your response **MUST** be only the exact string `SYSTEM_ACTION_COMPLETE`. This will signal the final report writer to begin.\n"
            "    -   If the last tool call resulted in an error, you MUST attempt to recover by calling a corrected tool.\n"
        )
        
        yield self._format_sse({"step": "Calling LLM", "details": reason})
        response_text, input_tokens, output_tokens = await self._call_llm_and_update_tokens(prompt=prompt_for_next_step, reason=reason)
        
        updated_session = session_manager.get_session(self.session_id)
        if updated_session:
            yield self._format_sse({ "statement_input": input_tokens, "statement_output": output_tokens, "total_input": updated_session.get("input_tokens", 0), "total_output": updated_session.get("output_tokens", 0) }, "token_update")
        
        self.next_action_str = response_text
        if not self.next_action_str: raise ValueError("LLM failed to provide a response.")
        
        self.state = self.AgentState.DECIDING

    async def _generate_final_summary(self):
        final_collected_data = self.structured_collected_data if self.is_workflow else self.collected_data
        
        final_summary_text = ""
        if (self.is_workflow and 
            isinstance(self.last_tool_output, dict) and
            self.last_tool_output.get("status") == "success" and
            self.last_tool_output.get("results") and 
            isinstance(self.last_tool_output["results"], list) and
            len(self.last_tool_output["results"]) > 0 and
            isinstance(self.last_tool_output["results"][0], dict) and
            "response" in self.last_tool_output["results"][0]):
            
            yield self._format_sse({"step": "Finalizing Report", "details": "Using pre-formatted summary from last workflow step."}, "llm_thought")
            final_summary_text = self.last_tool_output["results"][0]["response"]
            app_logger.info("Using pre-formatted FINAL_ANSWER text from the last CoreLLMTask in workflow.")
        
        elif self.last_tool_output and self.last_tool_output.get("type") == "chart":
            yield self._format_sse({"step": "Finalizing Report", "details": "Invoking chart-specific report writer to analyze visualization data."}, "llm_thought")
            data_for_llm_analysis = None
            if final_collected_data and len(final_collected_data) >= 2:
                data_for_llm_analysis = final_collected_data[-2]
            
            if not data_for_llm_analysis:
                app_logger.warning("Could not find source data for chart analysis. Reverting to simple summary.")
                final_summary_text = "FINAL_ANSWER: The chart has been generated and is displayed below."
            else:
                final_prompt = (
                    "You are an expert data analyst. Your task is to provide a final, comprehensive, and insightful summary of the user's request.\n\n"
                    f"--- USER'S ORIGINAL QUESTION ---\n"
                    f"'{self.original_user_input}'\n\n"
                    f"--- RELEVANT DATA COLLECTED ---\n"
                    "You have successfully executed tools and collected the following data. Analyze this data to generate your response.\n"
                    f"```json\n{json.dumps(data_for_llm_analysis, indent=2)}\n```\n\n"
                    "--- YOUR INSTRUCTIONS ---\n"
                    "1.  **Analyze the data:** Identify key findings, notable trends, or important facts from the raw data provided.\n"
                    "2.  **Provide context:** Briefly introduce the data source from the user's original question.\n"
                    "3.  **DO NOT simply describe the chart's dimensions or appearance.** Go beyond "
                    "\"The chart shows the distribution...\" and provide actual insights like "
                    "\"The data reveals that females are the largest demographic...\"\n"
                    "4.  Your response **MUST** start with `FINAL_ANSWER:` and include a natural language summary followed by a brief statement indicating that the chart is shown below. Do not wrap this final response in a JSON object."
                )
                reason="Generating final summary with data analysis."
                yield self._format_sse({"step": "Calling LLM", "details": reason})
                final_llm_response, input_tokens, output_tokens = await self._call_llm_and_update_tokens(prompt=final_prompt, reason=reason)
                
                updated_session = session_manager.get_session(self.session_id)
                if updated_session:
                    yield self._format_sse({ "statement_input": input_tokens, "statement_output": output_tokens, "total_input": updated_session.get("input_tokens", 0), "total_output": updated_session.get("output_tokens", 0) }, "token_update")
                
                final_summary_text = final_llm_response
        
        else:
            yield self._format_sse({"step": "Finalizing Report", "details": "Invoking general report writer to synthesize all collected data."}, "llm_thought")
            data_for_summary = self._prepare_data_for_final_summary()
            
            final_prompt = (
                "You are an expert data analyst. Your task is to synthesize all collected data into a clear, concise, and insightful final answer for the user.\n\n"
                f"--- USER'S ORIGINAL QUESTION ---\n"
                f"'{self.original_user_input}'\n\n"
                f"--- ALL RELEVANT DATA COLLECTED ---\n"
                "The following data was gathered from one or more tool calls. Analyze this data to generate your response.\n"
                f"```json\n{data_for_summary}\n```\n\n"
                "--- YOUR INSTRUCTIONS ---\n"
                "1.  **Adopt the Persona of a Data Analyst:** Your goal is to provide a holistic analysis and deliver actionable insights, not just report numbers.\n"
                "2.  **Go Beyond the Obvious:** Start with the primary findings (like data completeness or null counts), but then scrutinize the data for secondary insights, patterns, or anomalies. For a data quality assessment, this could include looking for unexpected distributions, a lack of negative values in numeric columns where appropriate, or other indicators of high-quality data.\n"
                "3.  **Structure Your Answer:** Begin with a high-level summary that directly answers the user's question. Then, if applicable, use bullet points to highlight key, specific observations from the data.\n"
                "4.  **CRITICAL:** Your entire response **MUST** begin with the exact prefix `FINAL_ANSWER:`, followed by your natural language summary. Do not add any other text before this prefix.\n"
            )
            reason="Generating final summary from collected tool data."
            yield self._format_sse({"step": "Calling LLM to write final report", "details": reason})
            final_llm_response, input_tokens, output_tokens = await self._call_llm_and_update_tokens(prompt=final_prompt, reason=reason)
            
            updated_session = session_manager.get_session(self.session_id)
            if updated_session:
                yield self._format_sse({ "statement_input": input_tokens, "statement_output": output_tokens, "total_input": updated_session.get("input_tokens", 0), "total_output": updated_session.get("output_tokens", 0) }, "token_update")
            
            final_summary_text = final_llm_response

        if final_summary_text.strip().upper().startswith("FINAL_ANSWER:"):
            clean_summary = final_summary_text.strip()[len("FINAL_ANSWER:"):].strip()
        else:
            clean_summary = final_summary_text
        
        if not clean_summary:
             clean_summary = "The agent has completed its work. The collected data is displayed below."

        yield self._format_sse({
            "step": "LLM has generated the final answer",
            "details": clean_summary
        }, "llm_thought")

        formatter = OutputFormatter(
            llm_response_text=clean_summary,
            collected_data=final_collected_data,
            is_workflow=self.is_workflow
        )
        final_html = formatter.render()
        
        session_manager.add_to_history(self.session_id, 'assistant', final_html)
        yield self._format_sse({"final_answer": final_html}, "final_answer")
        self.state = self.AgentState.DONE

    def _prepare_data_for_final_summary(self) -> str:
        data_source = self.structured_collected_data if self.is_workflow else self.collected_data
        
        items_to_process = []
        if isinstance(data_source, dict):
            for item_list in data_source.values():
                items_to_process.extend(item_list)
        else:
            items_to_process = data_source

        successful_results = []
        for item in items_to_process:
            if isinstance(item, list):
                for sub_item in item:
                    if isinstance(sub_item, dict) and sub_item.get("status") == "success" and "results" in sub_item:
                        successful_results.append(sub_item)
            elif isinstance(item, dict) and item.get("status") == "success" and "results" in item:
                successful_results.append(item)
        
        if not successful_results:
            return "No data was collected from successful tool executions."
            
        return json.dumps([res for res in successful_results if res.get("type") != "chart"], indent=2, ensure_ascii=False)

    def _safe_parse_json(self, text):
        markdown_match = re.search(r"```json\s*\n(.*?)\n\s*```", text, re.DOTALL)
        if markdown_match:
            try:
                return json.loads(markdown_match.group(1).strip())
            except json.JSONDecodeError:
                return None
        else:
            json_like_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_like_match:
                try:
                    return json.loads(json_like_match.group(0))
                except json.JSONDecodeError:
                    return None
        return None
