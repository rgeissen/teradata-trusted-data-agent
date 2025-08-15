# trusted_data_agent/agent/executor.py
import re
import json
import logging
import os
from enum import Enum, auto
from datetime import datetime, timedelta

# --- NEW: Import the correct function for loading prompts ---
from langchain_mcp_adapters.prompts import load_mcp_prompt

from trusted_data_agent.agent.formatter import OutputFormatter
from trusted_data_agent.core import session_manager
from trusted_data_agent.mcp import adapter as mcp_adapter
from trusted_data_agent.llm import handler as llm_handler
from trusted_data_agent.agent.workflow_manager import WorkflowManager
from trusted_data_agent.agent.prompts import NON_DETERMINISTIC_WORKFLOW_PROMPT, NON_DETERMINISTIC_WORKFLOW_RECOVERY_PROMPT

app_logger = logging.getLogger("quart.app")

def get_prompt_text_content(prompt_obj):
    """
    Extracts the text content from a loaded prompt object, handling different
    potential formats returned by the MCP adapter.
    """
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

def _format_sse(data: dict, event: str = None) -> str:
    msg = f"data: {json.dumps(data)}\n"
    if event is not None:
        msg += f"event: {event}\n"
    return f"{msg}\n"

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
        self.last_tool_result_str = None
        self.context_stack = []
        self.structured_collected_data = {}
        self.last_command_str = None
        self.charting_intent_detected = self._detect_charting_intent(original_user_input)
        self.last_tool_output = None
        self.temp_data_holder = None
        # --- NEW: Attributes for new workflow engine ---
        self.workflow_plan = None
        self.prompt_arguments = {}
        self.workflow_pointer = []
        self.workflow_loop_stack = []
        # --- NEW: Attributes for Mode 1 & 2 ---
        self.workflow_mode = 0  # 0: None, 1: Non-Deterministic, 2: Deterministic
        self.workflow_goal_prompt = ""
        self.workflow_history = []
        # --- NEW: Attribute to track last command in a non-deterministic workflow ---
        self.last_command_in_workflow = None
        # --- NEW: Debugging attribute to log every LLM response ---
        self.llm_debug_history = []


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
        # --- NEW: Add every LLM response to a debug history for analysis ---
        self.llm_debug_history.append({"reason": reason, "response": response_text})
        app_logger.info(f"LLM RESPONSE (DEBUG): Reason='{reason}', Response='{response_text}'")
        # --- END NEW: Debugging code ---
        return response_text, statement_input_tokens, statement_output_tokens

    def _detect_charting_intent(self, user_input: str) -> bool:
        """
        Detects if the user's original query explicitly asks for a chart or graphical representation.
        """
        chart_keywords = ["chart", "graph", "plot", "visualize", "diagram", "representation", "picture"]
        return any(keyword in user_input.lower() for keyword in chart_keywords)

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
        if self.is_workflow:
            context_key = " > ".join([ctx['list'][ctx['index']] for ctx in self.context_stack if 'list' in ctx and ctx['index'] != -1])
            if not context_key:
                # In non-deterministic mode, context stack isn't used, so provide a default key.
                context_key = f"Workflow: {self.active_prompt_name}" if self.active_prompt_name else "Workflow Results"
            if context_key not in self.structured_collected_data:
                self.structured_collected_data[context_key] = []
            self.structured_collected_data[context_key].append(tool_result)
            app_logger.info(f"Added tool result to structured data under key: '{context_key}' for workflow.")
        else:
            self.collected_data.append(tool_result)
            app_logger.info(f"Added tool result to collected data for non-workflow execution.")

    async def run(self):
        # Main execution loop for tool calls and decisions.
        for i in range(self.max_steps):
            if self.state in [AgentState.SUMMARIZING, AgentState.DONE, AgentState.ERROR]:
                break
            try:
                if self.is_workflow and self.workflow_mode == 1:
                    async for event in self._execute_nondeterministic_step():
                        yield event
                elif self.is_workflow and self.workflow_mode == 2:
                    # --- DEPRECATED: Old workflow engine logic is removed from here ---
                    # We can add the new deterministic logic here in the next step.
                    # For now, we will just complete the workflow.
                    yield _format_sse({"step": "Deterministic Workflow (Not Implemented)", "details": "Placeholder for deterministic workflow execution."})
                    self.state = AgentState.SUMMARIZING
                elif self.state == AgentState.DECIDING:
                    yield _format_sse({"step": "LLM has decided on an action", "details": self.next_action_str}, "llm_thought")
                    async for event in self._handle_deciding():
                        yield event
                elif self.state == AgentState.EXECUTING_TOOL:
                    async for event in self._execute_tool_with_orchestrators():
                        yield event

            except Exception as e:
                # --- MODIFIED: Unwrap exception before passing to recovery ---
                root_exception = unwrap_exception(e)
                app_logger.error(f"Error in state {self.state.name}: {root_exception}", exc_info=True)
                if self.is_workflow:
                    async for event in self._recover_with_llm(f"The plan failed with this error: {root_exception}"):
                        yield event
                else:
                    self.state = AgentState.ERROR
                    yield _format_sse({"error": "An error occurred during execution.", "details": str(root_exception)}, "error")
        
        if self.state == AgentState.SUMMARIZING:
            async for event in self._generate_final_summary():
                yield event
        elif self.state == AgentState.ERROR:
             yield _format_sse({"error": "Execution stopped due to an error.", "details": "The agent entered an unrecoverable error state."}, "error")

    async def _execute_nondeterministic_step(self):
        """
        Manages a single step of a non-deterministic workflow.
        It orchestrates the decision-making and execution in a single loop step.
        """
        # --- START: RESTRUCTURED CONTROL FLOW (FIXED) ---
        # 1. Handle a pending action if one exists.
        if self.next_action_str:
            # Check for a repetitive loop based on the raw response string
            if self.last_command_in_workflow and self.next_action_str == self.last_command_in_workflow:
                error_message = "Repetitive action detected in non-deterministic workflow."
                tool_result_str = json.dumps({"tool_input": self.next_action_str, "tool_output": {"status": "error", "error_message": error_message}})
                yield _format_sse({"step": "System Error: Repetitive Action Detected", "details": error_message, "type": "error"}, "tool_result")
                app_logger.warning(f"LOOP DETECTED: Non-deterministic workflow is trying to repeat the same command: {self.next_action_str}")
                
                async for event in self._recover_from_loop(tool_result_str):
                    yield event
                return
            
            # This is the critical update: We act on the LLM's response now.
            self.last_command_in_workflow = self.next_action_str
            
            async for event in self._handle_deciding():
                yield event
            
            if self.state == AgentState.EXECUTING_TOOL:
                async for event in self._execute_tool_with_orchestrators():
                    yield event
            
            # Reset the action string after handling it, so the next loop will call the LLM again.
            self.next_action_str = None
            return

        # 2. If no pending action, check for final answer or get one from the LLM.
        # This part of the code is now only executed if self.next_action_str is None,
        # which prevents the endless loop.

        if self.last_tool_output and isinstance(self.last_tool_output, dict) and self.last_tool_output.get("type") == "business_description":
            self.state = AgentState.SUMMARIZING
            self.next_action_str = f"FINAL_ANSWER: {self.last_tool_output.get('description', 'No description provided.')}"
            return

        # Prepare the prompt for the next LLM call
        if self.workflow_history:
            history_items = [f"- Executed tool `{item.get('tool_name')}` with arguments `{item.get('arguments', {})}`." for item in self.workflow_history]
            workflow_history_str = "\n".join(history_items)
        else:
            workflow_history_str = "No actions have been taken yet."
        
        prompt_for_next_step = NON_DETERMINISTIC_WORKFLOW_PROMPT.format(
            workflow_goal=self.workflow_goal_prompt,
            original_user_input=self.original_user_input,
            workflow_history_str=workflow_history_str,
            tool_result_str=self.last_tool_result_str or "No tool has been run yet. This is the first step."
        )
        reason = "Deciding next step in non-deterministic workflow."
        yield _format_sse({"step": "Calling LLM", "details": reason})
        response_text, input_tokens, output_tokens = await self._call_llm_and_update_tokens(prompt=prompt_for_next_step, reason=reason)
        
        updated_session = session_manager.get_session(self.session_id)
        if updated_session:
            yield _format_sse({
                "statement_input": input_tokens, "statement_output": output_tokens,
                "total_input": updated_session.get("input_tokens", 0), "total_output": updated_session.get("output_tokens", 0)
            }, "token_update")
        
        self.next_action_str = response_text
        if not self.next_action_str:
            raise ValueError("LLM failed to provide a response for the workflow.")

        self.state = AgentState.DECIDING
        # --- END: RESTRUCTURED CONTROL FLOW ---


    def _get_next_action_node(self):
        """Navigates the workflow_plan tree and returns the current action node."""
        if not self.workflow_plan or not self.workflow_pointer:
            return None

        current_level_nodes = self.workflow_plan
        
        for i in range(len(self.workflow_pointer) - 1):
            node_index = self.workflow_pointer[i]
            if node_index >= len(current_level_nodes): return None
            current_level_nodes = current_level_nodes[node_index].get("steps", [])
        
        final_index = self.workflow_pointer[-1]
        if final_index >= len(current_level_nodes):
            return None
            
        return current_level_nodes[final_index]

    def _advance_workflow_pointer(self):
        """Increments the pointer to the next step, handling nested structures."""
        if not self.workflow_plan or not self.workflow_pointer:
            return
        
        self.workflow_pointer[-1] += 1
        
    async def _execute_workflow_step(self):
        # This method is now deprecated in favor of the new workflow modes.
        # It can be removed or repurposed for the deterministic workflow later.
        app_logger.warning("DEPRECATED: _execute_workflow_step was called.")
        self.state = AgentState.SUMMARIZING
        return
        yield

    # --- START: NEW RECOVERY LOGIC FOR LOOPS ---
    async def _recover_from_loop(self, tool_result_str: str):
        """
        Attempts to recover from a detected repetitive action loop in non-deterministic workflow mode.
        """
        recovery_prompt = NON_DETERMINISTIC_WORKFLOW_RECOVERY_PROMPT.format(
            original_goal=self.workflow_goal_prompt,
            original_user_input=self.original_user_input,
            last_command=self.last_command_in_workflow
        )
        
        reason = "Recovering from repetitive action loop in non-deterministic workflow."
        yield _format_sse({"step": "Calling LLM for Recovery", "details": reason})
        response_text, input_tokens, output_tokens = await self._call_llm_and_update_tokens(
            prompt=recovery_prompt, 
            reason=reason,
            system_prompt_override="You are a tool-use assistant.",
            raise_on_error=True
        )

        updated_session = session_manager.get_session(self.session_id)
        if updated_session:
            yield _format_sse({ "statement_input": input_tokens, "statement_output": output_tokens, "total_input": updated_session.get("input_tokens", 0), "total_output": updated_session.get("output_tokens", 0) }, "token_update")
            
        self.next_action_str = response_text
        self.state = AgentState.DECIDING
        self.last_command_in_workflow = None # Reset the last command to avoid immediate re-triggering of the loop detection
    # --- END: NEW RECOVERY LOGIC FOR LOOPS ---

    async def _recover_with_llm(self, error_message: str):
        recovery_prompt = (
            "You are an expert troubleshooter for a multi-step workflow. The plan has failed. Your job is to get the plan back on track.\n\n"
            f"--- ORIGINAL GOAL ---\n{self.original_user_input}\n\n"
            f"--- THE FAILED STEP ---\n{json.dumps(self.current_command)}\n\n"
            f"--- THE ERROR ---\n{error_message}\n\n"
            "--- INSTRUCTIONS ---\n"
            "Analyze the error. Decide on a single, immediate action to recover. This is likely a new tool call with corrected parameters. Your response MUST be a single JSON object for a tool call."
        )
        reason = "Recovering from workflow error."
        yield _format_sse({"step": "Calling LLM for Recovery", "details": reason})
        response_text, input_tokens, output_tokens = await self._call_llm_and_update_tokens(prompt=recovery_prompt, reason=reason)
        
        updated_session = session_manager.get_session(self.session_id)
        if updated_session:
            yield _format_sse({"statement_input": input_tokens, "statement_output": output_tokens, "total_input": updated_session.get("input_tokens", 0), "total_output": updated_session.get("output_tokens", 0)}, "token_update")

        self.next_action_str = response_text
        self.is_workflow = False # Revert to standard mode for recovery
        self.state = AgentState.DECIDING

    async def _execute_tool_with_orchestrators(self):
        is_range_candidate, date_param_name, tool_supports_range = self._is_date_query_candidate()
        if is_range_candidate and not tool_supports_range:
            async for event in self._classify_date_query_type(): yield event
            if self.temp_data_holder and self.temp_data_holder.get('type') == 'range':
                async for event in self._execute_date_range_orchestrator(date_param_name, self.temp_data_holder.get('phrase')): yield event
                return

        tool_name = self.current_command.get("tool_name")
        # --- FIX: Check for the 'qlty_distinctCategories' tool specifically and ensure it's handled correctly ---
        if tool_name == 'qlty_distinctCategories':
            # This tool should always be run as a standard tool call, not an iteration
            # We don't need to check for the presence of column_name here because the LLM is expected to provide it
            async for event in self._execute_standard_tool():
                yield event
        elif self.dependencies['STATE'].get('tool_scopes', {}).get(tool_name) == 'column' and not self.current_command.get("arguments", {}).get("column_name"):
            async for event in self._execute_column_iteration(): yield event
        else:
            async for event in self._execute_standard_tool(): yield event

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
        yield _format_sse({"step": "Calling LLM", "details": reason})
        response_str, input_tokens, output_tokens = await self._call_llm_and_update_tokens(
            prompt=classification_prompt,
            reason=reason,
            system_prompt_override="You are a JSON-only responding assistant.",
            raise_on_error=True
        )

        updated_session = session_manager.get_session(self.session_id)
        if updated_session:
            yield _format_sse({ "statement_input": input_tokens, "statement_output": output_tokens, "total_input": updated_session.get("input_tokens", 0), "total_output": updated_session.get("output_tokens", 0) }, "token_update")
        
        try:
            self.temp_data_holder = json.loads(response_str)
        except (json.JSONDecodeError, KeyError):
            self.temp_data_holder = {'type': 'single', 'phrase': self.original_user_input}


    async def _execute_date_range_orchestrator(self, date_param_name: str, date_phrase: str):
        tool_name = self.current_command.get("tool_name")
        yield _format_sse({
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
        yield _format_sse({"step": "Calling LLM", "details": reason})
        range_response_str, input_tokens, output_tokens = await self._call_llm_and_update_tokens(
            prompt=conversion_prompt,
            reason=reason,
            system_prompt_override="You are a JSON-only responding assistant.",
            raise_on_error=True
        )
        updated_session = session_manager.get_session(self.session_id)
        if updated_session:
            yield _format_sse({ "statement_input": input_tokens, "statement_output": output_tokens, "total_input": updated_session.get("input_tokens", 0), "total_output": updated_session.get("output_tokens", 0) }, "token_update")
        
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

    def _enrich_arguments_from_history(self, prompt_name: str, arguments: dict) -> tuple[dict, list]:
        """
        Enriches arguments by looking at the prompt's definition and searching
        the conversation history for missing values. This version is simplified as
        parameter name correction is now handled by a dedicated shim.
        """
        events_to_yield = []
        enriched_args = arguments.copy()
        
        prompt_definition = self.dependencies['STATE'].get('mcp_prompts', {}).get(prompt_name)
        if not prompt_definition or not hasattr(prompt_definition, 'arguments'):
            return enriched_args, events_to_yield

        required_arg_names = {arg.name for arg in prompt_definition.arguments} if prompt_definition.arguments else set()
        
        for arg_name in required_arg_names:
            if arg_name in enriched_args: continue

            # If not found, search history
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
                events_to_yield.append(_format_sse({
                    "step": "System Correction",
                    "details": f"LLM omitted '{arg_name}'. System inferred it from history.",
                    "type": "workaround"
                }))

        return enriched_args, events_to_yield

    async def _handle_deciding(self):
        # This is the single entry point for deciding the next action
        if self.is_workflow:
            # In workflow mode, the next action is determined by the workflow loop,
            # not by parsing the LLM's free-form response.
            if self.workflow_mode == 1:
                # We let the non-deterministic loop handle the next action.
                return
            elif self.workflow_mode == 2:
                # The deterministic workflow would handle its own steps.
                return

        # --- ORIGINAL NON-WORKFLOW LOGIC PRESERVED BELOW ---
        if "SYSTEM_ACTION_COMPLETE" in self.next_action_str:
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
        
        command = json.loads(command_str)
        if "prompt_name" in command:
            self.is_workflow = True
            self.active_prompt_name = command.get("prompt_name")
            self.current_command = command.copy()
            self.workflow_history = []
            
            yaml_filename = f"{self.active_prompt_name}_workflow.yaml"
            if os.path.exists(yaml_filename):
                app_logger.info(f"Found deterministic workflow file: '{yaml_filename}'. Engaging Workflow Mode 2.")
                self.workflow_mode = 2
                self.active_prompt_plan = f"Deterministic plan from {yaml_filename}"
                self.state = AgentState.SUMMARIZING # Placeholder
                return
            else:
                app_logger.info(f"No deterministic workflow file found. Engaging Workflow Mode 1 for prompt '{self.active_prompt_name}'.")
                self.workflow_mode = 1

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
            
             # --- START: DEBUGGING LINES ---
            app_logger.info(f"DEBUG: Type of prompt_obj is {type(prompt_obj)}")
            app_logger.info(f"DEBUG: Content of prompt_obj: {prompt_obj}")
            # --- END: DEBUGGING LINES ---

            # --- START: MODIFIED and corrected text extraction logic ---
            prompt_text = get_prompt_text_content(prompt_obj)

            if not prompt_text:
                raise ValueError(f"Could not extract text content from rendered prompt '{self.active_prompt_name}'.")

            self.workflow_goal_prompt = prompt_text
            
            yield _format_sse({
                "step": f"Executing Prompt as a Non-Deterministic Workflow: {self.active_prompt_name}",
                "details": self.workflow_goal_prompt,
                "prompt_name": self.active_prompt_name
            }, "prompt_selected")
            
            self.next_action_str = None
            return

        # --- Original non-workflow logic continues below ---
        if command_str == self.last_command_str:
            app_logger.warning(f"LOOP DETECTED: The LLM is trying to repeat the exact same command. Command: {command_str}")
            error_message = f"Repetitive action detected."
            tool_result_str = json.dumps({"tool_input": json.loads(command_str), "tool_output": {"status": "error", "error_message": error_message}})
            yield _format_sse({"step": "System Error", "details": error_message, "type": "error"}, "tool_result")
            self.last_command_str = None 
            async for event in self._get_next_action_from_llm(tool_result_str=tool_result_str, reason="Recovering from repetitive action error."):
                yield event
            return
        
        self.last_command_str = command_str
        self.current_command = command
            
        if "tool_name" in command:
            self.state = AgentState.EXECUTING_TOOL
        else:
            self.state = AgentState.SUMMARIZING

    async def _execute_standard_tool(self, is_workflow_step=False):
        self.current_command, guardrail_events = self._apply_context_guardrail(self.current_command)
        for event in guardrail_events: yield event

        yield _format_sse({"step": "Tool Execution Intent", "details": self.current_command}, "tool_result")
        
        tool_name = self.current_command.get("tool_name")
        status_target = "chart" if tool_name == "viz_createChart" else "db"
        yield _format_sse({"target": status_target, "state": "busy"}, "status_indicator_update")
        
        tool_result = await mcp_adapter.invoke_mcp_tool(self.dependencies['STATE'], self.current_command)
        
        yield _format_sse({"target": status_target, "state": "idle"}, "status_indicator_update")

        if self.is_workflow and self.workflow_mode == 2: # Context stack is for deterministic workflows
            self._update_and_manage_context_stack(self.current_command, tool_result)

        if 'notification' in self.current_command:
            yield _format_sse({"step": "System Notification", "details": self.current_command['notification'], "type": "workaround"})
            del self.current_command['notification']

        self.last_tool_output = tool_result

        tool_result_str = ""
        if isinstance(tool_result, dict) and "error" in tool_result:
            error_details = tool_result.get("data", tool_result.get("error", ""))
            if "Function" in str(error_details) and "does not exist" in str(error_details):
                self.globally_failed_tools.add(tool_name)
            tool_result_str = json.dumps({"tool_input": self.current_command, "tool_output": {"status": "error", "error_message": error_details}})
        else:
            tool_result_str = json.dumps({"tool_name": tool_name, "tool_output": tool_result})
            # In workflow mode 1, we add to structured data.
            if self.is_workflow:
                self._add_to_structured_data(tool_result)
                if self.workflow_mode == 1:
                    self.workflow_history.append(self.current_command)
            else:
                 self.collected_data.append(tool_result)


        self.last_tool_result_str = tool_result_str # Store for the next step in the workflow

        if isinstance(tool_result, dict) and tool_result.get("error") == "parameter_mismatch":
            yield _format_sse({"details": tool_result}, "request_user_input")
            self.state = AgentState.ERROR
            return

        yield _format_sse({"step": "Tool Execution Result", "details": tool_result, "tool_name": tool_name}, "tool_result")
        
        if not self.is_workflow:
            async for event in self._get_next_action_from_llm(
                tool_result_str=tool_result_str, 
                reason="Deciding next action based on tool result."
            ):
                yield event

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
            self.last_tool_output = col_result
            self.state = AgentState.DECIDING
            return

        cols_command = {"tool_name": "base_columnDescription", "arguments": {"db_name": db_name, "obj_name": table_name}}
        yield _format_sse({"target": "db", "state": "busy"}, "status_indicator_update")
        cols_result = await mcp_adapter.invoke_mcp_tool(self.dependencies['STATE'], cols_command)
        yield _format_sse({"target": "db", "state": "idle"}, "status_indicator_update")
        if not (cols_result and isinstance(cols_result, dict) and cols_result.get('status') == 'success' and cols_result.get('results')):
            raise ValueError(f"Failed to retrieve column list for iteration. Response: {cols_result}")
        all_columns_metadata = cols_result.get('results', [])
        self._add_to_structured_data(cols_result)

        all_column_results = []
        
        reason="Determining tool constraints."
        yield _format_sse({"step": "Calling LLM", "details": reason})
        tool_constraints = await self._get_tool_constraints(tool_name)
        required_type = tool_constraints.get("dataType") if tool_constraints else None

        yield _format_sse({"target": "db", "state": "busy"}, "status_indicator_update")
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
        yield _format_sse({"target": "db", "state": "idle"}, "status_indicator_update")

        self._add_to_structured_data(all_column_results)
        self.last_tool_result_str = json.dumps({"tool_name": tool_name, "tool_output": all_column_results})
        self.last_tool_output = {"metadata": {"tool_name": tool_name}, "results": all_column_results, "status": "success"}
        self.state = AgentState.DECIDING

    async def _get_next_action_from_llm(self, tool_result_str: str | None = None, reason: str = "No reason provided."):
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

        prompt_for_next_step = (
            f"You are an assistant that has just received data from a tool call. Your task is to decide if this data is enough to answer the user's original question, or if another step is needed.\n\n"
            f"--- User's Original Question ---\n"
            f"'{self.original_user_input}'\n\n"
            f"--- Data from Last Tool Call ---\n"
            f"{tool_result_str}\n\n"
            f"{charting_guidance}"
            "--- Your Decision Process ---\n"
            "1.  **Analyze the Data:** Does the data above directly and completely answer the user's original question?\n"
            "2.  **Choose Your Action:**\n"
            "    -   If the data IS sufficient, your response **MUST** be only the exact string `SYSTEM_ACTION_COMPLETE`. Do not add any summary text.\n"
            "    -   If the data is NOT sufficient and you need more information, call another tool or prompt by providing the appropriate JSON block.\n"
            "    -   If the last tool call resulted in an error, you MUST attempt to recover.\n"
        )
        
        yield _format_sse({"step": "Calling LLM", "details": reason})
        response_text, input_tokens, output_tokens = await self._call_llm_and_update_tokens(prompt=prompt_for_next_step, reason=reason)
        
        updated_session = session_manager.get_session(self.session_id)
        if updated_session:
            yield _format_sse({ "statement_input": input_tokens, "statement_output": output_tokens, "total_input": updated_session.get("input_tokens", 0), "total_output": updated_session.get("output_tokens", 0) }, "token_update")
        
        self.next_action_str = response_text
        if not self.next_action_str: raise ValueError("LLM failed to provide a response.")
        
        self.state = AgentState.DECIDING

    async def _generate_final_summary(self):
        final_collected_data = self.structured_collected_data if self.is_workflow else self.collected_data
        
        final_summary_text = ""
        final_answer_match = re.search(r'FINAL_ANSWER:(.*)', self.next_action_str, re.DOTALL | re.IGNORECASE)

        if final_answer_match:
            potential_summary = final_answer_match.group(1).strip()
            if potential_summary:
                final_summary_text = potential_summary
                app_logger.info("Using pre-existing FINAL_ANSWER text provided by a workflow step.")

        if self.last_tool_output and self.last_tool_output.get("type") == "chart":
            data_for_llm_analysis = None
            if self.collected_data and len(self.collected_data) >= 2:
                data_for_llm_analysis = self.collected_data[-2]
            
            if not data_for_llm_analysis:
                app_logger.warning("Could not find source data for chart analysis. Reverting to simple summary.")
                final_summary_text = "The chart has been generated and is displayed below."
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
                yield _format_sse({"step": "Calling LLM", "details": reason})
                final_llm_response, input_tokens, output_tokens = await self._call_llm_and_update_tokens(prompt=final_prompt, reason=reason)
                
                updated_session = session_manager.get_session(self.session_id)
                if updated_session:
                    yield _format_sse({ "statement_input": input_tokens, "statement_output": output_tokens, "total_input": updated_session.get("input_tokens", 0), "total_output": updated_session.get("output_tokens", 0) }, "token_update")
                
                if final_llm_response:
                    final_answer_match = re.search(r'FINAL_ANSWER:(.*)', final_llm_response, re.DOTALL | re.IGNORECASE)
                    if final_answer_match:
                        final_summary_text = final_answer_match.group(1).strip()
                    else:
                        final_summary_text = final_llm_response

        if not final_summary_text:
            final_summary_text = "The agent has completed its work, but no summary was generated."

        yield _format_sse({
            "step": "LLM has generated the final answer",
            "details": final_summary_text
        }, "llm_thought")

        formatter = OutputFormatter(
            llm_response_text=final_summary_text, 
            collected_data=final_collected_data, 
            is_workflow=self.is_workflow
        )
        final_html = formatter.render()
        
        session_manager.add_to_history(self.session_id, 'assistant', final_html)
        yield _format_sse({"final_answer": final_html}, "final_answer")
        self.state = AgentState.DONE

    def _prepare_data_for_final_summary(self) -> str:
        """
        Prepares the collected data for the final summarization prompt.
        This now serializes the actual data results into a JSON string.
        """
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
        """Safely extracts and parses a JSON object from a string, handling markdown."""
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