# trusted_data_agent/agent/executor.py
import re
import json
import logging
import os
import copy
from enum import Enum, auto
from datetime import datetime, timedelta

from langchain_mcp_adapters.prompts import load_mcp_prompt

from trusted_data_agent.agent.formatter import OutputFormatter
from trusted_data_agent.core import session_manager
from trusted_data_agent.mcp import adapter as mcp_adapter
from trusted_data_agent.llm import handler as llm_handler
from trusted_data_agent.agent.prompts import (
    ERROR_RECOVERY_PROMPT,
    WORKFLOW_META_PLANNING_PROMPT,
    WORKFLOW_TACTICAL_PROMPT,
    WORKFLOW_PHASE_COMPLETION_PROMPT
)

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
    PLANNING = auto()
    EXECUTING = auto()
    SUMMARIZING = auto()
    DONE = auto()
    ERROR = auto()

def unwrap_exception(e: BaseException) -> BaseException:
    """Recursively unwraps ExceptionGroups to find the root cause."""
    if isinstance(e, ExceptionGroup) and e.exceptions:
        return unwrap_exception(e.exceptions[0])
    return e

class PlanExecutor:
    AgentState = AgentState

    def __init__(self, session_id: str, original_user_input: str, dependencies: dict):
        self.session_id = session_id
        self.original_user_input = original_user_input
        self.dependencies = dependencies
        self.state = self.AgentState.PLANNING
        
        # Unified State Properties
        self.structured_collected_data = {}
        self.workflow_state = {} 
        self.action_history = []
        self.meta_plan = None
        self.current_phase_index = 0
        self.last_tool_output = None
        
        # Workflow-specific context, potentially set by routes for manual invocation
        self.is_workflow = False
        self.active_prompt_name = None
        self.prompt_arguments = {}
        self.workflow_goal_prompt = ""

        # Non-Workflow Optimizations & State
        self.tool_constraints_cache = {}
        self.globally_skipped_tools = set()
        self.temp_data_holder = None
        self.last_failed_action_info = "None"
        self.events_to_yield = []
        self.last_action_str = None # For loop detection
        
        self.llm_debug_history = []
        self.max_steps = 40

    @staticmethod
    def _format_sse(data: dict, event: str = None) -> str:
        msg = f"data: {json.dumps(data)}\n"
        if event is not None:
            msg += f"event: {event}\n"
        return f"{msg}\n"

    async def _call_llm_and_update_tokens(self, prompt: str, reason: str, system_prompt_override: str = None, raise_on_error: bool = False) -> tuple[str, int, int]:
        """A centralized wrapper for calling the LLM that handles token updates."""
        response_text, statement_input_tokens, statement_output_tokens = await llm_handler.call_llm_api(
            self.dependencies['STATE']['llm'], prompt, self.session_id,
            dependencies=self.dependencies, reason=reason,
            system_prompt_override=system_prompt_override, raise_on_error=raise_on_error
        )
        self.llm_debug_history.append({"reason": reason, "response": response_text})
        app_logger.info(f"LLM RESPONSE (DEBUG): Reason='{reason}', Response='{response_text}'")
        return response_text, statement_input_tokens, statement_output_tokens

    def _add_to_structured_data(self, tool_result: dict, context_key_override: str = None):
        """Adds tool results to the structured data dictionary."""
        context_key = context_key_override or f"Plan Results: {self.active_prompt_name or 'Ad-hoc'}"
        if context_key not in self.structured_collected_data:
            self.structured_collected_data[context_key] = []
        
        if isinstance(tool_result, list):
             self.structured_collected_data[context_key].extend(tool_result)
        else:
             self.structured_collected_data[context_key].append(tool_result)
        app_logger.info(f"Added tool result to structured data under key: '{context_key}'.")

    async def run(self):
        """The main, unified execution loop for the agent."""
        try:
            if self.state == self.AgentState.PLANNING:
                async for event in self._generate_meta_plan(): yield event
                self.state = self.AgentState.EXECUTING

            if self.state == self.AgentState.EXECUTING:
                async for event in self._run_plan(): yield event

            if self.state == self.AgentState.SUMMARIZING:
                async for event in self._generate_final_summary(): yield event

        except Exception as e:
            root_exception = unwrap_exception(e)
            app_logger.error(f"Error in state {self.state.name}: {root_exception}", exc_info=True)
            self.state = self.AgentState.ERROR
            yield self._format_sse({"error": "Execution stopped due to an unrecoverable error.", "details": str(root_exception)}, "error")

    async def _generate_meta_plan(self):
        """The universal planner. It generates a meta-plan for ANY request."""
        if self.is_workflow:
            yield self._format_sse({"step": "Loading Workflow Prompt", "details": f"Loading '{self.active_prompt_name}'..."})
            mcp_client = self.dependencies['STATE'].get('mcp_client')
            if not mcp_client: raise RuntimeError("MCP client is not connected.")
            
            enriched_args, enrich_events = self._enrich_arguments_from_history(self.active_prompt_name, self.prompt_arguments)
            for event in enrich_events:
                yield event
            self.prompt_arguments = enriched_args

            async with mcp_client.session("teradata_mcp_server") as temp_session:
                prompt_obj = await load_mcp_prompt(
                    temp_session, name=self.active_prompt_name, arguments=self.prompt_arguments
                )
            if not prompt_obj: raise ValueError(f"Prompt '{self.active_prompt_name}' could not be loaded.")
            
            self.workflow_goal_prompt = get_prompt_text_content(prompt_obj)
            if not self.workflow_goal_prompt:
                raise ValueError(f"Could not extract text content from rendered prompt '{self.active_prompt_name}'.")
        else:
            self.workflow_goal_prompt = self.original_user_input

        reason = f"Generating a strategic meta-plan for the goal: '{self.workflow_goal_prompt[:100]}...'"
        yield self._format_sse({"step": "Calling LLM for Planning", "details": reason})

        planning_prompt = WORKFLOW_META_PLANNING_PROMPT.format(
            workflow_goal=self.workflow_goal_prompt,
            original_user_input=self.original_user_input
        )
        
        response_text, input_tokens, output_tokens = await self._call_llm_and_update_tokens(
            prompt=planning_prompt, 
            reason=reason
        )

        updated_session = session_manager.get_session(self.session_id)
        if updated_session:
            yield self._format_sse({ "statement_input": input_tokens, "statement_output": output_tokens, "total_input": updated_session.get("input_tokens", 0), "total_output": updated_session.get("output_tokens", 0) }, "token_update")
        
        try:
            json_str = response_text
            if response_text.strip().startswith("```json"):
                match = re.search(r"```json\s*\n(.*?)\n\s*```", response_text, re.DOTALL)
                if match:
                    json_str = match.group(1).strip()

            self.meta_plan = json.loads(json_str)
            if not isinstance(self.meta_plan, list) or not self.meta_plan:
                raise ValueError("LLM response for meta-plan was not a non-empty list.")

            yield self._format_sse({"step": "Strategic Meta-Plan Generated", "details": self.meta_plan})
        except (json.JSONDecodeError, ValueError) as e:
            raise RuntimeError(f"Failed to generate a valid meta-plan from the LLM. Response: {response_text}. Error: {e}")

    async def _run_plan(self):
        """Executes the generated meta-plan phase by phase."""
        if not self.meta_plan:
            raise RuntimeError("Cannot execute plan: meta_plan is not generated.")

        while self.current_phase_index < len(self.meta_plan):
            current_phase = self.meta_plan[self.current_phase_index]
            phase_goal = current_phase.get("goal", "No goal defined.")
            phase_num = current_phase.get("phase", self.current_phase_index + 1)
            relevant_tools = current_phase.get("relevant_tools", [])

            yield self._format_sse({
                "step": "Starting Plan Phase",
                "details": f"Phase {phase_num}/{len(self.meta_plan)}: {phase_goal}",
                "phase_details": current_phase
            })

            phase_attempts = 0
            max_phase_attempts = 5
            while True:
                phase_attempts += 1
                if phase_attempts > max_phase_attempts:
                    app_logger.error(f"Phase '{phase_goal}' failed after {max_phase_attempts} attempts. Attempting LLM recovery.")
                    async for event in self._recover_from_phase_failure(phase_goal):
                        yield event
                    self.current_phase_index -= 1 
                    break

                next_action, input_tokens, output_tokens = await self._get_next_tactical_action(phase_goal, relevant_tools)
                
                current_action_str = json.dumps(next_action, sort_keys=True)
                if current_action_str == self.last_action_str:
                    app_logger.warning(f"LOOP DETECTED: Repeating action: {current_action_str}")
                    self.last_failed_action_info = "Your last attempt failed because it was an exact repeat of the previous failed action. You MUST choose a different tool or different arguments."
                    yield self._format_sse({"step": "System Error", "details": "Repetitive action detected.", "type": "error"}, "tool_result")
                    self.last_action_str = None 
                    continue
                self.last_action_str = current_action_str
                
                if self.events_to_yield:
                    for event in self.events_to_yield: yield event
                    self.events_to_yield = []

                updated_session = session_manager.get_session(self.session_id)
                if updated_session:
                    yield self._format_sse({ "statement_input": input_tokens, "statement_output": output_tokens, "total_input": updated_session.get("input_tokens", 0), "total_output": updated_session.get("output_tokens", 0) }, "token_update")

                if isinstance(next_action, str) and next_action == "SYSTEM_ACTION_COMPLETE":
                    self.state = self.AgentState.SUMMARIZING
                    return

                if not isinstance(next_action, dict):
                    raise RuntimeError(f"Tactical LLM failed to provide a valid action. Received: {next_action}")

                async for event in self._execute_action_with_orchestrators(next_action, current_phase):
                    yield event

                phase_is_complete, input_tokens, output_tokens = await self._is_phase_complete(phase_goal, phase_num)
                
                if input_tokens > 0 or output_tokens > 0:
                    updated_session = session_manager.get_session(self.session_id)
                    if updated_session:
                        yield self._format_sse({ "statement_input": input_tokens, "statement_output": output_tokens, "total_input": updated_session.get("input_tokens", 0), "total_output": updated_session.get("output_tokens", 0) }, "token_update")

                if phase_is_complete:
                    yield self._format_sse({"step": f"Phase {phase_num} Complete", "details": "Goal has been achieved."})
                    self.last_action_str = None
                    break 

            self.current_phase_index += 1

        app_logger.info("Meta-plan has been fully executed. Transitioning to summarization.")
        self.state = self.AgentState.SUMMARIZING

    async def _execute_action_with_orchestrators(self, action: dict, phase: dict):
        """A wrapper that runs pre-flight checks (orchestrators) before executing a tool."""
        tool_name = action.get("tool_name")
        if not tool_name:
            raise ValueError("Action from tactical LLM is missing a 'tool_name'.")
        
        is_range_candidate, date_param_name, tool_supports_range = self._is_date_query_candidate(action)
        if is_range_candidate and not tool_supports_range:
            async for event in self._classify_date_query_type(): yield event
            if self.temp_data_holder and self.temp_data_holder.get('type') == 'range':
                async for event in self._execute_date_range_orchestrator(action, date_param_name, self.temp_data_holder.get('phrase')): yield event
                return

        tool_scope = self.dependencies['STATE'].get('tool_scopes', {}).get(tool_name)
        has_column_arg = "column_name" in action.get("arguments", {})
        if tool_scope == 'column' and not has_column_arg:
             async for event in self._execute_column_iteration(action): yield event
             return

        async for event in self._execute_tool(action, phase):
            yield event

    async def _execute_tool(self, action: dict, phase: dict):
        """Executes a single tool call as part of a plan phase."""
        tool_name = action.get("tool_name")
        phase_num = phase.get("phase", self.current_phase_index + 1)
        relevant_tools = phase.get("relevant_tools", [])

        if relevant_tools and tool_name not in relevant_tools:
            app_logger.warning(f"LLM proposed invalid tool '{tool_name}'. Retrying phase.")
            self.last_failed_action_info = f"Invalid tool '{tool_name}' was chosen. Permitted tools are: {relevant_tools}"
            yield self._format_sse({"step": "System Correction", "type": "workaround", "details": f"LLM chose invalid tool ('{tool_name}'). Retrying."})
            return

        # --- MODIFICATION START: Argument normalization is now handled in the Tool Invocation Layer ---
        # async for event in self._normalize_tool_arguments(action):
        #     yield event
        # --- MODIFICATION END ---
        
        if 'notification' in action:
            yield self._format_sse({"step": "System Notification", "details": action['notification'], "type": "workaround"})
            del action['notification']

        if tool_name == "CoreLLMTask":
            action.setdefault("arguments", {})["data"] = copy.deepcopy(self.workflow_state)

        yield self._format_sse({"step": "Tool Execution Intent", "details": action}, "tool_result")
        
        status_target = "chart" if tool_name == "viz_createChart" else "db"
        yield self._format_sse({"target": status_target, "state": "busy"}, "status_indicator_update")
        tool_result = await mcp_adapter.invoke_mcp_tool(self.dependencies['STATE'], action)
        yield self._format_sse({"target": status_target, "state": "idle"}, "status_indicator_update")

        self.last_tool_output = tool_result
        self.action_history.append({"action": action, "result": tool_result})
        
        phase_result_key = f"result_of_phase_{phase_num}"
        self.workflow_state.setdefault(phase_result_key, []).append(tool_result)
        
        self._add_to_structured_data(tool_result)

        if isinstance(tool_result, dict) and tool_result.get("status") == "error":
            yield self._format_sse({"details": tool_result, "tool_name": tool_name}, "tool_error")
        else:
            yield self._format_sse({"step": "Tool Execution Result", "details": tool_result, "tool_name": tool_name}, "tool_result")

    async def _get_next_tactical_action(self, current_phase_goal: str, relevant_tools: list[str]) -> tuple[dict | str, int, int]:
        """Makes a tactical LLM call to decide the single next best action for the current phase."""
        tactical_system_prompt = WORKFLOW_TACTICAL_PROMPT.format(
            workflow_goal=self.workflow_goal_prompt,
            current_phase_goal=current_phase_goal,
            relevant_tools_for_phase=json.dumps(relevant_tools),
            last_attempt_info=self.last_failed_action_info,
            workflow_history=json.dumps(self.action_history, indent=2),
            all_collected_data=json.dumps(self.workflow_state, indent=2)
        )

        response_text, input_tokens, output_tokens = await self._call_llm_and_update_tokens(
            prompt="Determine the next action based on the instructions and state provided in the system prompt.",
            reason=f"Deciding next tactical action for phase: {current_phase_goal}",
            system_prompt_override=tactical_system_prompt
        )
        
        self.last_failed_action_info = "None"

        if "FINAL_ANSWER:" in response_text.upper() or "SYSTEM_ACTION_COMPLETE" in response_text.upper():
            return "SYSTEM_ACTION_COMPLETE", input_tokens, output_tokens

        try:
            # --- MODIFICATION START: Robust JSON parsing and normalization ---
            json_match = re.search(r"```json\s*\n(.*?)\n\s*```|(\{.*\})", response_text, re.DOTALL)
            if not json_match: raise json.JSONDecodeError("No JSON object found", response_text, 0)
            
            json_str = json_match.group(1) or json_match.group(2)
            if not json_str: raise json.JSONDecodeError("Extracted JSON is empty", response_text, 0)

            raw_action = json.loads(json_str.strip())
            
            # --- Abstract Normalization Layer ---
            # This layer intelligently finds the tool call details, regardless of nesting.
            action_details = raw_action
            possible_wrapper_keys = ["action", "tool_call", "tool"]
            for key in possible_wrapper_keys:
                if key in action_details and isinstance(action_details[key], dict):
                    action_details = action_details[key]
                    break 

            # Find tool name using a list of synonyms
            tool_name_synonyms = ["tool_name", "name", "tool", "action_name"]
            found_tool_name = next((action_details.pop(key) for key in tool_name_synonyms if key in action_details), None)
            
            # Find arguments using a list of synonyms
            arg_synonyms = ["arguments", "args", "tool_input", "action_input", "parameters"]
            found_args = next((action_details.pop(key) for key in arg_synonyms if key in action_details), {})

            # Reconstruct the action into the canonical format
            normalized_action = {
                "tool_name": found_tool_name,
                "arguments": found_args if isinstance(found_args, dict) else {}
            }

            # If tool_name is still missing, try a final fallback for simple structures
            if not normalized_action["tool_name"] and len(action_details) == 1 and isinstance(list(action_details.values())[0], dict):
                 normalized_action["tool_name"] = list(action_details.keys())[0]
                 normalized_action["arguments"] = list(action_details.values())[0]

            # If tool_name is STILL missing, but there's only one permitted tool, infer it.
            if not normalized_action.get("tool_name") and len(relevant_tools) == 1:
                normalized_action["tool_name"] = relevant_tools[0]
                self.events_to_yield.append(self._format_sse({
                    "step": "System Correction", "type": "workaround",
                    "details": f"LLM omitted tool_name. System inferred '{relevant_tools[0]}'."
                }))
            
            if not normalized_action.get("tool_name"):
                 raise ValueError("Could not determine tool_name from LLM response.")

            return normalized_action, input_tokens, output_tokens
            # --- MODIFICATION END ---
        except (json.JSONDecodeError, ValueError) as e:
            raise RuntimeError(f"Failed to get a valid JSON action from the tactical LLM. Response: {response_text}. Error: {e}")

    async def _is_phase_complete(self, current_phase_goal: str, phase_num: int) -> tuple[bool, int, int]:
        """Checks if the current phase's goal has been met."""
        phase_result_key = f"result_of_phase_{phase_num}"
        if phase_result_key in self.workflow_state:
            if any(isinstance(r, dict) and r.get("status") == "success" for r in self.workflow_state[phase_result_key]):
                return True, 0, 0

        completion_system_prompt = WORKFLOW_PHASE_COMPLETION_PROMPT.format(
            current_phase_goal=current_phase_goal,
            workflow_history=json.dumps(self.action_history, indent=2),
            all_collected_data=json.dumps(self.workflow_state, indent=2)
        )

        response_text, input_tokens, output_tokens = await self._call_llm_and_update_tokens(
            prompt="Is the phase complete? Respond with only YES or NO.",
            reason=f"Checking for completion of phase: {current_phase_goal}",
            system_prompt_override=completion_system_prompt
        )
        
        return "yes" in response_text.lower(), input_tokens, output_tokens

    def _is_date_query_candidate(self, command: dict) -> tuple[bool, str, bool]:
        """Checks if a command is a candidate for the date-range orchestrator."""
        tool_name = command.get("tool_name")
        tool_spec = self.dependencies['STATE'].get('mcp_tools', {}).get(tool_name)
        if not tool_spec or not hasattr(tool_spec, 'args') or not isinstance(tool_spec.args, dict):
            return False, None, False

        tool_arg_names = set(tool_spec.args.keys())
        tool_supports_range = 'start_date' in tool_arg_names and 'end_date' in tool_arg_names
        
        args = command.get("arguments", {})
        date_param_name = next((param for param in args if 'date' in param.lower()), None)
        
        return bool(date_param_name), date_param_name, tool_supports_range

    async def _classify_date_query_type(self):
        """Uses LLM to classify a date query as 'single' or 'range'."""
        classification_prompt = (
            f"You are a query classifier. Analyze the following query: '{self.original_user_input}'. "
            "Determine if it refers to a 'single' date or a 'range' of dates. "
            "Extract the specific phrase that describes the date or range. "
            "Your response MUST be ONLY a JSON object with two keys: 'type' and 'phrase'."
        )
        reason="Classifying date query."
        yield self._format_sse({"step": "Calling LLM", "details": reason})
        response_str, input_tokens, output_tokens = await self._call_llm_and_update_tokens(
            prompt=classification_prompt, reason=reason,
            system_prompt_override="You are a JSON-only responding assistant.", raise_on_error=True
        )
        updated_session = session_manager.get_session(self.session_id)
        if updated_session:
            yield self._format_sse({ "statement_input": input_tokens, "statement_output": output_tokens, "total_input": updated_session.get("input_tokens", 0), "total_output": updated_session.get("output_tokens", 0) }, "token_update")
        try:
            self.temp_data_holder = json.loads(response_str)
        except (json.JSONDecodeError, KeyError):
            self.temp_data_holder = {'type': 'single', 'phrase': self.original_user_input}

    async def _execute_date_range_orchestrator(self, command: dict, date_param_name: str, date_phrase: str):
        """Executes a tool over a date range."""
        tool_name = command.get("tool_name")
        yield self._format_sse({
            "step": "System Orchestration", "type": "workaround",
            "details": f"Detected date range query ('{date_phrase}') for single-day tool ('{tool_name}')."
        })

        date_command = {"tool_name": "util_getCurrentDate"}
        date_result = await mcp_adapter.invoke_mcp_tool(self.dependencies['STATE'], date_command)
        if not (date_result and date_result.get("status") == "success" and date_result.get("results")):
            raise RuntimeError("Date Range Orchestrator failed to fetch current date.")
        current_date_str = date_result["results"][0].get("current_date")

        conversion_prompt = (
            f"Given the current date is {current_date_str}, "
            f"what are the start and end dates for '{date_phrase}'? "
            "Respond with ONLY a JSON object with 'start_date' and 'end_date' in YYYY-MM-DD format."
        )
        reason = "Calculating date range."
        yield self._format_sse({"step": "Calling LLM", "details": reason})
        range_response_str, _, _ = await self._call_llm_and_update_tokens(
            prompt=conversion_prompt, reason=reason,
            system_prompt_override="You are a JSON-only responding assistant.", raise_on_error=True
        )
        
        try:
            range_data = json.loads(range_response_str)
            start_date = datetime.strptime(range_data['start_date'], '%Y-%m-%d').date()
            end_date = datetime.strptime(range_data['end_date'], '%Y-%m-%d').date()
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise RuntimeError(f"Date Range Orchestrator failed to parse date range. Error: {e}")

        all_results = []
        yield self._format_sse({"target": "db", "state": "busy"}, "status_indicator_update")
        current_date_in_loop = start_date
        while current_date_in_loop <= end_date:
            date_str = current_date_in_loop.strftime('%Y-%m-%d')
            yield self._format_sse({"step": f"Processing data for: {date_str}"})
            
            day_command = {**command, 'arguments': {**command['arguments'], date_param_name: date_str}}
            day_result = await mcp_adapter.invoke_mcp_tool(self.dependencies['STATE'], day_command)
            
            if isinstance(day_result, dict) and day_result.get("status") == "success" and day_result.get("results"):
                all_results.extend(day_result["results"])
            
            current_date_in_loop += timedelta(days=1)
        yield self._format_sse({"target": "db", "state": "idle"}, "status_indicator_update")
        
        final_tool_output = {
            "status": "success",
            "metadata": {"tool_name": tool_name, "comment": f"Consolidated results for {date_phrase}"},
            "results": all_results
        }
        self._add_to_structured_data(final_tool_output)
        self.last_tool_output = final_tool_output

    async def _execute_column_iteration(self, command: dict):
        """Executes a tool over multiple columns, now with constraint checking."""
        tool_name = command.get("tool_name")
        base_args = command.get("arguments", {})
        db_name, table_name = base_args.get("database_name"), base_args.get("table_name")

        cols_command = {"tool_name": "base_columnDescription", "arguments": {"database_name": db_name, "table_name": table_name}}
        yield self._format_sse({"target": "db", "state": "busy"}, "status_indicator_update")
        cols_result = await mcp_adapter.invoke_mcp_tool(self.dependencies['STATE'], cols_command)
        yield self._format_sse({"target": "db", "state": "idle"}, "status_indicator_update")
        
        if not (cols_result and isinstance(cols_result, dict) and cols_result.get('status') == 'success' and cols_result.get('results')):
            raise ValueError(f"Failed to retrieve column list for iteration. Response: {cols_result}")
        
        all_columns_metadata = cols_result.get('results', [])
        all_column_results = [cols_result]
        
        tool_constraints = await self._get_tool_constraints(tool_name)
        required_type = tool_constraints.get("dataType") if tool_constraints else None
        
        yield self._format_sse({"target": "db", "state": "busy"}, "status_indicator_update")
        for column_info in all_columns_metadata:
            column_name = column_info.get("ColumnName")
            col_type = next((v for k, v in column_info.items() if "type" in k.lower()), "").upper()

            if required_type and col_type != "UNKNOWN":
                is_numeric = any(t in col_type for t in ["INT", "NUMERIC", "DECIMAL", "FLOAT"])
                is_char = any(t in col_type for t in ["CHAR", "VARCHAR", "TEXT"])
                if (required_type == "numeric" and not is_numeric) or \
                   (required_type == "character" and not is_char):
                    skipped_result = {"status": "skipped", "metadata": {"tool_name": tool_name, "column_name": column_name}, "results": [{"reason": f"Tool requires {required_type}, but '{column_name}' is {col_type}."}]}
                    all_column_results.append(skipped_result)
                    yield self._format_sse({"step": "Skipping incompatible column", "details": skipped_result}, "tool_result")
                    continue

            iter_command = {"tool_name": tool_name, "arguments": {**base_args, 'column_name': column_name}}
            col_result = await mcp_adapter.invoke_mcp_tool(self.dependencies['STATE'], iter_command)
            all_column_results.append(col_result)
        yield self._format_sse({"target": "db", "state": "idle"}, "status_indicator_update")

        self._add_to_structured_data(all_column_results)
        self.last_tool_output = {"metadata": {"tool_name": tool_name}, "results": all_column_results, "status": "success"}

    async def _generate_final_summary(self):
        """Generates the final summary for the user."""
        final_collected_data = self.structured_collected_data
        
        final_summary_text = ""
        if (self.last_tool_output and isinstance(self.last_tool_output, dict) and
            "response" in (self.last_tool_output.get("results", [{}])[0] or {})):
            
            final_summary_text = self.last_tool_output["results"][0]["response"]
        else:
            data_for_summary = self._prepare_data_for_final_summary()
            
            final_prompt = (
                "You are an expert data analyst. Synthesize all collected data into a clear, concise, and insightful final answer.\n\n"
                f"--- USER'S QUESTION ---\n'{self.original_user_input}'\n\n"
                f"--- DATA COLLECTED ---\n```json\n{data_for_summary}\n```\n\n"
                "--- INSTRUCTIONS ---\n"
                "1.  Provide a holistic analysis and actionable insights.\n"
                "2.  Begin with a high-level summary, then use bullet points for key observations.\n"
                "3.  Your entire response **MUST** begin with `FINAL_ANSWER:`.\n"
            )
            reason="Generating final summary from all collected tool data."
            yield self._format_sse({"step": "Calling LLM to write final report", "details": reason})
            final_llm_response, _, _ = await self._call_llm_and_update_tokens(prompt=final_prompt, reason=reason)
            final_summary_text = final_llm_response

        clean_summary = final_summary_text.replace("FINAL_ANSWER:", "").strip() or "The agent has completed its work."

        yield self._format_sse({"step": "LLM has generated the final answer", "details": clean_summary}, "llm_thought")

        formatter = OutputFormatter(
            llm_response_text=clean_summary,
            collected_data=final_collected_data,
            is_workflow=True
        )
        final_html = formatter.render()
        
        session_manager.add_to_history(self.session_id, 'assistant', final_html)
        yield self._format_sse({"final_answer": final_html}, "final_answer")
        self.state = self.AgentState.DONE

    def _prepare_data_for_final_summary(self) -> str:
        """Prepares all collected data for the final summarization prompt."""
        items_to_process = [item for sublist in self.structured_collected_data.values() for item in sublist]

        successful_results = []
        for item in items_to_process:
            if isinstance(item, list):
                successful_results.extend(sub_item for sub_item in item if isinstance(sub_item, dict) and sub_item.get("status") == "success")
            elif isinstance(item, dict) and item.get("status") == "success":
                successful_results.append(item)
        
        if not successful_results:
            return "No data was collected from successful tool executions."
            
        return json.dumps([res for res in successful_results if res.get("type") != "chart"], indent=2, ensure_ascii=False)

    def _enrich_arguments_from_history(self, prompt_name: str, arguments: dict) -> tuple[dict, list]:
        """Scans conversation history to find missing arguments for a prompt call."""
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

    async def _get_tool_constraints(self, tool_name: str) -> dict:
        """Uses an LLM to determine if a tool requires numeric or character columns."""
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
            
            reason="Determining tool constraints for column iteration."
            response_text, _, _ = await self._call_llm_and_update_tokens(
                prompt=prompt, reason=reason,
                system_prompt_override="You are a JSON-only responding assistant.",
                raise_on_error=True
            )

            try:
                constraints = json.loads(re.search(r'\{.*\}', response_text, re.DOTALL).group(0))
            except (json.JSONDecodeError, AttributeError):
                constraints = {}
        
        self.tool_constraints_cache[tool_name] = constraints
        return constraints

    async def _recover_from_phase_failure(self, failed_phase_goal: str):
        """Attempts to recover from a persistently failing phase by generating a new plan."""
        yield self._format_sse({"step": "Attempting LLM-based Recovery", "details": "The current plan is stuck. Asking LLM to generate a new plan."})

        last_error = "No specific error message found."
        failed_tool_name = "N/A (Phase Failed)"
        for action in reversed(self.action_history):
            result = action.get("result", {})
            if isinstance(result, dict) and result.get("status") == "error":
                last_error = result.get("data", result.get("error", "Unknown error"))
                failed_tool_name = action.get("action", {}).get("tool_name", failed_tool_name)
                # --- MIGRATE: Skipped Tool Tracking ---
                self.globally_skipped_tools.add(failed_tool_name)
                break

        recovery_prompt = ERROR_RECOVERY_PROMPT.format(
            user_question=self.original_user_input,
            error_message=last_error,
            failed_tool_name=failed_tool_name,
            all_collected_data=json.dumps(self.workflow_state, indent=2),
            workflow_goal_and_plan=f"The agent was trying to achieve this goal: '{failed_phase_goal}'"
        )
        
        reason = "Recovering from persistent phase failure."
        response_text, input_tokens, output_tokens = await self._call_llm_and_update_tokens(
            prompt=recovery_prompt, 
            reason=reason,
            raise_on_error=True
        )
        
        updated_session = session_manager.get_session(self.session_id)
        if updated_session:
            yield self._format_sse({"statement_input": input_tokens, "statement_output": output_tokens, "total_input": updated_session.get("input_tokens", 0), "total_output": updated_session.get("output_tokens", 0)}, "token_update")

        try:
            json_str = response_text
            if response_text.strip().startswith("```json"):
                match = re.search(r"```json\s*\n(.*?)\n\s*```", response_text, re.DOTALL)
                if match:
                    json_str = match.group(1).strip()
            
            new_plan = json.loads(json_str)
            if not isinstance(new_plan, list): raise ValueError("Recovery plan is not a list.")

            yield self._format_sse({"step": "Recovery Plan Generated", "details": new_plan})
            
            self.meta_plan = new_plan
            self.current_phase_index = 0
            self.action_history.append({"action": "RECOVERY_REPLAN", "result": "success"})

        except (json.JSONDecodeError, ValueError) as e:
            raise RuntimeError(f"LLM-based recovery failed. The LLM did not return a valid new plan. Response: {response_text}. Error: {e}")
