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
    # --- REFACTORED: Simplified states for the new architecture ---
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

    def __init__(self, session_id: str, original_user_input: str, dependencies: dict, initial_instruction: str = None):
        self.session_id = session_id
        self.original_user_input = original_user_input
        self.dependencies = dependencies
        
        # --- REFACTORED: State now starts at PLANNING ---
        self.state = self.AgentState.PLANNING
        
        # --- Unified State Properties ---
        self.collected_data = [] # Legacy, will be phased out
        self.structured_collected_data = {}
        self.workflow_state = {} 
        self.action_history = []
        self.meta_plan = None
        self.current_phase_index = 0
        self.last_tool_output = None
        
        # --- Workflow-specific context ---
        self.is_workflow = False
        self.active_prompt_name = None
        self.prompt_arguments = {} # For manually invoked prompts
        self.workflow_goal_prompt = ""

        # --- Non-Workflow Optimizations ---
        self.tool_constraints_cache = {}
        self.globally_skipped_tools = set()
        self.temp_data_holder = None
        self.last_failed_action_info = "None"
        self.events_to_yield = []
        
        # --- Debugging & Misc ---
        self.llm_debug_history = []
        self.max_steps = 40 # Safety break for loops

        # --- Handling manual prompt invocation from UI ---
        if initial_instruction:
            self._parse_initial_instruction(initial_instruction)

    def _parse_initial_instruction(self, instruction: str):
        """Parses the initial command to detect if it's a manual prompt invocation."""
        try:
            json_match = re.search(r"```json\s*\n(.*?)\n\s*```|(\{.*\})", instruction, re.DOTALL)
            if json_match:
                command_str = json_match.group(1) or json_match.group(2)
                command = json.loads(command_str.strip())
                if "prompt_name" in command:
                    self.is_workflow = True
                    self.active_prompt_name = command["prompt_name"]
                    self.prompt_arguments = command.get("arguments", {})
        except (json.JSONDecodeError, TypeError):
            app_logger.warning("Could not parse initial instruction as a prompt invocation.")


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
        context_key = context_key_override or f"Workflow: {self.active_prompt_name or 'Ad-hoc Plan'}"
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
                # Every execution starts by generating a plan.
                async for event in self._generate_meta_plan():
                    yield event
                self.state = self.AgentState.EXECUTING

            if self.state == self.AgentState.EXECUTING:
                # Once a plan exists, execute it.
                async for event in self._run_plan():
                    yield event

            if self.state == self.AgentState.SUMMARIZING:
                # After execution, generate the final summary.
                async for event in self._generate_final_summary():
                    yield event
            
            # The ERROR state is handled by the exception block below.

        except Exception as e:
            root_exception = unwrap_exception(e)
            app_logger.error(f"Error in state {self.state.name}: {root_exception}", exc_info=True)
            self.state = self.AgentState.ERROR
            yield self._format_sse({"error": "Execution stopped due to an unrecoverable error.", "details": str(root_exception)}, "error")

    async def _generate_meta_plan(self):
        """
        The universal planner. It generates a meta-plan for ANY request.
        """
        # Step 1: Determine the goal for the planner.
        if self.is_workflow:
            # For MCP Prompts, the goal is the rendered content of the prompt.
            yield self._format_sse({"step": "Loading Workflow Prompt", "details": f"Loading '{self.active_prompt_name}'..."})
            mcp_client = self.dependencies['STATE'].get('mcp_client')
            if not mcp_client: raise RuntimeError("MCP client is not connected.")
            
            async with mcp_client.session("teradata_mcp_server") as temp_session:
                prompt_obj = await load_mcp_prompt(
                    temp_session, name=self.active_prompt_name, arguments=self.prompt_arguments
                )
            if not prompt_obj: raise ValueError(f"Prompt '{self.active_prompt_name}' could not be loaded.")
            
            self.workflow_goal_prompt = get_prompt_text_content(prompt_obj)
            if not self.workflow_goal_prompt:
                raise ValueError(f"Could not extract text content from rendered prompt '{self.active_prompt_name}'.")
        else:
            # For direct user queries, the goal is the user's input itself.
            self.workflow_goal_prompt = self.original_user_input

        # Step 2: Call the LLM with the universal planning prompt.
        reason = f"Generating a strategic meta-plan for the goal: '{self.workflow_goal_prompt[:100]}...'"
        yield self._format_sse({"step": "Calling LLM for Planning", "details": reason})

        planning_system_prompt = WORKFLOW_META_PLANNING_PROMPT.format(
            workflow_goal=self.workflow_goal_prompt,
            original_user_input=self.original_user_input
        )
        
        response_text, input_tokens, output_tokens = await self._call_llm_and_update_tokens(
            prompt="Generate the meta-plan based on the instructions and context provided in the system prompt.", 
            reason=reason,
            system_prompt_override=planning_system_prompt
        )

        updated_session = session_manager.get_session(self.session_id)
        if updated_session:
            yield self._format_sse({ "statement_input": input_tokens, "statement_output": output_tokens, "total_input": updated_session.get("input_tokens", 0), "total_output": updated_session.get("output_tokens", 0) }, "token_update")
        
        # Step 3: Parse and store the plan.
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
        """
        The main, unified execution loop for the state machine.
        It orchestrates the execution of the meta-plan, phase by phase.
        """
        if not self.meta_plan:
            raise RuntimeError("Cannot execute plan: meta_plan is not generated.")

        while self.current_phase_index < len(self.meta_plan):
            current_phase = self.meta_plan[self.current_phase_index]
            phase_goal = current_phase.get("goal", "No goal defined for this phase.")
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
                    raise RuntimeError(f"Phase '{phase_goal}' failed to complete after {max_phase_attempts} attempts.")

                # --- Tactical LLM Call to get the next action ---
                next_action, input_tokens, output_tokens = await self._get_next_tactical_action(phase_goal, relevant_tools)
                
                if self.events_to_yield:
                    for event in self.events_to_yield: yield event
                    self.events_to_yield = []

                updated_session = session_manager.get_session(self.session_id)
                if updated_session:
                    yield self._format_sse({ "statement_input": input_tokens, "statement_output": output_tokens, "total_input": updated_session.get("input_tokens", 0), "total_output": updated_session.get("output_tokens", 0) }, "token_update")

                if isinstance(next_action, str) and next_action == "SYSTEM_ACTION_COMPLETE":
                    app_logger.info("Plan signaled early completion. Transitioning to summarization.")
                    self.state = self.AgentState.SUMMARIZING
                    return

                if not isinstance(next_action, dict):
                    raise RuntimeError(f"Tactical LLM failed to provide a valid action. Received: {next_action}")

                # --- Execute the action with embedded optimizers/orchestrators ---
                async for event in self._execute_action_with_orchestrators(next_action, current_phase):
                    yield event

                # --- Check if the phase is complete ---
                phase_is_complete, input_tokens, output_tokens = await self._is_phase_complete(phase_goal, phase_num)
                
                if input_tokens > 0 or output_tokens > 0:
                    updated_session = session_manager.get_session(self.session_id)
                    if updated_session:
                        yield self._format_sse({ "statement_input": input_tokens, "statement_output": output_tokens, "total_input": updated_session.get("input_tokens", 0), "total_output": updated_session.get("output_tokens", 0) }, "token_update")

                if phase_is_complete:
                    yield self._format_sse({"step": f"Phase {phase_num} Complete", "details": "Goal has been achieved."})
                    break 

            self.current_phase_index += 1

        app_logger.info("Meta-plan has been fully executed. Transitioning to summarization.")
        self.state = self.AgentState.SUMMARIZING


    async def _execute_action_with_orchestrators(self, action: dict, phase: dict):
        """
        A wrapper that runs pre-flight checks (orchestrators) before executing a tool.
        """
        tool_name = action.get("tool_name")
        if not tool_name:
            raise ValueError("Action from tactical LLM is missing a 'tool_name'.")
        
        # --- Pre-flight Check 1: Date-Range Orchestrator ---
        is_range_candidate, date_param_name, tool_supports_range = self._is_date_query_candidate(action)
        if is_range_candidate and not tool_supports_range:
            async for event in self._classify_date_query_type(): yield event
            if self.temp_data_holder and self.temp_data_holder.get('type') == 'range':
                async for event in self._execute_date_range_orchestrator(action, date_param_name, self.temp_data_holder.get('phrase')): yield event
                return # Orchestrator handles execution and state, so we exit early.

        # --- Pre-flight Check 2: Column-Iteration Orchestrator ---
        tool_scope = self.dependencies['STATE'].get('tool_scopes', {}).get(tool_name)
        has_column_arg = "column_name" in action.get("arguments", {})
        if tool_scope == 'column' and not has_column_arg:
             async for event in self._execute_column_iteration(action): yield event
             return # Orchestrator handles execution and state, so we exit early.

        # --- Standard Execution ---
        async for event in self._execute_tool(action, phase):
            yield event


    async def _execute_tool(self, action: dict, phase: dict):
        """Executes a single tool call as part of a plan phase."""
        tool_name = action.get("tool_name")
        phase_num = phase.get("phase", self.current_phase_index + 1)
        relevant_tools = phase.get("relevant_tools", [])

        # --- Self-Correction Guardrail ---
        if relevant_tools and tool_name not in relevant_tools:
            app_logger.warning(f"LLM proposed an invalid tool '{tool_name}' for the current phase. Expected one of: {relevant_tools}. Initiating self-correction.")
            self.last_failed_action_info = f"Your last attempt to use the tool '{tool_name}' was invalid because it is not in the list of permitted tools for this phase."
            yield self._format_sse({
                "step": "System Correction",
                "details": f"LLM chose an invalid tool ('{tool_name}'). Retrying with constraints.",
                "type": "workaround"
            })
            return # Return to the tactical loop to get a new action

        # --- Argument Enrichment & Normalization ---
        async for event in self._normalize_tool_arguments(action):
            yield event
        
        # --- Handle CoreLLMTask context injection ---
        if tool_name == "CoreLLMTask":
            if "arguments" not in action: action["arguments"] = {}
            action["arguments"]["data"] = copy.deepcopy(self.workflow_state)

        # --- Invoke the tool ---
        yield self._format_sse({"step": "Tool Execution Intent", "details": action}, "tool_result")
        
        status_target = "chart" if tool_name == "viz_createChart" else "db"
        yield self._format_sse({"target": status_target, "state": "busy"}, "status_indicator_update")
        tool_result = await mcp_adapter.invoke_mcp_tool(self.dependencies['STATE'], action)
        yield self._format_sse({"target": status_target, "state": "idle"}, "status_indicator_update")

        # --- Process the result ---
        self.last_tool_output = tool_result
        self.action_history.append({"action": action, "result": tool_result})
        
        phase_result_key = f"result_of_phase_{phase_num}"
        if phase_result_key not in self.workflow_state:
            self.workflow_state[phase_result_key] = []
        self.workflow_state[phase_result_key].append(tool_result)
        
        self._add_to_structured_data(tool_result, context_key_override="Plan Execution Results")

        if isinstance(tool_result, dict) and tool_result.get("status") == "error":
            # This is a tool execution error, not a planning error.
            # We will log it and let the phase completion check decide if we need to retry.
            yield self._format_sse({"details": tool_result, "tool_name": tool_name}, "tool_error")
        else:
            yield self._format_sse({"step": "Tool Execution Result", "details": tool_result, "tool_name": tool_name}, "tool_result")


    # --- Methods for Tactical Planning and Phase Completion (previously in WorkflowExecutor) ---

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
            app_logger.info("Tactical LLM signaled early completion.")
            return "SYSTEM_ACTION_COMPLETE", input_tokens, output_tokens

        try:
            json_match = re.search(r"```json\s*\n(.*?)\n\s*```|(\{.*\})", response_text, re.DOTALL)
            if not json_match: raise json.JSONDecodeError("No JSON object found", response_text, 0)
            
            json_str = json_match.group(1) or json_match.group(2)
            if not json_str: raise json.JSONDecodeError("Extracted JSON is empty", response_text, 0)

            action = json.loads(json_str.strip())
            
            # Normalize common LLM hallucinations
            tool_name_synonyms = ["tool_name", "name", "tool", "action"]
            found_tool_name = next((action.pop(key) for key in tool_name_synonyms if key in action), None)
            
            if found_tool_name: action["tool_name"] = found_tool_name
            
            if "tool_name" not in action and len(relevant_tools) == 1:
                action["tool_name"] = relevant_tools[0]
                self.events_to_yield.append(self._format_sse({
                    "step": "System Correction", "type": "workaround",
                    "details": f"LLM omitted tool_name. System inferred '{relevant_tools[0]}'."
                }))

            arg_synonyms = ["tool_input", "action_input", "tool_arguments", "parameters"]
            found_args = next((action.pop(key) for key in arg_synonyms if key in action), None)
            if found_args and "arguments" not in action:
                action["arguments"] = found_args

            return action, input_tokens, output_tokens
        except json.JSONDecodeError:
            raise RuntimeError(f"Failed to get a valid JSON action from the tactical LLM. Response: {response_text}")

    async def _is_phase_complete(self, current_phase_goal: str, phase_num: int) -> tuple[bool, int, int]:
        """Checks if the current phase's goal has been met."""
        phase_result_key = f"result_of_phase_{phase_num}"
        if phase_result_key in self.workflow_state:
            if any(isinstance(r, dict) and r.get("status") == "success" for r in self.workflow_state[phase_result_key]):
                app_logger.info(f"Deterministic check found success for phase {phase_num}. Marking complete.")
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

    async def _normalize_tool_arguments(self, next_action: dict):
        """Deterministically corrects common LLM argument name hallucinations."""
        if "arguments" not in next_action or not isinstance(next_action["arguments"], dict):
            return

        synonym_map = {
            "database": "database_name", "db": "database_name",
            "table": "table_name", "tbl": "table_name",
            "column": "column_name", "col": "column_name"
        }
        
        normalized_args = {}
        corrected = False
        for arg_name, arg_value in next_action["arguments"].items():
            canonical_name = synonym_map.get(arg_name.lower(), arg_name)
            if canonical_name != arg_name: corrected = True
            normalized_args[canonical_name] = arg_value
        
        if corrected:
            yield self._format_sse({
                "step": "System Correction", "type": "workaround",
                "details": "LLM used non-standard argument names. System corrected them."
            })
            next_action["arguments"] = normalized_args

    # --- Orchestrators (previously in non-workflow path) ---

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
        # This method remains unchanged from the previous version
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
        # This method remains largely unchanged, but now appends to structured_collected_data
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
            f"You are a date range calculation assistant. Given that the current date is {current_date_str}, "
            f"what are the start and end dates for '{date_phrase}'? "
            "Your response MUST be ONLY a JSON object with 'start_date' and 'end_date' in YYYY-MM-DD format."
        )
        reason = "Calculating date range."
        yield self._format_sse({"step": "Calling LLM", "details": reason})
        range_response_str, input_tokens, output_tokens = await self._call_llm_and_update_tokens(
            prompt=conversion_prompt, reason=reason,
            system_prompt_override="You are a JSON-only responding assistant.", raise_on_error=True
        )
        # ... (token update SSE) ...
        
        try:
            range_data = json.loads(range_response_str)
            start_date = datetime.strptime(range_data['start_date'], '%Y-%m-%d').date()
            end_date = datetime.strptime(range_data['end_date'], '%Y-%m-%d').date()
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise RuntimeError(f"Date Range Orchestrator failed to parse date range. Error: {e}")

        all_results = []
        yield self._format_sse({"target": "db", "state": "busy"}, "status_indicator_update")
        for day in range((end_date - start_date).days + 1):
            current_date_in_loop = start_date + timedelta(days=day)
            date_str = current_date_in_loop.strftime('%Y-%m-%d')
            yield self._format_sse({"step": f"Processing data for: {date_str}"})
            
            day_command = {**command, 'arguments': {**command['arguments'], date_param_name: date_str}}
            day_result = await mcp_adapter.invoke_mcp_tool(self.dependencies['STATE'], day_command)
            
            if isinstance(day_result, dict) and day_result.get("status") == "success" and day_result.get("results"):
                all_results.extend(day_result["results"])
        yield self._format_sse({"target": "db", "state": "idle"}, "status_indicator_update")
        
        final_tool_output = {
            "status": "success",
            "metadata": {"tool_name": tool_name, "comment": f"Consolidated results for {date_phrase}"},
            "results": all_results
        }
        self._add_to_structured_data(final_tool_output, context_key_override="Plan Execution Results")
        self.last_tool_output = final_tool_output

    async def _execute_column_iteration(self, command: dict):
        """Executes a tool over multiple columns."""
        # This method remains largely unchanged, but now appends to structured_collected_data
        tool_name = command.get("tool_name")
        base_args = command.get("arguments", {})
        
        db_name = base_args.get("database_name")
        table_name = base_args.get("table_name")

        cols_command = {"tool_name": "base_columnDescription", "arguments": {"database_name": db_name, "table_name": table_name}}
        yield self._format_sse({"target": "db", "state": "busy"}, "status_indicator_update")
        cols_result = await mcp_adapter.invoke_mcp_tool(self.dependencies['STATE'], cols_command)
        yield self._format_sse({"target": "db", "state": "idle"}, "status_indicator_update")
        
        if not (cols_result and isinstance(cols_result, dict) and cols_result.get('status') == 'success' and cols_result.get('results')):
            raise ValueError(f"Failed to retrieve column list for iteration. Response: {cols_result}")
        
        all_columns_metadata = cols_result.get('results', [])
        
        all_column_results = []
        
        # ... (logic to get tool constraints and iterate over columns) ...
        
        self._add_to_structured_data(all_column_results, context_key_override="Plan Execution Results")
        self.last_tool_output = {"metadata": {"tool_name": tool_name}, "results": all_column_results, "status": "success"}


    # --- Final Summary Generation ---
    async def _generate_final_summary(self):
        """Generates the final summary for the user."""
        # This method is largely the same, but now always uses structured_collected_data
        final_collected_data = self.structured_collected_data
        
        final_summary_text = ""
        if (self.last_tool_output and isinstance(self.last_tool_output, dict) and
            "response" in (self.last_tool_output.get("results", [{}])[0] or {})):
            
            yield self._format_sse({"step": "Finalizing Report", "details": "Using pre-formatted summary from last step."}, "llm_thought")
            final_summary_text = self.last_tool_output["results"][0]["response"]
        else:
            yield self._format_sse({"step": "Finalizing Report", "details": "Synthesizing all collected data."}, "llm_thought")
            data_for_summary = self._prepare_data_for_final_summary()
            
            final_prompt = (
                "You are an expert data analyst. Your task is to synthesize all collected data into a clear, concise, and insightful final answer for the user.\n\n"
                f"--- USER'S ORIGINAL QUESTION ---\n'{self.original_user_input}'\n\n"
                f"--- ALL RELEVANT DATA COLLECTED ---\n```json\n{data_for_summary}\n```\n\n"
                "--- YOUR INSTRUCTIONS ---\n"
                "1.  **Adopt the Persona of a Data Analyst:** Your goal is to provide a holistic analysis and deliver actionable insights, not just report numbers.\n"
                "2.  **Go Beyond the Obvious:** Start with the primary findings but then scrutinize the data for secondary insights, patterns, or anomalies.\n"
                "3.  **Structure Your Answer:** Begin with a high-level summary that directly answers the user's question. Then, use bullet points to highlight key, specific observations.\n"
                "4.  **CRITICAL:** Your entire response **MUST** begin with the exact prefix `FINAL_ANSWER:`, followed by your natural language summary.\n"
            )
            reason="Generating final summary from all collected tool data."
            yield self._format_sse({"step": "Calling LLM to write final report", "details": reason})
            final_llm_response, input_tokens, output_tokens = await self._call_llm_and_update_tokens(prompt=final_prompt, reason=reason)
            # ... (token update SSE) ...
            final_summary_text = final_llm_response

        clean_summary = final_summary_text.replace("FINAL_ANSWER:", "").strip()
        if not clean_summary:
             clean_summary = "The agent has completed its work. The collected data is displayed below."

        yield self._format_sse({"step": "LLM has generated the final answer", "details": clean_summary}, "llm_thought")

        formatter = OutputFormatter(
            llm_response_text=clean_summary,
            collected_data=final_collected_data,
            is_workflow=True # All executions are now considered workflows for formatting
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
