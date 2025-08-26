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
from trusted_data_agent.core.config import APP_CONFIG
from trusted_data_agent.agent.prompts import (
    ERROR_RECOVERY_PROMPT,
    WORKFLOW_META_PLANNING_PROMPT,
    WORKFLOW_TACTICAL_PROMPT,
    TACTICAL_SELF_CORRECTION_PROMPT,
    TACTICAL_SELF_CORRECTION_PROMPT_COLUMN_ERROR,
    TACTICAL_SELF_CORRECTION_PROMPT_TABLE_ERROR
)
from trusted_data_agent.agent import orchestrators

app_logger = logging.getLogger("quart.app")

DEFINITIVE_TOOL_ERRORS = {
    "Invalid query": "The generated query was invalid and could not be run against the database.",
    "3523": "The user does not have the necessary permissions for the requested object." # Example of a specific Teradata error code
}

RECOVERABLE_TOOL_ERRORS = {
    # This regex now captures the full object path (e.g., db.table) for better context
    "table_not_found": r"Object '([\w\.]+)' does not exist",
    "column_not_found": r"Column '(\w+)' does not exist"
}

class DefinitiveToolError(Exception):
    """Custom exception for unrecoverable tool errors."""
    def __init__(self, message, friendly_message):
        super().__init__(message)
        self.friendly_message = friendly_message

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

    def __init__(self, session_id: str, original_user_input: str, dependencies: dict, active_prompt_name: str = None, prompt_arguments: dict = None, execution_depth: int = 0, disabled_history: bool = False, previous_turn_data: list = None, force_history_disable: bool = False):
        self.session_id = session_id
        self.original_user_input = original_user_input
        self.dependencies = dependencies
        self.state = self.AgentState.PLANNING
        
        self.structured_collected_data = {}
        self.workflow_state = {} 
        self.turn_action_history = []
        self.meta_plan = None
        self.current_phase_index = 0
        self.last_tool_output = None
        
        self.active_prompt_name = active_prompt_name
        self.prompt_arguments = prompt_arguments or {}
        self.workflow_goal_prompt = ""

        self.is_in_loop = False
        self.current_loop_items = []
        self.processed_loop_items = []
        
        self.tool_constraints_cache = {}
        self.globally_skipped_tools = set()
        self.temp_data_holder = None
        self.last_failed_action_info = "None"
        self.events_to_yield = []
        self.last_action_str = None 
        
        self.llm_debug_history = []
        self.max_steps = 40
        
        self.session_known_entities = {}
        session_data = session_manager.get_session(self.session_id)
        if session_data:
            self.session_known_entities = session_data.get("session_known_entities", {}).copy()

        self.execution_depth = execution_depth
        self.MAX_EXECUTION_DEPTH = 5
        
        self.disabled_history = disabled_history or force_history_disable
        self.previous_turn_data = previous_turn_data or []
        self.is_delegation_only_plan = False
        self.is_synthesis_from_history = False
        self.is_conversational_plan = False


    @staticmethod
    def _format_sse(data: dict, event: str = None) -> str:
        msg = f"data: {json.dumps(data)}\n"
        if event is not None:
            msg += f"event: {event}\n"
        return f"{msg}\n"

    async def _call_llm_and_update_tokens(self, prompt: str, reason: str, system_prompt_override: str = None, raise_on_error: bool = False, disabled_history: bool = False) -> tuple[str, int, int]:
        """A centralized wrapper for calling the LLM that handles token updates."""
        final_disabled_history = disabled_history or self.disabled_history
        
        response_text, statement_input_tokens, statement_output_tokens = await llm_handler.call_llm_api(
            self.dependencies['STATE']['llm'], prompt, self.session_id,
            dependencies=self.dependencies, reason=reason,
            system_prompt_override=system_prompt_override, raise_on_error=raise_on_error,
            disabled_history=final_disabled_history
        )
        self.llm_debug_history.append({"reason": reason, "response": response_text})
        app_logger.debug(f"LLM RESPONSE (DEBUG): Reason='{reason}', Response='{response_text}'")
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
        final_answer_override = None
        try:
            if self.state == self.AgentState.PLANNING:
                should_replan = False
                planning_is_disabled_history = self.disabled_history

                while True: # Loop to allow for a single re-plan if necessary
                    async for event in self._generate_meta_plan(force_disable_history=planning_is_disabled_history):
                        yield event

                    if self.is_conversational_plan:
                        app_logger.info("Detected a conversational plan. Bypassing execution.")
                        self.state = self.AgentState.SUMMARIZING
                        break # Exit the planning loop and proceed to summarization

                    is_synthesis_plan = (
                        self.meta_plan and
                        len(self.meta_plan) == 1 and
                        self.meta_plan[0].get('relevant_tools') == ["CoreLLMTask"]
                    )

                    if is_synthesis_plan:
                        if APP_CONFIG.ALLOW_SYNTHESIS_FROM_HISTORY:
                            self.is_synthesis_from_history = True
                            app_logger.info("Detected a 'synthesis from history' plan. CoreLLMTask will run in full_context mode.")
                            yield self._format_sse({
                                "step": "Plan Optimization",
                                "type": "plan_optimization",
                                "details": "Agent determined the answer exists in history. Bypassing data collection and attempting direct synthesis."
                            })
                            break # Proceed with this plan
                        else:
                            if should_replan:
                                app_logger.error("Re-planning without history still resulted in a synthesis-only plan. Executing as is.")
                                break
                            
                            app_logger.warning("Planner suggested synthesis from history, but the feature is disabled. Forcing re-plan without history.")
                            yield self._format_sse({
                                "step": "System Correction",
                                "type": "workaround",
                                "details": "Agent is re-evaluating the plan without conversational history to ensure all necessary data is gathered."
                            })
                            should_replan = True
                            planning_is_disabled_history = True
                            continue # Re-run the planning loop with history disabled
                    else:
                        break # This is a normal plan, proceed
                
                if not self.is_conversational_plan:
                    self.state = self.AgentState.EXECUTING
                    self.is_delegation_only_plan = (
                        self.meta_plan and
                        len(self.meta_plan) == 1 and
                        'executable_prompt' in self.meta_plan[0]
                    )
            
            try:
                if self.state == self.AgentState.EXECUTING:
                    async for event in self._run_plan(): yield event
            except DefinitiveToolError as e:
                app_logger.error(f"Execution halted by definitive tool error: {e.friendly_message}")
                yield self._format_sse({"step": "Unrecoverable Error", "details": e.friendly_message, "type": "error"}, "tool_result")
                final_answer_override = f"I could not complete the request. Reason: {e.friendly_message}"
                self.state = self.AgentState.SUMMARIZING

            if self.state == self.AgentState.SUMMARIZING:
                if final_answer_override:
                    formatter = OutputFormatter(
                        llm_response_text=final_answer_override,
                        collected_data=self.structured_collected_data,
                        original_user_input=self.original_user_input
                    )
                    final_html = formatter.render()
                    session_manager.add_to_history(self.session_id, 'assistant', final_html)
                    yield self._format_sse({"final_answer": final_html}, "final_answer")
                    self.state = self.AgentState.DONE
                elif self.is_delegation_only_plan:
                    app_logger.info("This was a delegation-only plan. Skipping redundant final summary.")
                    self.state = self.AgentState.DONE
                else:
                    async for event in self._generate_final_summary(): yield event

        except Exception as e:
            root_exception = unwrap_exception(e)
            app_logger.error(f"Error in state {self.state.name}: {root_exception}", exc_info=True)
            self.state = self.AgentState.ERROR
            yield self._format_sse({"error": "Execution stopped due to an unrecoverable error.", "details": str(root_exception)}, "error")
        finally:
            if not self.disabled_history:
                final_known_entities = self._update_session_known_entities()
                session_manager.update_session_known_session_entities(self.session_id, final_known_entities)
                session_manager.update_last_turn_data(self.session_id, self.turn_action_history)
                app_logger.debug(f"Saved final known entities and last turn data to session {self.session_id}")

    def _distill_data_for_llm_context(self, data: any) -> any:
        """
        Recursively distills large data structures into metadata summaries to protect the LLM context window.
        """
        if isinstance(data, dict):
            if 'results' in data and isinstance(data['results'], list):
                results_list = data['results']
                is_large = len(results_list) > 5 or len(json.dumps(results_list)) > 2048

                if is_large and all(isinstance(item, dict) for item in results_list):
                    distilled_result = {
                        "status": data.get("status", "success"),
                        "metadata": {
                            "row_count": len(results_list),
                            "columns": list(results_list[0].keys()) if results_list else [],
                            **data.get("metadata", {})
                        },
                        "comment": "Full data is too large for context. This is a summary."
                    }
                    return distilled_result
            
            return {key: self._distill_data_for_llm_context(value) for key, value in data.items()}
        
        elif isinstance(data, list):
            return [self._distill_data_for_llm_context(item) for item in data]
        
        return data

    def _create_summary_from_history(self, history: list) -> str:
        """
        Creates a token-efficient, high-signal summary of a history list for the planner.
        This now uses the data distillation method to keep the context lean.
        """
        history_copy = copy.deepcopy(history)
        
        for entry in history_copy:
            if 'result' in entry:
                entry['result'] = self._distill_data_for_llm_context(entry['result'])
                
        return json.dumps(history_copy, indent=2)

    def _update_session_known_entities(self) -> dict:
        """
        Processes the current turn's history to update the session's long-term
        memory of known entities. It now uses lists to store multiple discovered
        values for an entity.
        """
        current_entities = self.session_known_entities.copy()
        all_known_tool_args = set(self.dependencies['STATE'].get('all_known_mcp_arguments', {}).get('tool', []))
        
        normalized_to_canonical_map = {
            arg.lower().replace('_', ''): arg for arg in all_known_tool_args
        }

        def _add_entity(entities_dict, key, value):
            """Helper to add a value to an entity list, avoiding duplicates."""
            if key not in entities_dict:
                entities_dict[key] = []
            
            if not isinstance(entities_dict[key], list):
                entities_dict[key] = [entities_dict[key]]

            if value not in entities_dict[key]:
                entities_dict[key].append(value)

        for entry in self.turn_action_history:
            action = entry.get("action", {})
            result = entry.get("result", {})

            if isinstance(result, dict) and result.get('status') == 'success':
                args = action.get("arguments", {})
                for arg_name, arg_value in args.items():
                    if arg_name in all_known_tool_args and arg_value:
                        _add_entity(current_entities, arg_name, arg_value)

                result_data = entry.get("result", {})
                if 'results' in result_data and isinstance(result_data.get('results'), list):
                    for row in result_data['results']:
                        if not isinstance(row, dict):
                            continue
                        
                        for row_key, row_value in row.items():
                            if not row_value:
                                continue

                            normalized_row_key = row_key.lower().replace('_', '')
                            
                            if normalized_row_key in normalized_to_canonical_map:
                                canonical_arg_name = normalized_to_canonical_map[normalized_row_key]
                                
                                if isinstance(row_value, (str, int, float)):
                                    _add_entity(current_entities, canonical_arg_name, row_value)
        
        return current_entities

    def _get_latest_entities(self) -> dict:
        """
        Abstracts the session's known entities, which may contain lists of values,
        into a dictionary with only the most recent (last) value for each entity.
        This provides a simple view for internal agent functions.
        """
        latest_entities = {}
        for key, value in self.session_known_entities.items():
            if isinstance(value, list) and value:
                latest_entities[key] = value[-1]
            else:
                latest_entities[key] = value
        return latest_entities

    async def _generate_meta_plan(self, force_disable_history: bool = False):
        """The universal planner. It generates a meta-plan for ANY request."""
        prompt_obj = None
        explicit_parameters_section = ""
        
        if self.active_prompt_name:
            yield self._format_sse({"step": "Loading Workflow Prompt", "details": f"Loading '{self.active_prompt_name}'"})
            mcp_client = self.dependencies['STATE'].get('mcp_client')
            if not mcp_client: raise RuntimeError("MCP client is not connected.")
            
            prompt_def = None
            structured_prompts = self.dependencies['STATE'].get('structured_prompts', {})
            for category_prompts in structured_prompts.values():
                for p in category_prompts:
                    if p.get("name") == self.active_prompt_name:
                        prompt_def = p
                        break
                if prompt_def: break

            if not prompt_def:
                raise ValueError(f"Could not find a definition for prompt '{self.active_prompt_name}' in the local cache.")

            required_args = {arg['name'] for arg in prompt_def.get('arguments', []) if arg.get('required')}
            
            enriched_args = self.prompt_arguments.copy()
            inferred_args = set()
            
            latest_entities = self._get_latest_entities()
            for arg_name in required_args:
                if arg_name not in enriched_args or enriched_args.get(arg_name) is None:
                    if arg_name in latest_entities:
                        enriched_args[arg_name] = latest_entities[arg_name]
                        inferred_args.add(arg_name)

            if inferred_args:
                yield self._format_sse({
                    "step": "System Correction",
                    "details": f"System inferred missing arguments {inferred_args} from conversation history.",
                    "type": "workaround",
                    "correction_type": "inferred_argument"
                })
            
            missing_args = {arg for arg in required_args if arg not in enriched_args or enriched_args.get(arg) is None}
            if missing_args:
                raise ValueError(
                    f"Cannot execute prompt '{self.active_prompt_name}' because the following required arguments "
                    f"are missing and could not be found in the session context: {missing_args}"
                )
            
            self.prompt_arguments = enriched_args

            try:
                async with mcp_client.session("teradata_mcp_server") as temp_session:
                    prompt_obj = await load_mcp_prompt(
                        temp_session, name=self.active_prompt_name, arguments=self.prompt_arguments
                    )
            except Exception as e:
                app_logger.error(f"Failed to load MCP prompt '{self.active_prompt_name}': {e}", exc_info=True)
                raise ValueError(f"Prompt '{self.active_prompt_name}' could not be loaded from the MCP server.") from e

            if not prompt_obj: raise ValueError(f"Prompt '{self.active_prompt_name}' could not be loaded.")
            
            self.workflow_goal_prompt = get_prompt_text_content(prompt_obj)
            if not self.workflow_goal_prompt:
                raise ValueError(f"Could not extract text content from rendered prompt '{self.active_prompt_name}'.")

            param_items = [f"- {key}: {json.dumps(value)}" for key, value in self.prompt_arguments.items()]
            explicit_parameters_section = (
                "\n--- EXPLICIT PARAMETERS ---\n"
                "The following parameters were explicitly provided for this prompt execution:\n"
                + "\n".join(param_items) + "\n"
            )
        else:
            self.workflow_goal_prompt = self.original_user_input

        summary = f"Generating a strategic meta-plan for the goal"
        details_payload = {
            "summary": summary,
            "full_text": self.workflow_goal_prompt
        }
        yield self._format_sse({"step": "Calling LLM for Planning", "details": details_payload})

        previous_turn_summary_str = self._create_summary_from_history(self.previous_turn_data)
        
        session_entities_str = json.dumps(self.session_known_entities, indent=2)

        active_prompt_context_section = ""
        if self.active_prompt_name:
            active_prompt_context_section = f"- Active Prompt: You are currently executing the '{self.active_prompt_name}' prompt. Your plan should execute the steps described in the goal, not re-call the same prompt."

        data_gathering_rule_str = ""
        answer_from_history_rule_str = ""
        if APP_CONFIG.ALLOW_SYNTHESIS_FROM_HISTORY:
            data_gathering_rule_str = (
                "**CRITICAL RULE (Grounding):** Your primary objective is to answer the user's `GOAL` using data from the available tools. You **MUST** prioritize using a data-gathering tool if the `Workflow History` does not contain a direct and complete answer to the user's `GOAL`."
            )
            answer_from_history_rule_str = (
                "2.  **CRITICAL RULE (Answer from History):** If the `Workflow History` or `Known Entities` contain enough information to fully answer the user's `GOAL`, your response **MUST be a single JSON object** for a one-phase plan. This plan **MUST** call the `CoreLLMTask` tool. You **MUST** write the complete, final answer text inside the `synthesized_answer` argument within that tool call. **You are acting as a planner; DO NOT use the `FINAL_ANSWER:` format.**"
            )

        planning_prompt = WORKFLOW_META_PLANNING_PROMPT.format(
            workflow_goal=self.workflow_goal_prompt,
            explicit_parameters_section=explicit_parameters_section,
            original_user_input=self.original_user_input,
            turn_action_history=previous_turn_summary_str,
            session_known_entities=session_entities_str,
            execution_depth=self.execution_depth,
            active_prompt_context_section=active_prompt_context_section,
            data_gathering_priority_rule=data_gathering_rule_str,
            answer_from_history_rule=answer_from_history_rule_str
        )
        
        yield self._format_sse({"target": "llm", "state": "busy"}, "status_indicator_update")
        response_text, input_tokens, output_tokens = await self._call_llm_and_update_tokens(
            prompt=planning_prompt, 
            reason=f"Generating a strategic meta-plan for the goal: '{self.workflow_goal_prompt[:100]}'",
            disabled_history=force_disable_history
        )
        yield self._format_sse({"target": "llm", "state": "idle"}, "status_indicator_update")

        app_logger.info(
            f"\n--- Meta-Planner Turn ---\n"
            f"** CONTEXT **\n"
            f"Original User Input: {self.original_user_input}\n"
            f"Execution Depth: {self.execution_depth}\n"
            f"Session Known Entities (for prompt):\n{session_entities_str}\n"
            f"Previous Turn History Summary (for prompt):\n{previous_turn_summary_str}\n"
            f"** GENERATED PLAN **\n{response_text}\n"
            f"-------------------------"
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

            plan_object = json.loads(json_str)
            
            if isinstance(plan_object, dict) and plan_object.get("plan_type") == "conversational":
                self.is_conversational_plan = True
                self.temp_data_holder = plan_object.get("response", "I'm sorry, I don't have a response for that.")
                yield self._format_sse({"step": "Conversational Response Identified", "details": self.temp_data_holder})
                return

            if isinstance(plan_object, dict) and ("tool_name" in plan_object or "prompt_name" in plan_object):
                yield self._format_sse({
                    "step": "System Correction",
                    "type": "workaround",
                    "details": "Planner returned a direct action instead of a plan. System is correcting the format."
                })
                tool_name = plan_object.get("tool_name") or plan_object.get("prompt_name")
                self.meta_plan = [{
                    "phase": 1,
                    "goal": f"Execute the action for the user's request: '{self.original_user_input}'",
                    "relevant_tools": [tool_name]
                }]
            elif not isinstance(plan_object, list) or not plan_object:
                raise ValueError("LLM response for meta-plan was not a non-empty list.")
            else:
                self.meta_plan = plan_object

            yield self._format_sse({"step": "Strategic Meta-Plan Generated", "details": self.meta_plan})
        except (json.JSONDecodeError, ValueError) as e:
            raise RuntimeError(f"Failed to generate a valid meta-plan from the LLM. Response: {response_text}. Error: {e}")

    async def _run_plan(self):
        """Executes the generated meta-plan, delegating to loop or standard executors."""
        if not self.meta_plan:
            raise RuntimeError("Cannot execute plan: meta_plan is not generated.")

        if (len(self.meta_plan) > 0 and 
            'executable_prompt' in self.meta_plan[0] and
            self.execution_depth < self.MAX_EXECUTION_DEPTH):
            
            phase_one = self.meta_plan[0]
            prompt_name = phase_one.get('executable_prompt')
            prompt_args = phase_one.get('arguments', {})
            
            yield self._format_sse({
                "step": "Prompt Execution Granted",
                "details": f"Executing prompt '{prompt_name}' as Phase 1 of the plan.",
                "type": "workaround"
            })
            
            sub_executor = PlanExecutor(
                session_id=self.session_id,
                original_user_input=phase_one.get('goal', f"Executing prompt: {prompt_name}"),
                dependencies=self.dependencies,
                active_prompt_name=prompt_name,
                prompt_arguments=prompt_args,
                execution_depth=self.execution_depth + 1,
                disabled_history=self.disabled_history,
                previous_turn_data=self.turn_action_history
            )
            
            async for event in sub_executor.run():
                yield event
            
            self.structured_collected_data.update(sub_executor.structured_collected_data)
            self.workflow_state.update(sub_executor.workflow_state)
            self.turn_action_history.extend(sub_executor.turn_action_history)
            
            self.meta_plan.pop(0)
            for phase in self.meta_plan:
                phase['phase'] -= 1

        while self.current_phase_index < len(self.meta_plan):
            current_phase = self.meta_plan[self.current_phase_index]
            
            is_hallucinated_loop = (
                current_phase.get("type") == "loop" and
                isinstance(current_phase.get("loop_over"), list) and
                all(isinstance(item, str) for item in current_phase.get("loop_over"))
            )
            
            if is_hallucinated_loop:
                async for event in orchestrators.execute_hallucinated_loop(self, current_phase):
                    yield event
            elif current_phase.get("type") == "loop":
                async for event in self._execute_looping_phase(current_phase):
                    yield event
            else:
                async for event in self._execute_standard_phase(current_phase):
                    yield event
            
            self.current_phase_index += 1

        app_logger.info("Meta-plan has been fully executed. Transitioning to summarization.")
        self.state = self.AgentState.SUMMARIZING
    
    def _extract_loop_items(self, source_phase_key: str) -> list:
        """
        Intelligently extracts the list of items to iterate over from a previous phase's results.
        """
        if source_phase_key not in self.workflow_state:
            app_logger.warning(f"Loop source '{source_phase_key}' not found in workflow state.")
            return []

        source_data = self.workflow_state[source_phase_key]
        
        def find_results_list(data):
            if isinstance(data, list):
                for item in data:
                    found = find_results_list(item)
                    if found is not None: return found
            elif isinstance(data, dict):
                if 'results' in data and isinstance(data['results'], list):
                    return data['results']
                for value in data.values():
                    found = find_results_list(value)
                    if found is not None: return found
            return None

        items = find_results_list(source_data)
        
        if items is None:
            app_logger.warning(f"Could not find a 'results' list in '{source_phase_key}'. Returning empty list.")
            return []
            
        return items

    async def _execute_looping_phase(self, phase: dict):
        """
        Orchestrates the execution of a looping phase. It uses a "fast path" for simple,
        repetitive tool calls to improve performance, and a standard, LLM-driven path
        for complex or synthesis-based loops.
        """
        phase_goal = phase.get("goal", "No goal defined.")
        phase_num = phase.get("phase", self.current_phase_index + 1)
        loop_over_key = phase.get("loop_over")
        relevant_tools = phase.get("relevant_tools", [])

        yield self._format_sse({
            "step": f"Starting Plan Phase {phase_num}/{len(self.meta_plan)}",
            "type": "phase_start",
            "details": {
                "phase_num": phase_num,
                "total_phases": len(self.meta_plan),
                "goal": phase_goal,
                "phase_details": phase
            }
        })

        self.current_loop_items = self._extract_loop_items(loop_over_key)
        
        if not self.current_loop_items:
            yield self._format_sse({"step": "Skipping Empty Loop", "details": f"No items found from '{loop_over_key}' to loop over."})
            yield self._format_sse({
                "step": f"Ending Plan Phase {phase_num}/{len(self.meta_plan)}",
                "type": "phase_end",
                "details": {"phase_num": phase_num, "total_phases": len(self.meta_plan), "status": "skipped"}
            })
            return

        is_fast_path_candidate = (
            len(relevant_tools) == 1 and 
            relevant_tools[0] not in ["CoreLLMTask", "viz_createChart"]
        )

        if is_fast_path_candidate:
            tool_name = relevant_tools[0]
            yield self._format_sse({
                "step": "Plan Optimization", 
                "type": "plan_optimization",
                "details": f"Engaging enhanced fast path for tool loop: '{tool_name}'"
            })
            
            session_context_args = self._get_latest_entities()
            phase_context_args = phase.get("arguments", {})
            
            synonym_map = {
                'tablename': ['table_name', 'obj_name', 'object_name'],
                'databasename': ['database_name', 'db_name'],
                'columnname': ['column_name', 'col_name']
            }

            all_loop_results = []
            yield self._format_sse({"target": "db", "state": "busy"}, "status_indicator_update")
            for i, item in enumerate(self.current_loop_items):
                yield self._format_sse({"step": f"Processing Loop Item {i+1}/{len(self.current_loop_items)}", "details": item})
                
                merged_args = {**session_context_args, **phase_context_args}
                if isinstance(item, dict):
                    for key, value in item.items():
                        normalized_key = key.lower().replace('_', '')
                        target_keys = synonym_map.get(normalized_key, [key])
                        for target_key in target_keys:
                            merged_args[target_key] = value

                command = {"tool_name": tool_name, "arguments": merged_args}
                async for event in self._execute_tool(command, phase, is_fast_path=True):
                    yield event
                
                self.turn_action_history.append({"action": command, "result": self.last_tool_output})
                all_loop_results.append(self.last_tool_output)

            yield self._format_sse({"target": "db", "state": "idle"}, "status_indicator_update")
            
            phase_result_key = f"result_of_phase_{phase_num}"
            self.workflow_state[phase_result_key] = all_loop_results
            self._add_to_structured_data(all_loop_results)
            self.last_tool_output = all_loop_results

        else: 
            self.is_in_loop = True
            self.processed_loop_items = []
            
            for i, item in enumerate(self.current_loop_items):
                yield self._format_sse({"step": f"Processing Loop Item {i+1}/{len(self.current_loop_items)}", "details": item})
                
                try:
                    async for event in self._execute_standard_phase(phase, is_loop_iteration=True):
                        yield event
                except Exception as e:
                    error_message = f"Error processing item {item}: {e}"
                    app_logger.error(error_message, exc_info=True)
                    error_result = {
                        "status": "error", 
                        "item": item, 
                        "error_message": {
                            "summary": f"An error occurred while processing the item.",
                            "details": str(e)
                        }
                    }
                    self._add_to_structured_data(error_result)
                    yield self._format_sse({"step": "Loop Item Failed", "details": error_result, "type": "error"}, "tool_result")

                self.processed_loop_items.append(item)

            self.is_in_loop = False
            self.current_loop_items = []
            self.processed_loop_items = []

        yield self._format_sse({
            "step": f"Ending Plan Phase {phase_num}/{len(self.meta_plan)}",
            "type": "phase_end",
            "details": {"phase_num": phase_num, "total_phases": len(self.meta_plan), "status": "completed"}
        })

    # --- MODIFICATION: Add deterministic data type validation for chart calls ---
    def _is_numeric(self, value: any) -> bool:
        """Checks if a value can be reliably converted to a number."""
        if isinstance(value, (int, float)):
            return True
        if isinstance(value, str):
            try:
                float(value.replace(',', ''))
                return True
            except (ValueError, TypeError):
                return False
        return False

    async def _execute_standard_phase(self, phase: dict, is_loop_iteration: bool = False):
        """Executes a single, non-looping phase or a single iteration of a complex loop."""
        phase_goal = phase.get("goal", "No goal defined.")
        phase_num = phase.get("phase", self.current_phase_index + 1)
        relevant_tools = phase.get("relevant_tools", [])
        strategic_args = phase.get("arguments", {})
        executable_prompt = phase.get("executable_prompt")

        if not is_loop_iteration:
            yield self._format_sse({
                "step": f"Starting Plan Phase {phase_num}/{len(self.meta_plan)}",
                "type": "phase_start",
                "details": {
                    "phase_num": phase_num,
                    "total_phases": len(self.meta_plan),
                    "goal": phase_goal,
                    "phase_details": phase
                }
            })

        tool_name = relevant_tools[0] if len(relevant_tools) == 1 else None
        if tool_name and tool_name != "viz_createChart":
            all_tools = self.dependencies['STATE'].get('mcp_tools', {})
            tool_def = all_tools.get(tool_name)
            if tool_def:
                required_args = {name for name, details in (tool_def.args.items() if hasattr(tool_def, 'args') and isinstance(tool_def.args, dict) else {}) if details.get('required')}
                
                if required_args.issubset(strategic_args.keys()):
                    yield self._format_sse({
                        "step": "Plan Optimization", 
                        "type": "plan_optimization",
                        "details": f"FASTPATH initiated for '{tool_name}'."
                    })
                    fast_path_action = {"tool_name": tool_name, "arguments": strategic_args}
                    async for event in self._execute_action_with_orchestrators(fast_path_action, phase):
                        yield event
                    
                    yield self._format_sse(
                        {"target": "context", "state": "processing_complete"}, 
                        "context_state_update"
                    )
                    if not is_loop_iteration:
                        yield self._format_sse({
                            "step": f"Ending Plan Phase {phase_num}/{len(self.meta_plan)}",
                            "type": "phase_end",
                            "details": {"phase_num": phase_num, "total_phases": len(self.meta_plan), "status": "completed"}
                        })
                    return

        phase_attempts = 0
        max_phase_attempts = 5
        while True:
            phase_attempts += 1
            if phase_attempts > max_phase_attempts:
                app_logger.error(f"Phase '{phase_goal}' failed after {max_phase_attempts} attempts. Attempting LLM recovery.")
                async for event in self._recover_from_phase_failure(phase_goal):
                    yield event
                return 

            enriched_args, enrich_events, _ = self._enrich_arguments_from_history(relevant_tools)
            
            for event in enrich_events:
                self.events_to_yield.append(event)

            yield self._format_sse({"target": "llm", "state": "busy"}, "status_indicator_update")
            next_action, input_tokens, output_tokens = await self._get_next_tactical_action(
                phase_goal, relevant_tools, enriched_args, strategic_args, executable_prompt
            )
            yield self._format_sse({"target": "llm", "state": "idle"}, "status_indicator_update")
            
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

            async for event in self._execute_action_with_orchestrators(next_action, phase):
                yield event
            
            if self.last_tool_output and isinstance(self.last_tool_output, dict) and self.last_tool_output.get("status") == "success":
                # --- MODIFICATION: Add deterministic data type validation for chart calls ---
                if next_action.get("tool_name") == "viz_createChart":
                    is_valid_chart = True
                    # 1. Check for basic mapping existence
                    spec = self.last_tool_output.get("spec", {})
                    options = spec.get("options", {})
                    mapping_keys = ['xField', 'yField', 'seriesField', 'angleField', 'colorField']
                    if not any(key in options for key in mapping_keys):
                        is_valid_chart = False
                        self.last_failed_action_info = "The last attempt to create a chart failed because the 'mapping' argument was incorrect or missing. You MUST provide a valid mapping with the correct keys (e.g., 'angle', 'color')."
                    
                    # 2. Check for data type mismatches
                    if is_valid_chart:
                        mapping = next_action.get("arguments", {}).get("mapping", {})
                        data = next_action.get("arguments", {}).get("data", [])
                        if data and mapping:
                            first_row = data[0]
                            numeric_roles = ['angle', 'y_axis', 'value']
                            for role, column_name in mapping.items():
                                if role.lower() in numeric_roles:
                                    if column_name in first_row and not self._is_numeric(first_row[column_name]):
                                        is_valid_chart = False
                                        self.last_failed_action_info = f"The last attempt failed. You mapped the non-numeric column '{column_name}' to the '{role}' role, which requires a number. You MUST map a numeric column to this role."
                                        break # Exit the loop on first error
                    
                    if not is_valid_chart:
                        app_logger.warning(f"Silent chart failure detected. Reason: {self.last_failed_action_info}")
                        continue # Force a retry of the phase

                self.last_action_str = None
                break 
            else:
                app_logger.warning(f"Action failed. Attempt {phase_attempts}/{max_phase_attempts} for phase.")
        
        if not is_loop_iteration:
            yield self._format_sse({
                "step": f"Ending Plan Phase {phase_num}/{len(self.meta_plan)}",
                "type": "phase_end",
                "details": {"phase_num": phase_num, "total_phases": len(self.meta_plan), "status": "completed"}
            })

    async def _execute_action_with_orchestrators(self, action: dict, phase: dict):
        """
        A wrapper that runs pre-flight checks (orchestrators) before executing a tool.
        These orchestrators act as a safety net for common planning mistakes.
        """
        tool_name = action.get("tool_name")
        prompt_name = action.get("prompt_name")

        if not tool_name and not prompt_name:
            raise ValueError("Action from tactical LLM is missing a 'tool_name' or 'prompt_name'.")

        if prompt_name:
            yield self._format_sse({
                "step": "Prompt Execution Granted",
                "details": f"Executing prompt '{prompt_name}' as a sub-task.",
                "type": "workaround"
            })
            sub_executor = PlanExecutor(
                session_id=self.session_id,
                original_user_input=f"Executing prompt: {prompt_name}",
                dependencies=self.dependencies,
                active_prompt_name=prompt_name,
                prompt_arguments=action.get("arguments", {}),
                execution_depth=self.execution_depth + 1,
                disabled_history=self.disabled_history,
                previous_turn_data=self.turn_action_history
            )
            async for event in sub_executor.run():
                yield event
            
            self.structured_collected_data.update(sub_executor.structured_collected_data)
            self.workflow_state.update(sub_executor.workflow_state)
            self.turn_action_history.extend(sub_executor.turn_action_history)
            self.last_tool_output = {"status": "success"} # Mark as success for the loop
            return

        is_range_candidate, date_param_name, tool_supports_range = self._is_date_query_candidate(action)
        if is_range_candidate and not tool_supports_range:
            yield self._format_sse({"target": "llm", "state": "busy"}, "status_indicator_update")
            async for event in self._classify_date_query_type(): yield event
            yield self._format_sse({"target": "llm", "state": "idle"}, "status_indicator_update")
            if self.temp_data_holder and self.temp_data_holder.get('type') == 'range':
                async for event in orchestrators.execute_date_range_orchestrator(self, action, date_param_name, self.temp_data_holder.get('phrase')):
                    yield event
                return

        tool_scope = self.dependencies['STATE'].get('tool_scopes', {}).get(tool_name)
        has_column_arg = "column_name" in action.get("arguments", {})
        if tool_scope == 'column' and not has_column_arg:
             async for event in orchestrators.execute_column_iteration(self, action):
                 yield event
             return
        
        async for event in self._execute_tool(action, phase):
            yield event

    def _resolve_arguments(self, arguments: dict) -> dict:
        """
        Scans tool arguments for placeholders (e.g., 'result_of_phase_1') and
        replaces them with the actual data from the workflow state.
        """
        if not isinstance(arguments, dict):
            return arguments

        resolved_args = {}
        for key, value in arguments.items():
            if isinstance(value, str):
                match = re.fullmatch(r"result_of_phase_(\d+)", value)
                if match:
                    phase_num = int(match.group(1))
                    source_key = f"result_of_phase_{phase_num}"
                    
                    if source_key in self.workflow_state:
                        data = self.workflow_state[source_key]
                        
                        if (isinstance(data, list) and len(data) == 1 and 
                            isinstance(data[0], dict) and "results" in data[0] and
                            isinstance(data[0]["results"], list) and len(data[0]["results"]) == 1 and
                            isinstance(data[0]["results"][0], dict) and len(data[0]["results"][0]) == 1):
                            
                            extracted_value = next(iter(data[0]["results"][0].values()))
                            app_logger.info(f"Resolved placeholder '{value}' to single extracted value: '{extracted_value}'")
                            resolved_args[key] = extracted_value
                        else:
                            app_logger.info(f"Resolved placeholder '{value}' to full data structure.")
                            resolved_args[key] = data
                    else:
                        app_logger.warning(f"Could not resolve placeholder '{value}': key '{source_key}' not in workflow state.")
                        resolved_args[key] = value 
                else:
                    resolved_args[key] = value
            else:
                resolved_args[key] = value
        
        return resolved_args

    async def _execute_tool(self, action: dict, phase: dict, is_fast_path: bool = False):
        """Executes a single tool call with a built-in retry and recovery mechanism."""
        tool_name = action.get("tool_name")
        arguments = action.get("arguments", {})
        
        if tool_name == "CoreLLMTask" and "synthesized_answer" in arguments:
            app_logger.info("Bypassing CoreLLMTask execution. Using synthesized answer from planner.")
            self.last_tool_output = {
                "status": "success",
                "results": [{"response": arguments["synthesized_answer"]}]
            }
            if not is_fast_path:
                yield self._format_sse({"step": "Tool Execution Result", "details": self.last_tool_output, "tool_name": tool_name}, "tool_result")
                self.turn_action_history.append({"action": action, "result": self.last_tool_output})
                phase_num = phase.get("phase", self.current_phase_index + 1)
                phase_result_key = f"result_of_phase_{phase_num}"
                self.workflow_state.setdefault(phase_result_key, []).append(self.last_tool_output)
                self._add_to_structured_data(self.last_tool_output)
            return
        
        max_retries = 3
        
        if 'arguments' in action:
            action['arguments'] = self._resolve_arguments(arguments)

        if tool_name == "CoreLLMTask" and self.is_synthesis_from_history:
            app_logger.info("Preparing CoreLLMTask for 'full_context' execution.")
            session_data = session_manager.get_session(self.session_id)
            session_history = session_data.get("session_history", []) if session_data else []
            
            action.setdefault("arguments", {})["mode"] = "full_context"
            action.setdefault("arguments", {})["session_history"] = session_history
            action["arguments"]["user_question"] = self.original_user_input
        
        for attempt in range(max_retries):
            if 'notification' in action:
                yield self._format_sse({"step": "System Notification", "details": action['notification'], "type": "workaround"})
                del action['notification']

            if tool_name == "CoreLLMTask" and not self.is_synthesis_from_history:
                distilled_workflow_state = self._distill_data_for_llm_context(copy.deepcopy(self.workflow_state))
                action.setdefault("arguments", {})["data"] = distilled_workflow_state
            
            if not is_fast_path:
                yield self._format_sse({"step": "Tool Execution Intent", "details": action}, "tool_result")
            
            status_target = "db"
            if tool_name == "CoreLLMTask":
                status_target = "llm"
            elif tool_name.startswith("util_"):
                status_target = "llm"
            
            yield self._format_sse({"target": status_target, "state": "busy"}, "status_indicator_update")
            
            tool_result, input_tokens, output_tokens = await mcp_adapter.invoke_mcp_tool(self.dependencies['STATE'], action)

            yield self._format_sse({"target": status_target, "state": "idle"}, "status_indicator_update")

            if input_tokens > 0 or output_tokens > 0:
                updated_session = session_manager.get_session(self.session_id)
                if updated_session:
                    yield self._format_sse({
                        "statement_input": input_tokens,
                        "statement_output": output_tokens,
                        "total_input": updated_session.get("input_tokens", 0),
                        "total_output": updated_session.get("output_tokens", 0)
                    }, "token_update")

            self.last_tool_output = tool_result 
            
            if isinstance(tool_result, dict) and tool_result.get("status") == "error":
                yield self._format_sse({"details": tool_result, "tool_name": tool_name}, "tool_error")
                
                error_data_str = str(tool_result.get('data', ''))
                
                for error_pattern, friendly_message in DEFINITIVE_TOOL_ERRORS.items():
                    if re.search(error_pattern, error_data_str, re.IGNORECASE):
                        raise DefinitiveToolError(error_data_str, friendly_message)
                
                if attempt < max_retries - 1:
                    correction_details = {
                        "summary": f"Tool failed. Attempting self-correction ({attempt + 1}/{max_retries - 1}).",
                        "details": tool_result
                    }
                    yield self._format_sse({"step": "System Self-Correction", "type": "workaround", "details": correction_details})
                    
                    corrected_action, correction_events = await self._attempt_tool_self_correction(action, tool_result)
                    for event in correction_events:
                        yield event
                    
                    if corrected_action:
                        if "FINAL_ANSWER:" in corrected_action:
                            self.last_tool_output = {"status": "success", "results": [{"response": corrected_action}]}
                            break
                        action = corrected_action
                        continue
                    else:
                        correction_failed_details = {
                            "summary": "Unable to find a correction. Aborting retries for this action.",
                            "details": tool_result
                        }
                        yield self._format_sse({"step": "System Self-Correction Failed", "type": "error", "details": correction_failed_details})
                        break
                else:
                    persistent_failure_details = {
                        "summary": f"Tool '{tool_name}' failed after {max_retries} attempts.",
                        "details": tool_result
                    }
                    yield self._format_sse({"step": "Persistent Failure", "type": "error", "details": persistent_failure_details})
            else:
                if not is_fast_path:
                    yield self._format_sse({"step": "Tool Execution Result", "details": tool_result, "tool_name": tool_name}, "tool_result")
                break 
        
        if not is_fast_path:
            self.turn_action_history.append({"action": action, "result": self.last_tool_output})
            phase_num = phase.get("phase", self.current_phase_index + 1)
            phase_result_key = f"result_of_phase_{phase_num}"
            self.workflow_state.setdefault(phase_result_key, []).append(self.last_tool_output)
            self._add_to_structured_data(self.last_tool_output)

    def _enrich_arguments_from_history(self, relevant_tools: list[str], current_args: dict = None) -> tuple[dict, list, bool]:
        """
        Scans the current turn's action history to find missing arguments for a tool call.
        It now only uses arguments from tool calls that were definitively successful.
        """
        events_to_yield = []
        initial_args = current_args.copy() if current_args else {}
        enriched_args = initial_args.copy()
        
        all_tools = self.dependencies['STATE'].get('mcp_tools', {})
        required_args_for_phase = set()
        for tool_name in relevant_tools:
            tool = all_tools.get(tool_name)
            if not tool: continue
            args_dict = tool.args if isinstance(tool.args, dict) else {}
            for arg_name, arg_details in args_dict.items():
                if arg_details.get('required', False):
                    required_args_for_phase.add(arg_name)

        args_to_find = {arg for arg in required_args_for_phase if arg not in enriched_args or not enriched_args.get(arg)}
        if not args_to_find:
            return enriched_args, [], False

        for entry in reversed(self.turn_action_history):
            if not args_to_find: break
            
            result = entry.get("result", {})
            is_successful_data_action = (
                isinstance(result, dict) and 
                result.get('status') == 'success' and 
                result.get('results')
            )
            is_successful_chart_action = (
                isinstance(result, dict) and
                result.get('type') == 'chart' and
                'spec' in result
            )

            if not (is_successful_data_action or is_successful_chart_action):
                continue

            action_args = entry.get("action", {}).get("arguments", {})
            for arg_name in list(args_to_find):
                if arg_name in action_args and action_args[arg_name] is not None:
                    enriched_args[arg_name] = action_args[arg_name]
                    args_to_find.remove(arg_name)

            if isinstance(result, dict):
                result_metadata = result.get("metadata", {})
                if result_metadata:
                    metadata_to_arg_map = {
                        "database": "database_name",
                        "table": "table_name",
                        "column": "column_name"
                    }
                    for meta_key, arg_name in metadata_to_arg_map.items():
                        if arg_name in args_to_find and meta_key in result_metadata:
                            enriched_args[arg_name] = result_metadata[meta_key]
                            args_to_find.remove(arg_name)
        
        was_enriched = enriched_args != initial_args
        if was_enriched:
            for arg_name, value in enriched_args.items():
                if arg_name not in initial_args:
                    app_logger.info(f"Proactively inferred '{arg_name}' from turn history: '{value}'")
                    events_to_yield.append(self._format_sse({
                        "step": "System Correction",
                        "details": f"System inferred '{arg_name}: {value}' from the current turn's actions.",
                        "type": "workaround",
                        "correction_type": "inferred_argument"
                    }))

        return enriched_args, events_to_yield, was_enriched

    async def _get_next_tactical_action(self, current_phase_goal: str, relevant_tools: list[str], enriched_args: dict, strategic_args: dict, executable_prompt: str = None) -> tuple[dict | str, int, int]:
        """Makes a tactical LLM call to decide the single next best action for the current phase."""
        
        permitted_tools_with_details = ""
        all_tools = self.dependencies['STATE'].get('mcp_tools', {})
        
        for tool_name in relevant_tools:
            tool = all_tools.get(tool_name)
            if not tool: continue

            tool_str = f"\n- Tool: `{tool.name}`\n  - Description: {tool.description}"
            args_dict = tool.args if isinstance(tool.args, dict) else {}
            
            if args_dict:
                tool_str += "\n  - Arguments:"
                for arg_name, arg_details in args_dict.items():
                    is_required = arg_details.get('required', False)
                    arg_type = arg_details.get('type', 'any')
                    req_str = "required" if is_required else "optional"
                    arg_desc = arg_details.get('description', 'No description.')
                    tool_str += f"\n    - `{arg_name}` ({arg_type}, {req_str}): {arg_desc}"
            permitted_tools_with_details += tool_str + "\n"
        
        permitted_prompts_with_details = "None"
        if executable_prompt:
            all_prompts = self.dependencies['STATE'].get('structured_prompts', {})
            prompt_info = None
            for category, prompts in all_prompts.items():
                for p in prompts:
                    if p['name'] == executable_prompt:
                        prompt_info = p
                        break
                if prompt_info: break
            
            if prompt_info:
                prompt_str = f"\n- Prompt: `{prompt_info['name']}`\n  - Description: {prompt_info.get('description', 'No description.')}"
                if prompt_info.get('arguments'):
                    prompt_str += "\n  - Arguments:"
                    for arg in prompt_info['arguments']:
                        req_str = "required" if arg.get('required') else "optional"
                        prompt_str += f"\n    - `{arg['name']}` ({arg.get('type', 'any')}, {req_str}): {arg.get('description', 'No description.')}"
                permitted_prompts_with_details = prompt_str + "\n"


        context_enrichment_section = ""
        if enriched_args:
            context_items = [f"- `{name}`: `{value}`" for name, value in enriched_args.items()]
            context_enrichment_section = (
                "\n--- CONTEXT FROM HISTORY ---\n"
                "The following critical information has been inferred from the conversation history. You MUST use it to fill in missing arguments.\n"
                + "\n".join(context_items) + "\n"
            )

        loop_context_section = ""
        if self.is_in_loop:
            next_item = next((item for item in self.current_loop_items if item not in self.processed_loop_items), None)
            if next_item:
                loop_context_section = (
                    f"\n--- LOOP CONTEXT ---\n"
                    f"- You are currently in a loop to process multiple items.\n"
                    f"- All Items in Loop: {json.dumps(self.current_loop_items)}\n"
                    f"- Items Already Processed: {json.dumps(self.processed_loop_items)}\n"
                    f"- Your task is to process this single item next: {json.dumps(next_item)}\n"
                )

        strategic_arguments_section = "None provided."
        if strategic_args:
            strategic_arguments_section = json.dumps(strategic_args, indent=2)

        distilled_workflow_state = self._distill_data_for_llm_context(copy.deepcopy(self.workflow_state))
        distilled_turn_history = self._distill_data_for_llm_context(copy.deepcopy(self.turn_action_history))

        tactical_system_prompt = WORKFLOW_TACTICAL_PROMPT.format(
            workflow_goal=self.workflow_goal_prompt,
            current_phase_goal=current_phase_goal,
            strategic_arguments_section=strategic_arguments_section,
            permitted_tools_with_details=permitted_tools_with_details,
            permitted_prompts_with_details=permitted_prompts_with_details,
            last_attempt_info=self.last_failed_action_info,
            turn_action_history=json.dumps(distilled_turn_history, indent=2),
            all_collected_data=json.dumps(distilled_workflow_state, indent=2),
            loop_context_section=loop_context_section,
            context_enrichment_section=context_enrichment_section
        )

        response_text, input_tokens, output_tokens = await self._call_llm_and_update_tokens(
            prompt="Determine the next action based on the instructions and state provided in the system prompt.",
            reason=f"Deciding next tactical action for phase: {current_phase_goal}",
            system_prompt_override=tactical_system_prompt,
            disabled_history=True
        )
        
        self.last_failed_action_info = "None"

        if "FINAL_ANSWER:" in response_text.upper() or "SYSTEM_ACTION_COMPLETE" in response_text.upper():
            return "SYSTEM_ACTION_COMPLETE", input_tokens, output_tokens

        try:
            json_match = re.search(r"```json\s*\n(.*?)\n\s*```|(\{.*\})", response_text, re.DOTALL)
            if not json_match: raise json.JSONDecodeError("No JSON object found", response_text, 0)
            
            json_str = json_match.group(1) or json_match.group(2)
            if not json_str: raise json.JSONDecodeError("Extracted JSON is empty", response_text, 0)

            raw_action = json.loads(json_str.strip())
            
            action_details = raw_action
            tool_name_synonyms = ["tool_name", "name", "tool", "action_name"]
            prompt_name_synonyms = ["prompt_name", "prompt"]
            arg_synonyms = ["arguments", "args", "tool_input", "action_input", "parameters"]
            
            possible_wrapper_keys = ["action", "tool_call", "tool", "prompt_call", "prompt"]
            for key in possible_wrapper_keys:
                if key in action_details and isinstance(action_details[key], dict):
                    action_details = action_details[key]
                    break 

            found_tool_name = None
            for key in tool_name_synonyms:
                if key in action_details:
                    found_tool_name = action_details.pop(key)
                    break
            
            found_prompt_name = None
            for key in prompt_name_synonyms:
                if key in action_details:
                    found_prompt_name = action_details.pop(key)
                    break

            found_args = None
            for key in arg_synonyms:
                if key in action_details and isinstance(action_details[key], dict):
                    found_args = action_details[key]
                    break
            
            if found_args is None:
                found_args = action_details

            normalized_action = {
                "tool_name": found_tool_name,
                "prompt_name": found_prompt_name,
                "arguments": found_args if isinstance(found_args, dict) else {}
            }

            if not normalized_action.get("tool_name") and not normalized_action.get("prompt_name"):
                if len(relevant_tools) == 1:
                    normalized_action["tool_name"] = relevant_tools[0]
                    self.events_to_yield.append(self._format_sse({
                        "step": "System Correction", "type": "workaround",
                        "correction_type": "inferred_tool_name",
                        "details": f"LLM omitted tool_name. System inferred '{relevant_tools[0]}'."
                    }))
                elif executable_prompt:
                    normalized_action["prompt_name"] = executable_prompt
                    self.events_to_yield.append(self._format_sse({
                        "step": "System Correction", "type": "workaround",
                        "correction_type": "inferred_prompt_name",
                        "details": f"LLM omitted prompt_name. System inferred '{executable_prompt}'."
                    }))
            
            if not normalized_action.get("tool_name") and not normalized_action.get("prompt_name"):
                 raise ValueError("Could not determine tool_name or prompt_name from LLM response.")

            return normalized_action, input_tokens, output_tokens
        except (json.JSONDecodeError, ValueError) as e:
            raise RuntimeError(f"Failed to get a valid JSON action from the tactical LLM. Response: {response_text}. Error: {e}")

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

    async def _generate_final_summary(self):
        """
        Generates the final summary. It intelligently handles different plan completion
        scenarios: dedicated summary tasks, looping summary tasks, and standard
        ad-hoc queries.
        """
        final_summary_text = ""
        
        if self.is_conversational_plan:
            final_summary_text = self.temp_data_holder or "I'm sorry, I don't have a response for that."
        elif self.meta_plan and self.turn_action_history:
            last_phase = self.meta_plan[-1]
            last_phase_num = last_phase.get("phase", len(self.meta_plan))
            phase_result_key = f"result_of_phase_{last_phase_num}"

            if last_phase.get("type") == "loop" and phase_result_key in self.workflow_state:
                app_logger.info(f"Final phase was a loop. Consolidating results from '{phase_result_key}'.")
                
                consolidated_texts = []
                phase_results = self.workflow_state.get(phase_result_key, [])
                
                items_to_check = []
                if isinstance(phase_results, list):
                    for item in phase_results:
                        if isinstance(item, list):
                            items_to_check.extend(item)
                        else:
                            items_to_check.append(item)
                
                for result in items_to_check:
                    if (isinstance(result, dict) and
                        result.get("status") == "success" and
                        "CoreLLMTask" in result.get("metadata", {}).get("tool_name", "") and
                        isinstance(result.get("results"), list) and result["results"]):
                        
                        summary_data = result["results"][0]
                        if isinstance(summary_data, dict) and "response" in summary_data:
                            consolidated_texts.append(summary_data["response"])

                if consolidated_texts:
                    final_summary_text = "\n\n<hr class='border-gray-600 my-4'>\n\n".join(consolidated_texts)
            
            elif not final_summary_text:
                last_action_entry = self.turn_action_history[-1] if self.turn_action_history else {}
                last_action = last_action_entry.get("action", {})
                last_result = last_action_entry.get("result", {})
                
                if (isinstance(last_action, dict) and last_action.get("tool_name") == "CoreLLMTask" and 
                    isinstance(last_result, dict) and last_result.get("status") == "success"):
                    
                    app_logger.info("Planner-defined single summary task found. Using its result directly.")
                    summary_data = last_result.get("results", [{}])[0]
                    final_summary_text = summary_data.get("response", "Planner summary task failed to produce text.")

        if not final_summary_text:
            app_logger.info("No planner-defined summary found. Generating standard ad-hoc summary.")
            standard_task_description = (
                "You are an expert data analyst. Your task is to create a final report for the user based on the provided data."
            )
            standard_formatting_instructions = (
                "Your entire response MUST be formatted in standard markdown and MUST be structured as follows:\n\n"
                "1.  **(Optional) Key Metric:** If the answer to the user's question can be summarized by a single primary value (either quantitative like a number, or qualitative like a status), you MUST provide it on the very first line in a specific JSON format. The line must start with `Key Metric: ` followed by a JSON object with a `value` (as a string) and a `label` (a short description).\n"
                "    - Quantitative Example: `Key Metric: {{\"value\": \"21\", \"label\": \"Databases on system\"}}`\n"
                "    - Qualitative Example: `Key Metric: {{\"value\": \"High\", \"label\": \"System Utilization\"}}`\n"
                "    If there is no single primary value, you MUST omit this line entirely.\n\n"
                "2.  **The Direct Answer:** This part MUST immediately follow the Key Metric (or be the first line if no metric is provided). It must be a single, concise sentence that directly and factually answers the user's question.\n\n"
                "3.  **Key Observations:** This section MUST start with a level-2 markdown heading (`## Key Observations`). It should contain a bulleted list of all supporting details and context."
            )
            
            distilled_workflow_state = self._distill_data_for_llm_context(copy.deepcopy(self.workflow_state))

            core_llm_command = {
                "tool_name": "CoreLLMTask",
                "arguments": {
                    "task_description": standard_task_description,
                    "formatting_instructions": standard_formatting_instructions,
                    "user_question": self.original_user_input,
                    "source_data": list(distilled_workflow_state.keys()),
                    "data": distilled_workflow_state
                }
            }
            
            yield self._format_sse({"step": "Calling LLM to write final report", "details": "Synthesizing markdown-formatted summary."})
            
            yield self._format_sse({"target": "llm", "state": "busy"}, "status_indicator_update")
            summary_result, input_tokens, output_tokens = await mcp_adapter.invoke_mcp_tool(self.dependencies['STATE'], core_llm_command)
            yield self._format_sse({"target": "llm", "state": "idle"}, "status_indicator_update")

            updated_session = session_manager.get_session(self.session_id)
            if updated_session:
                yield self._format_sse({ "statement_input": input_tokens, "statement_output": output_tokens, "total_input": updated_session.get("input_tokens", 0), "total_output": updated_session.get("output_tokens", 0) }, "token_update")

            if (summary_result and summary_result.get("status") == "success" and
                "response" in (summary_result.get("results", [{}])[0] or {})):
                final_summary_text = summary_result["results"][0]["response"]
            else:
                app_logger.error(f"CoreLLMTask failed to generate a standard summary. Fallback response will be used. Result: {summary_result}")
                final_summary_text = "The agent has completed its work, but an issue occurred while generating the final summary."

        clean_summary = final_summary_text.replace("FINAL_ANSWER:", "").strip() or "The agent has completed its work."
        yield self._format_sse({"step": "LLM has generated the final answer", "details": clean_summary}, "llm_thought")

        formatter = OutputFormatter(
            llm_response_text=clean_summary,
            collected_data=self.structured_collected_data,
            original_user_input=self.original_user_input,
            active_prompt_name=self.active_prompt_name
        )
        final_html = formatter.render()
        
        session_manager.add_to_history(self.session_id, 'assistant', final_html)
        yield self._format_sse({"final_answer": final_html}, "final_answer")
        self.state = self.AgentState.DONE

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
        """
        Attempts to recover from a persistently failing phase by generating a new plan.
        This version is robust to conversational text mixed with the JSON output.
        """
        yield self._format_sse({"step": "Attempting LLM-based Recovery", "details": "The current plan is stuck. Asking LLM to generate a new plan."})

        last_error = "No specific error message found."
        failed_tool_name = "N/A (Phase Failed)"
        for action in reversed(self.turn_action_history):
            result = action.get("result", {})
            if isinstance(result, dict) and result.get("status") == "error":
                last_error = result.get("data", result.get("error", "Unknown error"))
                failed_tool_name = action.get("action", {}).get("tool_name", failed_tool_name)
                self.globally_skipped_tools.add(failed_tool_name)
                break
        
        distilled_workflow_state = self._distill_data_for_llm_context(copy.deepcopy(self.workflow_state))

        recovery_prompt = ERROR_RECOVERY_PROMPT.format(
            user_question=self.original_user_input,
            error_message=last_error,
            failed_tool_name=failed_tool_name,
            all_collected_data=json.dumps(distilled_workflow_state, indent=2),
            workflow_goal_and_plan=f"The agent was trying to achieve this goal: '{failed_phase_goal}'"
        )
        
        reason = "Recovering from persistent phase failure."
        yield self._format_sse({"target": "llm", "state": "busy"}, "status_indicator_update")
        response_text, input_tokens, output_tokens = await self._call_llm_and_update_tokens(
            prompt=recovery_prompt, 
            reason=reason,
            raise_on_error=True
        )
        yield self._format_sse({"target": "llm", "state": "idle"}, "status_indicator_update")
        
        updated_session = session_manager.get_session(self.session_id)
        if updated_session:
            yield self._format_sse({"statement_input": input_tokens, "statement_output": output_tokens, "total_input": updated_session.get("input_tokens", 0), "total_output": updated_session.get("output_tokens", 0)}, "token_update")

        try:
            json_match = re.search(r'(\[.*\]|\{.*\})', response_text, re.DOTALL)
            if not json_match:
                raise ValueError("No valid JSON plan or action found in the recovery response.")
            
            json_str = json_match.group(1)
            plan_object = json.loads(json_str)

            if isinstance(plan_object, dict) and ("tool_name" in plan_object or "prompt_name" in plan_object):
                app_logger.warning("Recovery LLM returned a direct action; wrapping it in a plan.")
                tool_name = plan_object.get("tool_name") or plan_object.get("prompt_name")
                new_plan = [{
                    "phase": 1,
                    "goal": f"Recovered plan: Execute the action for the user's request: '{self.original_user_input}'",
                    "relevant_tools": [tool_name]
                }]
            elif isinstance(plan_object, list):
                new_plan = plan_object
            else:
                raise ValueError("Recovered plan is not a valid list or action object.")

            yield self._format_sse({"step": "Recovery Plan Generated", "details": new_plan})
            
            self.meta_plan = new_plan
            self.current_phase_index = 0
            self.turn_action_history.append({"action": "RECOVERY_REPLAN", "result": {"status": "success"}})

        except (json.JSONDecodeError, ValueError) as e:
            raise RuntimeError(f"LLM-based recovery failed. The LLM did not return a valid new plan. Response: {response_text}. Error: {e}")

    async def _attempt_tool_self_correction(self, failed_action: dict, error_result: dict) -> tuple[dict | None, list]:
        """
        Attempts to correct a failed tool call using a tiered, pattern-based approach.
        It first checks for specific, recoverable errors and uses specialized prompts
        before falling back to a generic correction attempt.
        """
        events = []
        tool_name = failed_action.get("tool_name")
        error_data_str = str(error_result.get('data', ''))
        correction_prompt = None
        system_prompt_override = None
        reason = ""

        # Tier 1: Check for "Table Not Found" error.
        table_error_match = re.search(RECOVERABLE_TOOL_ERRORS["table_not_found"], error_data_str, re.IGNORECASE)
        if table_error_match:
            invalid_table = table_error_match.group(1)
            invalid_table_name_only = invalid_table.split('.')[-1]
            failed_args = failed_action.get("arguments", {})
            db_name = failed_args.get("database_name", "the specified database")
            
            app_logger.warning(f"Detected recoverable 'table_not_found' error for table: {invalid_table}")
            
            correction_prompt = TACTICAL_SELF_CORRECTION_PROMPT_TABLE_ERROR.format(
                user_question=self.original_user_input,
                tool_name=tool_name,
                failed_arguments=json.dumps(failed_args),
                invalid_table_name=invalid_table_name_only,
                database_name=db_name,
                tools_context=self.dependencies['STATE'].get('tools_context', ''),
                prompts_context=self.dependencies['STATE'].get('prompts_context', '')
            )
            reason = f"Fact-based recovery for non-existent table '{invalid_table_name_only}'"
            system_prompt_override = "You are an expert troubleshooter. Follow the recovery directives precisely."

        # Tier 2: Check for "Column Not Found" error if no table error was found.
        if not correction_prompt:
            column_error_match = re.search(RECOVERABLE_TOOL_ERRORS["column_not_found"], error_data_str, re.IGNORECASE)
            if column_error_match:
                invalid_column = column_error_match.group(1)
                app_logger.warning(f"Detected recoverable 'column_not_found' error for column: {invalid_column}")
                
                correction_prompt = TACTICAL_SELF_CORRECTION_PROMPT_COLUMN_ERROR.format(
                    user_question=self.original_user_input,
                    tool_name=tool_name,
                    failed_arguments=json.dumps(failed_action.get("arguments", {})),
                    invalid_column_name=invalid_column,
                    tools_context=self.dependencies['STATE'].get('tools_context', ''),
                    prompts_context=self.dependencies['STATE'].get('prompts_context', '')
                )
                reason = f"Fact-based recovery for non-existent column '{invalid_column}'"
                system_prompt_override = "You are an expert troubleshooter. Follow the recovery directives precisely."
        
        # Tier 3 (Fallback): Generic self-correction for all other unknown errors.
        if not correction_prompt:
            tool_def = self.dependencies['STATE'].get('mcp_tools', {}).get(tool_name)
            if not tool_def: return None, events

            enriched_args, enrich_events, _ = self._enrich_arguments_from_history(
                {name for name, details in (tool_def.args.items() if hasattr(tool_def, 'args') and isinstance(tool_def.args, dict) else {}) if details.get('required')},
                failed_action.get("arguments", {})
            )
            events.extend(enrich_events)

            session_data = session_manager.get_session(self.session_id)
            session_history = session_data.get("session_history", []) if session_data else []

            correction_prompt = TACTICAL_SELF_CORRECTION_PROMPT.format(
                tool_definition=json.dumps(vars(tool_def), default=str),
                failed_command=json.dumps(failed_action),
                error_message=json.dumps(error_result.get('data', 'No error data.')),
                history_context=json.dumps(self._get_latest_entities()),
                session_history=json.dumps(session_history)
            )
            reason = f"Generic self-correction for failed tool call: {tool_name}"
            system_prompt_override = "You are a JSON-only responding assistant."

        events.append(self._format_sse({"target": "llm", "state": "busy"}, "status_indicator_update"))
        response_str, input_tokens, output_tokens = await self._call_llm_and_update_tokens(
            prompt=correction_prompt,
            reason=reason,
            system_prompt_override=system_prompt_override,
            raise_on_error=False
        )
        events.append(self._format_sse({"target": "llm", "state": "idle"}, "status_indicator_update"))
        
        updated_session = session_manager.get_session(self.session_id)
        if updated_session:
            events.append(self._format_sse({
                "statement_input": input_tokens, 
                "statement_output": output_tokens, 
                "total_input": updated_session.get("input_tokens", 0), 
                "total_output": updated_session.get("output_tokens", 0)
            }, "token_update"))
        
        if "FINAL_ANSWER:" in response_str:
            app_logger.info("Self-correction resulted in a FINAL_ANSWER. Halting retries.")
            final_answer_text = response_str.split("FINAL_ANSWER:", 1)[1].strip()
            return final_answer_text, events

        try:
            json_match = re.search(r"```json\s*\n(.*?)\n\s*```|(\{.*\})", response_text, re.DOTALL)
            if not json_match: raise json.JSONDecodeError("No JSON object found", response_str, 0)
            
            json_str = json_match.group(1) or json_match.group(2)
            if not json_str: raise json.JSONDecodeError("Extracted JSON is empty", response_str, 0)
            
            corrected_data = json.loads(json_str.strip())
            
            if "tool_name" in corrected_data and "arguments" in corrected_data:
                corrected_action = corrected_data
                correction_details = {
                    "summary": f"LLM proposed a new action. Retrying with tool '{corrected_action['tool_name']}'.",
                    "details": corrected_action
                }
                events.append(self._format_sse({"step": "System Self-Correction", "type": "workaround", "details": correction_details}))
                return corrected_action, events
            
            new_args = corrected_data.get("arguments", corrected_data)
            if isinstance(new_args, dict):
                corrected_action = {**failed_action, "arguments": new_args}
                correction_details = {
                    "summary": f"LLM proposed a fix. Retrying tool with new arguments.",
                    "details": new_args
                }
                events.append(self._format_sse({"step": "System Self-Correction", "type": "workaround", "details": correction_details}))
                return corrected_action, events

        except (json.JSONDecodeError, TypeError):
            correction_failed_details = {
                "summary": "LLM failed to provide a valid JSON correction.",
                "details": response_str
            }
            events.append(self._format_sse({"step": "System Self-Correction", "type": "error", "details": correction_failed_details}))
            return None, events
            
        return None, events
