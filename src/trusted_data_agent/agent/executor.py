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
    TACTICAL_SELF_CORRECTION_PROMPT
)
from trusted_data_agent.agent import orchestrators

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

    def __init__(self, session_id: str, original_user_input: str, dependencies: dict, active_prompt_name: str = None, prompt_arguments: dict = None, execution_depth: int = 0, disabled_history: bool = False):
        self.session_id = session_id
        self.original_user_input = original_user_input
        self.dependencies = dependencies
        self.state = self.AgentState.PLANNING
        
        self.structured_collected_data = {}
        self.workflow_state = {} 
        self.action_history = []
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
        
        self.known_entities = {}
        session_data = session_manager.get_session(self.session_id)
        if session_data:
            self.known_entities = session_data.get("known_entities", {}).copy()

        self.execution_depth = execution_depth
        self.MAX_EXECUTION_DEPTH = 5
        
        self.disabled_history = disabled_history
        # --- MODIFICATION START: Add flag to track the nature of the plan ---
        self.is_delegation_only_plan = False
        # --- MODIFICATION END ---


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
            system_prompt_override=system_prompt_override, raise_on_error=raise_on_error,
            disabled_history=self.disabled_history
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
                
                # --- MODIFICATION START: Determine if the plan is delegation-only immediately after creation ---
                self.is_delegation_only_plan = (
                    self.meta_plan and
                    len(self.meta_plan) == 1 and
                    'executable_prompt' in self.meta_plan[0]
                )
                # --- MODIFICATION END ---

            if self.state == self.AgentState.EXECUTING:
                async for event in self._run_plan(): yield event

            if self.state == self.AgentState.SUMMARIZING:
                # --- MODIFICATION START: Check the pre-determined flag instead of the (now modified) plan ---
                if self.is_delegation_only_plan:
                    app_logger.info("This was a delegation-only plan. Skipping redundant final summary.")
                    self.state = self.AgentState.DONE
                else:
                    async for event in self._generate_final_summary(): yield event
                # --- MODIFICATION END ---

        except Exception as e:
            root_exception = unwrap_exception(e)
            app_logger.error(f"Error in state {self.state.name}: {root_exception}", exc_info=True)
            self.state = self.AgentState.ERROR
            yield self._format_sse({"error": "Execution stopped due to an unrecoverable error.", "details": str(root_exception)}, "error")
        finally:
            _, final_known_entities_str = self._create_optimized_context()
            final_known_entities = json.loads(final_known_entities_str)
            session_manager.update_session_known_entities(self.session_id, final_known_entities)
            app_logger.info(f"Saved final known entities to session {self.session_id}: {final_known_entities_str}")

    def _create_optimized_context(self) -> tuple[str, str]:
        """
        Creates a token-efficient, high-signal context summary for the planner by
        summarizing past results and dynamically extracting known entities from both
        the tool call arguments and the tool call results.
        """
        optimized_history = []
        known_entities = self.known_entities.copy()
        
        all_known_tool_args = set(self.dependencies['STATE'].get('all_known_mcp_arguments', {}).get('tool', []))
        
        normalized_to_canonical_map = {
            arg.lower().replace('_', ''): arg for arg in all_known_tool_args
        }

        for entry in self.action_history:
            action = entry.get("action", {})
            result = entry.get("result", {})
            
            result_summary = {}
            if isinstance(result, dict):
                result_summary['status'] = result.get('status')
                metadata = result.get("metadata", {})
                if 'row_count' in metadata:
                    result_summary['row_count'] = metadata['row_count']
                if 'columns' in metadata:
                    result_summary['columns'] = [col.get('name') for col in metadata.get('columns', [])]
                if 'error' in result:
                     result_summary['error'] = result.get('error')
            
            optimized_history.append({
                "action": action,
                "result_summary": result_summary
            })

            if result_summary.get('status') == 'success':
                args = action.get("arguments", {})
                for arg_name, arg_value in args.items():
                    if arg_name in all_known_tool_args and arg_value:
                        known_entities[arg_name] = arg_value

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
                                
                                if not isinstance(known_entities.get(canonical_arg_name), set):
                                    known_entities[canonical_arg_name] = set()
                                
                                known_entities[canonical_arg_name].add(row_value)

        final_known_entities = {}
        for key, value in known_entities.items():
            if isinstance(value, set):
                sorted_list = sorted(list(value))
                final_known_entities[key] = sorted_list[0] if len(sorted_list) == 1 else sorted_list
            else:
                final_known_entities[key] = value

        return json.dumps(optimized_history, indent=2), json.dumps(final_known_entities, indent=2)

    async def _generate_meta_plan(self):
        """The universal planner. It generates a meta-plan for ANY request."""
        if self.active_prompt_name:
            yield self._format_sse({"step": "Loading Workflow Prompt", "details": f"Loading '{self.active_prompt_name}'"})
            mcp_client = self.dependencies['STATE'].get('mcp_client')
            if not mcp_client: raise RuntimeError("MCP client is not connected.")
            
            prompt_def = self.dependencies['STATE'].get('mcp_prompts', {}).get(self.active_prompt_name)
            required_args = {arg.name for arg in prompt_def.arguments} if prompt_def and hasattr(prompt_def, 'arguments') else set()
            
            enriched_args, enrich_events, _ = self._enrich_arguments_from_history(required_args, self.prompt_arguments, is_prompt=True)

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

        summary = f"Generating a strategic meta-plan for the goal"
        details_payload = {
            "summary": summary,
            "full_text": self.workflow_goal_prompt
        }
        yield self._format_sse({"step": "Calling LLM for Planning", "details": details_payload})

        optimized_history_str, known_entities_str = self._create_optimized_context()

        planning_prompt = WORKFLOW_META_PLANNING_PROMPT.format(
            workflow_goal=self.workflow_goal_prompt,
            original_user_input=self.original_user_input,
            workflow_history=optimized_history_str,
            known_entities=known_entities_str,
            execution_depth=self.execution_depth
        )
        
        yield self._format_sse({"target": "llm", "state": "busy"}, "status_indicator_update")
        response_text, input_tokens, output_tokens = await self._call_llm_and_update_tokens(
            prompt=planning_prompt, 
            reason=f"Generating a strategic meta-plan for the goal: '{self.workflow_goal_prompt[:100]}'"
        )
        yield self._format_sse({"target": "llm", "state": "idle"}, "status_indicator_update")

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
                disabled_history=self.disabled_history
            )
            
            async for event in sub_executor.run():
                yield event
            
            self.structured_collected_data.update(sub_executor.structured_collected_data)
            self.workflow_state.update(sub_executor.workflow_state)
            self.action_history.extend(sub_executor.action_history)
            
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
            "step": "Starting Looping Phase",
            "details": f"Phase {phase_num}/{len(self.meta_plan)}: {phase_goal}",
            "phase_details": phase
        })

        self.current_loop_items = self._extract_loop_items(loop_over_key)
        
        if not self.current_loop_items:
            yield self._format_sse({"step": "Skipping Empty Loop", "details": f"No items found from '{loop_over_key}' to loop over."})
            return

        is_fast_path_candidate = (
            len(relevant_tools) == 1 and 
            relevant_tools[0] != "CoreLLMTask"
        )

        if is_fast_path_candidate:
            tool_name = relevant_tools[0]
            yield self._format_sse({
                "step": "Plan Optimization", 
                "type": "plan_optimization",
                "details": f"Engaging enhanced fast path for tool loop: '{tool_name}'"
            })
            
            context_args = {}
            for history_item in self.action_history:
                action = history_item.get("action", {})
                if "arguments" in action:
                    context_args.update(action.get("arguments", {}))
            
            synonym_map = {
                'tablename': ['table_name', 'obj_name', 'object_name'],
                'databasename': ['database_name', 'db_name'],
                'columnname': ['column_name', 'col_name']
            }

            all_loop_results = []
            yield self._format_sse({"target": "db", "state": "busy"}, "status_indicator_update")
            for i, item in enumerate(self.current_loop_items):
                yield self._format_sse({"step": f"Processing Loop Item {i+1}/{len(self.current_loop_items)}", "details": item})
                
                merged_args = context_args.copy()
                if isinstance(item, dict):
                    for key, value in item.items():
                        normalized_key = key.lower().replace('_', '')
                        target_keys = synonym_map.get(normalized_key, [key])
                        for target_key in target_keys:
                            merged_args[target_key] = value

                command = {"tool_name": tool_name, "arguments": merged_args}
                async for event in self._execute_tool(command, phase, is_fast_path=True):
                    yield event
                
                self.action_history.append({"action": command, "result": self.last_tool_output})
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
                    error_result = {"status": "error", "item": item, "error_message": str(e)}
                    self._add_to_structured_data(error_result)
                    yield self._format_sse({"step": "Loop Item Failed", "details": error_result, "type": "error"}, "tool_result")

                self.processed_loop_items.append(item)

            self.is_in_loop = False
            self.current_loop_items = []
            self.processed_loop_items = []

        yield self._format_sse({"step": f"Looping Phase {phase_num} Complete", "details": "All items have been processed."})

    async def _execute_standard_phase(self, phase: dict, is_loop_iteration: bool = False):
        """Executes a single, non-looping phase or a single iteration of a complex loop."""
        phase_goal = phase.get("goal", "No goal defined.")
        phase_num = phase.get("phase", self.current_phase_index + 1)
        relevant_tools = phase.get("relevant_tools", [])

        if not is_loop_iteration:
            yield self._format_sse({
                "step": "Starting Plan Phase",
                "details": f"Phase {phase_num}/{len(self.meta_plan)}: {phase_goal}",
                "phase_details": phase
            })

        phase_attempts = 0
        max_phase_attempts = 5
        while True:
            phase_attempts += 1
            if phase_attempts > max_phase_attempts:
                app_logger.error(f"Phase '{phase_goal}' failed after {max_phase_attempts} attempts. Attempting LLM recovery.")
                async for event in self._recover_from_phase_failure(phase_goal):
                    yield event
                return 

            enriched_args, enrich_events, _ = self._get_required_args_and_enrich(relevant_tools)
            
            for event in enrich_events:
                self.events_to_yield.append(event)

            yield self._format_sse({"target": "llm", "state": "busy"}, "status_indicator_update")
            next_action, input_tokens, output_tokens = await self._get_next_tactical_action(
                phase_goal, relevant_tools, enriched_args
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
                self.last_action_str = None
                break
            else:
                app_logger.warning(f"Action failed. Attempt {phase_attempts}/{max_phase_attempts} for phase.")

    async def _execute_action_with_orchestrators(self, action: dict, phase: dict):
        """
        A wrapper that runs pre-flight checks (orchestrators) before executing a tool.
        These orchestrators act as a safety net for common planning mistakes.
        """
        tool_name = action.get("tool_name")
        if not tool_name:
            raise ValueError("Action from tactical LLM is missing a 'tool_name'.")

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

    async def _execute_tool(self, action: dict, phase: dict, is_fast_path: bool = False):
        """Executes a single tool call with a built-in retry and recovery mechanism."""
        tool_name = action.get("tool_name")
        max_retries = 3
        
        for attempt in range(max_retries):
            if 'notification' in action:
                yield self._format_sse({"step": "System Notification", "details": action['notification'], "type": "workaround"})
                del action['notification']

            if tool_name == "CoreLLMTask":
                action.setdefault("arguments", {})["data"] = copy.deepcopy(self.workflow_state)
            
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
                
                if attempt < max_retries - 1:
                    yield self._format_sse({"step": "System Self-Correction", "type": "workaround", "details": f"Tool failed. Attempting self-correction ({attempt + 1}/{max_retries - 1})."})
                    
                    corrected_action, correction_events = await self._attempt_tool_self_correction(action, tool_result)
                    for event in correction_events:
                        yield event
                    
                    if corrected_action:
                        action = corrected_action
                        continue
                    else:
                        yield self._format_sse({"step": "System Self-Correction Failed", "type": "error", "details": "Unable to find a correction. Aborting retries for this action."})
                        break
                else:
                    yield self._format_sse({"step": "Persistent Failure", "type": "error", "details": f"Tool '{tool_name}' failed after {max_retries} attempts."})
            else:
                if not is_fast_path:
                    yield self._format_sse({"step": "Tool Execution Result", "details": tool_result, "tool_name": tool_name}, "tool_result")
                break 
        
        if not is_fast_path:
            self.action_history.append({"action": action, "result": self.last_tool_output})
            phase_num = phase.get("phase", self.current_phase_index + 1)
            phase_result_key = f"result_of_phase_{phase_num}"
            self.workflow_state.setdefault(phase_result_key, []).append(self.last_tool_output)
            self._add_to_structured_data(self.last_tool_output)

    def _get_required_args_and_enrich(self, relevant_tools: list[str]) -> tuple[dict, list, bool]:
        """Helper to centralize getting required args and enriching them."""
        all_tools = self.dependencies['STATE'].get('mcp_tools', {})
        required_args_for_phase = set()
        for tool_name in relevant_tools:
            tool = all_tools.get(tool_name)
            if not tool: continue
            args_dict = tool.args if isinstance(tool.args, dict) else {}
            for arg_name, arg_details in args_dict.items():
                if arg_details.get('required', False):
                    required_args_for_phase.add(arg_name)
        
        return self._enrich_arguments_from_history(required_args_for_phase)

    async def _get_next_tactical_action(self, current_phase_goal: str, relevant_tools: list[str], enriched_args: dict) -> tuple[dict | str, int, int]:
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

        tactical_system_prompt = WORKFLOW_TACTICAL_PROMPT.format(
            workflow_goal=self.workflow_goal_prompt,
            current_phase_goal=current_phase_goal,
            permitted_tools_with_details=permitted_tools_with_details,
            last_attempt_info=self.last_failed_action_info,
            workflow_history=json.dumps(self.action_history, indent=2),
            all_collected_data=json.dumps(self.workflow_state, indent=2),
            loop_context_section=loop_context_section,
            context_enrichment_section=context_enrichment_section
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
            json_match = re.search(r"```json\s*\n(.*?)\n\s*```|(\{.*\})", response_text, re.DOTALL)
            if not json_match: raise json.JSONDecodeError("No JSON object found", response_text, 0)
            
            json_str = json_match.group(1) or json_match.group(2)
            if not json_str: raise json.JSONDecodeError("Extracted JSON is empty", response_text, 0)

            raw_action = json.loads(json_str.strip())
            
            action_details = raw_action
            tool_name_synonyms = ["tool_name", "name", "tool", "action_name"]
            arg_synonyms = ["arguments", "args", "tool_input", "action_input", "parameters"]
            
            possible_wrapper_keys = ["action", "tool_call", "tool"]
            for key in possible_wrapper_keys:
                if key in action_details and isinstance(action_details[key], dict):
                    action_details = action_details[key]
                    break 

            found_tool_name = None
            for key in tool_name_synonyms:
                if key in action_details:
                    found_tool_name = action_details.pop(key)
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
                "arguments": found_args if isinstance(found_args, dict) else {}
            }

            if not normalized_action.get("tool_name") and len(relevant_tools) == 1:
                normalized_action["tool_name"] = relevant_tools[0]
                self.events_to_yield.append(self._format_sse({
                    "step": "System Correction", "type": "workaround",
                    "correction_type": "inferred_tool_name",
                    "details": f"LLM omitted tool_name. System inferred '{relevant_tools[0]}'."
                }))
            
            if not normalized_action.get("tool_name"):
                 raise ValueError("Could not determine tool_name from LLM response.")

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
        Generates the final summary using a universal, parameterized CoreLLMTask.
        This ensures consistent, high-quality summaries for all workflows.
        """
        final_summary_text = ""
        final_collected_data = self.structured_collected_data

        app_logger.info("Generating final summary using the universal CoreLLMTask.")
        
        # --- MODIFICATION START: New, more precise instructions ---
        standard_task_description = (
            "You are an expert data analyst. Your task is to create a final report for the user based on the provided data."
        )
        
        standard_formatting_instructions = (
            "Your entire response MUST be formatted in standard markdown and MUST be separated into two distinct parts:\n\n"
            "1.  **The Direct Answer:** This MUST be the first part of your response. It must be a single, concise sentence that directly and factually answers the user's original question. It MUST NOT contain any additional context, analysis, or introductory phrases. For example, if the user asked 'How many databases are on the system?', your direct answer MUST be 'There are 21 databases on the system.' and nothing more.\n\n"
            "2.  **Key Observations:** This section MUST start with a level-2 markdown heading (`## Key Observations`). It should contain a bulleted list of all the supporting details, context, and deeper analysis derived from the data."
        )

        if not self.active_prompt_name:
            ad_hoc_rule = (
                "\n\n**CRITICAL RULE (Ad-hoc Queries):** For this report, you MUST NOT use special key-value formats like `***Key:*** Value`. "
                "Adhere strictly to the Direct Answer and Key Observations structure."
            )
            standard_formatting_instructions += ad_hoc_rule
        # --- MODIFICATION END ---

        core_llm_command = {
            "tool_name": "CoreLLMTask",
            "arguments": {
                "task_description": standard_task_description,
                "formatting_instructions": standard_formatting_instructions,
                "user_question": self.original_user_input,
                "source_data": list(self.workflow_state.keys()),
                "data": copy.deepcopy(self.workflow_state)
            }
        }
        
        yield self._format_sse({"step": "Calling LLM to write final report", "details": "Synthesizing a standardized, markdown-formatted summary for the user."})
        
        summary_result, input_tokens, output_tokens = await mcp_adapter.invoke_mcp_tool(self.dependencies['STATE'], core_llm_command)

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
            collected_data=final_collected_data,
            original_user_input=self.original_user_input,
            active_prompt_name=self.active_prompt_name
        )
        final_html = formatter.render()
        
        session_manager.add_to_history(self.session_id, 'assistant', final_html)
        yield self._format_sse({"final_answer": final_html}, "final_answer")
        self.state = self.AgentState.DONE

    def _enrich_arguments_from_history(self, required_args: set, current_args: dict = None, is_prompt: bool = False) -> tuple[dict, list, bool]:
        """
        Scans conversation history to find missing arguments for a tool or prompt call.
        This is a deterministic way to provide context to the LLM.
        Returns the enriched args, any UI events, and a boolean indicating if work was done.
        """
        events_to_yield = []
        initial_args = current_args.copy() if current_args else {}
        enriched_args = initial_args.copy()
        
        arg_type = "prompt" if is_prompt else "tool"
        known_args_for_type = self.dependencies['STATE'].get('all_known_mcp_arguments', {}).get(arg_type, [])
        
        args_to_find = {arg for arg in required_args if (arg not in enriched_args or enriched_args.get(arg) is None) and arg in known_args_for_type}
        if not args_to_find:
            return enriched_args, [], False

        session_data = session_manager.get_session(self.session_id)
        if not session_data:
            return enriched_args, [], False

        for entry in reversed(self.action_history):
            if not args_to_find: break
            
            action_args = entry.get("action", {}).get("arguments", {})
            for arg_name in list(args_to_find):
                if arg_name in action_args and action_args[arg_name] is not None:
                    enriched_args[arg_name] = action_args[arg_name]
                    args_to_find.remove(arg_name)

            result = entry.get("result", {})
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
                    app_logger.info(f"Proactively inferred '{arg_name}' from history: '{value}'")
                    events_to_yield.append(self._format_sse({
                        "step": "System Correction",
                        "details": f"System inferred '{arg_name}: {value}' from conversation history.",
                        "type": "workaround",
                        "correction_type": "inferred_argument"
                    }))

        return enriched_args, events_to_yield, was_enriched

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
        yield self._format_sse({"target": "llm", "state": "busy"}, "status_indicator_update")
        response_text, input_tokens, output_tokens = await self._call_llm_and_update_tokens(
            prompt=recovery_prompt, 
            reason=reason,
            raise_on_error=True
        )
        yield self._format_sse({"target": "llm", "state": "idle"}, "status_indicator_update")
        
        updated_session = session_manager.get_session(self.session_id)
        if updated_session:
            yield self._format_sse({"input_tokens": input_tokens, "output_tokens": output_tokens, "total_input": updated_session.get("input_tokens", 0), "total_output": updated_session.get("output_tokens", 0)}, "token_update")

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

    async def _attempt_tool_self_correction(self, failed_action: dict, error_result: dict) -> tuple[dict | None, list]:
        """
        Attempts to correct a failed tool call, first deterministically, then with an LLM.
        """
        events = []
        tool_name = failed_action.get("tool_name")
        tool_def = self.dependencies['STATE'].get('mcp_tools', {}).get(tool_name)
        if not tool_def:
            return None, events

        required_args = {name for name, details in (tool_def.args.items() if hasattr(tool_def, 'args') and isinstance(tool_def.args, dict) else {}) if details.get('required')}
        
        current_args = failed_action.get("arguments", {})
        
        enriched_args, enrich_events, was_enriched = self._enrich_arguments_from_history(required_args, current_args)
        if was_enriched:
            events.append(self._format_sse({"target": "context", "state": "busy"}, "status_indicator_update"))
            events.append(self._format_sse({"target": "context", "state": "idle"}, "status_indicator_update"))
        events.extend(enrich_events)
        
        if enriched_args != current_args and all(arg in enriched_args for arg in required_args):
            corrected_action = {**failed_action, "arguments": enriched_args}
            events.append(self._format_sse({"step": "System Self-Correction", "type": "workaround", "details": f"Deterministically corrected missing arguments. Retrying tool."}))
            return corrected_action, events

        history_context, _, was_enriched = self._enrich_arguments_from_history(set(self.dependencies['STATE'].get('all_known_mcp_arguments', {}).get('tool', [])))
        if was_enriched:
            events.append(self._format_sse({"target": "context", "state": "busy"}, "status_indicator_update"))
            events.append(self._format_sse({"target": "context", "state": "idle"}, "status_indicator_update"))

        session_data = session_manager.get_session(self.session_id)
        full_history = session_data.get("generic_history", []) if session_data else []

        correction_prompt = TACTICAL_SELF_CORRECTION_PROMPT.format(
            tool_definition=json.dumps(vars(tool_def), default=str),
            failed_command=json.dumps(failed_action),
            error_message=json.dumps(error_result.get('data', 'No error data.')),
            history_context=json.dumps(history_context),
            full_history=json.dumps(full_history)
        )
        
        events.append(self._format_sse({"target": "llm", "state": "busy"}, "status_indicator_update"))
        corrected_args_str, input_tokens, output_tokens = await self._call_llm_and_update_tokens(
            prompt=correction_prompt,
            reason=f"Self-correcting failed tool call for {tool_name}",
            system_prompt_override="You are a JSON-only responding assistant.",
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
        
        try:
            json_match = re.search(r"```json\s*\n(.*?)\n\s*```|(\{.*\})", corrected_args_str, re.DOTALL)
            if not json_match: raise json.JSONDecodeError("No JSON object found", corrected_args_str, 0)
            
            json_str = json_match.group(1) or json_match.group(2)
            if not json_str: raise json.JSONDecodeError("Extracted JSON is empty", corrected_args_str, 0)
            
            corrected_data = json.loads(json_str.strip())
            
            new_args = corrected_data.get("arguments", corrected_data)
            if isinstance(new_args, dict):
                corrected_action = {**failed_action, "arguments": new_args}
                events.append(self._format_sse({"step": "System Self-Correction", "type": "workaround", "details": f"LLM proposed a fix. Retrying tool with new arguments: {json.dumps(new_args)}"}))
                return corrected_action, events
        except (json.JSONDecodeError, TypeError):
            events.append(self._format_sse({"step": "System Self-Correction", "type": "error", "details": "LLM failed to provide a valid JSON correction."}))
            return None, events
            
        return None, events