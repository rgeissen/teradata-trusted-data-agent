# src/trusted_data_agent/agent/workflow_executor.py
import json
import logging
import re
import copy 

from trusted_data_agent.agent.prompts import (
    WORKFLOW_META_PLANNING_PROMPT,
    WORKFLOW_TACTICAL_PROMPT,
    WORKFLOW_PHASE_COMPLETION_PROMPT
)
from trusted_data_agent.core import session_manager
from trusted_data_agent.mcp import adapter as mcp_adapter

app_logger = logging.getLogger("quart.app")

class WorkflowExecutor:
    """
    A dedicated class to handle the execution of multi-step, stateful workflows.
    It implements a hybrid state machine, guided by a high-level meta-plan.
    """
    def __init__(self, parent_executor):
        self.parent = parent_executor
        self.session_id = self.parent.session_id
        self.dependencies = self.parent.dependencies
        
        self.original_user_input = self.parent.original_user_input
        self.active_prompt_name = self.parent.active_prompt_name
        self.workflow_goal_prompt = self.parent.workflow_goal_prompt
        
        # State machine properties
        self.meta_plan = None
        self.current_phase_index = 0
        self.workflow_state = {} # Stores results keyed by phase, e.g., {"result_of_phase_1": [...]}
        self.action_history = []
        # --- NEW: State for the self-correction loop ---
        self.last_failed_action_info = "None"

    async def _generate_meta_plan(self):
        """
        Generates the strategic, high-level meta-plan that will guide the state machine.
        This is a one-time call at the beginning of the workflow.
        """
        reason = f"Generating a strategic meta-plan for the '{self.active_prompt_name}' workflow."
        yield self.parent._format_sse({"step": "Calling LLM", "details": reason})

        planning_system_prompt = WORKFLOW_META_PLANNING_PROMPT.format(
            workflow_goal=self.workflow_goal_prompt,
            original_user_input=self.original_user_input
        )
        
        response_text, input_tokens, output_tokens = await self.parent._call_llm_and_update_tokens(
            prompt="Generate the meta-plan based on the instructions and context provided in the system prompt.", 
            reason=reason,
            system_prompt_override=planning_system_prompt
        )

        updated_session = session_manager.get_session(self.session_id)
        if updated_session:
            yield self.parent._format_sse({ "statement_input": input_tokens, "statement_output": output_tokens, "total_input": updated_session.get("input_tokens", 0), "total_output": updated_session.get("output_tokens", 0) }, "token_update")
        
        try:
            json_str = response_text
            if response_text.strip().startswith("```json"):
                match = re.search(r"```json\s*\n(.*?)\n\s*```", response_text, re.DOTALL)
                if match:
                    json_str = match.group(1).strip()

            self.meta_plan = json.loads(json_str)
            if not isinstance(self.meta_plan, list):
                raise ValueError("LLM response for meta-plan was not a list.")

            yield self.parent._format_sse({"step": "Strategic Meta-Plan Generated", "details": self.meta_plan})
        except (json.JSONDecodeError, ValueError) as e:
            raise RuntimeError(f"Failed to generate a valid meta-plan from the LLM. Response: {response_text}. Error: {e}")

    # --- MODIFIED: The tactical call now accepts and passes more context for validation and correction. ---
    async def _get_next_action(self, current_phase_goal: str, relevant_tools: list[str]) -> tuple[dict | str, int, int]:
        """
        Makes a tactical LLM call to decide the single next best action for the current phase.
        Can now return a string for early completion signals.
        """
        tactical_system_prompt = WORKFLOW_TACTICAL_PROMPT.format(
            workflow_goal=self.workflow_goal_prompt,
            current_phase_goal=current_phase_goal,
            relevant_tools_for_phase=json.dumps(relevant_tools),
            last_attempt_info=self.last_failed_action_info,
            workflow_history=json.dumps(self.action_history, indent=2),
            all_collected_data=json.dumps(self.workflow_state, indent=2)
        )

        response_text, input_tokens, output_tokens = await self.parent._call_llm_and_update_tokens(
            prompt="Determine the next action based on the instructions and state provided in the system prompt.",
            reason=f"Deciding next tactical action for phase: {current_phase_goal}",
            system_prompt_override=tactical_system_prompt
        )
        
        # --- NEW: Reset the failed action info after the LLM has been prompted with it. ---
        self.last_failed_action_info = "None"

        if "FINAL_ANSWER:" in response_text.upper():
            app_logger.info("Tactical LLM signaled early completion with FINAL_ANSWER.")
            return "SYSTEM_ACTION_COMPLETE", input_tokens, output_tokens

        try:
            json_match = re.search(r"```json\s*\n(.*?)\n\s*```|(\{.*\})", response_text, re.DOTALL)
            if not json_match:
                raise json.JSONDecodeError("No JSON object found in the response", response_text, 0)
            
            json_str = json_match.group(1) or json_match.group(2)
            if not json_str:
                 raise json.JSONDecodeError("Extracted JSON string is empty", response_text, 0)

            action = json.loads(json_str.strip())
            
            if "tool" in action and "tool_name" not in action:
                action["tool_name"] = action.pop("tool")
            if "action" in action and "tool_name" not in action:
                action["tool_name"] = action.pop("action")
            if "tool_input" in action and "arguments" not in action:
                action["arguments"] = action.pop("tool_input")
            if "action_input" in action and "arguments" not in action:
                action["arguments"] = action.pop("action_input")
            if "tool_arguments" in action and "arguments" not in action:
                action["arguments"] = action.pop("tool_arguments")
            if "parameters" in action and "arguments" not in action:
                action["arguments"] = action.pop("parameters")

            return action, input_tokens, output_tokens
        except json.JSONDecodeError:
            raise RuntimeError(f"Failed to get a valid JSON action from the tactical LLM. Response: {response_text}")

    async def _is_phase_complete(self, current_phase_goal: str) -> tuple[bool, int, int]:
        """
        Asks the LLM if the current phase's goal has been met.
        """
        completion_system_prompt = WORKFLOW_PHASE_COMPLETION_PROMPT.format(
            current_phase_goal=current_phase_goal,
            workflow_history=json.dumps(self.action_history, indent=2),
            all_collected_data=json.dumps(self.workflow_state, indent=2)
        )

        response_text, input_tokens, output_tokens = await self.parent._call_llm_and_update_tokens(
            prompt="Is the phase complete based on the system prompt? Respond with only YES or NO.",
            reason=f"Checking for completion of phase: {current_phase_goal}",
            system_prompt_override=completion_system_prompt
        )
        
        is_complete = "yes" in response_text.lower()
        return is_complete, input_tokens, output_tokens

    async def run(self):
        """
        The main execution loop for the state machine.
        It orchestrates the execution of the meta-plan, phase by phase.
        """
        try:
            if self.meta_plan is None:
                async for event in self._generate_meta_plan():
                    yield event

            while self.current_phase_index < len(self.meta_plan):
                current_phase = self.meta_plan[self.current_phase_index]
                phase_goal = current_phase.get("goal", "No goal defined for this phase.")
                phase_num = current_phase.get("phase", self.current_phase_index + 1)
                # --- NEW: Get the list of allowed tools for this phase ---
                relevant_tools = current_phase.get("relevant_tools", [])

                yield self.parent._format_sse({
                    "step": "Starting Workflow Phase",
                    "details": f"Phase {phase_num}/{len(self.meta_plan)}: {phase_goal}",
                    "phase_details": current_phase
                })

                phase_attempts = 0
                max_phase_attempts = 5 # Increased attempts for self-correction

                while True:
                    phase_attempts += 1
                    if phase_attempts > max_phase_attempts:
                        raise RuntimeError(f"Phase '{phase_goal}' failed to complete after {max_phase_attempts} attempts.")

                    reason_action = f"Deciding next tactical action for phase: {phase_goal}"
                    yield self.parent._format_sse({"step": "Calling LLM", "details": reason_action})
                    
                    # --- MODIFIED: Pass the relevant_tools to the tactical LLM call ---
                    next_action, input_tokens, output_tokens = await self._get_next_action(phase_goal, relevant_tools)
                    
                    updated_session = session_manager.get_session(self.session_id)
                    if updated_session:
                        yield self.parent._format_sse({ "statement_input": input_tokens, "statement_output": output_tokens, "total_input": updated_session.get("input_tokens", 0), "total_output": updated_session.get("output_tokens", 0) }, "token_update")

                    if isinstance(next_action, str) and next_action == "SYSTEM_ACTION_COMPLETE":
                        app_logger.info("Workflow signaled early completion. Transitioning to summarization.")
                        self.parent.state = self.parent.AgentState.SUMMARIZING
                        return

                    if not isinstance(next_action, dict):
                        raise RuntimeError(f"Tactical LLM failed to provide a valid action. Received: {next_action}")

                    tool_name = next_action.get("tool_name")
                    if not tool_name:
                        raise ValueError("Tactical LLM response is missing a 'tool_name'.")

                    # --- NEW: Validation Gate and Self-Correction Loop ---
                    if relevant_tools and tool_name not in relevant_tools:
                        app_logger.warning(f"LLM proposed an invalid tool '{tool_name}' for the current phase. Expected one of: {relevant_tools}. Initiating self-correction.")
                        self.last_failed_action_info = f"Your last attempt to use the tool '{tool_name}' was invalid because it is not in the list of permitted tools for this phase."
                        yield self.parent._format_sse({
                            "step": "System Correction",
                            "details": f"LLM chose an invalid tool ('{tool_name}'). Retrying with constraints.",
                            "type": "workaround"
                        })
                        continue # Restart the loop to get a new action

                    if current_phase.get("type") == "loop":
                        loop_data_key = current_phase.get("loop_over")
                        if loop_data_key and loop_data_key in self.workflow_state:
                             next_action['arguments']['data_from_previous_phase'] = self.workflow_state[loop_data_key]
                    
                    if tool_name == "CoreLLMTask":
                        if "arguments" not in next_action: next_action["arguments"] = {}
                        next_action["arguments"]["data"] = copy.deepcopy(self.workflow_state)

                    yield self.parent._format_sse({"step": "Tool Execution Intent", "details": next_action}, "tool_result")
                    
                    status_target = "db"
                    yield self.parent._format_sse({"target": status_target, "state": "busy"}, "status_indicator_update")
                    
                    tool_result = await mcp_adapter.invoke_mcp_tool(self.dependencies['STATE'], next_action)
                    
                    yield self.parent._format_sse({"target": status_target, "state": "idle"}, "status_indicator_update")

                    self.action_history.append({"action": next_action, "result": "success"})
                    phase_result_key = f"result_of_phase_{phase_num}"
                    if phase_result_key not in self.workflow_state:
                        self.workflow_state[phase_result_key] = []
                    self.workflow_state[phase_result_key].append(tool_result)
                    
                    self.parent.last_tool_output = tool_result
                    self.parent._add_to_structured_data(tool_result)

                    yield self.parent._format_sse({"step": "Tool Execution Result", "details": tool_result, "tool_name": tool_name}, "tool_result")

                    reason_check = f"Checking for completion of phase: {phase_goal}"
                    yield self.parent._format_sse({"step": "Calling LLM", "details": reason_check})
                    phase_is_complete, input_tokens, output_tokens = await self._is_phase_complete(phase_goal)
                    
                    updated_session = session_manager.get_session(self.session_id)
                    if updated_session:
                        yield self.parent._format_sse({ "statement_input": input_tokens, "statement_output": output_tokens, "total_input": updated_session.get("input_tokens", 0), "total_output": updated_session.get("output_tokens", 0) }, "token_update")

                    if phase_is_complete:
                        yield self.parent._format_sse({"step": f"Phase {phase_num} Complete", "details": "Goal has been achieved."})
                        break 

                self.current_phase_index += 1

            app_logger.info("Workflow meta-plan has been fully executed. Transitioning to summarization.")
            self.parent.state = self.parent.AgentState.SUMMARIZING
        
        except Exception as e:
            app_logger.error(f"Workflow failed during execution: {e}", exc_info=True)
            yield self.parent._format_sse({"error": "Workflow Execution Failed", "details": str(e)}, "error")
            self.parent.state = self.parent.AgentState.ERROR
            return
