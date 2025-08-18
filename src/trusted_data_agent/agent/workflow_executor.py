# src/trusted_data_agent/agent/workflow_executor.py
import json
import logging
import re

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

    async def _generate_meta_plan(self):
        """
        Generates the strategic, high-level meta-plan that will guide the state machine.
        This is a one-time call at the beginning of the workflow.
        """
        reason = f"Generating a strategic meta-plan for the '{self.active_prompt_name}' workflow."
        yield self.parent._format_sse({"step": "Calling LLM", "details": reason})

        planning_prompt = WORKFLOW_META_PLANNING_PROMPT.format(
            workflow_goal=self.workflow_goal_prompt,
            original_user_input=self.original_user_input
        )
        
        response_text, input_tokens, output_tokens = await self.parent._call_llm_and_update_tokens(
            prompt=planning_prompt, 
            reason=reason,
            system_prompt_override="You are a JSON-only strategic planning assistant."
        )

        updated_session = session_manager.get_session(self.session_id)
        if updated_session:
            yield self.parent._format_sse({ "statement_input": input_tokens, "statement_output": output_tokens, "total_input": updated_session.get("input_tokens", 0), "total_output": updated_session.get("output_tokens", 0) }, "token_update")
        
        try:
            json_str = response_text
            # Handle potential markdown code blocks
            if response_text.strip().startswith("```json"):
                match = re.search(r"```json\s*\n(.*?)\n\s*```", response_text, re.DOTALL)
                if match:
                    json_str = match.group(1).strip()

            self.meta_plan = json.loads(json_str)
            if not isinstance(self.meta_plan, list):
                raise ValueError("LLM response for meta-plan was not a list.")

            yield self.parent._format_sse({"step": "Strategic Meta-Plan Generated", "details": self.meta_plan})
        except (json.JSONDecodeError, ValueError) as e:
            # Raise a specific error that the run method can catch
            raise RuntimeError(f"Failed to generate a valid meta-plan from the LLM. Response: {response_text}. Error: {e}")

    async def _get_next_action(self, current_phase_goal: str) -> dict:
        """
        Makes a tactical LLM call to decide the single next best action for the current phase.
        This is now a standard async function that returns a value.
        """
        tactical_prompt = WORKFLOW_TACTICAL_PROMPT.format(
            workflow_goal=self.workflow_goal_prompt,
            current_phase_goal=current_phase_goal,
            workflow_history=json.dumps(self.action_history, indent=2),
            all_collected_data=json.dumps(self.workflow_state, indent=2)
        )

        response_text, _, _ = await self.parent._call_llm_and_update_tokens(
            prompt=tactical_prompt,
            reason=f"Deciding next tactical action for phase: {current_phase_goal}",
            system_prompt_override="You are a JSON-only tactical assistant."
        )

        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            raise RuntimeError(f"Failed to get a valid JSON action from the tactical LLM. Response: {response_text}")

    async def _is_phase_complete(self, current_phase_goal: str) -> bool:
        """
        Asks the LLM if the current phase's goal has been met.
        This is now a standard async function that returns a value.
        """
        completion_prompt = WORKFLOW_PHASE_COMPLETION_PROMPT.format(
            current_phase_goal=current_phase_goal,
            workflow_history=json.dumps(self.action_history, indent=2),
            all_collected_data=json.dumps(self.workflow_state, indent=2)
        )

        response_text, _, _ = await self.parent._call_llm_and_update_tokens(
            prompt=completion_prompt,
            reason=f"Checking for completion of phase: {current_phase_goal}",
            system_prompt_override="You are a YES/NO validation assistant."
        )
        
        return "yes" in response_text.lower()

    async def run(self):
        """
        The main execution loop for the state machine.
        It orchestrates the execution of the meta-plan, phase by phase.
        """
        # --- MODIFIED: Added self-contained error handling for the planning phase ---
        try:
            if self.meta_plan is None:
                async for event in self._generate_meta_plan():
                    yield event
        except RuntimeError as e:
            app_logger.error(f"Workflow failed during planning: {e}", exc_info=True)
            yield self.parent._format_sse({"error": "Workflow Planning Failed", "details": str(e)}, "error")
            self.parent.state = self.parent.AgentState.ERROR
            return # Gracefully exit the generator

        while self.current_phase_index < len(self.meta_plan):
            current_phase = self.meta_plan[self.current_phase_index]
            phase_goal = current_phase.get("goal", "No goal defined for this phase.")
            phase_num = current_phase.get("phase", self.current_phase_index + 1)

            yield self.parent._format_sse({
                "step": "Starting Workflow Phase",
                "details": f"Phase {phase_num}/{len(self.meta_plan)}: {phase_goal}",
                "phase_details": current_phase
            })

            # This inner loop executes all actions for the current phase
            while True:
                reason = f"Deciding next tactical action for phase: {phase_goal}"
                yield self.parent._format_sse({"step": "Calling LLM", "details": reason})
                next_action = await self._get_next_action(phase_goal)
                
                if not next_action:
                    raise RuntimeError("Tactical LLM failed to provide a next action.")

                tool_name = next_action.get("tool_name")
                if not tool_name:
                    raise ValueError("Tactical LLM response is missing a 'tool_name'.")

                if current_phase.get("type") == "loop":
                    loop_data_key = current_phase.get("loop_over")
                    if loop_data_key and loop_data_key in self.workflow_state:
                         next_action['arguments']['data_from_previous_phase'] = self.workflow_state[loop_data_key]

                try:
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

                except Exception as e:
                    app_logger.error(f"Error during workflow execution for step '{tool_name}': {e}", exc_info=True)
                    self.action_history.append({"action": next_action, "result": "error", "error_message": str(e)})
                    self.parent.state = self.parent.AgentState.ERROR
                    return
                
                reason = f"Checking for completion of phase: {phase_goal}"
                yield self.parent._format_sse({"step": "Calling LLM", "details": reason})
                phase_is_complete = await self._is_phase_complete(phase_goal)
                
                if phase_is_complete:
                    yield self.parent._format_sse({"step": f"Phase {phase_num} Complete", "details": phase_goal})
                    break

            self.current_phase_index += 1

        app_logger.info("Workflow meta-plan has been fully executed. Transitioning to summarization.")
        self.parent.state = self.parent.AgentState.SUMMARIZING
