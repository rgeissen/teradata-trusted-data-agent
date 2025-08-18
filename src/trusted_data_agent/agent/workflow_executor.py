# src/trusted_data_agent/agent/workflow_executor.py
import json
import logging
import re

from trusted_data_agent.agent.prompts import NON_DETERMINISTIC_WORKFLOW_RECOVERY_PROMPT, WORKFLOW_PLANNING_PROMPT
from trusted_data_agent.core import session_manager
from trusted_data_agent.mcp import adapter as mcp_adapter

app_logger = logging.getLogger("quart.app")

# --- NEW: Regex to extract the final output guidelines from the prompt text. ---
FINAL_OUTPUT_GUIDELINES_PATTERN = re.compile(
    r'## Final output guidelines:(.*?)(?:##|\Z)', 
    re.DOTALL | re.IGNORECASE
)

class WorkflowExecutor:
    """
    A dedicated class to handle the execution of multi-step, non-deterministic workflows.
    It is instantiated by the main PlanExecutor when a workflow is triggered.
    """
    def __init__(self, parent_executor):
        self.parent = parent_executor
        self.session_id = self.parent.session_id
        self.dependencies = self.parent.dependencies
        
        self.original_user_input = self.parent.original_user_input
        self.active_prompt_name = self.parent.active_prompt_name
        self.workflow_goal_prompt = self.parent.workflow_goal_prompt
        
        self.workflow_history = getattr(self.parent, 'workflow_history', [])
        self.last_tool_result_str = getattr(self.parent, 'last_workflow_tool_result_str', None)
        self.last_command_in_workflow = getattr(self.parent, 'last_command_in_workflow', None)
        self.plan_of_action = getattr(self.parent, 'plan_of_action', None)
        self.current_step_index = getattr(self.parent, 'current_step_index', 0)
        
        self.next_action_str = None

    def _persist_state_to_parent(self):
        """
        Writes the current workflow state back to the parent PlanExecutor instance.
        """
        self.parent.workflow_history = self.workflow_history
        self.parent.last_workflow_tool_result_str = self.last_tool_result_str
        self.parent.last_command_in_workflow = self.last_command_in_workflow
        self.parent.plan_of_action = self.plan_of_action
        self.parent.current_step_index = self.current_step_index

    async def _generate_plan_of_action(self):
        """
        Generates a deterministic plan of action by making a single, initial LLM call.
        """
        reason = f"Generating a phased plan of action for the '{self.active_prompt_name}' workflow."
        yield self.parent._format_sse({"step": "Calling LLM", "details": reason})

        planning_prompt = WORKFLOW_PLANNING_PROMPT.format(
            workflow_goal=self.workflow_goal_prompt,
            original_user_input=self.original_user_input
        )
        
        response_text, input_tokens, output_tokens = await self.parent._call_llm_and_update_tokens(
            prompt=planning_prompt, 
            reason=reason,
            system_prompt_override="You are a JSON-only planning assistant. Your response must be a single JSON list of tasks."
        )

        updated_session = session_manager.get_session(self.session_id)
        if updated_session:
            yield self.parent._format_sse({ "statement_input": input_tokens, "statement_output": output_tokens, "total_input": updated_session.get("input_tokens", 0), "total_output": updated_session.get("output_tokens", 0) }, "token_update")
        
        try:
            self.plan_of_action = json.loads(response_text)
            if not isinstance(self.plan_of_action, list):
                raise ValueError("LLM response was not a list.")

            # --- NEW: Inject final output guidelines into the last task's description ---
            final_guidelines_match = FINAL_OUTPUT_GUIDELINES_PATTERN.search(self.workflow_goal_prompt)
            final_guidelines_str = final_guidelines_match.group(1).strip() if final_guidelines_match else ""

            if self.plan_of_action and isinstance(self.plan_of_action[-1], dict) and self.plan_of_action[-1].get('tool_name') == 'CoreLLMTask':
                 final_task_desc = self.plan_of_action[-1]['arguments'].get('task_description', 'Synthesize final report for user.')
                 final_task_desc += f" The final response MUST adhere to the following output guidelines:\n{final_guidelines_str}"
                 self.plan_of_action[-1]['arguments']['task_description'] = final_task_desc

            yield self.parent._format_sse({"step": "Plan of Action Generated", "details": self.plan_of_action})
        except (json.JSONDecodeError, ValueError) as e:
            raise RuntimeError(f"Failed to generate a valid plan from the LLM. Response: {response_text}. Error: {e}")

    async def run(self):
        """
        The main execution loop for the deterministic workflow.
        It orchestrates the plan from start to finish.
        """
        if self.plan_of_action is None:
            async for event in self._generate_plan_of_action():
                yield event
            self._persist_state_to_parent()

        while self.current_step_index < len(self.plan_of_action):
            current_step = self.plan_of_action[self.current_step_index]
            
            yield self.parent._format_sse({
                "step": "Executing Plan Step",
                "details": f"Step {self.current_step_index + 1}/{len(self.plan_of_action)}: {current_step.get('task_name', 'Unnamed Task')}",
                "task": current_step
            })

            tool_name = current_step.get("tool_name")
            if tool_name is None:
                raise ValueError(f"Task at index {self.current_step_index} is missing a 'tool_name' key.")

            if tool_name == "CoreLLMTask":
                current_step['arguments']['data'] = self.parent.structured_collected_data
                app_logger.info(f"Populated CoreLLMTask data with collected results for task: {current_step.get('task_name')}")

            try:
                yield self.parent._format_sse({"step": "Tool Execution Intent", "details": current_step}, "tool_result")
                
                status_target = "db"
                yield self.parent._format_sse({"target": status_target, "state": "busy"}, "status_indicator_update")
                
                tool_result = await mcp_adapter.invoke_mcp_tool(self.dependencies['STATE'], current_step)
                
                yield self.parent._format_sse({"target": status_target, "state": "idle"}, "status_indicator_update")

                if isinstance(tool_result, dict) and tool_result.get("status") == "error":
                    error_details = tool_result.get("data", tool_result.get("error", ""))
                    tool_result_str = json.dumps({"tool_input": current_step, "tool_output": {"status": "error", "error_message": error_details}})
                    yield self.parent._format_sse({"details": tool_result, "tool_name": tool_name}, "tool_error")
                    self.parent.state = self.parent.AgentState.ERROR
                    return
                
                # --- NEW: Update parent's last_tool_output with the result of this step. ---
                self.parent.last_tool_output = tool_result
                self.parent._add_to_structured_data(tool_result)

                yield self.parent._format_sse({"step": "Tool Execution Result", "details": tool_result, "tool_name": tool_name}, "tool_result")
            except Exception as e:
                app_logger.error(f"Error during plan execution for step '{tool_name}': {e}", exc_info=True)
                self.parent.state = self.parent.AgentState.ERROR
                return

            self.current_step_index += 1
            self._persist_state_to_parent()
            
            if self.current_step_index >= len(self.plan_of_action):
                app_logger.info("Workflow plan has been fully executed. Transitioning to summarization.")
                self.parent.state = self.parent.AgentState.SUMMARIZING
                break
            
            return

    async def _recover_from_loop(self):
        """
        Attempts to recover from a detected repetitive action loop.
        """
        recovery_prompt = NON_DETERMINISTIC_WORKFLOW_RECOVERY_PROMPT.format(
            original_goal=self.workflow_goal_prompt,
            original_user_input=self.original_user_input,
            last_command=self.last_command_in_workflow,
            workflow_history_str="\n".join([str(item) for item in self.workflow_history]),
            tool_result_str=self.last_tool_result_str or "No tool result available for last command.",
            workflow_goal_and_plan=self.workflow_goal_prompt
        )
        
        reason = "Recovering from repetitive action loop in non-deterministic workflow."
        yield self.parent._format_sse({"step": "Calling LLM for Recovery", "details": reason, "type": "workaround"})
        
        response_text, input_tokens, output_tokens = await self.parent._call_llm_and_update_tokens(
            prompt=recovery_prompt, 
            reason=reason,
            system_prompt_override="You are a tool-use assistant.",
            raise_on_error=True
        )

        updated_session = session_manager.get_session(self.session_id)
        if updated_session:
            yield self.parent._format_sse({ "statement_input": input_tokens, "statement_output": output_tokens, "total_input": updated_session.get("input_tokens", 0), "total_output": updated_session.get("output_tokens", 0) }, "token_update")
            
        self.next_action_str = response_text
        self.last_command_in_workflow = None
