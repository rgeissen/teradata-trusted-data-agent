# src/trusted_data_agent/agent/workflow_executor.py
import json
import logging
import re

from trusted_data_agent.agent.prompts import WORKFLOW_STEP_TEMPLATE, NON_DETERMINISTIC_WORKFLOW_RECOVERY_PROMPT, FINAL_ANSWER_PROMPT
from trusted_data_agent.core import session_manager

app_logger = logging.getLogger("quart.app")

class WorkflowExecutor:
    """
    A dedicated class to handle the execution of multi-step, non-deterministic workflows.
    It is instantiated by the main PlanExecutor when a workflow is triggered.
    """
    def __init__(self, parent_executor):
        self.parent = parent_executor
        self.session_id = self.parent.session_id
        self.dependencies = self.parent.dependencies
        
        # State inherited from the parent executor at the start of the workflow
        self.original_user_input = self.parent.original_user_input
        self.active_prompt_name = self.parent.active_prompt_name
        self.workflow_goal_prompt = self.parent.workflow_goal_prompt
        
        # --- MODIFIED: State is now read from the parent to ensure persistence ---
        # Workflow-specific state is stored on the parent to persist across ephemeral instances of this class.
        self.workflow_history = getattr(self.parent, 'workflow_history', [])
        self.last_tool_result_str = getattr(self.parent, 'last_workflow_tool_result_str', None)
        self.last_command_in_workflow = getattr(self.parent, 'last_command_in_workflow', None)
        
        self.next_action_str = None

    def _persist_state_to_parent(self):
        """
        Writes the current workflow state back to the parent PlanExecutor instance.
        This ensures that if a new WorkflowExecutor is created for the next step,
        it can pick up the state from where the last one left off.
        """
        self.parent.workflow_history = self.workflow_history
        self.parent.last_workflow_tool_result_str = self.last_tool_result_str
        self.parent.last_command_in_workflow = self.last_command_in_workflow

    async def run(self):
        """
        The main execution loop for the non-deterministic workflow.
        This generator yields SSE events back to the parent PlanExecutor.
        """
        # The main run() loop in the parent PlanExecutor will handle the max_steps limit.
        # This method represents a single, complete turn of the workflow.
        
        while self.parent.state != self.parent.AgentState.SUMMARIZING:
            async for event in self._execute_nondeterministic_step():
                yield event
            # This break is important to yield control back to the parent's main loop
            break

    async def _execute_nondeterministic_step(self):
        """
        Manages a single, complete step of a non-deterministic workflow. This method
        is the central controller for Mode 1 workflows, ensuring state is
        correctly managed and passed between steps.
        """
        # --- 1. Get the next action from the LLM based on the current state ---
        if self.next_action_str is None:
            if self.workflow_history:
                history_items = []
                for item in self.workflow_history:
                    if 'tool_name' in item:
                        history_items.append(f"- Executed tool `{item.get('tool_name')}` with arguments `{item.get('arguments', {})}`.")
                    elif 'llm_response' in item:
                         history_items.append(f"- LLM provided non-tool response: `{item.get('llm_response')}`.")
                workflow_history_str = "\n".join(history_items)
            else:
                workflow_history_str = "No actions have been taken yet."
            
            prompt_for_next_step = WORKFLOW_STEP_TEMPLATE.format(
                workflow_goal=self.workflow_goal_prompt,
                original_user_input=self.original_user_input,
                workflow_history_str=workflow_history_str,
                tool_result_str=self.last_tool_result_str or "No tool has been run yet. This is the first step."
            )
            reason = "Deciding next step in non-deterministic workflow."
            yield self.parent._format_sse({"step": "Calling LLM", "details": reason})
            
            response_text, input_tokens, output_tokens = await self.parent._call_llm_and_update_tokens(prompt=prompt_for_next_step, reason=reason)
            
            updated_session = session_manager.get_session(self.session_id)
            if updated_session:
                yield self.parent._format_sse({ "statement_input": input_tokens, "statement_output": output_tokens, "total_input": updated_session.get("input_tokens", 0), "total_output": updated_session.get("output_tokens", 0) }, "token_update")
            
            self.next_action_str = response_text
            if not self.next_action_str:
                raise ValueError("LLM failed to provide a next step for the workflow.")

        # --- 2. Check for repetitive loops before execution ---
        if self.last_command_in_workflow and self.next_action_str == self.last_command_in_workflow:
            app_logger.warning(f"LOOP DETECTED: Non-deterministic workflow is trying to repeat the same command: {self.next_action_str}")
            async for event in self._recover_from_loop(): yield event
            return

        self.last_command_in_workflow = self.next_action_str

        # --- 3. Differentiate between tool calls and LLM-driven steps ---
        is_tool_call = self.next_action_str.strip().startswith('{')
        
        if is_tool_call:
            self.parent.next_action_str = self.next_action_str
            self.parent.state = self.parent.AgentState.DECIDING
            async for event in self.parent._handle_deciding(): yield event
            
            if self.parent.state == self.parent.AgentState.EXECUTING_TOOL:
                async for event in self.parent._execute_tool_with_orchestrators(): yield event
                
                # --- MODIFIED: Use a workflow-specific variable for the tool result ---
                self.last_tool_result_str = self.parent.last_tool_result_str
                if self.parent.last_tool_output and self.parent.last_tool_output.get("status") == "success":
                    self.workflow_history.append(self.parent.current_command)
        
        else: # This is a non-tool LLM-driven step
            yield self.parent._format_sse({"step": "LLM-Driven Step", "details": self.next_action_str}, "llm_thought")
            self.workflow_history.append({"llm_response": self.next_action_str})
            self.last_tool_result_str = f"LLM responded with non-tool step: '{self.next_action_str}'"
            self.parent.state = self.parent.AgentState.DECIDING

        # --- 4. Check for workflow completion ---
        yield self.parent._format_sse({"step": "Checking for completion", "details": "Asking LLM if enough data has been gathered."})
        is_complete, input_tokens, output_tokens = await self._check_for_workflow_completion()
        
        updated_session = session_manager.get_session(self.session_id)
        if updated_session:
             yield self.parent._format_sse({ "statement_input": input_tokens, "statement_output": output_tokens, "total_input": updated_session.get("input_tokens", 0), "total_output": updated_session.get("output_tokens", 0) }, "token_update")

        if is_complete:
            app_logger.info("Workflow completion check returned YES. Moving to summarization.")
            self.parent.state = self.parent.AgentState.SUMMARIZING
        else:
            app_logger.info("Workflow completion check returned NO. Continuing workflow.")
            self.next_action_str = None
            self.parent.state = self.parent.AgentState.DECIDING # Ensure parent state is ready for next loop
        
        # --- MODIFIED: Persist state back to the parent before exiting ---
        self._persist_state_to_parent()

    async def _check_for_workflow_completion(self) -> tuple[bool, int, int]:
        """
        Uses an LLM call to determine if the workflow has gathered enough data.
        Returns a boolean and the token counts for the call.
        """
        all_data_str = self.parent._prepare_data_for_final_summary()
        last_tool_result_str = self.last_tool_result_str or "No tool was run in the last step."

        prompt = FINAL_ANSWER_PROMPT.format(
            original_question=self.original_user_input,
            all_collected_data=all_data_str,
            last_tool_result=last_tool_result_str,
            # --- NEW: Pass the workflow_goal_prompt to the completion check ---
            workflow_goal_and_plan=self.workflow_goal_prompt
        )
        reason = "Checking for workflow completion."
        
        response_text, input_tokens, output_tokens = await self.parent._call_llm_and_update_tokens(
            prompt=prompt,
            reason=reason,
            system_prompt_override="You are an assistant that only responds with 'YES' or 'NO'."
        )
        
        is_complete = "YES" in response_text.upper()
        return is_complete, input_tokens, output_tokens

    async def _recover_from_loop(self):
        """
        Attempts to recover from a detected repetitive action loop.
        """
        recovery_prompt = NON_DETERMINISTIC_WORKFLOW_RECOVERY_PROMPT.format(
            original_goal=self.workflow_goal_prompt,
            original_user_input=self.original_user_input,
            last_command=self.last_command_in_workflow,
            # --- NEW: Pass additional context for better recovery ---
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