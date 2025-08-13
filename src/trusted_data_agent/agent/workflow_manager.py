# trusted_data_agent/agent/workflow_manager.py
import re
import logging

app_logger = logging.getLogger("quart.app")

class WorkflowManager:
    """
    Parses a structured workflow prompt into a deterministic execution tree.
    This class is a pure parser and does not manage execution state.
    """
    def __init__(self, prompt_text: str):
        self.execution_tree = self._parse_prompt_to_tree(prompt_text)
        app_logger.info(f"WorkflowManager parsed prompt into a tree with {len(self.execution_tree)} top-level phases.")

    def _find_tool_in_step(self, step_text: str) -> str | None:
        """Finds an explicitly mentioned tool in the step's content."""
        tool_match = re.search(r"using the `([a-zA-Z0-9_]+)` tool", step_text)
        return tool_match.group(1) if tool_match else None

    def _parse_prompt_to_tree(self, prompt_text: str) -> list[dict]:
        """
        Parses the entire prompt into a hierarchical tree of phases and steps.
        """
        lines = prompt_text.strip().split('\n')
        tree = []
        node_stack = []

        for line in lines:
            stripped_line = line.strip()
            if not stripped_line: continue

            indentation = len(line) - len(line.lstrip(' '))
            while len(node_stack) > 1 and indentation < node_stack[-1]['indent']:
                node_stack.pop()

            phase_match = re.match(r"^##\s+(?:Phase|Step)\s+\d+\s*[:\-]\s*(.*)", line)
            if phase_match:
                phase_node = {"type": "phase", "title": phase_match.group(1).strip(), "steps": [], "indent": indentation}
                tree.append(phase_node)
                node_stack = [phase_node]
                continue

            if not node_stack: continue

            loop_match = re.search(r"^(?:Cycle through|For each) the list of (\w+)", stripped_line, re.IGNORECASE)
            if loop_match:
                item_name_plural = loop_match.group(1).lower()
                item_name_singular = re.sub(r's$', '', item_name_plural)
                loop_node = {"type": "loop", "item_name": item_name_singular, "source_list_name": item_name_plural, "steps": [], "indent": indentation}
                node_stack[-1]["steps"].append(loop_node)
                node_stack.append(loop_node)
                continue

            step_match = re.match(r"^\s*-\s*(.*)", line)
            if step_match:
                step_content = step_match.group(1).strip()
                tool_name = self._find_tool_in_step(step_content)
                step_node = {"type": "llm_prompt" if not tool_name else "tool_call", "content": step_content, "tool_name": tool_name, "indent": indentation}
                node_stack[-1]["steps"].append(step_node)
        return tree
