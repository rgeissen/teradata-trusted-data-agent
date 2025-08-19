# trusted_data_agent/mcp/adapter.py
import json
import logging
import re
from datetime import datetime

from langchain_mcp_adapters.tools import load_mcp_tools
from trusted_data_agent.llm import handler as llm_handler

app_logger = logging.getLogger("quart.app")

VIZ_TOOL_DEFINITION = {
    "name": "viz_createChart",
    "description": "Generates a data visualization based on provided data. You must specify the chart type and map the data fields to the appropriate visual roles.",
    "args": {
        "chart_type": {
            "type": "string",
            "description": "The type of chart to generate (e.g., 'bar', 'pie', 'line', 'scatter'). This MUST be one of the types listed in the 'Charting Guidelines'.",
            "required": True
        },
        "data": {
            "type": "list[dict]",
            "description": "The data to be visualized, passed directly from the output of another tool.",
            "required": True
        },
        "title": {
            "type": "string",
            "description": "A descriptive title for the chart.",
            "required": True
        },
        "mapping": {
            "type": "dict",
            "description": "A dictionary that maps data keys to chart axes or roles (e.g., {'x_axis': 'product_name', 'y_axis': 'sales_total'}). The required keys for this mapping depend on the selected chart_type.",
            "required": True
        }
    }
}

UTIL_TOOL_DEFINITIONS = [
    {
        "name": "util_getCurrentDate",
        "description": "Returns the current system date in YYYY-MM-DD format. Use this as the first step for any user query involving relative dates like 'today', 'yesterday', or 'this week'.",
        "args": {}
    }
]

# --- MODIFIED: Added source_data argument for context scoping. ---
CORE_LLM_TASK_DEFINITION = {
    "name": "CoreLLMTask",
    "description": "Performs internal, LLM-driven tasks that are not direct calls to the Teradata database. This tool is used for text synthesis, summarization, and formatting based on a specific 'task_description' provided by the LLM itself.",
    "args": {
        "task_description": {
            "type": "string",
            "description": "A natural language description of the internal task to be executed (e.g., 'describe the table in a business context', 'format final output'). The LLM infers this from the workflow plan.",
            "required": True
        },
        "source_data": {
            "type": "list[string]",
            "description": "A list of keys (e.g., 'result_of_phase_1') identifying which data from the workflow history is relevant for this task. This is critical for providing the correct context.",
            "required": True
        }
    }
}


def _extract_and_clean_description(description: str | None) -> tuple[str, str]:
    """
    Parses a description string to find a datatype hint (e.g., "(type: str)")
    and cleans the description, returning both the cleaned description and the type.
    """
    if not isinstance(description, str):
        return "", "unknown"

    datatype = "unknown"
    match = re.search(r'\s*\((type:\s*(str|int|float|bool))\)', description, re.IGNORECASE)
    
    if match:
        datatype = match.group(2).lower()
        cleaned_description = description.replace(match.group(0), "").strip()
    else:
        cleaned_description = description
        
    return cleaned_description, datatype


async def load_and_categorize_teradata_resources(STATE: dict):
    mcp_client = STATE.get('mcp_client')
    llm_instance = STATE.get('llm')
    if not mcp_client or not llm_instance:
        raise Exception("MCP or LLM client not initialized.")

    async with mcp_client.session("teradata_mcp_server") as temp_session:
        app_logger.info("--- Loading and classifying Teradata tools and prompts... ---")

        loaded_tools = await load_mcp_tools(temp_session)
        loaded_prompts = []
        try:
            list_prompts_result = await temp_session.list_prompts()
            if hasattr(list_prompts_result, 'prompts'):
                loaded_prompts = list_prompts_result.prompts
        except Exception as e:
            app_logger.error(f"CRITICAL ERROR while loading prompts: {e}", exc_info=True)

        class SimpleTool:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
        
        viz_tool_obj = SimpleTool(**VIZ_TOOL_DEFINITION)
        loaded_tools.append(viz_tool_obj)
        for util_tool_def in UTIL_TOOL_DEFINITIONS:
            loaded_tools.append(SimpleTool(**util_tool_def))
        loaded_tools.append(SimpleTool(**CORE_LLM_TASK_DEFINITION))


        STATE['mcp_tools'] = {tool.name: tool for tool in loaded_tools}
        if loaded_prompts:
            STATE['mcp_prompts'] = {prompt.name: prompt for prompt in loaded_prompts}

        all_capabilities = []
        all_capabilities.extend([f"- {tool.name} (tool): {tool.description}" for tool in loaded_tools])
        
        for p in loaded_prompts:
            prompt_str = f"- {p.name} (prompt): {p.description or 'No description available.'}"
            if hasattr(p, 'arguments') and p.arguments:
                prompt_str += "\n  - Arguments:"
                for arg in p.arguments:
                    arg_dict = arg.model_dump()
                    arg_name = arg_dict.get('name', 'unknown_arg')
                    prompt_str += f"\n    - `{arg_name}`"
            all_capabilities.append(prompt_str)

        capabilities_list_str = "\n".join(all_capabilities)

        classification_prompt = (
            "You are a helpful assistant that analyzes a list of technical capabilities (tools and prompts) for a Teradata database system and classifies them. "
            "For each capability, you must determine a single user-friendly 'category' for a UI. "
            "Example categories might be 'Data Quality', 'Table Management', 'Performance', 'Utilities', 'Database Information', etc. Be concise and consistent.\n\n"
            "Your response MUST be a single, valid JSON object. The keys of this object must be the capability names, "
            "and the value for each key must be another JSON object containing only the 'category' you determined.\n\n"
            "Example format:\n"
            "{\n"
            '  "capability_name_1": {"category": "Some Category"},\n'
            '  "capability_name_2": {"category": "Another Category"}\n'
            "}\n\n"
            f"--- Capability List ---\n{capabilities_list_str}"
        )
        categorization_system_prompt = "You are an expert assistant that only responds with valid JSON."
        
        classified_capabilities_str, _, _ = await llm_handler.call_llm_api(
            llm_instance, classification_prompt, raise_on_error=True,
            system_prompt_override=categorization_system_prompt
        )
        
        match = re.search(r'\{.*\}', classified_capabilities_str, re.DOTALL)
        if match is None:
            raise ValueError(f"LLM failed to return a valid JSON for capability classification. Response: '{classified_capabilities_str}'")
        
        cleaned_str = match.group(0)
        classified_data = json.loads(cleaned_str)

        STATE['structured_tools'] = {}
        disabled_tools_list = STATE.get("disabled_tools", [])
        
        for tool in loaded_tools:
            classification = classified_data.get(tool.name, {})
            category = classification.get("category", "Uncategorized")

            if category not in STATE['structured_tools']:
                STATE['structured_tools'][category] = []

            is_disabled = tool.name in disabled_tools_list
            STATE['structured_tools'][category].append({
                "name": tool.name, "description": tool.description, "disabled": is_disabled
            })

        tool_context_parts = ["--- Available Tools ---"]
        for category, tools in sorted(STATE['structured_tools'].items()):
            enabled_tools_in_category = [t for t in tools if not t['disabled']]
            if enabled_tools_in_category:
                tool_context_parts.append(f"--- Category: {category} ---")
                for tool_info in enabled_tools_in_category:
                    tool_obj = STATE['mcp_tools'][tool_info['name']]
                    tool_str = f"- `{tool_obj.name}` (tool): {tool_obj.description}"
                    args_dict = tool_obj.args if isinstance(tool_obj.args, dict) else {}

                    if args_dict and "Arguments:" not in tool_obj.description:
                        tool_str += "\n  - Arguments:"
                        for arg_name, arg_details in args_dict.items():
                            arg_type = arg_details.get('type', 'any')
                            is_required = arg_details.get('required', False)
                            req_str = "required" if is_required else "optional"
                            arg_desc = arg_details.get('description', 'No description.')
                            tool_str += f"\n    - `{arg_name}` ({arg_type}, {req_str}): {arg_desc}"
                    tool_context_parts.append(tool_str)
        
        STATE['tools_context'] = "\n".join(tool_context_parts)

        STATE['structured_prompts'] = {}
        disabled_prompts_list = STATE.get("disabled_prompts", [])
        
        if loaded_prompts:
            for prompt_obj in loaded_prompts:
                classification = classified_data.get(prompt_obj.name, {})
                category = classification.get("category", "Uncategorized")
                
                if category not in STATE['structured_prompts']:
                    STATE['structured_prompts'][category] = []

                is_disabled = prompt_obj.name in disabled_prompts_list
                
                processed_args = []
                if hasattr(prompt_obj, 'arguments') and prompt_obj.arguments:
                    for arg in prompt_obj.arguments:
                        arg_dict = arg.model_dump()
                        cleaned_desc, arg_type = _extract_and_clean_description(arg_dict.get("description"))
                        arg_dict['description'] = cleaned_desc; arg_dict['type'] = arg_type
                        processed_args.append(arg_dict)
                
                STATE['structured_prompts'][category].append({
                    "name": prompt_obj.name,
                    "description": prompt_obj.description or "No description available.",
                    "arguments": processed_args,
                    "disabled": is_disabled
                })

        prompt_context_parts = ["--- Available Prompts ---"]
        for category, prompts in sorted(STATE['structured_prompts'].items()):
            enabled_prompts_in_category = [p for p in prompts if not p['disabled']]
            if enabled_prompts_in_category:
                prompt_context_parts.append(f"--- Category: {category} ---")
                for prompt_info in enabled_prompts_in_category:
                    prompt_description = prompt_info.get("description", "No description available.")
                    prompt_str = f"- `{prompt_info['name']}` (prompt): {prompt_description}"
                    
                    processed_args = prompt_info.get('arguments', [])
                    if processed_args:
                        prompt_str += "\n  - Arguments:"
                        for arg_details in processed_args:
                            arg_name = arg_details.get('name', 'unknown')
                            arg_type = arg_details.get('type', 'any')
                            is_required = arg_details.get('required', False)
                            req_str = "required" if is_required else "optional"
                            arg_desc = arg_details.get('description', 'No description.')
                            prompt_str += f"\n    - `{arg_name}` ({arg_type}, {req_str}): {arg_desc}"
                    prompt_context_parts.append(prompt_str)

        if len(prompt_context_parts) > 1:
            STATE['prompts_context'] = "\n".join(prompt_context_parts)
        else:
            STATE['prompts_context'] = "--- No Prompts Available ---"


def _transform_chart_data(data: any) -> list[dict]:
    if isinstance(data, dict) and 'labels' in data and 'values' in data:
        app_logger.warning("Correcting hallucinated chart data format from labels/values to list of dicts.")
        labels = data.get('labels', [])
        values = data.get('values', [])
        if isinstance(labels, list) and isinstance(values, list) and len(labels) == len(values):
            return [{"label": l, "value": v} for l, v in zip(labels, values)]
    if isinstance(data, dict) and 'columns' in data and 'rows' in data:
        app_logger.warning("Correcting hallucinated chart data format from columns/rows to list of dicts.")
        if isinstance(data.get('rows'), list):
            return data['rows']
    
    if isinstance(data, list) and data and isinstance(data[0], dict):
        if "ColumnName" in data[0] and "DistinctValue" in data[0] and "DistinctValueCount" in data[0]:
            app_logger.info("Detected qlty_distinctCategories output pattern. Renaming 'ColumnName' to 'SourceColumnName'.")
            transformed_data = []
            for row in data:
                new_row = row.copy()
                if "ColumnName" in new_row:
                    new_row["SourceColumnName"] = new_row.pop("ColumnName")
                transformed_data.append(new_row)
            return transformed_data

    return data

def _build_g2plot_spec(args: dict, data: list[dict]) -> dict:
    chart_type = args.get("chart_type", "").lower()
    mapping = args.get("mapping", {})
    
    canonical_map = {
        'x_axis': 'xField', 
        'y_axis': 'yField', 
        'color': 'seriesField',
        'angle': 'angleField',
        'category': 'xField', 
        'value': 'yField'      
    }

    reverse_canonical_map = {
        alias.lower(): canonical for canonical, aliases in canonical_map.items() 
        for alias in [canonical] + [key for key in aliases]
    }
    
    options = {"title": {"text": args.get("title", "Generated Chart")}}
    
    first_row_keys_lower = {k.lower(): k for k in data[0].keys()} if data and data[0] else {}
    
    processed_mapping = {}
    for llm_key, data_col_name in mapping.items():
        canonical_key = reverse_canonical_map.get(llm_key.lower())
        if canonical_key:
            actual_col_name = first_row_keys_lower.get(data_col_name.lower())
            if not actual_col_name:
                raise KeyError(f"The mapped column '{data_col_name}' (from '{llm_key}') was not found in the provided data.")
            processed_mapping[canonical_map[canonical_key]] = actual_col_name
        else:
            app_logger.warning(f"Unknown mapping key from LLM: '{llm_key}'. Skipping.")

    options.update(processed_mapping)

    if chart_type == 'pie' and 'seriesField' in options:
        options['colorField'] = options.pop('seriesField')

    final_data = []
    if data:
        for row in data:
            new_row = row.copy()
            for g2plot_key, actual_col_name in options.items():
                if g2plot_key in ['yField', 'angleField', 'size']:
                    cell_value = new_row.get(actual_col_name)
                    if cell_value is not None:
                        try:
                            new_row[actual_col_name] = float(cell_value)
                        except (ValueError, TypeError):
                            app_logger.warning(f"Non-numeric value '{cell_value}' encountered for numeric field '{actual_col_name}'. Conversion failed.")
            final_data.append(new_row)
    
    options["data"] = final_data
    
    g2plot_type_map = {
        "bar": "Column", "column": "Column", "line": "Line", "area": "Area",
        "pie": "Pie", "scatter": "Scatter", "histogram": "Histogram", 
        "heatmap": "Heatmap", "boxplot": "Box", "wordcloud": "WordCloud"
    }
    g2plot_type = g2plot_type_map.get(chart_type, chart_type.capitalize())

    return {"type": g2plot_type, "options": options}

# --- MODIFIED: Added a CRITICAL RULE to the internal prompt to enforce strict formatting adherence. ---
async def _invoke_core_llm_task(STATE: dict, command: dict) -> dict:
    """
    Executes a task handled by the LLM itself, based on a generic task_description
    and a focused subset of the workflow's collected data.
    """
    args = command.get("arguments", {})
    task_description = args.get("task_description")
    source_data_keys = args.get("source_data", [])
    
    # This is the full history of all data collected in the workflow so far.
    full_workflow_state = args.get("data", {}) 
    
    app_logger.info(f"Executing client-side LLM task: {task_description}")

    # Build a focused data payload containing only the data specified by the tactical LLM.
    focused_data_for_task = {}
    if isinstance(full_workflow_state, dict):
        for key in source_data_keys:
            if key in full_workflow_state:
                focused_data_for_task[key] = full_workflow_state[key]
    
    if not focused_data_for_task:
        app_logger.warning(f"CoreLLMTask was called for '{task_description}' but no source data was found for keys: {source_data_keys}. Passing all data as a fallback.")
        focused_data_for_task = full_workflow_state

    final_prompt = (
        "You are a highly capable text processing and synthesis assistant. Your task is to perform the following operation based on the provided data context.\n\n"
        "--- TASK ---\n"
        f"{task_description}\n\n"
        "--- RELEVANT DATA (Selected from Previous Phases) ---\n"
        f"{json.dumps(focused_data_for_task, indent=2)}\n\n"
        "--- CRITICAL RULE ---\n"
        "You MUST adhere to any and all formatting instructions contained in the 'TASK' description with absolute precision. Do not deviate, simplify, or change the requested format in any way.\n\n"
        "Your response should be the direct result of the task. Do not add any conversational text or extra formatting unless explicitly requested by the task description."
    )

    response_text, _, _ = await llm_handler.call_llm_api(
        llm_instance=STATE.get('llm'),
        prompt=final_prompt,
        reason=f"Executing CoreLLMTask: {task_description}",
        system_prompt_override="You are a text processing and synthesis assistant.",
        raise_on_error=True
    )

    return {"status": "success", "results": [{"response": response_text}]}

async def invoke_mcp_tool(STATE: dict, command: dict) -> any:
    mcp_client = STATE.get('mcp_client')
    tool_name = command.get("tool_name")
    
    if tool_name == "CoreLLMTask":
        return await _invoke_core_llm_task(STATE, command)

    if tool_name == "util_getCurrentDate":
        app_logger.info("Executing client-side tool: util_getCurrentDate")
        current_date = datetime.now().strftime('%Y-%m-%d')
        return {
            "status": "success",
            "metadata": {"tool_name": "util_getCurrentDate"},
            "results": [{"current_date": current_date}]
        }

    if tool_name == "viz_createChart":
        app_logger.info(f"Handling abstract chart generation for: {command}")
        
        try:
            args = command.get("arguments", {})
            data = args.get("data")
            data = _transform_chart_data(data) 
            
            if not isinstance(data, list) or not data:
                return {"error": "Validation failed", "data": "The 'data' argument must be a non-empty list of dictionaries."}
            
            chart_spec = _build_g2plot_spec(args, data)
            
            return {"type": "chart", "spec": chart_spec, "metadata": {"tool_name": "viz_createChart"}}
        except Exception as e:
            app_logger.error(f"Error building G2Plot spec: {e}", exc_info=True)
            return {"error": "Chart Generation Failed", "data": str(e)}

    args = command.get("arguments") or command.get("parameters") or command.get("tool_input") or command.get("action_input") or command.get("tool_arguments") or {}
    
    app_logger.debug(f"Invoking tool '{tool_name}' with args: {args}")
    try:
        async with mcp_client.session("teradata_mcp_server") as temp_session:
            call_tool_result = await temp_session.call_tool(tool_name, args)
    except Exception as e:
        app_logger.error(f"Error during tool invocation for '{tool_name}': {e}", exc_info=True)
        return {"status": "error", "error": f"An exception occurred while invoking tool '{tool_name}'.", "data": str(e)}
    
    if hasattr(call_tool_result, 'content') and isinstance(call_tool_result.content, list) and len(call_tool_result.content) > 0:
        text_content = call_tool_result.content[0]
        if hasattr(text_content, 'text') and isinstance(text_content.text, str):
            try:
                return json.loads(text_content.text)
            except json.JSONDecodeError:
                app_logger.warning(f"Tool '{tool_name}' returned a non-JSON string: '{text_content.text}'")
                return {"status": "error", "error": "Tool returned non-JSON string", "data": text_content.text}
    
    raise RuntimeError(f"Unexpected tool result format for '{tool_name}': {call_tool_result}")
