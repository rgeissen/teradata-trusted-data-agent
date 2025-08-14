# trusted_data_agent/mcp/adapter.py
import json
import logging
import re
from datetime import datetime

from langchain_mcp_adapters.tools import load_mcp_tools
from trusted_data_agent.llm import handler as llm_handler

app_logger = logging.getLogger("quart.app")

PARAMETER_ALIASES = {
    "database_name": ["db_name", "database"],
    "table_name": ["tbl_name", "object_name", "obj_name"],
    "column_name": ["col_name", "column"]
}

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


async def load_and_categorize_teradata_resources(STATE: dict):
    mcp_client = STATE.get('mcp_client')
    llm_instance = STATE.get('llm')
    if not mcp_client or not llm_instance:
        raise Exception("MCP or LLM client not initialized.")

    async with mcp_client.session("teradata_mcp_server") as temp_session:
        app_logger.info("--- Loading Teradata tools and prompts... ---")
        loaded_tools = await load_mcp_tools(temp_session)

        class SimpleTool:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
        
        viz_tool_obj = SimpleTool(**VIZ_TOOL_DEFINITION)
        loaded_tools.append(viz_tool_obj)
        
        for util_tool_def in UTIL_TOOL_DEFINITIONS:
            loaded_tools.append(SimpleTool(**util_tool_def))

        STATE['mcp_tools'] = {tool.name: tool for tool in loaded_tools}
        STATE['tool_scopes'] = classify_tool_scopes(loaded_tools)
        
        disabled_tools_list = STATE.get("disabled_tools", [])
        enabled_tools = [t for t in loaded_tools if t.name not in disabled_tools_list]

        tool_details_list = []
        for tool in enabled_tools:
            tool_str = f"- `{tool.name}`: {tool.description}"
            args_dict = tool.args if isinstance(tool.args, dict) else {}
            if args_dict:
                tool_str += "\n  - Arguments:"
                for arg_name, arg_details in args_dict.items():
                    arg_type = arg_details.get('type', 'any')
                    is_required = arg_details.get('required', False)
                    req_str = "required" if is_required else "optional"
                    arg_desc = arg_details.get('description', 'No description.')
                    tool_str += f"\n    - `{arg_name}` ({arg_type}, {req_str}): {arg_desc}"
            tool_details_list.append(tool_str)
        
        STATE['tools_context'] = "--- Available Tools ---\n" + "\n".join(tool_details_list)
        
        tool_list_for_prompt = "\n".join([f"- {tool.name}: {tool.description}" for tool in loaded_tools])
        categorization_prompt = (
            "You are a helpful assistant that organizes lists of technical tools for a **Teradata database system** into logical categories for a user interface. "
            "Your response MUST be a single, valid JSON object. The keys should be the category names, "
            f"and the values should be an array of tool names belonging to that category.\n\n"
            f"--- Tool List ---\n{tool_list_for_prompt}"
        )
        categorization_system_prompt = "You are a helpful assistant that organizes lists into JSON format."
        categorized_tools_str, _, _ = await llm_handler.call_llm_api(
            llm_instance, categorization_prompt, raise_on_error=True,
            system_prompt_override=categorization_system_prompt
        )
        
        match = re.search(r'\{.*\}', categorized_tools_str, re.DOTALL)
        if match is None:
            raise ValueError(f"LLM failed to return a valid JSON for tool categorization. Response: '{categorized_tools_str}'")
        cleaned_str = match.group(0)
        categorized_tools = json.loads(cleaned_str)

        STATE['structured_tools'] = {}
        for category, tool_names in categorized_tools.items():
            tool_list = []
            for name in tool_names:
                if name in STATE['mcp_tools']:
                    tool_obj = STATE['mcp_tools'][name]
                    is_disabled = name in disabled_tools_list
                    tool_list.append({
                        "name": tool_obj.name,
                        "description": tool_obj.description,
                        "disabled": is_disabled
                    })
            STATE['structured_tools'][category] = tool_list

        loaded_prompts = []
        try:
            list_prompts_result = await temp_session.list_prompts()
            if hasattr(list_prompts_result, 'prompts'):
                loaded_prompts = list_prompts_result.prompts
        except Exception as e:
            app_logger.error(f"CRITICAL ERROR while loading prompts: {e}", exc_info=True)

        if loaded_prompts:
            STATE['mcp_prompts'] = {prompt.name: prompt for prompt in loaded_prompts}
            
            disabled_prompts_list = STATE.get("disabled_prompts", [])
            enabled_prompts = [
                p for p in loaded_prompts 
                if p.name not in disabled_prompts_list
            ]
            
            if enabled_prompts:
                STATE['prompts_context'] = "--- Available Prompts ---\n" + "\n".join([f"- `{p.name}`: {p.description or 'No description available.'}" for p in enabled_prompts])
            else:
                STATE['prompts_context'] = "--- No Prompts Available ---"

            prompt_list_for_prompt = "\n".join([f"- {p.name}: {p.description or 'No description available.'}" for p in loaded_prompts])
            
            categorization_prompt_for_prompts = (
                "You are a JSON formatting expert. Your task is to categorize the following list of Teradata system prompts into a single JSON object."
                "\n\n**CRITICAL RULES:**"
                "\n1. Your entire response MUST be a single, raw JSON object."
                "\n2. DO NOT include ```{json} markdown wrappers, conversational text, or any explanations."
                "\n3. The JSON keys MUST be the category names."
                "\n4. The JSON values MUST be an array of the prompt names."
                f"\n\n--- Prompt List to Categorize ---\n{prompt_list_for_prompt}"
            )

            categorized_prompts_str, _, _ = await llm_handler.call_llm_api(
                llm_instance, categorization_prompt_for_prompts, raise_on_error=True,
                system_prompt_override=categorization_system_prompt
            )
            
            match_prompts = re.search(r'\{.*\}', categorized_prompts_str, re.DOTALL)
            if match_prompts is None:
                raise ValueError(f"LLM failed to return valid JSON for prompt categorization. Response: '{categorized_prompts_str}'")
            cleaned_str_prompts = match_prompts.group(0)
            categorized_prompts = json.loads(cleaned_str_prompts)
            
            STATE['structured_prompts'] = {}
            for category, prompt_names in categorized_prompts.items():
                prompt_list = []
                for raw_name_from_llm in prompt_names:
                    name = raw_name_from_llm.split(':', 1)[0].strip()
                    is_found = name in STATE['mcp_prompts']
                    
                    app_logger.debug(f"Categorization Check: Raw='{raw_name_from_llm}', Parsed='{name}', Found? -> {is_found}")

                    if is_found:
                        prompt_obj = STATE['mcp_prompts'][name]
                        is_disabled = name in disabled_prompts_list
                        prompt_list.append({
                            "name": prompt_obj.name,
                            "description": prompt_obj.description or "No description available.",
                            "arguments": [arg.model_dump() for arg in prompt_obj.arguments],
                            "disabled": is_disabled
                        })
                STATE['structured_prompts'][category] = prompt_list
        else:
            STATE['prompts_context'] = "--- No Prompts Available ---"
            STATE['structured_prompts'] = {}


async def validate_and_correct_parameters(STATE: dict, command: dict) -> dict:
    """
    Corrects common LLM parameter hallucinations by translating between
    canonical names (e.g., 'database_name') and tool-specific aliases (e.g., 'db_name').
    """
    mcp_tools = STATE.get('mcp_tools', {})
    tool_name = command.get("tool_name")
    if not tool_name or tool_name not in mcp_tools:
        return command

    args = command.get("arguments", {})
    tool_spec = mcp_tools[tool_name]
    spec_arg_names = set(tool_spec.args.keys() if isinstance(tool_spec.args, dict) else [])
    
    corrected_args = args.copy()
    
    reverse_alias_map = {alias: canon for canon, aliases in PARAMETER_ALIASES.items() for alias in aliases}

    for arg_name in list(corrected_args.keys()):
        if arg_name in reverse_alias_map:
            canonical_name = reverse_alias_map[arg_name]
            if canonical_name not in corrected_args:
                app_logger.info(f"SHIM APPLIED (Pre-Correction): Translating provided alias '{arg_name}' to canonical '{canonical_name}' for tool '{tool_name}'.")
                corrected_args[canonical_name] = corrected_args.pop(arg_name)

    for canonical_name, aliases in PARAMETER_ALIASES.items():
        if canonical_name in corrected_args:
            for alias in aliases:
                if alias in spec_arg_names and alias not in corrected_args:
                    app_logger.info(f"SHIM APPLIED (Post-Correction): Translating canonical '{canonical_name}' to required alias '{alias}' for tool '{tool_name}'.")
                    corrected_args[alias] = corrected_args.pop(canonical_name)
                    break 
    
    command['arguments'] = corrected_args
    return command

def _transform_chart_data(data: any) -> list[dict]:
    """
    Checks for and corrects common LLM data format hallucinations for charts.
    Also renames 'ColumnName' to 'SourceColumnName' for clarity in charting.
    """
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
    """
    Constructs the G2Plot JSON specification from the LLM's request.
    
    This version uses a more robust mapping approach, relying on canonical keys from the prompt
    and dynamically matching them to column names in the data.
    """
    chart_type = args.get("chart_type", "").lower()
    mapping = args.get("mapping", {})
    
    # Define a canonical to G2Plot mapping
    # This centralizes the logic and makes it robust to minor LLM variations.
    canonical_map = {
        'x_axis': 'xField', 
        'y_axis': 'yField', 
        'color': 'seriesField',
        'angle': 'angleField',
        'category': 'xField', # Add aliases for flexibility
        'value': 'yField'      # Add aliases for flexibility
    }

    # Reverse map for checking against LLM output
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

    # Special handling for pie charts where 'seriesField' needs to be 'colorField'
    if chart_type == 'pie' and 'seriesField' in options:
        options['colorField'] = options.pop('seriesField')

    # Ensure numeric fields are correctly typed.
    final_data = []
    if data:
        for row in data:
            new_row = row.copy()
            for g2plot_key, actual_col_name in options.items():
                if g2plot_key in ['yField', 'angleField', 'size']:
                    cell_value = new_row.get(actual_col_name)
                    if cell_value is not None:
                        try:
                            # Convert to float for G2Plot compatibility
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

async def invoke_mcp_tool(STATE: dict, command: dict) -> any:
    mcp_client = STATE.get('mcp_client')
    tool_name = command.get("tool_name")
    
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
            # This transform is still useful for correcting LLM hallucinations
            # about data structure, but the hardcoded mapping logic is removed.
            data = _transform_chart_data(data) 
            
            if not isinstance(data, list) or not data:
                return {"error": "Validation failed", "data": "The 'data' argument must be a non-empty list of dictionaries."}
            
            chart_spec = _build_g2plot_spec(args, data)
            
            return {"type": "chart", "spec": chart_spec, "metadata": {"tool_name": "viz_createChart"}}
        except Exception as e:
            app_logger.error(f"Error building G2Plot spec: {e}", exc_info=True)
            return {"error": "Chart Generation Failed", "data": str(e)}

    validated_command = await validate_and_correct_parameters(STATE, command)
    if "error" in validated_command:
        return validated_command

    args = validated_command.get("arguments", validated_command.get("parameters", {}))
    
    app_logger.debug(f"Invoking tool '{tool_name}' with args: {args}")
    try:
        async with mcp_client.session("teradata_mcp_server") as temp_session:
            call_tool_result = await temp_session.call_tool(tool_name, args)
    except Exception as e:
        app_logger.error(f"Error during tool invocation for '{tool_name}': {e}", exc_info=True)
        return {"error": f"An exception occurred while invoking tool '{tool_name}'.", "data": str(e)}
    
    if hasattr(call_tool_result, 'content') and isinstance(call_tool_result.content, list) and len(call_tool_result.content) > 0:
        text_content = call_tool_result.content[0]
        if hasattr(text_content, 'text') and isinstance(text_content.text, str):
            try:
                return json.loads(text_content.text)
            except json.JSONDecodeError:
                app_logger.warning(f"Tool '{tool_name}' returned a non-JSON string: '{text_content.text}'")
                return {"error": "Tool returned non-JSON string", "data": text_content.text}
    
    raise RuntimeError(f"Unexpected tool result format for '{tool_name}': {call_tool_result}")

def classify_tool_scopes(tools: list) -> dict:
    scopes = {}
    for tool in tools:
        arg_names = set(tool.args.keys()) if isinstance(tool.args, dict) else set()
        if 'col_name' in arg_names or 'column_name' in arg_names:
            scopes[tool.name] = 'column'
        elif 'table_name' in arg_names or 'obj_name' in arg_names:
            scopes[tool.name] = 'table'
        else:
            scopes[tool.name] = 'database'
    return scopes