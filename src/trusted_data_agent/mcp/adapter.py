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


def _extract_and_clean_description(description: str | None) -> tuple[str, str]:
    """
    Parses a description string to find a datatype hint (e.g., "(type: str)")
    and cleans the description, returning both the cleaned description and the type.
    """
    if not isinstance(description, str):
        return "", "unknown"

    datatype = "unknown"
    # Regex to find "(type: xxx)" where xxx is one of the allowed types, case-insensitive
    match = re.search(r'\s*\((type:\s*(str|int|float|bool))\)', description, re.IGNORECASE)
    
    if match:
        # Extract the specific type (e.g., "str") from the second main capture group
        datatype = match.group(2).lower()
        # Remove the entire matched part (e.g., " (type: str)") from the description
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

        # Step 1: Load all capabilities (tools and prompts) from the server
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

        STATE['mcp_tools'] = {tool.name: tool for tool in loaded_tools}
        if loaded_prompts:
            STATE['mcp_prompts'] = {prompt.name: prompt for prompt in loaded_prompts}

        # Step 2: Prepare a single list of all capabilities for the LLM
        all_capabilities = []
        all_capabilities.extend([f"- {tool.name} (tool): {tool.description}" for tool in loaded_tools])
        
        # --- MODIFIED: Enhance prompt descriptions with their arguments for better classification ---
        for p in loaded_prompts:
            prompt_str = f"- {p.name} (prompt): {p.description or 'No description available.'}"
            # Check if the prompt has arguments and append them to the description string
            if hasattr(p, 'arguments') and p.arguments:
                prompt_str += "\n  - Arguments:"
                for arg in p.arguments:
                    arg_dict = arg.model_dump()
                    arg_name = arg_dict.get('name', 'unknown_arg')
                    prompt_str += f"\n    - `{arg_name}`"
            all_capabilities.append(prompt_str)
        # --- END MODIFICATION ---

        capabilities_list_str = "\n".join(all_capabilities)

        # Step 3: Create a single, unified prompt for categorization and scope inference
        classification_prompt = (
            "You are a helpful assistant that analyzes a list of technical capabilities (tools and prompts) for a Teradata database system and classifies them. "
            "For each capability, you must determine two things: a user-friendly 'category' for a UI, and its operational 'scope'.\n\n"
            "The 'scope' must be one of the following exact values: 'database', 'table', 'column', or 'none'.\n"
            " - Use 'database' for capabilities that operate on the entire database or list databases. A capability has a 'database' scope if it takes a `database_name` or `db_name` argument but no `table_name` or `column_name`.\n"
            " - Use 'table' for capabilities that primarily operate on a specific table. A capability has a 'table' scope if it takes a `table_name` or `obj_name` argument.\n"
            " - Use 'column' for capabilities that require a specific column name to function. A capability has a 'column' scope if it takes a `column_name` argument.\n"
            " - Use 'none' for utilities or general prompts that don't operate on a specific database object.\n\n"
            "Your response MUST be a single, valid JSON object. The keys of this object must be the capability names, "
            "and the value for each key must be another JSON object containing the 'category' and 'scope' you determined.\n\n"
            "Example format:\n"
            "{\n"
            '  "capability_name_1": {"category": "Some Category", "scope": "table"},\n'
            '  "capability_name_2": {"category": "Another Category", "scope": "database"}\n'
            "}\n\n"
            f"--- Capability List ---\n{capabilities_list_str}"
        )
        categorization_system_prompt = "You are an expert assistant that only responds with valid JSON."
        
        # Step 4: Make a single LLM call for all classifications
        classified_capabilities_str, _, _ = await llm_handler.call_llm_api(
            llm_instance, classification_prompt, raise_on_error=True,
            system_prompt_override=categorization_system_prompt
        )
        
        match = re.search(r'\{.*\}', classified_capabilities_str, re.DOTALL)
        if match is None:
            raise ValueError(f"LLM failed to return a valid JSON for capability classification. Response: '{classified_capabilities_str}'")
        
        cleaned_str = match.group(0)
        classified_data = json.loads(cleaned_str)

        # Step 5: Process the unified response for Tools
        STATE['structured_tools'] = {}
        STATE['tool_scopes'] = {}
        tool_details_list = []
        disabled_tools_list = STATE.get("disabled_tools", [])

        for tool in loaded_tools:
            classification = classified_data.get(tool.name, {})
            category = classification.get("category", "Uncategorized")
            scope = classification.get("scope")

            if category not in STATE['structured_tools']:
                STATE['structured_tools'][category] = []

            is_disabled = tool.name in disabled_tools_list
            STATE['structured_tools'][category].append({
                "name": tool.name, "description": tool.description, "disabled": is_disabled
            })

            if not is_disabled:
                description_prefix = f"(scope: {scope}) " if scope and scope != 'none' else ""
                if scope and scope in ['database', 'table', 'column']:
                    STATE['tool_scopes'][tool.name] = scope

                tool_str = f"- `{tool.name}`: {description_prefix}{tool.description}"
                args_dict = tool.args if isinstance(tool.args, dict) else {}

                # Only add arguments from tool.args if not already in the description, to prevent duplicates.
                if args_dict and "Arguments:" not in tool.description:
                    tool_str += "\n  - Arguments:"
                    for arg_name, arg_details in args_dict.items():
                        arg_type = arg_details.get('type', 'any')
                        is_required = arg_details.get('required', False)
                        req_str = "required" if is_required else "optional"
                        arg_desc = arg_details.get('description', 'No description.')
                        tool_str += f"\n    - `{arg_name}` ({arg_type}, {req_str}): {arg_desc}"
                tool_details_list.append(tool_str)
        
        STATE['tools_context'] = "--- Available Tools ---\n" + "\n".join(tool_details_list)

        # Step 6: Process the unified response for Prompts
        STATE['structured_prompts'] = {}
        prompt_details_list = []
        disabled_prompts_list = STATE.get("disabled_prompts", [])
        
        if loaded_prompts:
            for prompt_obj in loaded_prompts:
                classification = classified_data.get(prompt_obj.name, {})
                category = classification.get("category", "Uncategorized")
                scope = classification.get("scope")
                
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

                if not is_disabled:
                    description_prefix = f"(scope: {scope}) " if scope and scope != 'none' else ""
                    prompt_description = prompt_obj.description or "No description available."
                    prompt_str = f"- `{prompt_obj.name}`: {description_prefix}{prompt_description}"

                    # Only add arguments if not already in the description, to prevent duplicates.
                    if processed_args and "Arguments:" not in prompt_description:
                        prompt_str += "\n  - Arguments:"
                        for arg_details in processed_args:
                            arg_name = arg_details.get('name', 'unknown')
                            arg_type = arg_details.get('type', 'any')
                            is_required = arg_details.get('required', False)
                            req_str = "required" if is_required else "optional"
                            arg_desc = arg_details.get('description', 'No description.')
                            prompt_str += f"\n    - `{arg_name}` ({arg_type}, {req_str}): {arg_desc}"
                    prompt_details_list.append(prompt_str)

        if prompt_details_list:
            STATE['prompts_context'] = "--- Available Prompts ---\n" + "\n".join(prompt_details_list)
        else:
            STATE['prompts_context'] = "--- No Prompts Available ---"


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

    args = command.get("arguments", command.get("parameters", {}))
    
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