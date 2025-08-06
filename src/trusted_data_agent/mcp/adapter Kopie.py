# src/trusted_data_agent/mcp/adapter.py
import json
import logging
import re

from langchain_mcp_adapters.tools import load_mcp_tools
from trusted_data_agent.llm import handler as llm_handler

app_logger = logging.getLogger("quart.app")

# --- NEW: Formalized Parameter Alias Map ---
# This map acts as a translation layer to correct common parameter name variations
# from the LLM, enhancing agent robustness without LLM intervention.
# It maps a canonical parameter name to a list of its common aliases.
PARAMETER_ALIASES = {
    "database_name": ["db_name", "database"],
    "table_name": ["tbl_name", "object_name", "obj_name"],
    "column_name": ["col_name", "column"]
}

# --- NEW: Abstract definition for the single visualization tool ---
# This provides a consistent, abstract interface for the LLM to use for all charting tasks.
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


def _patch_legacy_tool_definitions(tools: list):
    """
    Patches the definitions of legacy tools in-place.

    This function is a temporary shim to address inconsistencies in the tool
    definitions provided by the MCP server. It adds the 'db_name' argument
    to older 'qlty_' tools that do not explicitly define it, ensuring that
    the LLM is aware of this parameter.

    Once the MCP tool definitions are permanently corrected on the server,
    this function can be safely removed.
    """
    LEGACY_QUALITY_TOOLS = [
        "qlty_missingValues", "qlty_negativeValues", "qlty_distinctCategories",
        "qlty_standardDeviation", "qlty_columnSummary", "qlty_univariateStatistics",
        "qlty_rowsWithMissingValues"
    ]

    for tool in tools:
        if tool.name in LEGACY_QUALITY_TOOLS:
            if "db_name" not in tool.args:
                app_logger.info(f"PATCHING TOOL DEFINITION: Adding optional 'db_name' to '{tool.name}'.")
                tool.args["db_name"] = {
                    "type": "string",
                    "description": "Optional: The name of the database where the table resides. If not provided, the default database is used.",
                    "required": False
                }

async def load_and_categorize_teradata_resources(STATE: dict):
    """
    Loads all tools and prompts from the Teradata MCP server,
    categorizes them using an LLM, and stores them in the application state.
    """
    mcp_client = STATE.get('mcp_client')
    llm_instance = STATE.get('llm')
    if not mcp_client or not llm_instance:
        raise Exception("MCP or LLM client not initialized.")

    async with mcp_client.session("teradata_mcp_server") as temp_session:
        app_logger.info("--- Loading Teradata tools and prompts... ---")

        # Load Tools from MCP
        loaded_tools = await load_mcp_tools(temp_session)

        # Apply the patch for legacy tool definitions.
        # This can be removed once the MCP definitions are updated.
        _patch_legacy_tool_definitions(loaded_tools)

        # --- MODIFIED: Add the abstract viz_createChart tool to the list ---
        # We convert our dict definition to a simple object that matches the structure
        # of the tools loaded from the MCP for consistency.
        class SimpleTool:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
        
        viz_tool_obj = SimpleTool(**VIZ_TOOL_DEFINITION)
        loaded_tools.append(viz_tool_obj)
        # --- END MODIFICATION ---

        STATE['mcp_tools'] = {tool.name: tool for tool in loaded_tools}
        STATE['tool_scopes'] = classify_tool_scopes(loaded_tools)
        
        tool_strings = []
        for tool in loaded_tools:
            # Ensure args is a dictionary before calling .keys()
            args_dict = tool.args if isinstance(tool.args, dict) else {}
            arg_list = ", ".join(args_dict.keys())
            tool_strings.append(f"- `{tool.name}({arg_list})`: {tool.description}")
        
        STATE['tools_context'] = "--- Available Tools ---\n" + "\n".join(tool_strings)
        
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
        STATE['structured_tools'] = {category: [{"name": name, "description": STATE['mcp_tools'][name].description} for name in tool_names if name in STATE['mcp_tools']] for category, tool_names in categorized_tools.items()}

        loaded_prompts = []
        try:
            list_prompts_result = await temp_session.list_prompts()
            if hasattr(list_prompts_result, 'prompts'):
                loaded_prompts = list_prompts_result.prompts
        except Exception as e:
            app_logger.error(f"CRITICAL ERROR while loading prompts: {e}", exc_info=True)

        if loaded_prompts:
            STATE['mcp_prompts'] = {prompt.name: prompt for prompt in loaded_prompts}
            STATE['prompts_context'] = "--- Available Prompts ---\n" + "\n".join([f"- `{p.name}`: {p.description}" for p in loaded_prompts])
            
            serializable_prompts = [{"name": p.name, "description": p.description, "arguments": [arg.model_dump() for arg in p.arguments]} for p in loaded_prompts]
            prompt_list_for_prompt = "\n".join([f"- {p['name']}: {p['description']}" for p in serializable_prompts])
            
            categorization_prompt_for_prompts = (
                "You are a JSON formatting expert. Your task is to categorize the following list of Teradata system prompts into a single JSON object."
                "\n\n**CRITICAL RULES:**"
                "\n1. Your entire response MUST be a single, raw JSON object."
                "\n2. DO NOT include ```json markdown wrappers, conversational text, or any explanations."
                "\n3. The JSON keys MUST be the category names."
                "\n4. The JSON values MUST be an array of the prompt names."
                "\n\n**EXAMPLE OUTPUT:**"
                "\n{"
                "\n  \"Testing Suite\": [\"test_dbaTools\", \"test_secTools\"],"
                "\n  \"Database Administration\": [\"dba_tableArchive\", \"dba_databaseLineage\"]"
                "\n}"
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
            STATE['structured_prompts'] = {category: [p for p in serializable_prompts if p['name'] in prompt_names] for category, prompt_names in categorized_prompts.items()}
        else:
            STATE['prompts_context'] = "--- No Prompts Available ---"
            STATE['structured_prompts'] = {}

# --- NEW: Function to dynamically generate charting guidelines ---
def _generate_charting_guidelines(chart_tools: list) -> str:
    """
    Inspects the loaded chart tools and generates a markdown-formatted
    string of guidelines for the LLM's system prompt. This serves as the
    LLM's "rulebook" for how to use the abstract viz_createChart tool.
    """
    if not chart_tools:
        return "No charting tools are available."

    guidelines = []
    for tool in chart_tools:
        # e.g., 'generate_bar_chart' -> 'bar'
        chart_type = tool.name.replace("generate_", "").replace("_chart", "")
        
        guideline = f"- **{chart_type}**: {tool.description}"
        
        # Ensure args is a dictionary before processing
        args_dict = tool.args if isinstance(tool.args, dict) else {}
        
        required_args = [name for name, details in args_dict.items() if details.get("required", True)]
        # We only care about the mapping arguments for the guideline
        mapping_args = [arg for arg in required_args if arg not in ['data', 'title']]

        if mapping_args:
            guideline += "\n  - **Required mappings**: "
            arg_details = []
            for arg_name in mapping_args:
                arg_spec = args_dict.get(arg_name, {})
                # Provide a default description if none exists
                arg_desc = arg_spec.get('description', 'No description').split('.')[0]
                arg_details.append(f"`{arg_name}` ({arg_desc})")
            guideline += ", ".join(arg_details)

        guidelines.append(guideline)
    
    return "\n".join(guidelines)
# --- END NEW ---

async def load_and_categorize_chart_resources(STATE: dict):
    """
    Loads all chart generation tools, generates dynamic guidelines for the LLM,
    and categorizes them for the UI.
    """
    mcp_client = STATE.get('mcp_client')
    llm_instance = STATE.get('llm')
    if not mcp_client or not llm_instance:
        raise Exception("MCP or LLM client not initialized for chart loading.")

    async with mcp_client.session("chart_mcp_server") as temp_session:
        app_logger.info("--- Loading Chart tools... ---")
        loaded_charts = await load_mcp_tools(temp_session)
        
        STATE['mcp_charts'] = {tool.name: tool for tool in loaded_charts}
        
        # --- MODIFIED: Generate guidelines and use them for the LLM context ---
        chart_guidelines = _generate_charting_guidelines(loaded_charts)
        STATE['charting_guidelines'] = chart_guidelines
        # The 'charts_context' will now be populated by the routes with these guidelines
        # This keeps the old key for compatibility but fills it with new, better content.
        STATE['charts_context'] = chart_guidelines
        # --- END MODIFICATION ---
        
        # This part remains for populating the UI gallery as per our hybrid approach
        chart_list_for_prompt = "\n".join([f"- {tool.name}: {tool.description}" for tool in loaded_charts])
        categorization_prompt = (
            "You are a helpful assistant that organizes a list of data visualization tools into logical categories for a user interface. "
            "Your response MUST be a single, valid JSON object. The keys should be the category names (e.g., 'Bar & Column', 'Pie & Donut', 'Line & Area'), "
            f"and the values should be an array of tool names belonging to that category.\n\n"
            f"--- Chart Tool List ---\n{chart_list_for_prompt}"
        )
        categorization_system_prompt = "You are a helpful assistant that organizes lists into JSON format."
        
        categorized_charts_str, _, _ = await llm_handler.call_llm_api(
            llm_instance, categorization_prompt, raise_on_error=True,
            system_prompt_override=categorization_system_prompt
        )
        
        match = re.search(r'\{.*\}', categorized_charts_str, re.DOTALL)
        if match is None:
            raise ValueError(f"LLM failed to return a valid JSON for chart categorization. Response: '{categorized_charts_str}'")
        
        cleaned_str = match.group(0)
        categorized_charts = json.loads(cleaned_str)
        STATE['structured_charts'] = {category: [{"name": name, "description": STATE['mcp_charts'][name].description} for name in tool_names if name in STATE['mcp_charts']] for category, tool_names in categorized_charts.items()}


async def validate_and_correct_parameters(STATE: dict, command: dict) -> dict:
    """
    Validates tool parameters, applies shims for common aliases, and attempts
    to correct mismatches using an LLM before finally asking the user for input.
    """
    mcp_tools = STATE.get('mcp_tools', {})
    llm_instance = STATE.get('llm')
    tool_name = command.get("tool_name")
    if not tool_name or tool_name not in mcp_tools:
        return command

    args = command.get("arguments", {})
    tool_spec = mcp_tools[tool_name]
    
    # Ensure tool_spec.args is a dictionary
    spec_args_dict = tool_spec.args if isinstance(tool_spec.args, dict) else {}
    spec_arg_names = set(spec_args_dict.keys())

    # --- REFACTORED: Use the module-level alias map ---
    corrected_args = args.copy()
    for canonical_name, aliases in PARAMETER_ALIASES.items():
        if canonical_name in spec_arg_names and canonical_name not in corrected_args:
            for alias in aliases:
                if alias in corrected_args:
                    app_logger.info(f"SHIM APPLIED: Translating parameter alias '{alias}' to '{canonical_name}' for tool '{tool_name}'.")
                    corrected_args[canonical_name] = corrected_args.pop(alias)
                    break 
    
    command['arguments'] = corrected_args
    args = corrected_args
    # --- END REFACTOR ---

    llm_arg_names = set(args.keys())
    required_params = {name for name, field in spec_args_dict.items() if field.get("required", False)}

    if required_params.issubset(llm_arg_names):
        return command

    app_logger.info(f"Parameter mismatch for tool '{tool_name}'. Attempting correction with LLM.")
    correction_prompt = f"""
        You are a parameter-mapping specialist. Your task is to map the 'LLM-Generated Parameters' to the 'Official Tool Parameters'.
        The user wants to call the tool '{tool_name}', which is described as: '{tool_spec.description}'.

        Official Tool Parameters: {list(spec_arg_names)}
        LLM-Generated Parameters: {list(llm_arg_names)}

        Respond with a single JSON object that maps each generated parameter name to its correct official name.
        If a generated parameter does not sensibly map to any official parameter, use `null` as the value.
        Example response: {{"database": "db_name", "table": "table_name", "extra_param": null}}
    """
    
    correction_response_text, _, _ = await llm_handler.call_llm_api(llm_instance, prompt=correction_prompt, chat_history=[])
    
    try:
        json_match = re.search(r"```json\s*\n(.*?)\n\s*```", correction_response_text, re.DOTALL)
        if not json_match:
            json_match = re.search(r'\{.*\}', correction_response_text, re.DOTALL)
        
        if not json_match:
             raise ValueError("LLM did not return a valid JSON object for parameter mapping.")

        name_mapping = json.loads(json_match.group(0).strip())

        if any(v is None for v in name_mapping.values()):
            raise ValueError("LLM could not confidently map all parameters.")

        final_corrected_args = {}
        for llm_name, spec_name in name_mapping.items():
            if llm_name in args and spec_name in spec_arg_names:
                final_corrected_args[spec_name] = args[llm_name]
        
        if not required_params.issubset(set(final_corrected_args.keys())):
             raise ValueError(f"Corrected parameters are still missing required arguments. Missing: {required_params - set(final_corrected_args.keys())}")

        app_logger.info(f"Successfully corrected parameters for tool '{tool_name}'. New args: {final_corrected_args}")
        command['arguments'] = final_corrected_args
        return command

    except (ValueError, json.JSONDecodeError, AttributeError) as e:
        app_logger.warning(f"Parameter correction failed for '{tool_name}': {e}. Requesting user input.")
        spec_arguments = [{k: v for k, v in arg.items() if k != 'default'} for arg in spec_args_dict.values()]
        return {
            "error": "parameter_mismatch",
            "tool_name": tool_name,
            "message": "The agent could not determine the correct parameters for the tool. Please provide them below.",
            "specification": {
                "name": tool_name,
                "description": tool_spec.description,
                "arguments": spec_arguments
            }
        }

async def invoke_mcp_tool(STATE: dict, command: dict) -> any:
    """
    Invokes a tool. It now includes a validation and translation layer for
    the abstract 'viz_createChart' tool before calling the concrete MCP tool.
    """
    mcp_client = STATE.get('mcp_client')
    mcp_charts = STATE.get('mcp_charts', {})
    
    # --- NEW: Intercept and validate viz_createChart calls ---
    if command.get("tool_name") == "viz_createChart":
        app_logger.info(f"Handling abstract chart generation for: {command}")
        
        args = command.get("arguments", {})
        chart_type = args.get("chart_type")
        data = args.get("data")
        mapping = args.get("mapping")
        
        # 1. Basic validation
        if not all([chart_type, data, mapping]):
            return {"error": "Validation failed", "data": "The 'chart_type', 'data', and 'mapping' arguments are all required for viz_createChart."}
        
        if not isinstance(data, list) or not data:
            return {"error": "Validation failed", "data": "The 'data' argument must be a non-empty list of dictionaries."}
            
        # 2. Find the corresponding concrete tool
        target_tool_name = f"generate_{chart_type}_chart"
        if target_tool_name not in mcp_charts:
            # Fallback for simple names like 'bar' -> 'generate_bar_chart'
            # This handles cases where a chart tool might not follow the exact naming convention.
            simple_name_match = [name for name in mcp_charts if chart_type in name]
            if len(simple_name_match) == 1:
                target_tool_name = simple_name_match[0]
                app_logger.info(f"Matched '{chart_type}' to concrete tool '{target_tool_name}'")
            else:
                return {"error": "Validation failed", "data": f"The specified chart_type '{chart_type}' does not correspond to a unique chart tool on the MCP server."}

        tool_spec = mcp_charts[target_tool_name]
        tool_spec_args = tool_spec.args if isinstance(tool_spec.args, dict) else {}

        # 3. Validate required mappings against the tool's spec
        required_mappings = {name for name, details in tool_spec_args.items() if details.get("required", True) and name not in ['data', 'title']}
        provided_mappings = set(mapping.keys())
        
        if not required_mappings.issubset(provided_mappings):
            missing = required_mappings - provided_mappings
            return {"error": "Validation failed", "data": f"Missing required mappings for chart type '{chart_type}': {', '.join(missing)}"}

        # 4. Validate data types for mapped columns
        first_row = data[0]
        for map_key, col_name in mapping.items():
            if col_name not in first_row:
                return {"error": "Validation failed", "data": f"The column '{col_name}' specified in the mapping does not exist in the provided data."}
            
            # A simple heuristic for numeric type checking
            is_numeric_arg = 'value' in map_key.lower() or 'y_axis' in map_key.lower() or 'angle' in map_key.lower() or 'size' in map_key.lower()
            if is_numeric_arg:
                try:
                    # Check if the value can be converted to a float.
                    float(first_row[col_name])
                except (ValueError, TypeError):
                     return {"error": "Validation failed", "data": f"The mapping '{map_key}' for chart type '{chart_type}' requires a numeric data column, but the column '{col_name}' contains non-numeric data ('{first_row[col_name]}')."}

        # 5. If all validation passes, construct the concrete tool call
        app_logger.info(f"Validation successful. Translating 'viz_createChart' to concrete tool '{target_tool_name}'.")
        final_args = {
            "data": data,
            "title": args.get("title", "Generated Chart")
        }
        final_args.update(mapping) # Add the axis/field mappings
        
        # Re-assign the command to be the concrete, validated tool call
        command = {"tool_name": target_tool_name, "arguments": final_args}
    # --- END NEW ---

    if command.get("tool_name") not in mcp_charts:
        validated_command = await validate_and_correct_parameters(STATE, command)
        if "error" in validated_command:
            return validated_command
    else:
        validated_command = command

    if not mcp_client:
        return {"error": "MCP client is not connected."}

    tool_name = validated_command.get("tool_name")
    args = validated_command.get("arguments", validated_command.get("parameters", {}))
    
    # --- MODIFIED: This entire block is now replaced by the MCP server call ---
    # The MCP server is now responsible for generating the final spec.
    # The old local spec generation logic is removed.
    if tool_name in mcp_charts:
        app_logger.info(f"Invoking chart tool '{tool_name}' on chart MCP server with args: {args}")
        try:
            async with mcp_client.session("chart_mcp_server") as temp_session:
                # The chart MCP server is expected to return the final JSON spec
                call_tool_result = await temp_session.call_tool(tool_name, args)
            
            if hasattr(call_tool_result, 'content') and isinstance(call_tool_result.content, list) and len(call_tool_result.content) > 0:
                text_content = call_tool_result.content[0]
                if hasattr(text_content, 'text') and isinstance(text_content.text, str):
                    try:
                        # The server's response IS the chart spec
                        chart_spec = json.loads(text_content.text)
                        return {"type": "chart", "spec": chart_spec, "metadata": {"tool_name": tool_name}}
                    except json.JSONDecodeError:
                        app_logger.warning(f"Chart tool '{tool_name}' returned a non-JSON string: '{text_content.text}'")
                        return {"error": "Chart tool returned non-JSON string", "data": text_content.text}
            
            raise RuntimeError(f"Unexpected chart tool result format for '{tool_name}': {call_tool_result}")

        except Exception as e:
            app_logger.error(f"Error during chart tool invocation for '{tool_name}': {e}", exc_info=True)
            return {"error": f"An exception occurred while invoking chart tool '{tool_name}'."}
    # --- END MODIFICATION ---

    try:
        app_logger.debug(f"Invoking tool '{tool_name}' with args: {args}")
        async with mcp_client.session("teradata_mcp_server") as temp_session:
            call_tool_result = await temp_session.call_tool(tool_name, args)
        
        if hasattr(call_tool_result, 'content') and isinstance(call_tool_result.content, list) and len(call_tool_result.content) > 0:
            text_content = call_tool_result.content[0]
            if hasattr(text_content, 'text') and isinstance(text_content.text, str):
                try:
                    return json.loads(text_content.text)
                except json.JSONDecodeError:
                    app_logger.warning(f"Tool '{tool_name}' returned a non-JSON string: '{text_content.text}'")
                    return {"error": "Tool returned non-JSON string", "data": text_content.text}
        
        raise RuntimeError(f"Unexpected tool result format for '{tool_name}': {call_tool_result}")
    except Exception as e:
        app_logger.error(f"Error during tool invocation for '{tool_name}': {e}", exc_info=True)
        return {"error": f"An exception occurred while invoking tool '{tool_name}'."}

def classify_tool_scopes(tools: list) -> dict:
    """
    Classifies tools based on their argument names to determine if they
    operate at a database, table, or column level.
    """
    scopes = {}
    for tool in tools:
        # Ensure tool.args is a dictionary before accessing keys
        arg_names = set(tool.args.keys()) if isinstance(tool.args, dict) else set()
        if 'col_name' in arg_names or 'column_name' in arg_names:
            scopes[tool.name] = 'column'
        elif 'table_name' in arg_names or 'obj_name' in arg_names:
            scopes[tool.name] = 'table'
        else:
            scopes[tool.name] = 'database'
    return scopes