# src/trusted_data_agent/mcp/adapter.py
import json
import logging
import re

from langchain_mcp_adapters.tools import load_mcp_tools
from trusted_data_agent.llm import handler as llm_handler

app_logger = logging.getLogger("quart.app")

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

        STATE['mcp_tools'] = {tool.name: tool for tool in loaded_tools}
        # --- MODIFICATION: Classify and store tool scopes ---
        STATE['tool_scopes'] = classify_tool_scopes(loaded_tools)
        # --- END MODIFICATION ---
        
        # Build a descriptive tools_context for the LLM, including arguments.
        tool_strings = []
        for tool in loaded_tools:
            arg_list = ", ".join(tool.args.keys())
            tool_strings.append(f"- `{tool.name}({arg_list})`: {tool.description}")
        
        STATE['tools_context'] = "--- Available Tools ---\n" + "\n".join(tool_strings)
        
        # Categorize tools for the UI
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

        # Load and categorize prompts
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

async def validate_and_correct_parameters(STATE: dict, command: dict) -> dict:
    """
    Validates tool parameters, applies shims for legacy tools, and attempts
    to correct mismatches using an LLM before finally asking the user for input.
    """
    mcp_tools = STATE.get('mcp_tools', {})
    llm_instance = STATE.get('llm')
    tool_name = command.get("tool_name")
    if not tool_name or tool_name not in mcp_tools:
        return command

    args = command.get("arguments", {})
    
    # --- SHIM FOR LEGACY QUALITY TOOLS ---
    # This logic intercepts calls to older 'qlty_' tools and reformats the
    # arguments to what the tool expects (a single fully qualified table name).
    LEGACY_QUALITY_TOOLS = [
        "qlty_missingValues", "qlty_negativeValues", "qlty_distinctCategories",
        "qlty_standardDeviation", "qlty_columnSummary", "qlty_univariateStatistics",
        "qlty_rowsWithMissingValues"
    ]
    if tool_name in LEGACY_QUALITY_TOOLS:
        db_name = args.get("db_name")
        table_name = args.get("table_name")
        if db_name and table_name and '.' not in table_name:
            # This is the improved notification message for the user.
            notification_message = f"Applying compatibility fix for legacy tool '{tool_name}': combining database and table name for context."
            app_logger.info(f"SHIM APPLIED: {notification_message}")
            command['notification'] = notification_message
            
            args["table_name"] = f"{db_name}.{table_name}"
            del args["db_name"]
    # --- END SHIM ---

    llm_arg_names = set(args.keys())
    tool_spec = mcp_tools[tool_name]
    spec_arg_names = set(tool_spec.args.keys())
    required_params = {name for name, field in tool_spec.args.items() if field.get("required", False)}

    # If all required parameters are present, proceed.
    if required_params.issubset(llm_arg_names):
        return command

    # If parameters are missing, attempt correction with an LLM.
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

        corrected_args = {}
        for llm_name, spec_name in name_mapping.items():
            if llm_name in args and spec_name in spec_arg_names:
                corrected_args[spec_name] = args[llm_name]
        
        if not required_params.issubset(set(corrected_args.keys())):
             raise ValueError(f"Corrected parameters are still missing required arguments. Missing: {required_params - set(corrected_args.keys())}")

        app_logger.info(f"Successfully corrected parameters for tool '{tool_name}'. New args: {corrected_args}")
        command['arguments'] = corrected_args
        return command

    except (ValueError, json.JSONDecodeError, AttributeError) as e:
        # If correction fails, ask the user for the parameters directly.
        app_logger.warning(f"Parameter correction failed for '{tool_name}': {e}. Requesting user input.")
        spec_arguments = list(tool_spec.args.values())
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
    Invokes a tool, either by calling the MCP client or by handling it
    locally (e.g., for chart generation). It validates and corrects parameters first.
    """
    mcp_client = STATE.get('mcp_client')
    mcp_charts = STATE.get('mcp_charts', {})

    if command.get("tool_name") not in mcp_charts:
        # The command dict is modified in-place by this function if a shim is applied
        validated_command = await validate_and_correct_parameters(STATE, command)
        if "error" in validated_command:
            return validated_command
    else:
        validated_command = command

    if not mcp_client:
        return {"error": "MCP client is not connected."}

    tool_name = validated_command.get("tool_name")
    args = validated_command.get("arguments", validated_command.get("parameters", {}))

    # Handle chart generation locally
    if tool_name in mcp_charts:
        app_logger.info(f"Locally handling chart generation for tool: {tool_name}")
        try:
            is_bar_chart = "generate_bar_chart" in tool_name
            data = args.get("data", [])
            
            x_field = args.get("x_axis") or args.get("x_field")
            y_field = args.get("y_axis") or args.get("y_field")
            angle_field = args.get("angle_field")
            color_field = args.get("color_field")

            # Infer axes if not provided
            if not x_field or not y_field:
                if data:
                    first_row = data[0]
                    x_field = next((k for k, v in first_row.items() if isinstance(v, str)), None)
                    y_field = next((k for k, v in first_row.items() if isinstance(v, (int, float)) or (isinstance(v, str) and v.replace('.', '', 1).isdigit())), None)

            # Ensure numeric data for y-axis
            if y_field and data:
                for row in data:
                    try:
                        row[y_field] = float(row[y_field])
                    except (ValueError, TypeError):
                        row[y_field] = 0

            spec_options = {
                "data": data,
                "xField": y_field if is_bar_chart else x_field,
                "yField": x_field if is_bar_chart else y_field,
                "angleField": angle_field,
                "colorField": color_field,
                "seriesField": args.get("series_field", args.get("series")),
                "title": { "visible": True, "text": args.get("title", "Generated Chart") }
            }
            chart_type_mapping = { "generate_bar_chart": "Bar", "generate_column_chart": "Column", "generate_pie_chart": "Pie", "generate_line_chart": "Line", "generate_area_chart": "Area", "generate_scatter_chart": "Scatter", "generate_histogram_chart": "Histogram", "generate_boxplot_chart": "Box", "generate_dual_axes_chart": "DualAxes", }
            plot_type = next((v for k, v in chart_type_mapping.items() if k in tool_name), "Column")
            final_spec_options = {k: v for k, v in spec_options.items() if v is not None}
            chart_spec = { "type": plot_type, "options": final_spec_options }
            return {"type": "chart", "spec": chart_spec, "metadata": {"tool_name": tool_name}}
        except Exception as e:
            app_logger.error(f"Error during local chart generation: {e}", exc_info=True)
            return {"error": f"Failed to generate chart spec locally: {e}"}

    # Invoke tool via MCP client
    try:
        app_logger.debug(f"Invoking tool '{tool_name}' with args: {args}")
        async with mcp_client.session("teradata_mcp_server") as temp_session:
            call_tool_result = await temp_session.call_tool(tool_name, args)
        
        # Parse the tool's response
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

# --- NEW: Function to classify tool scopes ---
def classify_tool_scopes(tools: list) -> dict:
    """
    Classifies tools based on their argument names to determine if they
    operate at a database, table, or column level.
    """
    scopes = {}
    for tool in tools:
        arg_names = set(tool.args.keys())
        if 'col_name' in arg_names or 'column_name' in arg_names:
            scopes[tool.name] = 'column'
        elif 'table_name' in arg_names or 'obj_name' in arg_names:
            scopes[tool.name] = 'table' # Changed from 'database' for clarity
        else:
            scopes[tool.name] = 'database'
    return scopes
# --- END NEW ---
