# src/trusted_data_agent/agent/shims/prompt_shims.py

"""
This file contains client-side shims for MCP prompts that are not yet
compliant with the "MCP Workflow Prompt Guidelines".

The logic in `mcp/adapter.py` will use this dictionary to override the
content of prompts loaded from the server.

- The dictionary key must be the exact prompt name from the MCP server.
- The dictionary value must be the corrected, compliant prompt content as a string.
"""

PROMPT_SHIMS = {
    # This is a corrected version of the non-compliant prompt from our discussion.
    # It adheres to the workflow guidelines.
    "base_tableBusinessDesc": """# Name: base_tableBusinessDesc
# Description: Gathers technical details for a table and then provides a business-oriented description of its purpose and columns.

## Phase 1 - Get Table DDL
- Get the DDL for the table '{table_name}' in database '{db_name}' using the `base_tableDDL` tool.

## Phase 2 - Final Summary
- You now have the DDL for the table. Analyze the DDL structure (table name, column names, data types) to infer and describe the table in a business context. The description should include the purpose of the table and the likely business purpose of its key columns. Your final output must be in markdown.
"""
}