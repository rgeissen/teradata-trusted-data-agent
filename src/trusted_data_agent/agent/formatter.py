# trusted_data_agent/agent/formatter.py
import re
import uuid
import json

class OutputFormatter:
    """
    Parses raw LLM output and structured tool data to generate professional,
    failure-safe HTML for the UI.
    """
    def __init__(self, llm_response_text: str, collected_data: list | dict, is_workflow: bool = False):
        self.raw_summary = llm_response_text
        self.collected_data = collected_data
        self.is_workflow = is_workflow
        self.processed_data_indices = set()

    def _has_renderable_tables(self) -> bool:
        """Checks if there is any data that will be rendered as a table."""
        data_source = []
        if isinstance(self.collected_data, dict):
            for item_list in self.collected_data.values():
                data_source.extend(item_list)
        else:
            data_source = self.collected_data

        for item in data_source:
            if isinstance(item, dict) and "results" in item:
                results = item.get("results")
                if isinstance(results, list) and results and all(isinstance(row, dict) for row in results):
                    return True
        return False

    def _sanitize_summary(self) -> str:
        """
        Cleans the LLM's summary text, removing markdown elements that will be
        rendered separately and replacing them intelligently.
        """
        clean_summary = self.raw_summary
        
        # Remove markdown tables and replace with placeholder if renderable tables exist
        markdown_table_pattern = re.compile(r"\|.*\|[\n\r]*\|[-| :]*\|[\n\r]*(?:\|.*\|[\n\r]*)*", re.MULTILINE)
        if markdown_table_pattern.search(clean_summary):
            replacement_text = "\n(Data table is shown below)\n" if self._has_renderable_tables() else ""
            clean_summary = re.sub(markdown_table_pattern, replacement_text, clean_summary)

        # Remove DDL blocks, as they are rendered by a dedicated function.
        sql_ddl_pattern = re.compile(r"```sql\s*CREATE MULTISET TABLE.*?;?\s*```|CREATE MULTISET TABLE.*?;", re.DOTALL | re.IGNORECASE)
        clean_summary = re.sub(sql_ddl_pattern, "\n(Formatted DDL shown below)\n", clean_summary)
        
        # --- MODIFIED: Enhanced markdown parser to handle key-value pairs and nested lists ---
        lines = clean_summary.strip().split('\n')
        html_output = []
        
        def process_inline_markdown(text_content):
            # Process backticks for code blocks first
            text_content = re.sub(r'`(.*?)`', r'<code class="bg-gray-900/70 text-teradata-orange rounded-md px-1.5 py-0.5 font-mono text-sm">\1</code>', text_content)
            # Process bold/strong tags
            text_content = re.sub(r'\*{2,3}(.*?):\*{1,3}', r'<strong>\1:</strong>', text_content)
            text_content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text_content)
            return text_content

        current_list_items = []

        for line in lines:
            stripped_line = line.strip()
            
            # --- MODIFIED: Regex now accepts 2 or 3 asterisks for key-value pairs ---
            key_value_match = re.match(r'^\*{2,3}(.*?)\*{2,3}\s*(.*)$', stripped_line)
            # Pattern for list items
            list_item_match = re.match(r'^[*-]\s*(.*)$', stripped_line)

            if key_value_match:
                # This is a key-value line, so render any pending list first
                if current_list_items:
                    html_output.append('<ul class="list-disc list-inside space-y-2 text-gray-300 mb-4 pl-4">')
                    for item in current_list_items:
                        html_output.append(f'<li>{item}</li>')
                    html_output.append('</ul>')
                    current_list_items = []

                key = key_value_match.group(1).strip()
                value = key_value_match.group(2).strip()
                
                # Special handling for 'Description' to treat its value as a paragraph
                if "description" in key.lower():
                     html_output.append(f'<p class="text-gray-300 mb-2"><strong>{key}</strong></p>')
                     # The value part might be long and needs its own paragraph tag
                     html_output.append(f'<p class="text-gray-300 mb-4">{process_inline_markdown(value)}</p>')
                else:
                    html_output.append(f'<p class="text-gray-300 mb-2"><strong>{key}</strong> {process_inline_markdown(value)}</p>')

            elif list_item_match:
                # This is a list item, collect it
                content = list_item_match.group(1).strip()
                if content:
                    current_list_items.append(process_inline_markdown(content))
            
            else:
                # This is not a list item or a key-value pair. Render any pending list.
                if current_list_items:
                    html_output.append('<ul class="list-disc list-inside space-y-2 text-gray-300 mb-4 pl-4">')
                    for item in current_list_items:
                        html_output.append(f'<li>{item}</li>')
                    html_output.append('</ul>')
                    current_list_items = []

                # Process as a header or paragraph
                if stripped_line.startswith('# '):
                    content = stripped_line[2:]
                    html_output.append(f'<h3 class="text-xl font-bold text-white mb-3 border-b border-gray-700 pb-2">{content}</h3>')
                elif stripped_line.startswith('## '):
                    content = stripped_line[3:]
                    html_output.append(f'<h4 class="text-lg font-semibold text-white mt-4 mb-2">{content}</h4>')
                elif stripped_line:
                    html_output.append(f'<p class="text-gray-300 mb-4">{process_inline_markdown(stripped_line)}</p>')

        # After the loop, render any remaining list items
        if current_list_items:
            html_output.append('<ul class="list-disc list-inside space-y-2 text-gray-300 mb-4 pl-4">')
            for item in current_list_items:
                html_output.append(f'<li>{item}</li>')
            html_output.append('</ul>')

        return "".join(html_output)

    def _render_ddl(self, tool_result: dict, index: int) -> str:
        if not isinstance(tool_result, dict) or "results" not in tool_result: return ""
        results = tool_result.get("results")
        if not isinstance(results, list) or not results: return ""
        ddl_text = results[0].get('Request Text', 'DDL not available.')
        ddl_text_sanitized = ddl_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        metadata = tool_result.get("metadata", {})
        table_name = metadata.get("table", "DDL")
        self.processed_data_indices.add(index)
        return f"""
        <div class="response-card">
            <div class="sql-code-block">
                <div class="sql-header">
                    <span>SQL DDL: {table_name}</span>
                    <button class="copy-button" onclick="copyToClipboard(this)">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M4 1.5H3a2 2 0 0 0-2 2V14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V3.5a2 2 0 0 0-2-2h-1v1h1a1 1 0 0 1 1 1V14a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1V3.5a1 1 0 0 1 1-1h1v-1z"/><path d="M9.5 1a.5.5 0 0 1 .5.5v1a.5.5 0 0 1-.5.5h-3a.5.5 0 0 1-.5-.5v-1a.5.5 0 0 1 .5-.5h3zM-1 7a.5.5 0 0 1 .5-.5h15a.5.5 0 0 1 0 1H-.5A.5.5 0 0 1-1 7z"/></svg>
                        Copy
                    </button>
                </div>
                <pre><code class="language-sql">{ddl_text_sanitized}</code></pre>
            </div>
        </div>
        """

    def _render_table(self, tool_result: dict, index: int, default_title: str) -> str:
        if not isinstance(tool_result, dict) or "results" not in tool_result: return ""
        results = tool_result.get("results")
        if not isinstance(results, list) or not results or not all(isinstance(item, dict) for item in results): return ""
        
        metadata = tool_result.get("metadata", {})
        # --- MODIFIED: Prioritize metadata.tool_name, then check for a 'response' key, then use default. ---
        title = metadata.get("tool_name", default_title)
        
        # If the result is from a CoreLLMTask (identified by a 'response' key in results[0])
        # and it's still using a generic title, try to make it more descriptive.
        if "response" in results[0] and title == default_title:
            # Attempt to extract a more meaningful title from the response content
            # For example, if the response starts with "# Business Description: TableName"
            first_line_match = re.match(r'#\s*(.*?)(?:\n|$)', results[0]["response"])
            if first_line_match:
                title = first_line_match.group(1).strip()
            else:
                title = "LLM Generated Content" # Fallback if no clear title in response

        headers = results[0].keys()
        
        table_data_json = json.dumps(results)

        html = f"""
        <div class="response-card">
            <div class="flex justify-between items-center mb-2">
                <h4 class="text-lg font-semibold text-white">Data: Result for <code>{title}</code></h4>
                <button class="copy-button" onclick="copyTableToClipboard(this)" data-table='{table_data_json}'>
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M4 1.5H3a2 2 0 0 0-2 2V14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V3.5a2 2 0 0 0-2-2h-1v1h1a1 1 0 0 1 1 1V14a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1V3.5a1 1 0 0 1 1-1h1v-1z"/><path d="M9.5 1a.5.5 0 0 1 .5.5v1a.5.5 0 0 1-.5.5h-3a.5.5 0 0 1-.5-.5v-1a.5.5 0 0 1 .5-.5h3zM-1 7a.5.5 0 0 1 .5-.5h15a.5.5 0 0 1 0 1H-.5A.5.5 0 0 1-1 7z"/></svg>
                        Copy Table
                    </button>
                </div>
            <div class='table-container'>
                <table class='assistant-table'>
                    <thead><tr>{''.join(f'<th>{h}</th>' for h in headers)}</tr></thead>
                    <tbody>
        """
        for row in results:
            html += "<tr>"
            for header in headers:
                cell_data = str(row.get(header, ''))
                sanitized_cell = cell_data.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                html += f"<td>{sanitized_cell}</td>"
            html += "</tr>"
        html += "</tbody></table></div></div>"
        self.processed_data_indices.add(index)
        return html
        
    def _render_chart_with_details(self, chart_data: dict, table_data: dict, chart_index: int, table_index: int) -> str:
        chart_id = f"chart-render-target-{uuid.uuid4()}"
        chart_spec_json = json.dumps(chart_data.get("spec", {}))
        
        table_html = "" 
        results = table_data.get("results")
        if isinstance(results, list) and results and all(isinstance(item, dict) for item in results):
            headers = results[0].keys()
            table_data_json = json.dumps(results)
            
            table_html += f"""
            <div class="flex justify-between items-center mt-4 mb-2">
                <h5 class="text-md font-semibold text-white">Chart Data</h5>
                <button class="copy-button" onclick="copyTableToClipboard(this)" data-table='{table_data_json}'>
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M4 1.5H3a2 2 0 0 0-2 2V14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V3.5a2 2 0 0 0-2-2h-1v1h1a1 1 0 0 1 1 1V14a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1V3.5a1 1 0 0 1 1-1h1v-1z"/><path d="M9.5 1a.5.5 0 0 1 .5.5v1a.5.5 0 0 1-.5.5h-3a.5.5 0 0 1-.5-.5v-1a.5.5 0 0 1 .5-.5h3zM-1 7a.5.5 0 0 1 .5-.5h15a.5.5 0 0 1 0 1H-.5A.5.5 0 0 1-1 7z"/></svg>
                    Copy Table
                </button>
            </div>
            """
            
            table_html += "<div class='table-container'><table class='assistant-table'><thead><tr>"
            table_html += ''.join(f'<th>{h}</th>' for h in headers)
            table_html += "</tr></thead><tbody>"
            for row in results:
                table_html += "<tr>"
                for header in headers:
                    cell_data = str(row.get(header, ''))
                    sanitized_cell = cell_data.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    table_html += f"<td>{sanitized_cell}</td>"
                table_html += "</tr>"
            table_html += "</tbody></table></div>"

        self.processed_data_indices.add(chart_index)
        self.processed_data_indices.add(table_index)

        return f"""
        <div class="response-card">
            <div id="{chart_id}" class="chart-render-target" data-spec='{chart_spec_json}'></div>
            <details class="mt-4">
                <summary class="text-sm font-semibold text-gray-400 cursor-pointer hover:text-white">Show Details</summary>
                {table_html}
            </details>
        </div>
        """

    def _format_workflow_summary(self) -> str:
        """
        A specialized formatter to render the results of a multi-step workflow
        that produces structured, grouped data.
        """
        sanitized_summary = self._sanitize_summary()
        html = f"<div class='response-card summary-card'>{sanitized_summary}</div>"
        
        # Ensure collected_data is treated as a list or converted for rendering
        if isinstance(self.collected_data, dict) and self.collected_data:
            data_to_process = self.collected_data
        elif isinstance(self.collected_data, list) and self.collected_data:
            data_to_process = {"Overall Workflow Results": self.collected_data}
        else:
            return html

        for context_key, data_items in data_to_process.items():
            display_key = context_key.replace(">", "&gt;")
            html += f"<details class='response-card bg-white/5 open:pb-4 mb-4 rounded-lg border border-white/10'><summary class='p-4 font-bold text-xl text-white cursor-pointer hover:bg-white/10 rounded-t-lg'>Report for: <code>{display_key}</code></summary><div class='px-4'>"
            
            for i, item in enumerate(data_items):
                # Handle lists of results (e.g., from column iteration)
                if isinstance(item, list) and item and isinstance(item[0], dict):
                    combined_results = []
                    metadata = {}
                    for sub_item in item:
                        if isinstance(sub_item, dict) and sub_item.get('status') == 'success':
                            if not metadata: metadata = sub_item.get("metadata", {})
                            combined_results.extend(sub_item.get("results", []))
                    
                    if combined_results:
                        table_to_render = {"results": combined_results, "metadata": metadata}
                        html += self._render_table(table_to_render, i, "Column Iteration Result")
                    elif any(isinstance(sub_item, dict) and (sub_item.get('status') == 'skipped' or sub_item.get('status') == 'error') for sub_item in item):
                        html += f"<div class='response-card'><p class='text-sm text-gray-400 italic'>No data results for '{display_key}' due to skipped or errored sub-steps.</p></div>"
                    continue

                # Handle individual dictionary items (e.g., direct tool results, business descriptions)
                if isinstance(item, dict):
                    tool_name = item.get("metadata", {}).get("tool_name")
                    if item.get("type") == "business_description":
                        html += f"<div class='response-card'><h4 class='text-lg font-semibold text-white mb-2'>Business Description</h4><p class='text-gray-300'>{item.get('description')}</p></div>"
                    elif tool_name == 'base_tableDDL':
                        html += self._render_ddl(item, i)
                    elif "results" in item:
                        html += self._render_table(item, i, f"Result for {tool_name}")
                    elif item.get("status") == "skipped":
                        html += f"<div class='response-card'><p class='text-sm text-gray-400 italic'>Skipped Step: <strong>{tool_name or 'N/A'}</strong>. Reason: {item.get('reason')}</p></div>"
                    elif item.get("status") == "error":
                        html += f"<div class='response-card'><p class='text-sm text-red-400 italic'>Error in Step: <strong>{tool_name or 'N/A'}</strong>. Details: {item.get('error_message', item.get('data', ''))}</p></div>"
            html += "</div></details>"

        return html

    def render(self) -> str:
        """
        Main rendering method. It decides which formatting strategy to use
        based on whether the execution was a workflow or a standard query.
        """
        if self.is_workflow:
            return self._format_workflow_summary()

        final_html = ""
        data_source = self.collected_data if isinstance(self.collected_data, list) else []

        # 1. Always render the summary first.
        clean_summary_html = self._sanitize_summary()
        if clean_summary_html:
            final_html += f'<div class="response-card summary-card">{clean_summary_html}</div>'

        # 2. Identify and render all charts next.
        charts = []
        for i, tool_result in enumerate(data_source):
            if isinstance(tool_result, dict) and tool_result.get("type") == "chart":
                charts.append((i, tool_result))
        
        for i, chart_result in charts:
            table_data_result = data_source[i-1] if i > 0 else None
            if table_data_result and isinstance(table_data_result, dict) and "results" in table_data_result:
                final_html += self._render_chart_with_details(chart_result, table_data_result, i, i-1)
            else:
                chart_id = f"chart-render-target-{uuid.uuid4()}"
                chart_spec_json = json.dumps(chart_result.get("spec", {}))
                final_html += f"""
                <div class="response-card">
                    <div id="{chart_id}" class="chart-render-target" data-spec='{chart_spec_json}'></div>
                </div>
                """
                self.processed_data_indices.add(i)

        # 3. Render all remaining data tables and DDLs last.
        for i, tool_result in enumerate(data_source):
            if i in self.processed_data_indices or not isinstance(tool_result, dict):
                continue
            
            metadata = tool_result.get("metadata", {})
            tool_name = metadata.get("tool_name")

            if tool_name == 'base_tableDDL':
                final_html += self._render_ddl(tool_result, i)
            elif "results" in tool_result:
                 final_html += self._render_table(tool_result, i, tool_name or "Result")

        if not final_html.strip():
            return "<p>The agent completed its work but did not produce a visible output.</p>"
            
        return final_html
