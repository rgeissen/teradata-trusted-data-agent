# trusted_data_agent/agent/formatter.py
import re
import uuid
import json

class OutputFormatter:
    """
    Parses raw LLM output and structured tool data to generate professional,
    failure-safe HTML for the UI.
    """
    def __init__(self, llm_response_text: str, collected_data: list, is_workflow: bool = False):
        self.raw_summary = llm_response_text
        self.collected_data = collected_data
        self.is_workflow = is_workflow
        self.processed_data_indices = set()

        if self.is_workflow:
            try:
                # FIX: Sanitize the response to handle non-standard whitespace before parsing.
                sanitized_response = llm_response_text.replace(u'\xa0', u' ')
                
                # FIX: Robustly extract JSON, ignoring potential markdown wrappers from the LLM.
                json_match = re.search(r"\{.*\}", sanitized_response, re.DOTALL)
                if not json_match:
                    raise json.JSONDecodeError("No JSON object found in the LLM response.", sanitized_response, 0)
                
                report_data = json.loads(json_match.group(0))
                self.raw_summary = report_data.get("summary", "Workflow completed, but no summary was generated.")
                self.collected_data = report_data.get("structured_data", collected_data)
            except (json.JSONDecodeError, TypeError):
                # Fallback if the LLM fails to produce valid JSON
                self.raw_summary = "The agent finished the workflow but failed to generate a structured report. Displaying raw data instead."


    def _has_renderable_tables(self) -> bool:
        """Checks if there is any data that will be rendered as a table."""
        for item in self.collected_data:
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
        
        # Check for markdown tables and decide whether to add a placeholder.
        markdown_table_pattern = re.compile(r"\|.*\|[\n\r]*\|[-| :]*\|[\n\r]*(?:\|.*\|[\n\r]*)*", re.MULTILINE)
        if markdown_table_pattern.search(clean_summary):
            replacement_text = "\n(Data table is shown below)\n" if self._has_renderable_tables() else ""
            clean_summary = re.sub(markdown_table_pattern, replacement_text, clean_summary)

        # Remove DDL blocks, as they are rendered by a dedicated function.
        sql_ddl_pattern = re.compile(r"```sql\s*CREATE MULTISET TABLE.*?;?\s*```|CREATE MULTISET TABLE.*?;", re.DOTALL | re.IGNORECASE)
        clean_summary = re.sub(sql_ddl_pattern, "\n(Formatted DDL shown below)\n", clean_summary)
        
        lines = clean_summary.strip().split('\n')
        html_output = ""
        in_list = False

        def process_line_markdown(line):
            line = re.sub(r'\*{2,3}(.*?):\*{1,3}', r'<strong>\1:</strong>', line)
            line = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', line)
            line = re.sub(r'`(.*?)`', r'<code class="bg-gray-900/70 text-teradata-orange rounded-md px-1.5 py-0.5 font-mono text-sm">\1</code>', line)
            return line

        for line in lines:
            line = line.strip()
            if not line:
                if in_list:
                    html_output += '</ul>'
                    in_list = False
                continue

            if line.startswith(('* ', '- ')):
                if not in_list:
                    html_output += '<ul class="list-disc list-inside space-y-2 text-gray-300 mb-4">'
                    in_list = True
                content = line[2:]
                processed_content = process_line_markdown(content)
                html_output += f'<li>{processed_content}</li>'
            elif line.startswith('# '):
                if in_list: html_output += '</ul>'; in_list = False
                content = line[2:]
                html_output += f'<h3 class="text-xl font-bold text-white mb-3 border-b border-gray-700 pb-2">{content}</h3>'
            elif line.startswith('## '):
                if in_list: html_output += '</ul>'; in_list = False
                content = line[3:]
                html_output += f'<h4 class="text-lg font-semibold text-white mt-4 mb-2">{content}</h4>'
            else:
                if in_list:
                    html_output += '</ul>'
                    in_list = False
                processed_line = process_line_markdown(line)
                html_output += f'<p class="text-gray-300 mb-4">{processed_line}</p>'
        
        if in_list:
            html_output += '</ul>'
            
        return html_output

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

    # --- MODIFIED: Added a header with a copy button to generic tables ---
    def _render_table(self, tool_result: dict, index: int, default_title: str) -> str:
        if not isinstance(tool_result, dict) or "results" not in tool_result: return ""
        results = tool_result.get("results")
        if not isinstance(results, list) or not results or not all(isinstance(item, dict) for item in results): return ""
        
        metadata = tool_result.get("metadata", {})
        title = metadata.get("tool_name", default_title)
        headers = results[0].keys()
        
        # Embed the raw results as a JSON string in a data attribute for the copy function
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
        
    # --- MODIFIED: Added a header with a copy button to the table within chart details ---
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
        A specialized formatter to render the results of any multi-step workflow
        that produces a structured report with collapsible sections.
        """
        data_by_table = {}
        for item in self.collected_data:
            if isinstance(item, list):
                for sub_item in item:
                    if isinstance(sub_item, dict):
                        metadata = sub_item.get("metadata", {})
                        table_name = metadata.get("table_name", metadata.get("table"))
                        if table_name and '.' in table_name: table_name = table_name.split('.')[1]
                        if not table_name: continue
                        if table_name not in data_by_table: data_by_table[table_name] = []
                        data_by_table[table_name].append(sub_item)
                continue
            
            if not isinstance(item, dict): continue
            
            metadata = item.get("metadata", {})
            table_name = metadata.get("table_name", metadata.get("table"))
            if not table_name and item.get("type") == "business_description": table_name = item.get("table_name")
            if table_name and '.' in table_name: table_name = table_name.split('.')[1]
            if not table_name: continue

            if table_name not in data_by_table: data_by_table[table_name] = []
            data_by_table[table_name].append(item)

        sanitized_summary = self._sanitize_summary()
        html = f"<div class='response-card summary-card'>{sanitized_summary}</div>"
        
        for table_name, data in data_by_table.items():
            html += f"<details class='response-card bg-white/5 open:pb-4 mb-4 rounded-lg border border-white/10'><summary class='p-4 font-bold text-xl text-white cursor-pointer hover:bg-white/10 rounded-t-lg'>Report for: <code>{table_name}</code></summary><div class='px-4'>"
            
            for item in data:
                tool_name = item.get("metadata", {}).get("tool_name")
                if item.get("type") == "business_description":
                    html += f"<div class='response-card'><h4 class='text-lg font-semibold text-white mb-2'>Business Description</h4><p class='text-gray-300'>{item.get('description')}</p></div>"
                elif tool_name == 'base_tableDDL':
                    html += self._render_ddl(item, 0)
                elif tool_name == 'qlty_columnSummary':
                    html += self._render_table(item, 0, f"Column Summary for {table_name}")
                elif tool_name == 'qlty_univariateStatistics':
                    if isinstance(item, list):
                        flat_results = [res for col_res in item if isinstance(col_res, dict) and col_res.get('status') == 'success']
                        if flat_results:
                            combined_results = [res for col_res in flat_results for res in col_res.get('results', [])]
                            if combined_results:
                                first_successful_metadata = next((res.get('metadata') for res in flat_results if res.get('metadata')), {})
                                html += self._render_table({"results": combined_results, "metadata": first_successful_metadata}, 0, f"Univariate Statistics for {table_name}")
                    elif item.get('status') == 'success':
                        html += self._render_table(item, 0, f"Univariate Statistics for {table_name}")
                elif item.get("status") == "skipped":
                     html += f"<div class='response-card'><p class='text-sm text-gray-400 italic'>Skipped Step: <strong>{tool_name or 'N/A'}</strong>. Reason: {item.get('reason')}</p></div>"
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

        # 1. Always render the summary first.
        clean_summary_html = self._sanitize_summary()
        if clean_summary_html:
            final_html += f'<div class="response-card summary-card">{clean_summary_html}</div>'

        # 2. Identify and render all charts next.
        charts = []
        chart_source_indices = set()
        for i, tool_result in enumerate(self.collected_data):
            if isinstance(tool_result, dict) and tool_result.get("type") == "chart":
                charts.append((i, tool_result))
                if i > 0:
                    chart_source_indices.add(i - 1)
        
        for i, chart_result in charts:
            table_data_result = self.collected_data[i-1] if i > 0 else None
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
        for i, tool_result in enumerate(self.collected_data):
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