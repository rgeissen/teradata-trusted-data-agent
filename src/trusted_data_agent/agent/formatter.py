# trusted_data_agent/agent/formatter.py
import re
import uuid
import json

class OutputFormatter:
    """
    Parses raw LLM output and structured tool data to generate professional,
    failure-safe HTML for the UI.
    """
    def __init__(self, llm_response_text: str, collected_data: list | dict, original_user_input: str = None, active_prompt_name: str = None):
        # --- MODIFICATION START: Corrected typo from ll_response_text to llm_response_text ---
        self.raw_summary = llm_response_text
        # --- MODIFICATION END ---
        self.collected_data = collected_data
        self.original_user_input = original_user_input
        self.active_prompt_name = active_prompt_name
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

    def _parse_structured_markdown(self, text: str) -> dict | None:
        """
        Parses a specific markdown format like '***Key:*** value' into a structured dictionary,
        making it robust to formatting errors like missing newlines, commas, and key name variations.
        """
        data = {'columns': []}
        pattern = re.compile(r'\*{3}(.*?):\*{3}\s*`?(.*?)`?(?=\s*(?:- )?\*{3}|$)', re.DOTALL)
        
        matches = pattern.findall(text)
        
        if not matches:
            return None

        header_key_synonyms = {
            'table name': 'table_name',
            'database name': 'database_name',
            'database': 'database_name',
            'description': 'description',
            'table description': 'description',
            'business purpose': 'description'
        }
        
        for key, value in matches:
            key_clean = key.strip().lower()
            value_clean = value.strip()

            canonical_key = header_key_synonyms.get(key_clean)

            if canonical_key:
                data[canonical_key] = value_clean
            elif 'column descriptions' in key_clean or 'columns' in key_clean:
                col_match = re.match(r'^\*+(.*?):\*+\s*(.*)$', value_clean, re.DOTALL)
                if col_match:
                    data['columns'].append({'name': col_match.group(1).strip(), 'description': col_match.group(2).strip()})
            else:
                data['columns'].append({'name': key.strip(), 'description': value_clean})

        if 'table_name' in data or 'description' in data or data.get('columns'):
            return data
        return None

    def _render_structured_report(self, data: dict) -> str:
        """
        Renders the parsed structured data into its inner HTML format, with enhanced visual hierarchy.
        """
        html = f'<p><strong class="text-white">Table Name:</strong> <code class="text-orange-500 font-bold font-mono text-sm">{data.get("table_name", "N/A")}</code></p>'
        html += f'<p><strong class="text-white">Database Name:</strong> <code class="text-orange-500 font-bold font-mono text-sm">{data.get("database_name", "N/A")}</code></p>'
        
        description = data.get("description", "No description provided.")
        if description:
             html += f'<p class="mt-2 text-gray-300">{description}</p>'
        
        if data.get('columns'):
            html += '<hr class="border-gray-600 my-4">'
            html += '<h4 class="text-base font-semibold text-white mb-3">Column Details</h4>'
            html += '<ul class="list-none space-y-3 text-gray-300">'
            for col in data['columns']:
                html += f'<li><strong class="text-white font-mono bg-gray-900/70 rounded-md px-1.5 py-0.5 text-sm">{col.get("name")}:</strong> {col.get("description")}</li>'
            html += '</ul>'

        return f'<div class="response-card bg-white/5 p-4 rounded-lg mb-4">{html}</div>'

    def _render_standard_markdown(self, text: str) -> str:
        """Renders a block of text by processing standard markdown elements, including nested lists."""
        lines = text.strip().split('\n')
        html_output = []
        list_level_stack = []

        def get_indent_level(line_text):
            return len(line_text) - len(line_text.lstrip(' '))

        def process_inline_markdown(text_content):
            text_content = text_content.replace(r'\_', '_')
            text_content = re.sub(r'`(.*?)`', r'<code class="bg-gray-900/70 text-teradata-orange rounded-md px-1.5 py-0.5 font-mono text-sm">\1</code>', text_content)
            text_content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text_content)
            return text_content

        for line in lines:
            stripped_line = line.lstrip(' ')
            current_indent = get_indent_level(line)
            
            list_item_match = re.match(r'^([*-])\s+(.*)$', stripped_line)

            while list_level_stack and (not list_item_match or current_indent < list_level_stack[-1]):
                html_output.append('</ul>')
                list_level_stack.pop()

            if list_item_match:
                if not list_level_stack or current_indent > list_level_stack[-1]:
                    html_output.append('<ul class="list-disc list-outside space-y-2 text-gray-300 mb-4 pl-5">')
                    list_level_stack.append(current_indent)
                
                content = list_item_match.group(2).strip()
                if content:
                    html_output.append(f'<li>{process_inline_markdown(content)}</li>')
            else:
                heading_match = re.match(r'^(#{1,6})\s+(.*)$', stripped_line)
                hr_match = re.match(r'^-{3,}$', stripped_line)

                if heading_match:
                    level = len(heading_match.group(1))
                    content = process_inline_markdown(heading_match.group(2).strip())
                    if level == 1:
                        html_output.append(f'<h2 class="text-xl font-bold text-white mb-3 border-b border-gray-700 pb-2">{content}</h2>')
                    elif level == 2:
                        html_output.append(f'<h3 class="text-lg font-bold text-white mb-3 border-b border-gray-700 pb-2">{content}</h3>')
                    else:
                        html_output.append(f'<h4 class="text-base font-semibold text-white mt-4 mb-2">{content}</h4>')
                elif hr_match:
                    html_output.append('<hr class="border-gray-600 my-4">')
                elif stripped_line:
                    html_output.append(f'<p class="text-gray-300 mb-4">{process_inline_markdown(stripped_line)}</p>')

        while list_level_stack:
            html_output.append('</ul>')
            list_level_stack.pop()

        return "".join(html_output)

    def _sanitize_summary(self) -> str:
        """
        Cleans the LLM's summary text, applying special formatting for simple
        queries to emphasize the direct answer, while being robust to different
        LLM provider outputs. This method is now intended for non-structured,
        ad-hoc query summaries.
        """
        clean_summary = self.raw_summary

        markdown_block_match = re.search(r"```(?:markdown)?\s*\n(.*?)\n\s*```", clean_summary, re.DOTALL)
        if markdown_block_match:
            clean_summary = markdown_block_match.group(1).strip()
        
        markdown_table_pattern = re.compile(r"\|.*\|[\n\r]*\|[-| :]*\|[\n\r]*(?:\|.*\|[\n\r]*)*", re.MULTILINE)
        if markdown_table_pattern.search(clean_summary):
            replacement_text = "\n(Data table is shown below)\n" if self._has_renderable_tables() else ""
            clean_summary = re.sub(markdown_table_pattern, replacement_text, clean_summary)

        sql_ddl_pattern = re.compile(r"```sql\s*CREATE MULTISET TABLE.*?;?\s*```|CREATE MULTISET TABLE.*?;", re.DOTALL | re.IGNORECASE)
        clean_summary = re.sub(sql_ddl_pattern, "\n(Formatted DDL shown below)\n", clean_summary)
        
        summary_to_process = clean_summary
        if self.original_user_input:
            question_for_regex = re.escape(self.original_user_input.strip())
            pattern = re.compile(f"^{question_for_regex}\\s*", re.IGNORECASE)
            
            match = pattern.match(summary_to_process)
            if match:
                summary_to_process = summary_to_process[match.end():]
        
        summary_to_process = summary_to_process.lstrip(': ').strip()

        lines = [line for line in summary_to_process.strip().split('\n') if line.strip()]
        if not lines:
            return ""

        direct_answer_text = ""
        remaining_content = ""
        
        first_line = lines[0]
        heading_match = re.match(r'^(#{1,6})\s+(.*)$', first_line)

        if heading_match and len(lines) > 1:
            direct_answer_text = lines[1].strip()
            remaining_content = '\n'.join(lines[2:])
        else:
            paragraphs = re.split(r'\n\s*\n', summary_to_process.strip())
            direct_answer_text = paragraphs[0].strip()
            remaining_content = '\n\n'.join(paragraphs[1:])
        
        def process_inline_markdown(text_content):
            text_content = text_content.replace(r'\_', '_')
            text_content = re.sub(r'`(.*?)`', r'<code class="bg-gray-900/70 text-teradata-orange rounded-md px-1.5 py-0.5 font-mono text-sm">\1</code>', text_content)
            text_content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text_content)
            return text_content

        styled_answer = f'<p class="text-lg font-semibold text-white mb-4">{process_inline_markdown(direct_answer_text)}</p>'
        remaining_html = self._render_standard_markdown(remaining_content)
        
        final_html = styled_answer
        if remaining_html and remaining_html.strip():
            final_html += '<hr class="border-gray-600 my-4">'
            final_html += remaining_html
        
        return final_html

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
        title = metadata.get("tool_name", default_title)
        
        if "response" in results[0] and title == default_title:
            first_line_match = re.match(r'#\s*(.*?)(?:\n|$)', results[0]["response"])
            if first_line_match:
                title = first_line_match.group(1).strip()
            else:
                title = "LLM Generated Content"

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
                    html += f"<td>{sanitized_cell}</td>"
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

    def _format_workflow_report(self) -> str:
        """
        A specialized formatter to render the results of a multi-step workflow
        that produces structured, grouped data. This is the only path that should
        attempt to render a special "structured report" from the LLM summary.
        """
        sanitized_summary = ""
        parsed_data = self._parse_structured_markdown(self.raw_summary)
        if parsed_data:
            sanitized_summary = self._render_structured_report(parsed_data)
        else:
            sanitized_summary = self._sanitize_summary()

        html = f"<div class='response-card summary-card'>{sanitized_summary}</div>"
        
        if isinstance(self.collected_data, dict) and self.collected_data:
            data_to_process = self.collected_data
        elif isinstance(self.collected_data, list) and self.collected_data:
            data_to_process = {"Execution Report": self.collected_data}
        else:
            return html

        for context_key, data_items in data_to_process.items():
            display_key = context_key.replace("Workflow: ", "").replace(">", "&gt;")
            html += f"<details class='response-card bg-white/5 open:pb-4 mb-4 rounded-lg border border-white/10'><summary class='p-4 font-bold text-lg text-white cursor-pointer hover:bg-white/10 rounded-t-lg'>Report for: <code>{display_key}</code></summary><div class='px-4'>"
            
            for i, item in enumerate(data_items):
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

    def _format_standard_query_report(self) -> str:
        """
        A dedicated formatter for standard (non-workflow) queries. It creates a
        two-part report: a high-level summary followed by a collapsible
        "Execution Report" containing all detailed data tables.
        """
        final_html = ""
        
        clean_summary_html = self._sanitize_summary()
        if clean_summary_html:
            final_html += f'<div class="response-card summary-card">{clean_summary_html}</div>'

        data_source = []
        if isinstance(self.collected_data, dict):
            for item_list in self.collected_data.values():
                data_source.extend(item_list)
        elif isinstance(self.collected_data, list):
            data_source = self.collected_data
            
        if not data_source:
            return final_html

        details_html = ""
        charts = []
        for i, tool_result in enumerate(data_source):
            if isinstance(tool_result, dict) and tool_result.get("type") == "chart":
                charts.append((i, tool_result))
        
        for i, chart_result in charts:
            table_data_result = data_source[i-1] if i > 0 else None
            if table_data_result and isinstance(table_data_result, dict) and "results" in table_data_result:
                details_html += self._render_chart_with_details(chart_result, table_data_result, i, i-1)
            else:
                chart_id = f"chart-render-target-{uuid.uuid4()}"
                chart_spec_json = json.dumps(chart_result.get("spec", {}))
                details_html += f"""
                <div class="response-card">
                    <div id="{chart_id}" class="chart-render-target" data-spec='{chart_spec_json}'></div>
                </div>
                """
                self.processed_data_indices.add(i)

        for i, tool_result in enumerate(data_source):
            if i in self.processed_data_indices or not isinstance(tool_result, dict):
                continue
            
            metadata = tool_result.get("metadata", {})
            tool_name = metadata.get("tool_name")

            if tool_name == 'base_tableDDL':
                details_html += self._render_ddl(tool_result, i)
            elif "results" in tool_result:
                 details_html += self._render_table(tool_result, i, tool_name or "Result")

        if details_html:
            final_html += (
                f"<details class='response-card bg-white/5 open:pb-4 mb-4 rounded-lg border border-white/10'>"
                f"<summary class='p-4 font-bold text-lg text-white cursor-pointer hover:bg-white/10 rounded-t-lg'>Execution Report</summary>"
                f"<div class='px-4'>{details_html}</div>"
                f"</details>"
            )
            
        return final_html

    def render(self) -> str:
        """
        Main rendering method. It now acts as a router, deciding which
        formatting strategy to use based on the execution type.
        """
        if self.active_prompt_name:
            return self._format_workflow_report()
        else:
            return self._format_standard_query_report()