# trusted_data_agent/agent/formatter.py
import re
import uuid
import json

class OutputFormatter:
    """
    Parses raw LLM output and structured tool data to generate professional,
    failure-safe HTML for the UI.
    """
    def __init__(self, llm_summary_text: str, collected_data: list):
        self.raw_summary = llm_summary_text
        self.collected_data = collected_data
        self.processed_data_indices = set()

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
            # Only add the placeholder if we actually have table data to render.
            # This prevents a confusing placeholder if the LLM hallucinates a table.
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

    def _render_table(self, tool_result: dict, index: int, default_title: str) -> str:
        if not isinstance(tool_result, dict) or "results" not in tool_result: return ""
        results = tool_result.get("results")
        if not isinstance(results, list) or not results or not all(isinstance(item, dict) for item in results): return ""
        metadata = tool_result.get("metadata", {})
        title = metadata.get("table_name", default_title)
        headers = results[0].keys()
        html = f"""
        <div class="response-card">
            <h4 class="text-lg font-semibold text-white mb-2">Data: <code>{title}</code></h4>
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
        
    def _render_chart(self, chart_data: dict, index: int) -> str:
        chart_id = f"chart-render-target-{uuid.uuid4()}"
        # The data-spec attribute will hold the JSON string for the chart
        chart_spec_json = json.dumps(chart_data.get("spec", {}))
        self.processed_data_indices.add(index)
        return f"""
        <div class="response-card">
            <div id="{chart_id}" class="chart-render-target" data-spec='{chart_spec_json}'></div>
        </div>
        """

    def render(self) -> str:
        final_html = ""
        clean_summary_html = self._sanitize_summary()
        if clean_summary_html:
            final_html += f'<div class="response-card summary-card">{clean_summary_html}</div>'

        for i, tool_result in enumerate(self.collected_data):
            if i in self.processed_data_indices or not isinstance(tool_result, dict):
                continue
            
            # Check for chart type first
            if tool_result.get("type") == "chart":
                final_html += self._render_chart(tool_result, i)
                continue

            metadata = tool_result.get("metadata", {})
            tool_name = metadata.get("tool_name")

            if tool_name == 'base_tableDDL':
                final_html += self._render_ddl(tool_result, i)
            elif tool_name and "results" in tool_result:
                 final_html += self._render_table(tool_result, i, f"Result for {tool_name}")

        if not final_html.strip():
            return "<p>The agent completed its work but did not produce a visible output.</p>"
        return final_html
