#!/usr/bin/env python3
"""Export notebook to a beautifully styled HTML for PDF printing.

Usage:
  python3 export_html.py [notebook.ipynb] [--no-execute]

By default the notebook is executed in-place (outputs refreshed) before
exporting so that newly-added chart cells (e.g. chart_4a_boxplot, etc.)
produce their PNG outputs in the HTML.  Pass --no-execute to skip execution
and use whatever outputs are already stored.
"""
import json, sys, re, html as html_mod
import markdown

# ── CLI args ──
args = sys.argv[1:]
no_execute = "--no-execute" in args
args = [a for a in args if not a.startswith("--")]
nb_path = args[0] if args else "eda_report_clean.ipynb"
out_path = nb_path.replace(".ipynb", ".html")

# ── Optionally execute the notebook ──
if not no_execute:
    try:
        import nbformat
        from nbconvert.preprocessors import ExecutePreprocessor

        print(f"⏳ Executing notebook: {nb_path}  (this may take ~30 s) …")
        with open(nb_path, encoding="utf-8") as f:
            nb_node = nbformat.read(f, as_version=4)

        ep = ExecutePreprocessor(
            timeout=300,          # 5-minute timeout per cell
            kernel_name="python3",
            allow_errors=True,    # don't abort on cell errors
        )
        ep.preprocess(nb_node, {"metadata": {"path": "."}})

        # Write executed notebook back so Jupyter also shows the outputs
        with open(nb_path, "w", encoding="utf-8") as f:
            nbformat.write(nb_node, f)
        print(f"✅ Notebook executed and saved: {nb_path}")

        # Convert nbformat node → plain dict so the rest of the script works
        nb = json.loads(nbformat.writes(nb_node))

    except Exception as exc:
        print(f"⚠️  Could not execute notebook ({exc}). Falling back to stored outputs.")
        nb = json.load(open(nb_path, encoding="utf-8"))
else:
    print(f"ℹ️  Skipping execution (--no-execute). Using stored outputs.")
    nb = json.load(open(nb_path, encoding="utf-8"))

# ── CSS ──
CSS = """
@import url("https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap");
body {
  font-family: Inter, -apple-system, BlinkMacSystemFont, sans-serif;
  max-width: 860px; margin: 40px auto; padding: 0 24px;
  color: #1a1a2e; line-height: 1.7; font-size: 14px; background: #fff;
}
h1 { font-size: 26px; font-weight: 700; border-bottom: 3px solid #2563eb; padding-bottom: 10px; }
h2 { font-size: 19px; font-weight: 700; margin-top: 32px; border-bottom: 2px solid #e2e8f0; padding-bottom: 8px; }
h3 { font-size: 14px; font-weight: 600; color: #0891b2; margin-top: 20px; }
table { width: 100%; border-collapse: collapse; margin: 14px 0; font-size: 12.5px; }
th {
  background: #f1f5f9; color: #2563eb; font-size: 10px;
  letter-spacing: .5px; text-transform: uppercase;
  padding: 8px 10px; text-align: left; border-bottom: 2px solid #e2e8f0;
}
td { padding: 7px 10px; border-bottom: 1px solid #e2e8f0; color: #475569; }
td:first-child { color: #1a1a2e; font-weight: 500; }
blockquote {
  border-left: 3px solid #2563eb; background: #eff6ff;
  padding: 10px 14px; margin: 14px 0; border-radius: 0 6px 6px 0;
  font-size: 12.5px; color: #1e40af;
}
code { background: #f1f5f9; padding: 1px 5px; border-radius: 3px; font-size: 11.5px; }
pre { background: #f1f5f9; padding: 14px; border-radius: 6px; font-size: 11.5px;
      border: 1px solid #e2e8f0; overflow-x: auto; white-space: pre-wrap; word-wrap: break-word; }
pre code { background: none; padding: 0; }
hr { border: none; border-top: 1px solid #e2e8f0; margin: 28px 0; }
img { max-width: 100%; height: auto; margin: 14px auto; display: block; }
p { color: #475569; margin: 8px 0; }
strong { color: #1a1a2e; }
ul, ol { color: #475569; margin: 8px 0; padding-left: 24px; }
li { margin: 4px 0; line-height: 1.6; }
em { font-style: italic; }
.cell-output-text {
  background: #f8fafc; padding: 12px 14px; border-radius: 6px;
  border: 1px solid #e2e8f0; margin: 10px 0; font-size: 11.5px;
  font-family: 'Courier New', Courier, monospace; color: #334155;
  white-space: pre-wrap; word-wrap: break-word; overflow-x: auto;
}
.cell-output-html {
  margin: 10px 0; overflow-x: auto;
}
.cell-output-html table {
  font-size: 12px; border-collapse: collapse; width: 100%;
}
.cell-output-html th, .cell-output-html td {
  padding: 6px 10px; border: 1px solid #e2e8f0; text-align: left;
}
.cell-output-html th {
  background: #f1f5f9; font-weight: 600;
}
@media print {
  body { max-width: 100%; margin: 0; padding: 20px; }
  img { page-break-inside: avoid; }
  h2 { page-break-before: auto; }
}
"""

parts = [
    '<!DOCTYPE html><html><head><meta charset="utf-8">',
    f'<title>Mule Account Detection - EDA Report</title>',
    f'<style>{CSS}</style>',
    '</head><body>'
]

for cell in nb["cells"]:
    if cell["cell_type"] == "markdown":
        src = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
        parts.append(markdown.markdown(src, extensions=["tables", "fenced_code"]))

    elif cell["cell_type"] == "code":
        for o in cell.get("outputs", []):
            output_type = o.get("output_type", "")

            # Handle display_data and execute_result (both have 'data' dict)
            if output_type in ("display_data", "execute_result"):
                data = o.get("data", {})
                # Prefer image/png (matplotlib charts)
                img = data.get("image/png", "")
                if img:
                    # img may already be a plain string or a list of strings
                    if isinstance(img, list):
                        img = "".join(img)
                    parts.append(f'<img src="data:image/png;base64,{img.strip()}"/>')
                # Then text/html (e.g. pandas DataFrames)
                elif "text/html" in data:
                    html_content = data["text/html"]
                    if isinstance(html_content, list):
                        html_content = "".join(html_content)
                    parts.append(f'<div class="cell-output-html">{html_content}</div>')
                # Fallback to text/plain
                elif "text/plain" in data:
                    text = data["text/plain"]
                    if isinstance(text, list):
                        text = "".join(text)
                    escaped = html_mod.escape(text)
                    parts.append(f'<div class="cell-output-text">{escaped}</div>')

            # Handle stream outputs (print() calls)
            elif output_type == "stream":
                text = o.get("text", "")
                if isinstance(text, list):
                    text = "".join(text)
                escaped = html_mod.escape(text)
                parts.append(f'<div class="cell-output-text">{escaped}</div>')

            # Handle error outputs
            elif output_type == "error":
                traceback_lines = o.get("traceback", [])
                text = "\n".join(traceback_lines)
                # Strip ANSI escape codes
                text = re.sub(r'\x1b\[[0-9;]*m', '', text)
                escaped = html_mod.escape(text)
                parts.append(f'<div class="cell-output-text" style="color:#dc2626;">{escaped}</div>')

parts.append("</body></html>")

with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(parts))

print(f"✅ Exported: {out_path}")
print(f"   Open in Chrome → ⌘P → Save as PDF")
