#!/usr/bin/env python3
"""Export notebook to a beautifully styled HTML for PDF printing."""
import json, sys, markdown

# ── Which notebook to export ──
nb_path = sys.argv[1] if len(sys.argv) > 1 else "eda_report_clean.ipynb"
out_path = nb_path.replace(".ipynb", ".html")

nb = json.load(open(nb_path))

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
      border: 1px solid #e2e8f0; overflow-x: auto; }
pre code { background: none; padding: 0; }
hr { border: none; border-top: 1px solid #e2e8f0; margin: 28px 0; }
img { max-width: 100%; height: auto; margin: 14px 0; display: block; }
p { color: #475569; margin: 8px 0; }
strong { color: #1a1a2e; }
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
            img = o.get("data", {}).get("image/png", "")
            if img:
                parts.append(f'<img src="data:image/png;base64,{img}"/>')

parts.append("</body></html>")

with open(out_path, "w") as f:
    f.write("\n".join(parts))

print(f"✅ Exported: {out_path}")
print(f"   Open in Chrome → ⌘P → Save as PDF")
