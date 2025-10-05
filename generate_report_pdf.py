#!/usr/bin/env python3
"""
Generate evaluation_report.pdf from technical_report.md using Markdown + WeasyPrint.
"""

from pathlib import Path
import markdown
from weasyprint import HTML
from weasyprint.text.fonts import FontConfiguration

def generate_pdf(markdown_file: Path, output_pdf: Path) -> None:
    """Convert markdown file to styled PDF."""

    # Read markdown content
    md_content = markdown_file.read_text(encoding='utf-8')

    # Convert markdown to HTML
    md = markdown.Markdown(extensions=['tables', 'fenced_code'])
    html_body = md.convert(md_content)

    # Create full HTML document with styling - using system fonts
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>RAG System Evaluation Report</title>
    <style>
        @page {{
            size: letter;
            margin: 1in;
            @top-right {{
                content: "Assignment 2: Ground the Domain RAG";
                font-size: 9pt;
                color: #666;
            }}
            @bottom-center {{
                content: "Page " counter(page);
                font-size: 9pt;
                color: #666;
            }}
        }}

        body {{
            font-family: Arial, Helvetica, sans-serif;
            font-size: 11pt;
            line-height: 1.6;
            color: #000;
        }}

        h1 {{
            font-size: 22pt;
            font-weight: bold;
            color: #000;
            margin-top: 0;
            margin-bottom: 14pt;
            border-bottom: 2pt solid #000;
            padding-bottom: 6pt;
        }}

        h2 {{
            font-size: 16pt;
            font-weight: bold;
            color: #000;
            margin-top: 18pt;
            margin-bottom: 10pt;
            border-bottom: 1pt solid #999;
            padding-bottom: 4pt;
        }}

        h3 {{
            font-size: 13pt;
            font-weight: bold;
            color: #000;
            margin-top: 14pt;
            margin-bottom: 6pt;
        }}

        p {{
            margin: 8pt 0;
        }}

        code {{
            font-family: Courier, monospace;
            font-size: 9pt;
            background-color: #f0f0f0;
            padding: 2pt 4pt;
        }}

        pre {{
            font-family: Courier, monospace;
            font-size: 9pt;
            background-color: #f0f0f0;
            padding: 10pt;
            border-left: 3pt solid #999;
            overflow-x: auto;
            margin: 10pt 0;
        }}

        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 10pt 0;
            font-size: 10pt;
        }}

        th {{
            background-color: #333;
            color: white;
            font-weight: bold;
            padding: 6pt 8pt;
            text-align: left;
            border: 1pt solid #999;
        }}

        td {{
            padding: 5pt 8pt;
            border: 1pt solid #999;
        }}

        tr:nth-child(even) {{
            background-color: #f5f5f5;
        }}

        ul, ol {{
            margin: 8pt 0;
            padding-left: 24pt;
        }}

        li {{
            margin: 3pt 0;
        }}

        strong {{
            font-weight: bold;
        }}

        em {{
            font-style: italic;
        }}

        blockquote {{
            margin: 10pt 0;
            padding-left: 14pt;
            border-left: 3pt solid #999;
            color: #555;
            font-style: italic;
        }}
    </style>
</head>
<body>
{html_body}
</body>
</html>"""

    # Generate PDF with font configuration
    font_config = FontConfiguration()
    HTML(string=html_content).write_pdf(
        output_pdf,
        font_config=font_config
    )

    print(f"✓ Generated: {output_pdf}")
    print(f"  Source: {markdown_file}")
    print(f"  Size: {output_pdf.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    # Paths
    docs_dir = Path(__file__).parent / "docs"
    output_dir = Path(__file__).parent

    # Generate evaluation_report.pdf from technical_report.md
    markdown_file = docs_dir / "technical_report.md"
    output_pdf = output_dir / "evaluation_report.pdf"

    if not markdown_file.exists():
        print(f"✗ Error: {markdown_file} not found")
        exit(1)

    print("=" * 60)
    print("GENERATING EVALUATION REPORT PDF")
    print("=" * 60)
    print()

    generate_pdf(markdown_file, output_pdf)

    print()
    print("=" * 60)
    print("PDF GENERATION COMPLETE")
    print("=" * 60)
