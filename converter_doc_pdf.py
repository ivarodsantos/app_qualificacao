"""
Script para converter documentação Markdown para PDF
"""

from weasyprint import HTML, CSS
from markdown import markdown
import os

# Ler o arquivo markdown
with open('relatorio_implementacao_pdf.md', 'r', encoding='utf-8') as f:
    md_content = f.read()

# Converter Markdown para HTML
html_content = markdown(md_content, extensions=['fenced_code', 'tables'])

# CSS para estilização
css_style = CSS(string='''
    @page {
        size: A4;
        margin: 25mm;
    }
    
    body {
        font-family: Arial, sans-serif;
        font-size: 10pt;
        line-height: 1.5;
        color: #333;
    }
    
    h1 {
        color: #C00000;
        font-size: 18pt;
        border-bottom: 3px solid #5B9BD5;
        padding-bottom: 5px;
        page-break-before: always;
    }
    
    h1:first-of-type {
        page-break-before: avoid;
    }
    
    h2 {
        color: #003366;
        font-size: 14pt;
        margin-top: 15px;
        margin-bottom: 10px;
    }
    
    h3 {
        color: #5B9BD5;
        font-size: 12pt;
        margin-top: 12px;
        margin-bottom: 8px;
    }
    
    code {
        background-color: #f5f5f5;
        padding: 2px 5px;
        border-radius: 3px;
        font-family: 'Courier New', monospace;
        font-size: 9pt;
    }
    
    pre {
        background-color: #f8f8f8;
        border-left: 3px solid #5B9BD5;
        padding: 10px;
        margin: 10px 0;
        overflow-x: auto;
        font-family: 'Courier New', monospace;
        font-size: 8.5pt;
        line-height: 1.3;
    }
    
    pre code {
        background-color: transparent;
        padding: 0;
    }
    
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 10px 0;
        font-size: 9pt;
    }
    
    table th {
        background-color: #5B9BD5;
        color: white;
        padding: 8px;
        text-align: left;
        font-weight: bold;
    }
    
    table td {
        border-bottom: 1px solid #ddd;
        padding: 6px;
    }
    
    ul, ol {
        margin: 8px 0;
        padding-left: 20px;
    }
    
    li {
        margin: 3px 0;
    }
    
    hr {
        border: none;
        border-top: 2px solid #ccc;
        margin: 15px 0;
    }
    
    blockquote {
        border-left: 4px solid #FFC000;
        padding-left: 15px;
        margin: 10px 0;
        color: #666;
        font-style: italic;
    }
''')

# Criar HTML completo
full_html = f'''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Documentação - Implementação PDF</title>
</head>
<body>
    {html_content}
</body>
</html>
'''

# Definir pasta de saída (mesmo com .gitignore vamos tentar)
output_path = 'docs/relatorio_implementacao_pdf.pdf'

# Criar pasta se não existir
os.makedirs('docs', exist_ok=True)

# Gerar PDF
try:
    HTML(string=full_html).write_pdf(output_path, stylesheets=[css_style])
    print(f"✅ PDF gerado com sucesso em: {output_path}")
    print(f"📄 Tamanho: {os.path.getsize(output_path) / 1024:.1f} KB")
except Exception as e:
    print(f"❌ Erro ao gerar PDF: {e}")
    # Tentar salvar em local alternativo
    output_path = 'relatorio_implementacao_pdf.pdf'
    HTML(string=full_html).write_pdf(output_path, stylesheets=[css_style])
    print(f"✅ PDF salvo em local alternativo: {output_path}")
