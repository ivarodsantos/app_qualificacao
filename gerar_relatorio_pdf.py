"""
Módulo para geração de relatórios PDF a partir dos dados filtrados do app.
Versão 2.0 com identidade visual do Ceará Sem Fome
"""

from fpdf import FPDF
from datetime import datetime
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo
import matplotlib.pyplot as plt
from io import BytesIO
import os
import unicodedata


def sanitize_text(text):
    """
    Remove acentos e caracteres especiais para compatibilidade com FPDF.
    Converte texto Unicode para ASCII removendo acentuações.
    """
    if text is None or text == '':
        return ''
    
    # Converter para string se não for
    text = str(text)
    
    # Normalizar Unicode (NFD = decompor caracteres acentuados)
    nfd = unicodedata.normalize('NFD', text)
    
    # Remover marcas diacríticas (acentos)
    ascii_text = ''.join(char for char in nfd if unicodedata.category(char) != 'Mn')
    
    # Substituir alguns caracteres problemáticos remanescentes
    replacements = {
        'ç': 'c', 'Ç': 'C',
        'ñ': 'n', 'Ñ': 'N',
        'ß': 'ss',
        'º': 'o', 'ª': 'a',
    }
    
    for old, new in replacements.items():
        ascii_text = ascii_text.replace(old, new)
    
    # Manter apenas caracteres ASCII imprimíveis
    safe_text = ''.join(char for char in ascii_text if ord(char) < 128 and ord(char) >= 32)
    
    return safe_text


# Cores da identidade visual Ceará Sem Fome
class Cores:
    AZUL = (91, 155, 213)  # #5B9BD5
    AMARELO = (255, 192, 0)  # #FFC000
    VERDE = (146, 208, 80)  # #92D050
    VERMELHO = (192, 0, 0)  # #C00000
    BEGE = (250, 248, 240)  # #FAF8F0 - fundo
    CINZA_ESCURO = (95, 95, 95)  # #5F5F5F
    AZUL_ESCURO = (0, 51, 102)  # #003366 - original
    BRANCO = (255, 255, 255)


class RelatorioPDF(FPDF):
    """Classe para geração de relatórios PDF personalizados."""
    
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        
    def header(self):
        """Cabeçalho do PDF com logos."""
        # Fundo bege claro
        self.set_fill_color(*Cores.BEGE)
        self.rect(0, 0, 210, 55, 'F')
        
        # Logo Ceará Sem Fome (esquerda) - subir um pouco
        logo_csf = 'icons/neg_color.png'
        if os.path.exists(logo_csf):
            self.image(logo_csf, 10, 5, 60)
        
        # Logo Governo do Ceará (direita) - subir um pouco
        logo_gov = 'icons/govce_hor_pos.png'
        if os.path.exists(logo_gov):
            self.image(logo_gov, 150, 7, 50)
        
        # Título centralizado com mais destaque - baixar mais
        self.set_y(35)
        self.set_font('Arial', 'B', 16)
        self.set_text_color(*Cores.VERMELHO)
        self.cell(0, 8, sanitize_text('RELATORIO DE ANALISE DE DADOS'), 0, 1, 'C')
        
        # Linha separadora grossa
        self.ln(3)
        self.set_draw_color(*Cores.AZUL)
        self.set_line_width(1.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.set_line_width(0.2)  # Reset
        self.ln(8)
        
    def footer(self):
        """Rodapé do PDF."""
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(*Cores.CINZA_ESCURO)
        self.cell(0, 10, f'Página {self.page_no()}', 0, 0, 'C')
        
    def adicionar_metadados(self, filtros_aplicados):
        """Adiciona informações sobre quando e com quais filtros o relatório foi gerado."""
        self.set_font('Arial', '', 9)
        self.set_text_color(*Cores.CINZA_ESCURO)
        
        # Data e hora de geração
        agora = datetime.now().strftime('%d/%m/%Y às %H:%M:%S')
        self.cell(0, 5, f'Gerado em: {agora}', 0, 1)
        
        # Filtros aplicados
        if filtros_aplicados:
            self.ln(2)
            self.set_font('Arial', 'B', 10)
            self.set_text_color(*Cores.AZUL_ESCURO)
            self.cell(0, 5, 'Filtros Aplicados:', 0, 1)
            
            self.set_font('Arial', '', 8)
            self.set_text_color(*Cores.CINZA_ESCURO)
            for chave, valor in filtros_aplicados.items():
                if valor:
                    if isinstance(valor, list):
                        valor_str = ', '.join(map(str, valor[:3]))
                        if len(valor) > 3:
                            valor_str += f' (+{len(valor)-3} outros)'
                    else:
                        valor_str = str(valor)
                    # Sanitizar texto antes de adicionar
                    self.cell(0, 4, f'  . {sanitize_text(chave)}: {sanitize_text(valor_str)}', 0, 1)
        else:
            self.set_text_color(*Cores.VERDE)
            self.set_font('Arial', 'B', 9)
            self.cell(0, 5, 'Nenhum filtro aplicado (dados completos)', 0, 1)
        
        self.ln(5)
        
    def rounded_rect(self, x, y, w, h, r, style=''):
        """Desenha um retângulo com cantos arredondados."""
        k = self.k
        hp = self.h
        
        if style == 'F':
            op = 'f'
        elif style == 'FD' or style == 'DF':
            op = 'B'
        else:
            op = 'S'
        
        self._out(f'{(x+r)*k:.2f} {(hp-y)*k:.2f} m')
        
        # Linha do topo
        self._out(f'{(x+w-r)*k:.2f} {(hp-y)*k:.2f} l')
        # Canto superior direito
        self._arc(x+w-r, y+r, r, 270, 360)
        # Linha direita
        self._out(f'{(x+w)*k:.2f} {(hp-y-h+r)*k:.2f} l')
        # Canto inferior direito
        self._arc(x+w-r, y+h-r, r, 0, 90)
        # Linha inferior
        self._out(f'{(x+r)*k:.2f} {(hp-y-h)*k:.2f} l')
        # Canto inferior esquerdo
        self._arc(x+r, y+h-r, r, 90, 180)
        # Linha esquerda
        self._out(f'{(x)*k:.2f} {(hp-y-r)*k:.2f} l')
        # Canto superior esquerdo
        self._arc(x+r, y+r, r, 180, 270)
        
        self._out(f'{op}')
    
    def _arc(self, x, y, r, a1, a2):
        """Desenha um arco de círculo."""
        import math
        k = self.k
        hp = self.h
        
        a1 = math.radians(a1)
        a2 = math.radians(a2)
        
        x1 = x + r * math.cos(a1)
        y1 = y + r * math.sin(a1)
        x2 = x + r * math.cos(a2)
        y2 = y + r * math.sin(a2)
        
        self._out(f'{x2*k:.2f} {(hp-y2)*k:.2f} l')
    
    def adicionar_kpis_coloridos(self, kpis):
        """Adiciona seção de KPIs principais com boxes coloridos e cantos arredondados."""
        self.set_font('Arial', 'B', 12)
        self.set_text_color(*Cores.AZUL_ESCURO)
        self.cell(0, 7, 'Indicadores Principais', 0, 1)
        self.ln(3)
        
        # KPIs em grid 2x3 (incluindo taxa de conclusão)
        kpi_width = 90
        kpi_height = 22
        spacing = 10
        x_start = 15
        y_start = self.get_y()
        
        kpi_list = [
            ('Total de Turmas', kpis.get('total_turmas', 0), Cores.AZUL),
            ('Vagas Ofertadas', kpis.get('total_vagas', 0), Cores.AMARELO),
            ('Total Inscritos', kpis.get('total_inscritos', 0), Cores.VERDE),
            ('Total Concludentes', kpis.get('total_concludentes', 0), Cores.AZUL),
        ]
        
        for idx, (label, valor, cor) in enumerate(kpi_list):
            col = idx % 2
            row = idx // 2
            
            x = x_start + (col * (kpi_width + spacing))
            y = y_start + (row * (kpi_height + 5))
            
            self.set_xy(x, y)
            
            # Box colorido com cantos arredondados
            self.set_fill_color(*cor)
            self.set_draw_color(*cor)
            self.set_line_width(0.5)
            self.rounded_rect(x, y, kpi_width, kpi_height, 3, 'DF')
            
            # Label em branco
            self.set_xy(x + 3, y + 3)
            self.set_font('Arial', 'B', 9)
            self.set_text_color(*Cores.BRANCO)
            self.cell(kpi_width - 6, 6, sanitize_text(label), 0, 0)
            
            # Valor grande em branco
            self.set_xy(x + 3, y + 11)
            self.set_font('Arial', 'B', 16)
            valor_fmt = f'{valor:,}'.replace(',', '.')
            self.cell(kpi_width - 6, 8, valor_fmt, 0, 0)
        
        # Taxa de conclusão do mesmo tamanho
        if 'taxa_conclusao' in kpis:
            taxa = kpis['taxa_conclusao']
            
            # Posição: terceira linha, primeira coluna
            col = 0
            row = 2
            x = x_start + (col * (kpi_width + spacing))
            y = y_start + (row * (kpi_height + 5))
            
            self.set_xy(x, y)
            
            # Box vermelho com cantos arredondados
            self.set_fill_color(*Cores.VERMELHO)
            self.set_draw_color(*Cores.VERMELHO)
            self.rounded_rect(x, y, kpi_width, kpi_height, 3, 'DF')
            
            # Label
            self.set_xy(x + 3, y + 3)
            self.set_font('Arial', 'B', 9)
            self.set_text_color(*Cores.BRANCO)
            self.cell(kpi_width - 6, 6, 'Taxa de Conclusao', 0, 0)
            
            # Valor
            self.set_xy(x + 3, y + 11)
            self.set_font('Arial', 'B', 16)
            self.cell(kpi_width - 6, 8, f'{taxa:.2f}%', 0, 0)
        
        self.set_y(y_start + (3 * (kpi_height + 5)) + 3)
        self.ln(5)
        
    def adicionar_tabela(self, titulo, df, colunas=None, max_linhas=15):
        """Adiciona uma tabela minimalista ao PDF."""
        if df.empty:
            return
            
        self.set_font('Arial', 'B', 11)
        self.set_text_color(*Cores.AZUL_ESCURO)
        self.cell(0, 7, sanitize_text(titulo), 0, 1)
        self.ln(2)
        
        if colunas:
            df = df[colunas]
        
        df = df.head(max_linhas)
        
        # Sanitizar nomes das colunas e dados do DataFrame
        df_sanitized = df.copy()
        df_sanitized.columns = [sanitize_text(col) for col in df_sanitized.columns]
        for col in df_sanitized.columns:
            df_sanitized[col] = df_sanitized[col].apply(lambda x: sanitize_text(str(x)))
        
        col_width = 175 / len(df_sanitized.columns)
        
        # Cabeçalho minimalista - apenas texto com fundo sutil
        self.set_font('Arial', 'B', 9)
        self.set_fill_color(240, 248, 255)  # Azul muito claro
        self.set_text_color(*Cores.AZUL_ESCURO)
        self.set_draw_color(200, 200, 200)  # Borda cinza clara
        
        for col in df_sanitized.columns:
            self.cell(col_width, 7, str(col)[:28], 'B', 0, 'L', True)
        self.ln()
        
        # Dados - apenas linha inferior para separação
        self.set_font('Arial', '', 8)
        self.set_text_color(*Cores.CINZA_ESCURO)
        self.set_draw_color(230, 230, 230)  # Borda bem clara
        
        for idx, row in df_sanitized.iterrows():
            # Sem cor de fundo alternada para look mais limpo
            self.set_fill_color(*Cores.BRANCO)
            
            for col_idx, col in enumerate(df_sanitized.columns):
                valor = str(row[col])[:28]
                # Apenas borda inferior para separar linhas
                border = 'B' if idx < len(df_sanitized) - 1 else ''
                self.cell(col_width, 6, valor, border, 0, 'L', False)
            self.ln()
        
        self.ln(5)
    
    def adicionar_grafico(self, img_bytes, titulo=None, largura=180):
        """Adiciona um gráfico (imagem) ao PDF."""
        if titulo:
            self.set_font('Arial', 'B', 11)
            self.set_text_color(*Cores.AZUL_ESCURO)
            self.cell(0, 7, titulo, 0, 1)
            self.ln(2)
        
        # Salvar imagem temporária
        temp_img = 'temp_chart.png'
        with open(temp_img, 'wb') as f:
            f.write(img_bytes)
        
        # Adicionar ao PDF
        self.image(temp_img, x=15, w=largura)
        
        # Remover arquivo temporário
        if os.path.exists(temp_img):
            os.remove(temp_img)
        
        self.ln(5)


def gerar_grafico_barras(df, coluna_x, coluna_y, titulo, cor=Cores.AZUL, top_n=10):
    """
    Gera um gráfico de barras horizontais.
    
    Returns:
        bytes: Imagem PNG do gráfico
    """
    # Preparar dados e sanitizar texto
    df_plot = df.nlargest(top_n, coluna_y).copy()
    df_plot[coluna_x] = df_plot[coluna_x].apply(sanitize_text)
    
    # Criar figura
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Gráfico de barras
    bars = ax.barh(df_plot[coluna_x], df_plot[coluna_y], 
                   color=[c/255 for c in cor])
    
    # Adicionar valores no final das barras
    for i, (idx, row) in enumerate(df_plot.iterrows()):
        valor = row[coluna_y]
        ax.text(valor + max(df_plot[coluna_y]) * 0.01, i, 
                f'{int(valor):,}'.replace(',', '.'),
                va='center', fontsize=9)
    
    # Estilização
    ax.set_xlabel(sanitize_text(coluna_y), fontsize=10, color=[c/255 for c in Cores.CINZA_ESCURO])
    ax.set_title(sanitize_text(titulo), fontsize=12, fontweight='bold', color=[c/255 for c in Cores.AZUL_ESCURO])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Inverter ordem (maior no topo)
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    # Converter para bytes
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    img_bytes = buffer.read()
    plt.close()
    
    return img_bytes


def gerar_grafico_pizza(df, coluna_labels, coluna_valores, titulo, top_n=10):
    """
    Gera um gráfico de pizza/rosca.
    
    Returns:
        bytes: Imagem PNG do gráfico
    """
    # Preparar dados e sanitizar labels
    df_plot = df.nlargest(top_n, coluna_valores).copy()
    df_plot[coluna_labels] = df_plot[coluna_labels].apply(sanitize_text)
    
    # Cores variadas
    cores = [
        [c/255 for c in Cores.AZUL],
        [c/255 for c in Cores.VERDE],
        [c/255 for c in Cores.AMARELO],
        [c/255 for c in Cores.VERMELHO],
        [c/255 for c in (100, 150, 200)],
        [c/255 for c in (200, 150, 100)],
        [c/255 for c in (150, 100, 200)],
        [c/255 for c in (100, 200, 150)],
        [c/255 for c in (200, 100, 150)],
        [c/255 for c in (150, 200, 100)],
    ]
    
    # Criar figura
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Gráfico de rosca
    wedges, texts, autotexts = ax.pie(
        df_plot[coluna_valores],
        labels=df_plot[coluna_labels],
        autopct='%1.1f%%',
        startangle=90,
        colors=cores[:len(df_plot)],
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    
    # Estilização dos textos
    for text in texts:
        text.set_fontsize(9)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(9)
        autotext.set_fontweight('bold')
    
    ax.set_title(sanitize_text(titulo), fontsize=12, fontweight='bold', 
                 color=[c/255 for c in Cores.AZUL_ESCURO], pad=20)
    
    plt.tight_layout()
    
    # Converter para bytes
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    img_bytes = buffer.read()
    plt.close()
    
    return img_bytes


def gerar_relatorio(df_filtrado, kpis, filtros_aplicados=None):
    """
    Gera um relatório PDF completo com os dados filtrados.
    
    Args:
        df_filtrado: DataFrame com os dados filtrados
        kpis: Dicionário com os KPIs principais
        filtros_aplicados: Dicionário com os filtros aplicados (opcional)
    
    Returns:
        bytes: Conteúdo do PDF em bytes para download
    """
    pdf = RelatorioPDF()
    pdf.add_page()
    
    # Página 1: Metadados e KPIs
    pdf.adicionar_metadados(filtros_aplicados or {})
    pdf.adicionar_kpis_coloridos(kpis)
    
    # Página 2: Análise por Cursos
    pdf.add_page()
    
    if not df_filtrado.empty and 'CURSO' in df_filtrado.columns and 'CONCLUDENTES' in df_filtrado.columns:
        # Gráfico de barras
        top_cursos = (
            df_filtrado.groupby('CURSO')['CONCLUDENTES']
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        
        if not top_cursos.empty:
            img_cursos = gerar_grafico_barras(
                top_cursos, 'CURSO', 'CONCLUDENTES',
                'Top 10 Cursos por Concludentes',
                cor=Cores.AZUL
            )
            pdf.adicionar_grafico(img_cursos)
        
        # Tabela complementar
        top_cursos.columns = ['Curso', 'Concludentes']
        pdf.adicionar_tabela('Dados Detalhados', top_cursos)
    
    # Página 3: Análise por Executoras
    pdf.add_page()
    
    if not df_filtrado.empty and 'EXECUTORA' in df_filtrado.columns:
        resumo_exec = (
            df_filtrado.groupby('EXECUTORA')
            .agg({
                'CURSO': 'count',
                'VAGAS OFERTADAS': 'sum',
                'CONCLUDENTES': 'sum'
            })
            .reset_index()
        )
        resumo_exec.columns = ['Executora', 'Turmas', 'Vagas', 'Concludentes']
        
        if not resumo_exec.empty:
            # Gráfico de pizza
            img_exec = gerar_grafico_pizza(
                resumo_exec, 'Executora', 'Concludentes',
                'Distribuição de Concludentes por Executora'
            )
            pdf.adicionar_grafico(img_exec)
        
        # Tabela
        pdf.adicionar_tabela('Resumo por Executora', resumo_exec, max_linhas=20)
    
    # Página 4: Análise por Municípios
    pdf.add_page()
    
    if not df_filtrado.empty and 'Nome_Município' in df_filtrado.columns:
        resumo_mun = (
            df_filtrado.groupby('Nome_Município')
            .agg({
                'CURSO': 'count',
                'CONCLUDENTES': 'sum'
            })
            .sort_values('CONCLUDENTES', ascending=False)
            .head(15)
            .reset_index()
        )
        resumo_mun.columns = ['Município', 'Turmas', 'Concludentes']
        
        if not resumo_mun.empty:
            # Gráfico de barras
            img_mun = gerar_grafico_barras(
                resumo_mun, 'Município', 'Concludentes',
                'Top 15 Municípios por Concludentes',
                cor=Cores.VERDE,
                top_n=15
            )
            pdf.adicionar_grafico(img_mun)
        
        # Tabela
        pdf.adicionar_tabela('Dados Detalhados', resumo_mun)
    
    # Retornar PDF como bytes
    return bytes(pdf.output())
