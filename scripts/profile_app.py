"""
Script de Profiling para o Aplicativo de Qualificação
Mede o tempo de execução de componentes críticos para identificar gargalos
"""

import time
import pandas as pd
import json
import sys
import io
from pathlib import Path

# Configurar encoding UTF-8 para o stdout no Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Adicionar o diretório do projeto ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

class PerformanceProfiler:
    """Classe para medir e reportar tempos de execução"""
    
    def __init__(self):
        self.timings = {}
        self.current_operation = None
        self.start_time = None
    
    def start(self, operation_name):
        """Inicia medição de uma operação"""
        self.current_operation = operation_name
        self.start_time = time.time()
        print(f"\n⏱️  Iniciando: {operation_name}...")
    
    def end(self):
        """Finaliza medição da operação atual"""
        if self.current_operation and self.start_time:
            elapsed = time.time() - self.start_time
            self.timings[self.current_operation] = elapsed
            print(f"✅ Concluído em {elapsed:.3f}s")
            self.current_operation = None
            self.start_time = None
    
    def report(self):
        """Gera relatório de performance"""
        print("\n" + "="*60)
        print("📊 RELATÓRIO DE PERFORMANCE")
        print("="*60)
        
        # Ordenar por tempo (mais lento primeiro)
        sorted_timings = sorted(self.timings.items(), key=lambda x: x[1], reverse=True)
        
        total_time = sum(self.timings.values())
        
        for operation, elapsed in sorted_timings:
            percentage = (elapsed / total_time) * 100 if total_time > 0 else 0
            
            # Emoji baseado no tempo
            if elapsed > 2:
                icon = "🔴"
            elif elapsed > 1:
                icon = "🟡"
            else:
                icon = "🟢"
            
            print(f"{icon} {operation:.<50} {elapsed:>6.3f}s ({percentage:>5.1f}%)")
        
        print("-"*60)
        print(f"⏱️  TEMPO TOTAL: {total_time:.3f}s")
        print("="*60)
        
        # Identificar top 3 gargalos
        print("\n🎯 TOP 3 GARGALOS:")
        for i, (operation, elapsed) in enumerate(sorted_timings[:3], 1):
            print(f"   {i}. {operation}: {elapsed:.3f}s")
        
        return self.timings

def profile_google_sheets_loading():
    """Perfil de carregamento do Google Sheets"""
    from google_sheets_api import carregar_google_sheet_por_aba
    
    link = "https://docs.google.com/spreadsheets/d/1M2huy5RGW5D28zWRnBiHI4kSGWZNi5ejyygnxQjx7uo/edit?gid=0#gid=0"
    nome_aba = "Compilado"
    intervalo = "A:AN"
    
    df = carregar_google_sheet_por_aba(link, nome_aba, intervalo)
    return df

def profile_csv_loading():
    """Perfil de carregamento de arquivos CSV"""
    files = [
        "data2/planilha de referência dos municipios com codigo do ibge - planilha de referência dos municipios com codigo do ibge.csv",
        "data2/data-1762178638816_kitchen.csv",
        "data2/compilado_trilha_sebrae.csv",
        "data2/compilado_mentoria_sebrae.csv",
    ]
    
    dfs = []
    for file in files:
        try:
            df = pd.read_csv(file, encoding="utf-8")
            dfs.append((file.split('/')[-1], len(df), df.memory_usage(deep=True).sum()))
        except Exception as e:
            print(f"   ⚠️  Erro ao carregar {file}: {e}")
    
    return dfs

def profile_geojson_loading():
    """Perfil de carregamento de arquivos GeoJSON"""
    files = [
        "data/municipios_latlon.geojson",
        "data2/cozinhas_geo_ipece_01122025_simplificado.geojson",
        "data2/municipios_com_qualificacao_simplificado.geojson",
    ]
    
    geojsons = []
    for file in files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                file_size = Path(file).stat().st_size / (1024 * 1024)  # MB
                num_features = len(data.get('features', []))
                geojsons.append((file.split('/')[-1], file_size, num_features))
        except Exception as e:
            print(f"   ⚠️  Erro ao carregar {file}: {e}")
    
    return geojsons

def profile_data_processing(df_compilado):
    """Perfil de processamento de dados"""
    from tratamento_compilado import tratamento_compilado
    
    # Carregar dependências
    df_lotes = pd.read_csv(
        "data2/planilha de referência dos municipios com codigo do ibge - planilha de referência dos municipios com codigo do ibge.csv",
        encoding="utf-8"
    )
    
    df_kitchen = pd.read_csv("data2/data-1762178638816_kitchen.csv")
    df_cozinhas_simp = df_kitchen[['sda_id', 'name']].copy()
    
    # Processar
    cursos_df = tratamento_compilado(df_compilado, df_lotes, df_cozinhas_simp)
    
    return cursos_df

def profile_merge_operations(cursos_df, df_kitchen):
    """Perfil de operações de merge"""
    from merge_id_plataforma import merge_id_plataforma
    
    merged_df = merge_id_plataforma(cursos_df, df_kitchen)
    return merged_df

def profile_filtering(merged_df):
    """Perfil de operações de filtro"""
    # Simular filtros
    filters = [
        ("Filtro por município", merged_df[merged_df['Nome_Município'] == 'Fortaleza']),
        ("Filtro por executora", merged_df[merged_df['EXECUTORA'] == 'Instituto Maria da Hora']),
        ("Filtro combinado", merged_df[(merged_df['Nome_Município'] == 'Fortaleza') & 
                                       (merged_df['EXECUTORA'] == 'Instituto Maria da Hora')]),
    ]
    
    return filters

def main():
    """Executa profiling completo do aplicativo"""
    profiler = PerformanceProfiler()
    
    print("🔍 PROFILING DO APLICATIVO DE QUALIFICAÇÃO")
    print("="*60)
    
    # 1. Google Sheets
    profiler.start("1. Carregamento Google Sheets API")
    df_compilado = profile_google_sheets_loading()
    profiler.end()
    print(f"   ℹ️  Linhas carregadas: {len(df_compilado):,}")
    
    # 2. CSVs
    profiler.start("2. Carregamento de arquivos CSV")
    csv_info = profile_csv_loading()
    profiler.end()
    for name, rows, size_bytes in csv_info:
        size_mb = size_bytes / (1024 * 1024)
        print(f"   ℹ️  {name}: {rows:,} linhas, {size_mb:.2f} MB")
    
    # 3. GeoJSON
    profiler.start("3. Carregamento de arquivos GeoJSON")
    geojson_info = profile_geojson_loading()
    profiler.end()
    for name, size_mb, features in geojson_info:
        print(f"   ℹ️  {name}: {size_mb:.2f} MB, {features:,} features")
    
    # 4. Processamento de dados
    profiler.start("4. Tratamento e processamento de dados")
    cursos_df = profile_data_processing(df_compilado.copy())
    profiler.end()
    print(f"   ℹ️  Dados processados: {len(cursos_df):,} linhas")
    
    # 5. Merge com plataforma
    profiler.start("5. Merge com dados da plataforma")
    df_kitchen = pd.read_csv("data2/data-1762178638816_kitchen.csv")
    merged_df = profile_merge_operations(cursos_df, df_kitchen)
    profiler.end()
    print(f"   ℹ️  Dados após merge: {len(merged_df):,} linhas")
    
    # 6. Operações de filtro
    profiler.start("6. Operações de filtro (3 exemplos)")
    filters = profile_filtering(merged_df)
    profiler.end()
    for filter_name, filtered_df in filters:
        print(f"   ℹ️  {filter_name}: {len(filtered_df):,} linhas resultantes")
    
    # 7. Normalização de dados
    profiler.start("7. Normalização e conversão de tipos")
    # Simular normalizações do app
    test_df = merged_df.copy()
    test_df["STATUS"] = test_df["STATUS"].astype(str).str.strip()
    test_df["EXECUTORA"] = test_df["EXECUTORA"].astype(str).str.strip()
    test_df["Nome_Município"] = test_df["Nome_Município"].astype(str).str.strip()
    profiler.end()
    
    # Gerar relatório
    timings = profiler.report()
    
    # Salvar resultados em arquivo
    results_file = Path(__file__).parent / "performance_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'timings': timings,
            'data_info': {
                'google_sheets_rows': len(df_compilado),
                'processed_rows': len(merged_df),
                'csv_files': len(csv_info),
                'geojson_files': len(geojson_info),
                'total_geojson_size_mb': sum(size for _, size, _ in geojson_info),
            }
        }, f, indent=2)
    
    print(f"\n💾 Resultados salvos em: {results_file}")
    
    # Recomendações
    print("\n💡 RECOMENDAÇÕES:")
    
    # Encontrar operação mais lenta
    slowest = max(timings.items(), key=lambda x: x[1])
    print(f"\n1. PRIORIDADE MÁXIMA: Otimizar '{slowest[0]}' ({slowest[1]:.3f}s)")
    
    if timings.get("1. Carregamento Google Sheets API", 0) > 1:
        print("   → Implementar cache com @st.cache_data")
    
    geojson_time = timings.get("3. Carregamento de arquivos GeoJSON", 0)
    if geojson_time > 0.5:
        total_size = sum(size for _, size, _ in geojson_info)
        print(f"   → Simplificar geometrias GeoJSON (total: {total_size:.1f}MB)")
    
    processing_time = timings.get("4. Tratamento e processamento de dados", 0)
    if processing_time > 1:
        print("   → Adicionar cache em funções de processamento")
    
    print("\n✅ Profiling concluído!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Erro durante profiling: {e}")
        import traceback
        traceback.print_exc()
