"""
Script para simplificar geometrias de arquivos GeoJSON
Reduz o tamanho dos arquivos mantendo a qualidade visual
"""

import geopandas as gpd
import json
from pathlib import Path
import sys
import io

# Configurar encoding UTF-8 para o stdout no Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def simplify_geojson(input_file, output_file, tolerance=0.001):
    """
    Simplifica geometrias de um arquivo GeoJSON
    
    Args:
        input_file: caminho do arquivo GeoJSON original
        output_file: caminho para salvar o arquivo simplificado
        tolerance: tolerância de simplificação (menor = mais detalhes, maior = mais simples)
                  Valores típicos: 0.0001 (muito detalhado) a 0.01 (muito simplificado)
    """
    print(f"\n🔄 Processando: {input_file}")
    
    # Verificar se arquivo existe
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"❌ Arquivo não encontrado: {input_file}")
        return False
    
    # Obter tamanho original
    original_size = input_path.stat().st_size / (1024 * 1024)  # MB
    print(f"📏 Tamanho original: {original_size:.2f} MB")
    
    try:
        # Carregar GeoJSON
        print("📥 Carregando arquivo...")
        gdf = gpd.read_file(input_file)
        
        print(f"ℹ️  Features: {len(gdf)}")
        print(f"ℹ️  CRS: {gdf.crs}")
        
        # Simplificar geometrias
        print(f"⚙️  Simplificando com tolerância {tolerance}...")
        gdf['geometry'] = gdf['geometry'].simplify(tolerance, preserve_topology=True)
        
        # Salvar arquivo simplificado
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"💾 Salvando em: {output_file}")
        gdf.to_file(output_file, driver='GeoJSON')
        
        # Obter novo tamanho
        new_size = output_path.stat().st_size / (1024 * 1024)  # MB
        reduction = ((original_size - new_size) / original_size) * 100
        
        print(f"✅ Concluído!")
        print(f"📏 Tamanho novo: {new_size:.2f} MB")
        print(f"📉 Redução: {reduction:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro ao processar arquivo: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Simplifica todos os arquivos GeoJSON pesados do projeto"""
    print("=" * 60)
    print("🗺️  SIMPLIFICAÇÃO DE ARQUIVOS GEOJSON")
    print("=" * 60)
    
    # Arquivos para simplificar
    files_to_simplify = [
        {
            'input': 'data2/municipios_com_qualificacao.geojson',
            'output': 'data2/municipios_com_qualificacao_simplificado.geojson',
            'tolerance': 0.001,
        },
        {
            'input': 'data2/lotes_2025.geojson',
            'output': 'data2/lotes_2025_simplificado.geojson',
            'tolerance': 0.001,
        },
        {
            'input': 'data2/cozinhas_geo_ipece_01122025.geojson',
            'output': 'data2/cozinhas_geo_ipece_01122025_simplificado.geojson',
            'tolerance': 0.0005,  # Menor tolerância para pontos/cozinhas
        },
    ]
    
    results = []
    total_original = 0
    total_simplified = 0
    
    for file_info in files_to_simplify:
        success = simplify_geojson(
            file_info['input'],
            file_info['output'],
            file_info['tolerance']
        )
        results.append((file_info['input'], success))
        
        if success:
            original = Path(file_info['input']).stat().st_size / (1024 * 1024)
            simplified = Path(file_info['output']).stat().st_size / (1024 * 1024)
            total_original += original
            total_simplified += simplified
    
    # Resumo
    print("\n" + "=" * 60)
    print("📊 RESUMO")
    print("=" * 60)
    
    for file_path, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {Path(file_path).name}")
    
    if total_original > 0:
        total_reduction = ((total_original - total_simplified) / total_original) * 100
        print(f"\n📏 Tamanho total original: {total_original:.2f} MB")
        print(f"📏 Tamanho total simplificado: {total_simplified:.2f} MB")
        print(f"📉 Redução total: {total_reduction:.1f}%")
    
    print("\n" + "=" * 60)
    print("💡 PRÓXIMO PASSO:")
    print("Atualizar qualificacao_app2.py para usar os arquivos simplificados:")
    print("  - Trocar 'municipios_com_qualificacao.geojson' por")
    print("    'municipios_com_qualificacao_simplificado.geojson'")
    print("  - E assim por diante para os outros arquivos")
    print("=" * 60)


if __name__ == "__main__":
    main()
