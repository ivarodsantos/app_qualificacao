import pandas as pd
import requests
import json
import os
import time

# --- CONFIG ---
API_KEY = "AIzaSyCzCCCEss2JnTKtBsGWwpvCJMFg19svQpU" # User provided key in qualificacao_app2.py
INPUT_FILE = "data2/compilado_novos_lotes_merge_nomes_cozinhas_05122025.csv"
OUTPUT_FILE = "data2/coordenadas_locais.csv"
CACHE_FILE = "data2/coordenadas_locais_cache.json" # To avoid re-fetching
# --------------

def get_lat_lon(address, city):
    """
    Fetch latitude and longitude for a given address and city using Google Geocoding API.
    """
    full_address = f"{address}, {city}, Ceará, Brazil"
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "address": full_address,
        "key": API_KEY
    }
    
    try:
        response = requests.get(base_url, params=params)
        data = response.json()
        
        if data['status'] == 'OK':
            location = data['results'][0]['geometry']['location']
            return location['lat'], location['lng']
        else:
            print(f"Error finding '{full_address}': {data['status']}")
            return None, None
            
    except Exception as e:
        print(f"Request failed for '{full_address}': {e}")
        return None, None

def main():
    print("Loading data...")
    try:
        df = pd.read_csv(INPUT_FILE, sep=';', encoding='utf-8')
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    # Filter columns
    cols_needed = ["ENDEREÇO DO LOCAL DO CURSO", "CIDADE", "LOCAL DO CURSO"]
    for col in cols_needed:
        if col not in df.columns:
            print(f"Missing column: {col}")
            return

    # Get unique combinations of Address + City + Location Name
    # We include Location Name because sometimes the address is generic but the location name helps (though API mostly uses address)
    # Actually, for geocoding, Address + City is usually enough. 
    # Let's normalize data slightly
    df['search_address'] = df["ENDEREÇO DO LOCAL DO CURSO"].fillna("").astype(str).str.strip()
    df['search_city'] = df["CIDADE"].fillna("").astype(str).str.strip()
    
    # Filter out empty addresses
    unique_places = df[df['search_address'] != ""][['search_address', 'search_city']].drop_duplicates()
    
    print(f"Found {len(unique_places)} unique addresses to geocode.")
    
    # Load cache if exists
    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            cache = json.load(f)
            print(f"Loaded {len(cache)} cached coordinates.")

    results = []
    
    # Process
    count = 0
    for index, row in unique_places.iterrows():
        addr = row['search_address']
        city = row['search_city']
        key = f"{addr}|{city}"
        
        if key in cache:
            lat, lon = cache[key]
        else:
            print(f"Geocoding: {addr}, {city}...")
            lat, lon = get_lat_lon(addr, city)
            cache[key] = (lat, lon)
            time.sleep(0.1) # Respect API rate limits
            
        if lat is not None and lon is not None:
            results.append({
                "ENDEREÇO DO LOCAL DO CURSO": addr,
                "CIDADE": city,
                "LATITUDE": lat,
                "LONGITUDE": lon
            })
        
        count += 1
        if count % 10 == 0:
            # Save cache periodically
            with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(cache, f, ensure_ascii=False, indent=4)

    # Final save of cache
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=4)

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    print(f"Saved {len(results_df)} coordinates to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
