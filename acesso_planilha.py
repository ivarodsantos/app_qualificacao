import pandas as pd

def carregar_google_sheet_aba(link):
    try:
        planilha_id = link.split("/d/")[1].split("/")[0]

        if "gid=" in link:
            gid = link.split("gid=")[1].split("&")[0]
        else:
            gid = "0"

        link_csv = f"https://docs.google.com/spreadsheets/d/{planilha_id}/export?format=csv&gid={gid}"

        df = pd.read_csv(link_csv)
        return df

    except Exception as e:
        print("Erro ao carregar a planilha:", e)
        return None
