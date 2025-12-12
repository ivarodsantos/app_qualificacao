# Guia de Configuração para Deploy no Streamlit Cloud

## 1. Dependências (Resolvido)
O erro inicial `ModuleNotFoundError` ocorria porque as bibliotecas do Google e de PDF não estavam listadas no `requirements.txt`.
**Ação realizada:** O arquivo `requirements.txt` foi atualizado com todas as bibliotecas necessárias (`google-auth`, `google-api-python-client`, `fpdf`, etc.).

## 2. Autenticação do Google (Ação Necessária)
O GitHub ignora arquivos sensíveis como `token.json` e `credentials.json` (por segurança). Portanto, o Streamlit Cloud não tem acesso a eles e o login falharia.

**Para corrigir isso, você deve configurar os "Secrets" no painel do Streamlit Cloud:**

1. Abra o arquivo `token.json` no seu computador (ele está na pasta do projeto).
2. Copie todo o conteúdo dele.
3. Vá para o painel do seu app no [Streamlit Cloud](https://share.streamlit.io/).
4. Clique em **Settings** (botão de engrenagem) -> **Secrets**.
5. Cole o conteúdo no formato TOML. Como o `token.json` é um JSON, você precisará adaptá-lo levemente para o formato TOML.

### Exemplo de como formatar nos Secrets:

Se o seu `token.json` for assim:
```json
{
    "token": "ya29.a0Aa...",
    "refresh_token": "1//04...",
    "token_uri": "https://oauth2.googleapis.com/token",
    "client_id": "123456...",
    "client_secret": "GOCSPX...",
    "scopes": ["https://www.googleapis.com/auth/spreadsheets.readonly"],
    "universe_domain": "googleapis.com",
    "account": "",
    "expiry": "2024-12-12T12:00:00Z"
}
```

Escreva assim na caixa de text **Secrets** do Streamlit:

```toml
[google_oauth]
token = "ya29.a0Aa..."
refresh_token = "1//04..."
token_uri = "https://oauth2.googleapis.com/token"
client_id = "123456..."
client_secret = "GOCSPX..."
scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
universe_domain = "googleapis.com"
account = ""
expiry = "2024-12-12T12:00:00Z"
```

**Nota:** 
- A seção deve se chamar `[google_oauth]`.
- As chaves (token, refresh_token, etc.) devem ser iguais às do seu arquivo.
- Os valores devem estar entre aspas duplas.
- Se houver listas (como `scopes`), mantenha os colchetes.

### 🔒 Nota sobre Segurança
**Isso é seguro? Sim.**
Os "Secrets" do Streamlit funcionam como variáveis de ambiente criptografadas.
- **Não ficam no GitHub:** O código no seu repositório continua sem as senhas.
- **Acesso Restrito:** Apenas você (no painel do Streamlit) e a aplicação rodando têm acesso a esses dados.
- **Padrão de Indústria:** Essa é a forma recomendada de gerenciar credenciais em nuvem (AWS, Azure, Heroku, etc. usam mecanismos similares).
- **Risco Zero de Vazamento:** Como você não está commitando o arquivo `token.json`, se alguém clonar seu repositório, não terá acesso à sua conta.

## 3. Próximos Passos
Após configurar os Secrets, faça um "Reboot" ou "Redeploy" do app no Streamlit Cloud para garantir que ele pegue as novas configurações e dependências.
