# CorretorBot — Assistente de Apólice (RAG + LLM)

Aplicação Streamlit para corretores consultarem uma apólice em PDF: faz upload do documento, extrai o texto, busca os trechos mais relevantes (RAG com embeddings) e gera uma resposta com um LLM, exibindo referências (arquivo/página/trechos) de forma expansível.

## Funcionalidades
- Upload de 1 PDF por vez (apólice ativa)
- Chat com histórico na sessão
- RAG simples:
  - chunking por parágrafo (por página)
  - embeddings com SentenceTransformers
  - busca por similaridade de cosseno (top_k = 3)
- Resposta do LLM baseada apenas no contexto
- Trechos de referência expansíveis (“Fonte | Página”)

## Estrutura do projeto
- `app.py`: interface Streamlit + orquestração (upload, chat, referências)
- `rag.py`: chunking, embeddings e busca de chunks relevantes
- `llm.py`: chamada ao LLM (Groq)
- `.env`: variáveis de ambiente (não commitar)

## Requisitos
- Python 3.10+ (recomendado)
- Dependências Python:
  - `streamlit`
  - `pypdf`
  - `sentence-transformers`
  - `numpy`
  - `groq`
  - `python-dotenv`

## Instalação (Windows / PowerShell)
Crie e ative um ambiente virtual:

Instale as dependências:

```powershell
pip install streamlit pypdf sentence-transformers numpy groq python-dotenv
```

## Configuração
Crie um arquivo `.env` na raiz com sua chave:

```env
GROQ_API_KEY=coloque_sua_chave_aqui
```

Importante: não versionar o `.env`.

## Como rodar
```powershell
streamlit run app.py
```

Abra a URL que o Streamlit mostrar no terminal.

## Como usar
1. Faça upload do PDF da apólice.
2. Digite uma pergunta (ex.: “qual o prêmio total?”, “qual a franquia?”).
3. Leia a resposta.
4. Abra “Trechos de referência” para ver as citações com página e fonte.

## Limitações atuais (MVP)
- Apenas 1 PDF por vez (não faz multi-apólice)
- Se o PDF for imagem/scan e não tiver texto extraível, o app não responde (precisaria OCR)
- Chunking simples por parágrafo pode não ser ideal para tabelas

## Próximos passos sugeridos
- OCR fallback para PDFs escaneados
- Multi-apólice (selecionar apólice ativa ou indexar várias)
- Mostrar também score de similaridade nas referências (debug)
- Guardar embeddings em cache persistente (evitar recomputar ao recarregar)