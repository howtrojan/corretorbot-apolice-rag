import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv
from rag import criar_chunks, gerar_embeddings, buscar_chunks_relevantes
from llm import gerar_resposta_llm

load_dotenv()

# Inicializa histórico de mensagens
if "messages" not in st.session_state:
    st.session_state.messages = []  # cada item: {"role": "user"/"assistant", "content": "..."}

# Mensagem de boas-vindas (só uma vez, se ainda não tiver nada)
if not st.session_state.messages:
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": "Olá! Envie uma pergunta sobre esta apólice, como 'qual o prêmio total?' ou 'qual a franquia?'.",
        }
    )


st.title("CorretorBot - Assistente de Apólice")

st.write("Envie um PDF de apólice para visualizar o texto.")

uploaded_file = st.file_uploader("Selecione o PDF da apólice", type=["pdf"])

# Botão global para limpar conversa
if st.button("Limpar conversa"):
    st.session_state.messages = []

if uploaded_file is not None:
    reader = PdfReader(uploaded_file)
    chunks = []
    chunk_metas = []

    for idx_pagina, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        if not page_text.strip():
            continue

        page_chunks = criar_chunks(page_text, tamanho_max=800)
        for chunk in page_chunks:
            if not chunk.strip():
                continue
            chunks.append(chunk)
            chunk_metas.append({"arquivo": uploaded_file.name, "pagina": idx_pagina})

    # Resetar histórico de chat quando um novo PDF é carregado
    if "ultimo_nome_arquivo" not in st.session_state:
        st.session_state.ultimo_nome_arquivo = uploaded_file.name
    elif st.session_state.ultimo_nome_arquivo != uploaded_file.name:
        st.session_state.ultimo_nome_arquivo = uploaded_file.name
        st.session_state.messages = []
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "Nova apólice carregada. Faça suas perguntas sobre este documento.",
            }
        )

    if not chunks:
        st.warning("Não foi possível extrair texto desse PDF.")
        st.stop()

    embeddings = gerar_embeddings(chunks)

    st.subheader("Chat sobre a apólice")

    # Mostrar histórico de mensagens
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg.get("content", ""))
            if msg["role"] == "assistant" and msg.get("references_md"):
                with st.expander("Trechos de referência"):
                    st.markdown(msg["references_md"])

    # Campo de entrada de nova pergunta
    pergunta = st.chat_input(
        "Faça uma pergunta sobre esta apólice (ex: 'qual o prêmio total?')"
    )

    if pergunta:
        # Adiciona a pergunta ao histórico
        st.session_state.messages.append({"role": "user", "content": pergunta})

        # Recupera chunks relevantes
        chunks_rel, scores, idx_top = buscar_chunks_relevantes(
            pergunta, chunks, embeddings, top_k=3
        )

        referencias = []
        for pos, idx in enumerate(idx_top):
            meta = chunk_metas[idx]
            trecho = chunks[idx].strip()
            if len(trecho) > 900:
                trecho = trecho[:900].rstrip() + "..."

            referencias.append(
                {
                    "arquivo": meta["arquivo"],
                    "pagina": meta["pagina"],
                    "trecho": trecho,
                    "contexto": f"Fonte: {meta['arquivo']} | Página: {meta['pagina']}\n{trecho}",
                }
            )

        contexto_llm = [ref["contexto"] for ref in referencias]

        referencias_md = "\n\n".join(
            f"- {ref['arquivo']} — pág. {ref['pagina']}\n\n> "
            + "\n> ".join(ref["trecho"].splitlines())
            for ref in referencias
        )

        # Gera resposta
        with st.chat_message("assistant"):
            with st.spinner("Lendo a apólice e respondendo..."):
                resposta = gerar_resposta_llm(pergunta, contexto_llm)
                st.markdown(resposta)
                with st.expander("Trechos de referência"):
                    st.markdown(referencias_md)

        # Adiciona resposta ao histórico
        st.session_state.messages.append(
            {"role": "assistant", "content": resposta, "references_md": referencias_md}
        )

        
