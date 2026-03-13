import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np


def criar_chunks(texto, tamanho_max=800):
    chunks = []
    texto_atual = ""

    for paragrafo in texto.split("\n"):
        if not paragrafo.strip():
            continue

        if len(texto_atual) + len(paragrafo) < tamanho_max:
            texto_atual += paragrafo + "\n"
        else:
            chunks.append(texto_atual)
            texto_atual = paragrafo + "\n"

    if texto_atual:
        chunks.append(texto_atual)

    return chunks


@st.cache_resource
def carregar_modelo_embedding(model_name="all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)


def gerar_embeddings(chunks, modelo=None):
    if modelo is None:
        modelo = carregar_modelo_embedding()
    vetores = modelo.encode(chunks)
    return np.array(vetores)


def buscar_chunks_relevantes(pergunta, chunks, embeddings, top_k=3, modelo=None):
    if modelo is None:
        modelo = carregar_modelo_embedding()

    pergunta_vec = modelo.encode([pergunta])
    pergunta_vec = np.array(pergunta_vec)[0]

    pergunta_norm = pergunta_vec / np.linalg.norm(pergunta_vec)
    emb_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    scores = emb_norm @ pergunta_norm

    idx_ordenados = np.argsort(-scores)
    idx_top = idx_ordenados[:top_k]

    chunks_selecionados = [chunks[i] for i in idx_top]
    scores_selecionados = [float(scores[i]) for i in idx_top]

    idx_top = [int(i) for i in idx_top]

    return chunks_selecionados, scores_selecionados, idx_top
