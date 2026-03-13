import os
from groq import Groq


def gerar_resposta_llm(
    pergunta,
    chunks_selecionados,
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_completion_tokens=512,
    api_key=None,
):
    api_key = api_key or os.environ.get("GROQ_API_KEY")
    client = Groq(api_key=api_key)

    contexto = "\n\n---\n\n".join(chunks_selecionados)

    system_prompt = (
        "Você é um assistente para corretores de seguros. "
        "Responda APENAS com base no contexto da apólice abaixo. "
        "Quando a pergunta for sobre valores, responda no formato:\n"
        "- Resposta direta\n"
        "- Trechos de referência (1–3 citações)\n"
        "Se não encontrar a informação, diga claramente que não encontrou. "
        "Não invente coberturas. Se não achar algo no contexto, diga que não encontrou. "
        "Nunca responda com ‘acho que’ ou ‘provavelmente’. "
        "Seja categórico em dizer que não encontrou quando o contexto não tiver a informação. "
        "Quando citar, copie literalmente do contexto e mantenha a linha 'Fonte: ... | Página: ...'."
    )

    user_content = (
        f"Pergunta do corretor: {pergunta}\n\nContexto da apólice:\n{contexto}"
    )

    resposta = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
    )

    return resposta.choices[0].message.content.strip()
