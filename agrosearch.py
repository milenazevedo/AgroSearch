"""
AgroSearch - Motor de Busca Inteligente
Laboratório Prático 04 - Desafio Integrador (UNIPÊ)

FASE 1 (Parte 1): Pipeline de Pré-processamento
FASE 2 (Parte 2): Índice Invertido
FASE 3 (Parte 3): TF-IDF e ranqueamento  -> A INTEGRAR
------------------------------------------------
Etapas implementadas do zero (sem scikit-learn, NLTK ou similares):
    1. Tokenização    -> quebra o texto em palavras
    2. Normalização   -> minúsculas + remoção de acentos
    3. Stopwords      -> remove palavras vazias (LIGA/DESLIGA na UI)
    4. Stemming       -> reduz a palavra ao radical (LIGA/DESLIGA na UI)

Como rodar:
    pip install streamlit
    python -m streamlit run .\app_agrosearch.py

Como integrar com as Partes 2 e 3 (app único):
    - Tudo entre "PRÉ-PROCESSAMENTO" e "UI" é a API que as outras partes usam:
          preprocessar(texto, usar_stopwords=True, usar_stemming=True) -> list[str]
      Use a MESMA função para os documentos e para a query.
    - DOCUMENTOS é o dicionário {id: texto} com a base sugerida pelo professor.
    - Os checkboxes ficam em st.session_state["usar_stopwords"] e
      st.session_state["usar_stemming"], então as Fases 2 e 3 podem ler esses
      valores e reagir à mesma configuração.
    - A UI da Fase 1 está em renderizar_fase1(); basta chamá-la dentro de uma
      aba/seção do app final.
"""

import re
import unicodedata
from collections import Counter, defaultdict

import streamlit as st

# =============================================================================
# BASE DE DOCUMENTOS (hardcode sugerido no enunciado)
# =============================================================================
DOCUMENTOS = {
    1: "A soja requer irrigação constante durante o período de floração para garantir a produtividade.",
    2: "O controle biológico de lagartas na soja pode ser feito com a vespa Trichogramma.",
    3: "A adubação verde com leguminosas melhora o nitrogênio no solo para o milho.",
    4: "Lagartas desfolhadoras causam grande prejuízo na cultura da soja e do algodão.",
    5: "A irrigação por gotejamento economiza água e é ideal para o cultivo orgânico.",
}


# =============================================================================
# PRÉ-PROCESSAMENTO
# =============================================================================

# ---------- Etapa 1: Tokenização ----------
def tokenizar(texto):
    """Quebra o texto em tokens (sequências de letras/dígitos).

    A pontuação é descartada e palavras acentuadas são mantidas inteiras
    (o \\w do Python é compatível com Unicode).
    """
    return re.findall(r"[^\W_]+", texto)


# ---------- Etapa 2: Normalização ----------
def normalizar(token):
    """Converte para minúsculas e remove acentos (ex.: 'Irrigação' -> 'irrigacao')."""
    token = token.lower()
    decomposto = unicodedata.normalize("NFKD", token)
    return "".join(c for c in decomposto if not unicodedata.combining(c))


# ---------- Etapa 3: Stopwords ----------
_STOPWORDS_BRUTAS = """
a o as os um uma uns umas
de do da dos das em no na nos nas num numa nuns numas dum duma
por pelo pela pelos pelas para pra com sem sobre entre até ao aos à às
e ou mas que se como já também não mais menos muito muitos pouco só ainda então
porque pois logo assim aqui ali lá quando onde qual quais quem cujo
eu tu ele ela eles elas você me te lhe lhes nós
meu minha teu tua seu sua seus suas nosso nossa
este esta esse essa isto isso aquele aquela aquilo
ser são é foi era sido ter tem tinha haver há estar está estão pode podem deve devem
durante após antes desde depois
cada todo toda todos todas outro outra outros outras mesmo mesma
"""

# O conjunto é normalizado com a MESMA função dos tokens, para que
# 'é' -> 'e', 'até' -> 'ate', 'não' -> 'nao' etc. casem corretamente.
STOPWORDS = {normalizar(p) for p in _STOPWORDS_BRUTAS.split()}


def remover_stopwords(tokens):
    """Remove tokens (já normalizados) que estejam na lista de stopwords."""
    return [t for t in tokens if t not in STOPWORDS]


# ---------- Etapa 4: Stemming (redutor de sufixos para o português) ----------
# Stemmer simplificado, inspirado no RSLP, implementado do zero.
# Opera sobre palavras JÁ normalizadas (minúsculas e sem acento).
# Cada regra é (sufixo, tamanho mínimo do radical que deve sobrar).
_SUFIXOS = [
    # nominalização / substantivos
    ("amento", 3), ("imento", 3), ("mento", 3),
    ("acao", 3), ("icao", 3), ("ucao", 3), ("cao", 3),
    ("ividade", 3), ("idade", 3),
    ("adora", 3), ("ador", 3), ("edor", 3), ("idor", 3),
    ("ante", 4), ("ente", 4),
    ("ismo", 3), ("ista", 3),
    # advérbio
    ("mente", 4),
    # adjetivos
    ("avel", 3), ("ivel", 3),
    ("osa", 3), ("oso", 3),
    ("ico", 3), ("ica", 3),
    ("ivo", 5), ("iva", 5),
    ("ia", 3),
    # formas verbais
    ("ando", 3), ("endo", 3), ("indo", 3),
    ("aram", 3), ("ado", 4), ("ada", 4), ("ido", 4), ("ida", 4),
    ("ou", 3), ("am", 3),
    ("ar", 3), ("er", 3), ("ir", 3),
]
# testa sempre o sufixo mais longo primeiro
_SUFIXOS.sort(key=lambda regra: len(regra[0]), reverse=True)

_VOGAIS = "aeiou"


def _reduzir_plural(palavra):
    """Passo 1 do stemmer: tira o plural (animais -> animal, flores -> flor...)."""
    if len(palavra) < 4 or not palavra.endswith("s"):
        return palavra
    if palavra.endswith("ns"):                      # jardins  -> jardim
        return palavra[:-2] + "m"
    if palavra.endswith(("oes", "aes", "aos")):     # acoes    -> acao
        return palavra[:-3] + "ao"
    if palavra.endswith("ais"):                     # animais  -> animal
        return palavra[:-3] + "al"
    if palavra.endswith("eis"):                     # papeis   -> papel
        return palavra[:-3] + "el"
    if palavra.endswith("ois"):                     # lencois  -> lencol
        return palavra[:-3] + "ol"
    if palavra.endswith(("res", "zes")):            # flores   -> flor
        return palavra[:-2]
    if palavra.endswith(("us", "is", "ss")):        # exceções: onibus, lapis...
        return palavra
    return palavra[:-1]                             # lagartas -> lagarta


def _remover_sufixo(palavra):
    """Passo 2 do stemmer: remove o maior sufixo que casar (respeitando o radical mínimo)."""
    for sufixo, radical_minimo in _SUFIXOS:
        if palavra.endswith(sufixo) and len(palavra) - len(sufixo) >= radical_minimo:
            return palavra[: -len(sufixo)]
    return palavra


def _remover_vogal_final(palavra):
    """Passo 3 do stemmer: tira a vogal final (a/e/o) se vier depois de consoante."""
    if len(palavra) > 3 and palavra[-1] in "aeo" and palavra[-2] not in _VOGAIS:
        return palavra[:-1]
    return palavra


def stemmizar(palavra):
    """Reduz uma palavra normalizada ao seu radical.

    Exemplos: irrigacao -> irrig | irrigar -> irrig | lagartas -> lagart
    """
    if len(palavra) < 4:      # palavras muito curtas não são alteradas
        return palavra
    palavra = _reduzir_plural(palavra)
    palavra = _remover_sufixo(palavra)
    palavra = _remover_vogal_final(palavra)
    return palavra


# ---------- Pipeline completo ----------
def preprocessar(texto, usar_stopwords=True, usar_stemming=True):
    """Pipeline completo: tokenização -> normalização -> [stopwords] -> [stemming].

    Retorna a lista de tokens finais. Deve ser usada tanto nos documentos
    quanto na query do usuário (Fase 3), sempre com a mesma configuração.
    """
    tokens = [normalizar(t) for t in tokenizar(texto)]
    if usar_stopwords:
        tokens = remover_stopwords(tokens)
    if usar_stemming:
        tokens = [stemmizar(t) for t in tokens]
    return tokens


def detalhar_pipeline(texto, usar_stopwords=True, usar_stemming=True):
    """Versão 'passo a passo' do pipeline, usada pela UI para mostrar cada etapa."""
    linhas = []
    for token in tokenizar(texto):
        normalizado = normalizar(token)
        eh_stopword = normalizado in STOPWORDS
        if usar_stopwords and eh_stopword:
            final = "— (removida)"
        elif usar_stemming:
            final = stemmizar(normalizado)
        else:
            final = normalizado
        linhas.append(
            {
                "1. Token": token,
                "2. Normalizado": normalizado,
                "3. Stopword?": "sim" if eh_stopword else "não",
                "4. Stem": stemmizar(normalizado),
                "Resultado final": final,
            }
        )
    return linhas


def construir_vocabulario(documentos, usar_stopwords=True, usar_stemming=True):
    """Vocabulário do corpus: Counter {termo: frequência total}."""
    vocabulario = Counter()
    for texto in documentos.values():
        vocabulario.update(preprocessar(texto, usar_stopwords, usar_stemming))
    return vocabulario


def agrupar_por_stem(documentos, usar_stopwords=True):
    """Mostra quais palavras (normalizadas) colapsaram em cada radical."""
    grupos = defaultdict(set)
    for texto in documentos.values():
        for token in preprocessar(texto, usar_stopwords, usar_stemming=False):
            grupos[stemmizar(token)].add(token)
    return grupos


# =============================================================================
# UI (Streamlit) - Fase 1
# =============================================================================
def renderizar_fase1(documentos=DOCUMENTOS):
    st.header("Fase 1 — Pipeline de Pré-processamento")
    st.write(
        "Tokenização → Normalização → Stopwords → Stemming. "
        "Ligue e desligue as etapas opcionais e veja o vocabulário mudar."
    )

    # ---- Controles (checkboxes exigidos no enunciado) ----
    col_a, col_b = st.columns(2)
    with col_a:
        usar_stopwords = st.checkbox(
            "Remover stopwords",
            value=True,
            key="usar_stopwords",
            help="Remove palavras muito comuns e sem valor de busca (a, de, para, com...).",
        )
    with col_b:
        usar_stemming = st.checkbox(
            "Aplicar stemming",
            value=True,
            key="usar_stemming",
            help="Reduz cada palavra ao radical (irrigação e irrigar → irrig).",
        )

    # ---- Documentos originais ----
    with st.expander("Ver documentos originais", expanded=False):
        for doc_id, texto in documentos.items():
            st.markdown(f"**Doc {doc_id}:** {texto}")

    # ---- Passo a passo por documento ----
    st.subheader("Passo a passo por documento")
    doc_id = st.selectbox(
        "Escolha o documento",
        options=list(documentos.keys()),
        format_func=lambda i: f"Doc {i}",
    )
    st.info(documentos[doc_id])
    st.dataframe(
        detalhar_pipeline(documentos[doc_id], usar_stopwords, usar_stemming),
        hide_index=True,
    )
    tokens_finais = preprocessar(documentos[doc_id], usar_stopwords, usar_stemming)
    st.markdown("**Tokens finais:**")
    st.code(str(tokens_finais), language="python")

    # ---- Vocabulário do corpus ----
    st.subheader("Vocabulário do corpus")
    vocabulario = construir_vocabulario(documentos, usar_stopwords, usar_stemming)
    base = construir_vocabulario(documentos, usar_stopwords=False, usar_stemming=False)

    m1, m2, m3 = st.columns(3)
    m1.metric(
        "Tamanho do vocabulário (configuração atual)",
        len(vocabulario),
        delta=len(vocabulario) - len(base),
        delta_color="off",
    )
    m2.metric("Só normalização (sem stopwords/stemming)", len(base))
    m3.metric("Total de tokens no corpus", sum(vocabulario.values()))

    st.dataframe(
        [
            {"Termo": termo, "Frequência no corpus": freq}
            for termo, freq in sorted(vocabulario.items())
        ],
        hide_index=True,
    )

    if usar_stemming:
        with st.expander("Quais palavras o stemming agrupou em cada radical?"):
            grupos = agrupar_por_stem(documentos, usar_stopwords)
            st.dataframe(
                [
                    {
                        "Radical": radical,
                        "Palavras originais (normalizadas)": ", ".join(sorted(palavras)),
                        "Qtd": len(palavras),
                    }
                    for radical, palavras in sorted(grupos.items())
                ],
                hide_index=True,
            )

    # ---- Teste livre (útil para conferir como a query será tratada na Fase 3) ----
    st.subheader("Teste com seu próprio texto")
    texto_livre = st.text_input("Digite uma frase ou consulta", value="irrigação da soja")
    if texto_livre.strip():
        st.write("Tokens finais:")
        st.code(str(preprocessar(texto_livre, usar_stopwords, usar_stemming)), language="python")


# =============================================================================
# FASE 2 - ÍNDICE INVERTIDO (implementado do zero)
# =============================================================================
def tokenizar_corpus(documentos, usar_stopwords=True, usar_stemming=True):
    """Aplica o pipeline da Fase 1 em cada documento: {id: [tokens finais]}."""
    return {
        doc_id: preprocessar(texto, usar_stopwords, usar_stemming)
        for doc_id, texto in documentos.items()
    }


def construir_indice_invertido(docs_tokens):
    """Índice invertido: {termo: [IDs dos docs onde o termo aparece]}.

    Cada ID aparece uma única vez por termo (usamos um set) e as listas e os
    termos saem ordenados para facilitar a leitura.
    """
    indice = defaultdict(set)
    for doc_id, tokens in docs_tokens.items():
        for token in tokens:
            indice[token].add(doc_id)
    return {termo: sorted(indice[termo]) for termo in sorted(indice)}


def construir_indice_com_frequencias(docs_tokens):
    """Índice com postings {termo: {doc_id: contagem}}.

    Guarda quantas vezes o termo aparece em cada documento, ou seja, o TF bruto
    que a Fase 3 precisa (evita recontar tokens).
    """
    indice = defaultdict(lambda: defaultdict(int))
    for doc_id, tokens in docs_tokens.items():
        for token in tokens:
            indice[token][doc_id] += 1
    return {t: dict(sorted(indice[t].items())) for t in sorted(indice)}


def buscar_documentos(indice, termos, modo="OR"):
    """Consulta booleana no índice. `termos` já pré-processados; modo 'OR' ou 'AND'."""
    if not termos:
        return []
    conjuntos = [set(indice.get(t, [])) for t in termos]
    resultado = set.intersection(*conjuntos) if modo == "AND" else set.union(*conjuntos)
    return sorted(resultado)


def renderizar_fase2(documentos=DOCUMENTOS):
    st.header("Fase 2 — Índice Invertido")
    st.write(
        "Estrutura **Termo → [IDs dos documentos]** construída a partir dos tokens "
        "pré-processados. Ela acompanha os checkboxes da Fase 1."
    )

    # Lê a mesma configuração dos checkboxes da Fase 1
    usar_stopwords = st.session_state.get("usar_stopwords", True)
    usar_stemming = st.session_state.get("usar_stemming", True)
    st.caption(
        f"Configuração atual: stopwords {'ligadas' if usar_stopwords else 'desligadas'} · "
        f"stemming {'ligado' if usar_stemming else 'desligado'}"
    )

    indice = construir_indice_invertido(
        tokenizar_corpus(documentos, usar_stopwords, usar_stemming)
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Termos no índice", len(indice))
    c2.metric("Postings (termo, doc)", sum(len(ids) for ids in indice.values()))
    c3.metric("Documentos indexados", len(documentos))

    filtro = st.text_input("Filtrar termos (contém):", key="filtro_indice").strip().lower()
    visivel = {t: ids for t, ids in indice.items() if filtro in t}

    formato = st.radio(
        "Formato de exibição", ["Tabela", "JSON"], horizontal=True, key="fmt_indice"
    )
    if formato == "JSON":
        st.json(visivel)
    else:
        st.dataframe(
            [
                {"Termo": t, "Nº de docs": len(ids), "Documentos (IDs)": ", ".join(map(str, ids))}
                for t, ids in visivel.items()
            ],
            hide_index=True,
        )

    st.subheader("Consulta rápida no índice")
    consulta = st.text_input("Digite uma consulta", value="irrigação soja", key="consulta_indice")
    modo = st.radio("Combinação dos termos", ["OR", "AND"], horizontal=True, key="modo_indice")
    termos = preprocessar(consulta, usar_stopwords, usar_stemming)
    st.write("Termos da consulta após o pré-processamento:")
    st.code(str(termos), language="python")
    achados = buscar_documentos(indice, termos, modo)
    if achados:
        st.success("Documentos encontrados: " + ", ".join(f"Doc {i}" for i in achados))
    else:
        st.warning("Nenhum documento encontrado.")
    return indice


# =============================================================================
# EXECUÇÃO (Fases 1 e 2 integradas; falta a Fase 3)
# =============================================================================
def main():
    st.set_page_config(page_title="AgroSearch", page_icon="🌱", layout="wide")
    st.title("🌱 AgroSearch — Motor de Busca Inteligente")

    # A Fase 1 vem primeiro: ela cria os checkboxes usados pelas outras fases.
    aba1, aba2 = st.tabs(["Fase 1 — Pré-processamento", "Fase 2 — Índice Invertido"])
    with aba1:
        renderizar_fase1()
    with aba2:
        renderizar_fase2()
    # TODO (Parte 3): adicionar a aba "Fase 3 — Busca TF-IDF" aqui.


if __name__ == "__main__":
    main()