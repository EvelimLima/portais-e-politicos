# app.py
import datetime
import torch
from transformers import pipeline, BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import feedparser
from sqlalchemy import create_engine, text
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import config
import re

# CORREÇÃO #11: importa o parser centralizado do coletor
from coletor_sbbd import parsear_resposta_llama, ColetorParaMySQL
from classifica import extrair_trecho, parsear_resposta as parsear_resposta_classifica

# ==========================================
# 0. CONFIG GLOBAL E FUNÇÕES ÚTEIS
# ==========================================
st.set_page_config(page_title="Portais e Políticos - Llama 3.2", page_icon="👥", layout="wide")


@st.cache_resource
def get_engine():
    return create_engine(f"mysql+pymysql://{config.USUARIO_MYSQL}:{config.SENHA_MYSQL}@{config.HOST_MYSQL}:{config.PORTA_MYSQL}/{config.BANCO_MYSQL}")


@st.cache_resource
def carregar_modelo_llama():
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    print("\n--- DIAGNÓSTICO DE GPU ---")
    cuda_disponivel = torch.cuda.is_available()
    print(f"CUDA Disponível? {cuda_disponivel}")
    if cuda_disponivel:
        props = torch.cuda.get_device_properties(0)
        vram_total = props.total_memory / 1024**3
        vram_livre = (props.total_memory - torch.cuda.memory_allocated(0)) / 1024**3
        print(f"Placa: {props.name}  |  VRAM total: {vram_total:.1f}GB  |  Livre: {vram_livre:.1f}GB")
    print("--------------------------\n")

    MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=100,
        do_sample=False,
        temperature=None,
        top_p=None,
        repetition_penalty=1.1,
    )

    if cuda_disponivel:
        vram_pos = (props.total_memory - torch.cuda.memory_allocated(0)) / 1024**3
        print(f"Llama carregado. VRAM livre após: {vram_pos:.1f}GB")
    return pipe


# mapa de modelos disponíveis
MODELOS_DISPONIVEIS = {
    "Llama 3.2 3B — Decoder (Prompt)": {
        "tipo": "decoder",
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "nome_db": "Llama-3.2-3B-PyTorch",
        "descricao": "Modelo generativo local. Maior contexto, zero-shot via prompt.",
    },
    "Twitter-XLM-RoBERTa — Encoder (Direto)": {
        "tipo": "encoder_roberta",
        "model_id": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        "nome_db": "twitter-xlm-roberta-base-sentiment",
        "descricao": "Treinado em tweets. Referência de domain mismatch.",
    },
}


@st.cache_resource
def carregar_modelo_encoder(model_id, tipo):
    """Carrega modelo encoder (RoBERTa) para classificação direta."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nCarregando encoder: {model_id}")
    # RoBERTa requer use_fast=False
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

    model = AutoModelForSequenceClassification.from_pretrained(model_id).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.eval()
    print("Encoder carregado.")
    return tokenizer, model


def classificar_com_encoder(trecho, tokenizer, model, tipo):
    """Classifica um trecho com modelo encoder e retorna (motivo, classificacao)."""
    import torch
    inputs = tokenizer(
        trecho, return_tensors="pt", truncation=True, max_length=512, padding=True
    ).to(next(model.parameters()).device)

    with torch.no_grad():
        logits = model(**inputs).logits

    idx = logits.argmax(-1).item()

    # cardiffnlp: negative/neutral/positive
    label = model.config.id2label[idx].lower()
    mapa = {"negative": "Negativo", "neutral": "Neutro", "positive": "Positivo"}
    classificacao = mapa.get(label, "Neutro")
    motivo = f"RoBERTa: {label}"

    return motivo, classificacao


def descobrir_rss(url_site):
    if not url_site.startswith(('http://', 'https://')):
        url_site = 'https://' + url_site
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    try:
        response = requests.get(url_site, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for link in soup.find_all('link', type=['application/rss+xml', 'application/atom+xml']):
            href = link.get('href')
            if href:
                return urljoin(url_site, href)
        fallbacks = ['/feed/', '/rss/', '/feed.xml', '/rss.xml']
        for fb in fallbacks:
            test_url = url_site.rstrip('/') + fb
            try:
                r = requests.head(test_url, headers=headers, timeout=5)
                if r.status_code == 200:
                    d = feedparser.parse(test_url)
                    if not d.bozo:
                        return test_url
            except Exception:
                continue
    except Exception as e:
        print(f"Erro ao acessar {url_site}: {e}")
    return url_site


# ==========================================
# FUNÇÕES DE GERENCIAMENTO DE POLÍTICOS
# ==========================================

def carregar_politicos(engine):
    try:
        df = pd.read_sql("SELECT id, nome, aliases, ativo FROM config_politicos ORDER BY id", engine)
        resultado = []
        for _, row in df.iterrows():
            aliases_lista = [a.strip() for a in str(row['aliases']).split(',') if a.strip()]
            resultado.append({
                "id": int(row['id']),
                "nome": row['nome'],
                "aliases": aliases_lista,
                "ativo": bool(row['ativo'])
            })
        return resultado
    except Exception:
        return []


def montar_dicionario_politicos(engine):
    politicos = {}
    for p in carregar_politicos(engine):
        if p['ativo']:
            politicos[p['nome']] = p['aliases']
    return politicos


# ==========================================
# PROMPT PADRÃO (única definição — evita divergência)
# ==========================================
PROMPT_SISTEMA = """Você é um analista político objetivo. Avalie o sentimento do trecho em relação ao político.
Responda APENAS no formato abaixo com quebra de linha:
Motivo: [Explique em 1 a 2 frases por que o trecho é positivo, negativo ou neutro em relação ao político]
Classificacao: [Positivo, Negativo ou Neutro]"""


st.sidebar.title("Navegação do Sistema")
tela_atual = st.sidebar.radio(
    "Selecione a etapa:",
    [" Configuração", " Monitor de Processo", " Análise Analítica", " Laboratório IA"]
)
st.sidebar.divider()
st.sidebar.subheader("Modelo Ativo")
_modelo_sidebar = st.sidebar.selectbox(
    "Modelo de Classificação",
    list(MODELOS_DISPONIVEIS.keys()),
    key="modelo_global"
)
st.sidebar.caption(MODELOS_DISPONIVEIS[_modelo_sidebar]["descricao"])
st.sidebar.divider()
st.sidebar.warning("⚠️ Este sistema está em desenvolvimento. As classificações de sentimento são geradas automaticamente por modelos de linguagem e podem conter erros. Consulte sempre os links das notícias originais para uma leitura completa e contextualizada.")


# ==========================================
# TELA 1: CONFIGURAÇÃO
# ==========================================
if tela_atual == " Configuração":

    st.title(" Configuração do Sistema")
    engine = get_engine()

    st.subheader("Motor de Inteligência")
    modelo_selecionado = st.session_state.get("modelo_global", list(MODELOS_DISPONIVEIS.keys())[0])
    cfg_modelo = MODELOS_DISPONIVEIS[modelo_selecionado]
    st.info(f"**Modelo ativo:** {modelo_selecionado}  \n{cfg_modelo['descricao']}")
    st.success("Todos os modelos rodam localmente — privacidade total e custo zero de API.")

    st.divider()
    st.info("Atenção!! Para **Portais** e **Políticos**, você pode Adicionar, "
            "deixar Ativo **ou** Inativo e Excluir")

    col1, col2 = st.columns(2)

    # ── Coluna 1: Portais RSS ─────────────────────────────────────────────────
    with col1:
        st.subheader("Portais de Notícias (RSS)")


        with st.form("form_rss"):
            novo_site    = st.text_input("Nome do Portal", placeholder="Ex: Amazonas Atual — Política")
            url_digitada = st.text_input(
                "URL da seção ou categoria a monitorar",
                placeholder="Ex: amazonasatual.com.br/categoria/politica/",
                help="Cole a URL da página de notícias que você acompanha. "
                        "Se o portal tiver abas por tema, use o link da aba correta."
            )
            submit_rss   = st.form_submit_button("Descobrir e Adicionar")

            if submit_rss:
                if not novo_site.strip() or not url_digitada.strip():
                    st.error("Preencha o nome e a URL do portal.")
                else:
                    with st.spinner(f"Procurando RSS em {url_digitada}..."):
                        url_rss = descobrir_rss(url_digitada.strip())
                        feed = feedparser.parse(url_rss)
                        if feed.bozo and len(feed.entries) == 0:
                            st.error("RSS não encontrado. Insira o link direto do feed.")
                        else:
                            try:
                                with engine.connect() as conn:
                                    conn.execute(text(
                                        "INSERT INTO config_feeds (nome_site, url_rss, ativo) VALUES (:n, :u, 1)"
                                    ), {"n": novo_site.strip(), "u": url_rss})
                                    conn.commit()
                                st.success(f"Cadastrado: {url_rss}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Erro ao salvar: {e}")

        try:
            df_feeds = pd.read_sql("SELECT id, nome_site, url_rss, ativo FROM config_feeds ORDER BY id", engine)
        except Exception:
            df_feeds = pd.DataFrame()

        if df_feeds.empty:
            st.info("Nenhum portal cadastrado.")
        else:
            for _, row in df_feeds.iterrows():
                fid   = int(row['id'])
                ativo = bool(row['ativo'])
                ca, cb, cc, cd = st.columns([1, 3, 1, 1])
                ca.markdown(f"**{row['nome_site']}**")
                cb.caption(row['url_rss'][:50] + "..." if len(row['url_rss']) > 50 else row['url_rss'])
                label_toggle = "Ativo *" if ativo else "Inativo"
                if cc.button(label_toggle, key=f"feed_toggle_{fid}"):
                    with engine.connect() as conn:
                        conn.execute(text("UPDATE config_feeds SET ativo = :a WHERE id = :id"),
                                        {"a": int(not ativo), "id": fid})
                        conn.commit()
                    st.rerun()
                if cd.button("Excluir", key=f"feed_del_{fid}", help="Remover portal?"):
                    with engine.connect() as conn:
                        conn.execute(text("DELETE FROM config_feeds WHERE id = :id"), {"id": fid})
                        conn.commit()
                    st.rerun()

    # ── Coluna 2: Políticos ───────────────────────────────────────────────────
    with col2:
        st.subheader("Políticos Monitorados")

        with st.form("form_politico"):
            nome_pol    = st.text_input("Nome oficial do político", placeholder="Ex: Arthur Bisneto")
            aliases_raw = st.text_input("Variações do nome (separadas por vírgula)",
                                        placeholder="Ex: Arthur Bisneto, Dep. Arthur, Bisneto")
            submit_pol  = st.form_submit_button(" Adicionar Político")

            if submit_pol:
                nome_pol    = nome_pol.strip()
                aliases_raw = aliases_raw.strip()
                LIMITE_NOME  = 100
                LIMITE_ALIAS = 100
                aliases_lista_raw = [a.strip() for a in aliases_raw.split(',') if a.strip()]
                alias_longo = next((a for a in aliases_lista_raw if len(a) > LIMITE_ALIAS), None)
                politicos_existentes = [p['nome'].lower() for p in carregar_politicos(engine)]

                if not nome_pol:
                    st.error("Informe o nome do político.")
                elif len(nome_pol) > LIMITE_NOME:
                    st.error(f"Nome muito longo ({len(nome_pol)} chars, máx {LIMITE_NOME}).")
                elif not aliases_raw:
                    st.error("Informe ao menos uma variação do nome.")
                elif alias_longo:
                    st.error(f'Variação "{alias_longo}" muito longa (máx {LIMITE_ALIAS} chars).')
                elif nome_pol.lower() in politicos_existentes:
                    st.warning(f"'{nome_pol}' já está cadastrado.")
                else:
                    if nome_pol not in aliases_lista_raw:
                        aliases_lista_raw.insert(0, nome_pol)
                    try:
                        with engine.connect() as conn:
                            conn.execute(text(
                                "INSERT INTO config_politicos (nome, aliases, ativo) VALUES (:n, :a, 1)"
                            ), {"n": nome_pol, "a": ", ".join(aliases_lista_raw)})
                            conn.commit()
                        st.success(f"'{nome_pol}' adicionado.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Erro ao salvar: {e}")

        todos_politicos = carregar_politicos(engine)
        if not todos_politicos:
            st.info("Nenhum político cadastrado.")
        else:
            for p in todos_politicos:
                pid   = p['id']
                ativo = p['ativo']
                pa, pb, pc, pd_ = st.columns([1, 3, 1, 1])
                pa.markdown(f"**{p['nome']}**")
                pb.caption(", ".join(p['aliases']))
                label_toggle = "Ativo *" if ativo else "Inativo"
                if pc.button(label_toggle, key=f"pol_toggle_{pid}"):
                    with engine.connect() as conn:
                        conn.execute(text("UPDATE config_politicos SET ativo = :a WHERE id = :id"),
                                        {"a": int(not ativo), "id": pid})
                        conn.commit()
                    st.rerun()
                if pd_.button("Excluir", key=f"pol_del_{pid}", help=f"Remover Figura? {p['nome']}"):
                    with engine.connect() as conn:
                        conn.execute(text("DELETE FROM config_politicos WHERE id = :id"), {"id": pid})
                        conn.commit()
                    st.toast(f"'{p['nome']}' removido.")
                    st.rerun()


# ==========================================
# TELA 2: MONITOR DE PROCESSO
# ==========================================
elif tela_atual == " Monitor de Processo":
    st.title(" Monitor de Processamento (Pipeline)")
    st.markdown("Controle manual da extração de notícias e classificação pela Inteligência Artificial.")

    colA, colB = st.columns(2)
    engine = get_engine()

    with colA:
        st.subheader("Etapa 1: Ingestão de Dados")

        politicos_ativos = montar_dicionario_politicos(engine)
        with st.expander(f"Políticos nesta coleta ({len(politicos_ativos)})"):
            for nome in politicos_ativos.keys():
                st.markdown(f"- {nome}")

        try:
            n_feeds = int(pd.read_sql(
                "SELECT COUNT(*) as n FROM config_feeds WHERE ativo = 1", engine
            ).iloc[0]["n"])
        except Exception:
            n_feeds = 0

        if n_feeds == 0:
            st.warning(" Nenhum portal ativo. Vá em **Configuração → Fontes de Dados** e ative ao menos um.")

        if st.button(" Iniciar Coleta nos Portais (RSS)", use_container_width=True, disabled=(n_feeds == 0)):
            with st.spinner("Conectando aos portais..."):
                try:
                    coletor = ColetorParaMySQL(
                        politicos=politicos_ativos,
                        db_user=config.USUARIO_MYSQL,
                        db_pass=config.SENHA_MYSQL,
                        db_name=config.BANCO_MYSQL,
                        db_host=config.HOST_MYSQL
                    )
                    coletor.executar_coleta_e_processamento_incremental()
                    st.success("Coleta e limpeza finalizadas com sucesso!")
                except Exception as e:
                    st.error(f"Erro na coleta: {e}")

    with colB:
        nome_modelo_etapa2 = st.session_state.get("modelo_global", list(MODELOS_DISPONIVEIS.keys())[0])
        cfg_etapa2 = MODELOS_DISPONIVEIS[nome_modelo_etapa2]
        nome_modelo_str_etapa2 = cfg_etapa2["nome_db"]
        st.subheader(f"Etapa 2: Classificação — {nome_modelo_etapa2.split('—')[0].strip()}")

        try:
            tabelas = pd.read_sql("SHOW TABLES LIKE 'analises_llama'", engine)
            if tabelas.empty:
                n_pendentes = int(pd.read_sql(
                    "SELECT COUNT(*) as n FROM noticias_processadas", engine
                ).iloc[0]["n"])
                query_busca = "SELECT id, titulo, politico_busca, conteudo_limpo FROM noticias_processadas"
            else:
                # Filtra por modelo ativo: pendente = ainda não classificado por ESTE modelo
                n_pendentes = int(pd.read_sql("""
                    SELECT COUNT(*) as n FROM noticias_processadas
                    WHERE id NOT IN (
                        SELECT DISTINCT noticia_id FROM analises_llama
                        WHERE modelo = %(modelo)s
                    )
                """, engine, params={"modelo": nome_modelo_str_etapa2}).iloc[0]["n"])
                query_busca = """
                    SELECT id, titulo, politico_busca, conteudo_limpo
                    FROM noticias_processadas
                    WHERE id NOT IN (
                        SELECT DISTINCT noticia_id FROM analises_llama
                        WHERE modelo = %(modelo)s
                    )
                """
        except Exception as e:
            st.error(f"Erro ao verificar notícias pendentes: {e}")
            n_pendentes = 0
            query_busca = ""

        if n_pendentes == 0:
            st.info("ℹ Nenhuma notícia pendente. Execute a **Etapa 1** primeiro.")
        else:
            st.success(f" {n_pendentes} notícia(s) aguardando classificação.")

        if st.button(" Iniciar Análise de Sentimento", use_container_width=True, disabled=(n_pendentes == 0)):
            try:
                df_pendentes = pd.read_sql(query_busca, engine, params={"modelo": nome_modelo_str_etapa2}) if "%(modelo)s" in query_busca else pd.read_sql(query_busca, engine)
                if df_pendentes.empty:
                    st.info("Não há novas notícias para classificar.")
                else:
                    total = len(df_pendentes)
                    # Pega modelo selecionado na Tela de Configuração
                    nome_modelo_ativo = st.session_state.get("modelo_global", list(MODELOS_DISPONIVEIS.keys())[0])
                    cfg = MODELOS_DISPONIVEIS[nome_modelo_ativo]
                    tipo = cfg["tipo"]
                    nome_modelo_str = cfg["nome_db"]

                    st.write(f"Analisando {total} trechos com **{nome_modelo_ativo}**...")
                    st.toast(f"Carregando {nome_modelo_ativo}... Isso pode levar 1 minuto.")

                    if tipo == "decoder":
                        pipe = carregar_modelo_llama()
                        encoder_tok, encoder_model = None, None
                    else:
                        pipe = None
                        encoder_tok, encoder_model = carregar_modelo_encoder(cfg["model_id"], tipo)

                    barra_progresso = st.progress(0)
                    status_texto = st.empty()
                    contador = 0

                    for idx, (_, row) in enumerate(df_pendentes.iterrows()):
                        nome_politico = row['politico_busca']
                        status_texto.text(f"Classificando {idx+1}/{total}: {nome_politico}")

                        trecho = extrair_trecho(str(row['conteudo_limpo']), nome_politico)

                        try:
                            if tipo == "decoder":
                                messages = [
                                    {"role": "system", "content": PROMPT_SISTEMA},
                                    {"role": "user", "content": f"Político alvo: {nome_politico}\nTrecho: \"{trecho}\""}
                                ]
                                outputs = pipe(messages, max_new_tokens=100)
                                resposta = outputs[0]["generated_text"][-1]["content"].strip()
                                motivo, classificacao = parsear_resposta_classifica(resposta)
                            else:
                                motivo, classificacao = classificar_com_encoder(
                                    trecho, encoder_tok, encoder_model, tipo
                                )
                        except Exception as e:
                            classificacao, motivo = "Neutro", f"Erro no modelo: {str(e)}"

                        # Salva incrementalmente
                        try:
                            with engine.connect() as conn:
                                ja_existe = conn.execute(text(
                                    "SELECT COUNT(*) FROM analises_llama WHERE noticia_id = :nid AND modelo = :mod"
                                ), {"nid": int(row['id']), "mod": nome_modelo_str}).scalar()
                                if not ja_existe:
                                    conn.execute(text("""
                                        INSERT INTO analises_llama
                                            (noticia_id, politico, trecho, classificacao, motivo, modelo)
                                        VALUES (:nid, :pol, :trecho, :classif, :motivo, :modelo)
                                    """), {
                                        "nid": int(row['id']), "pol": nome_politico,
                                        "trecho": trecho, "classif": classificacao,
                                        "motivo": motivo, "modelo": nome_modelo_str
                                    })
                                    conn.commit()
                                    contador += 1
                        except Exception as e:
                            st.warning(f"Erro ao salvar id={row['id']}: {e}")

                        barra_progresso.progress(int((idx + 1) / total * 100))

                    status_texto.text("Classificação concluída!")
                    st.success(f"{contador} análises salvas no banco!")
                    st.cache_data.clear()

            except Exception as e:
                st.error(f"Erro no processamento: {e}")


# ==========================================
# TELA 3: ANÁLISE ANALÍTICA
# ==========================================
elif tela_atual == " Análise Analítica":
    modelo_ativo_t3 = st.session_state.get("modelo_global", list(MODELOS_DISPONIVEIS.keys())[0])
    nome_db_t3 = MODELOS_DISPONIVEIS[modelo_ativo_t3]["nome_db"]

    st.title(" Monitoramento Político do Amazonas")
    st.caption(f"Projeto de Monitoramento Político com NLP · Modelo: {nome_db_t3}")
    st.markdown(f"**Análise de Sentimento com {modelo_ativo_t3.split('—')[0].strip()} sobre notícias locais.**")
    st.divider()

    @st.cache_data(ttl=600)
    def carregar_dados_do_banco(nome_modelo_db):
        query = """
            SELECT
                DATE(n.data_publicacao) as data,
                n.portal,
                n.link,
                n.titulo,
                a.politico,
                a.trecho,
                a.classificacao,
                a.motivo
            FROM analises_llama a
            JOIN noticias_processadas n ON a.noticia_id = n.id
            WHERE a.modelo = %(modelo)s
        """
        df = pd.read_sql(query, get_engine(), params={"modelo": nome_modelo_db})
        df['data'] = pd.to_datetime(df['data'], errors='coerce').dt.date
        df = df.dropna(subset=['data', 'classificacao'])
        df['classificacao'] = df['classificacao'].astype(str).str.strip().str.capitalize()
        df['classificacao'] = df['classificacao'].replace({
            "Positive": "Positivo", "Negative": "Negativo", "Neutral": "Neutro",
            "Positiva": "Positivo", "Negativa": "Negativo"
        })
        return df

    try:
        df_noticias = carregar_dados_do_banco(nome_db_t3)
    except Exception as e:
        st.error(f"Erro ao conectar com o banco de dados na Tela 3: {e}")
        st.stop()

    if df_noticias.empty:
        st.warning(f"Nenhum dado encontrado para o modelo **{nome_db_t3}**. Vá em Monitor de Processo e execute a classificação.")
        st.stop()

    st.sidebar.header(" Filtros da Análise")
    min_date = df_noticias['data'].min()
    max_date = df_noticias['data'].max()

    atalho = st.sidebar.radio(
        "Período rápido",
        ["Personalizado", "Última semana", "Último mês", "Últimos 3 meses", "Tudo"],
        horizontal=True,
    )
    if atalho == "Última semana":
        _inicio_padrao = max(min_date, max_date - datetime.timedelta(days=7))
        _fim_padrao = max_date
    elif atalho == "Último mês":
        _inicio_padrao = max(min_date, max_date - datetime.timedelta(days=30))
        _fim_padrao = max_date
    elif atalho == "Últimos 3 meses":
        _inicio_padrao = max(min_date, max_date - datetime.timedelta(days=90))
        _fim_padrao = max_date
    else:
        _inicio_padrao = min_date
        _fim_padrao = max_date

    data_inicio, data_fim = st.sidebar.date_input(
        "Intervalo de tempo",
        [_inicio_padrao, _fim_padrao],
        min_value=min_date, max_value=max_date,
        disabled=(atalho != "Personalizado"),
    )
    politicos_unicos = sorted(df_noticias['politico'].dropna().unique())
    politicos_selecionados = st.sidebar.multiselect("Políticos", politicos_unicos, default=politicos_unicos)
    portais_unicos = sorted(df_noticias['portal'].dropna().unique())
    portais_selecionados = st.sidebar.multiselect("Portais", portais_unicos, default=portais_unicos)

    df_filtrado = df_noticias[
        (df_noticias['data'] >= data_inicio) &
        (df_noticias['data'] <= data_fim) &
        (df_noticias['politico'].isin(politicos_selecionados)) &
        (df_noticias['portal'].isin(portais_selecionados))
    ]

    if len(df_filtrado) == 0:
        st.warning("Nenhum dado encontrado para os filtros selecionados.")
        st.stop()

    total    = len(df_filtrado)
    positivos = len(df_filtrado[df_filtrado['classificacao'] == "Positivo"])
    negativos = len(df_filtrado[df_filtrado['classificacao'] == "Negativo"])
    taxa_neg  = (negativos / total) * 100 if total > 0 else 0

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Menções",      total)
    col2.metric("Portais",      df_filtrado['portal'].nunique())
    col3.metric("Positivas",    positivos)
    col4.metric("Negativas",    negativos)
    col5.metric("Negatividade", f"{taxa_neg:.1f}%")
    st.divider()

    cores_sentimento = {"Positivo": "#00CC96", "Negativo": "#EF553B", "Neutro": "#636EFA"}
    col_graf1, col_graf2 = st.columns([2, 1])

    with col_graf1:
        st.subheader(" Evolução temporal do sentimento")
        df_filtrado = df_filtrado.copy()
        df_filtrado['data_dt'] = pd.to_datetime(df_filtrado['data'])
        df_timeline = (
            df_filtrado
            .groupby([df_filtrado['data_dt'].dt.to_period("W").apply(lambda r: r.start_time), 'classificacao'])
            .size()
            .reset_index(name='contagem')
        )
        fig_timeline = px.area(
            df_timeline, x="data_dt", y="contagem", color="classificacao",
            color_discrete_map=cores_sentimento
        )
        fig_timeline.update_layout(height=450, xaxis_title="Semana", yaxis_title="Quantidade")
        st.plotly_chart(fig_timeline, use_container_width=True)

    with col_graf2:
        st.subheader(" Balanço por político")
        df_barras = df_filtrado.groupby(['politico', 'classificacao']).size().reset_index(name='contagem')
        fig_barras = px.bar(
            df_barras, y="politico", x="contagem", color="classificacao",
            orientation="h", barmode="group", color_discrete_map=cores_sentimento
        )
        fig_barras.update_layout(height=450)
        evento_barras = st.plotly_chart(
            fig_barras, use_container_width=True,
            on_select="rerun", key="grafico_barras"
        )

    _pontos = evento_barras.selection.get("points", [])
    _politico_clique = _pontos[0].get("y") if _pontos else None
    _sentimento_clique = _pontos[0].get("legendgroup") if _pontos else None

    st.divider()
    st.subheader(" Nuvem de Palavras Frequentes")

    def gerar_nuvem(textos):
        texto_unico = " ".join(textos.dropna().astype(str))
        stopwords = set([
            "de","do","da","dos","das","em","no","na","nos","nas","para","por",
            "com","sem","que","e","ou","se","um","uma","uns","umas","ao","aos",
            "à","á","às","ás","pelo","pela","pelos","pelas","num","numa","nuns","numas",
            "dum","duma","duns","dumas","este","esta","estes","estas","esse",
            "essa","esses","essas","aquele","aquela","aqueles","aquelas","isso",
            "isto","aquilo","seu","sua","seus","suas","meu","minha","mais",
            "mas","como","até","após","sobre","entre","contra","já","também",
            "ainda","então","quando","onde","porque","pois","nem","não","foi",
            "ser","ter","há","vai","vão","tem","têm","são","está","estão",
            "ele","ela","eles","elas","nós","vós","lhe","lhes","me","te",
            "nos","vos","isso","isto","tudo","nada","muito","pouco","outro",
            "outra","outros","outras","todo","toda","todos","todas","cada",
            "segunda","terça","quarta","quinta","sexta","sábado","domingo",
            "janeiro","fevereiro","março","abril","maio","junho","julho",
            "agosto","setembro","outubro","novembro","dezembro",
            "by","audio","áudio","compartilhe","compartilhar","veja","vídeo",
            "video","siga","podcast","spotify","whatsapp","telegram","facebook",
            "instagram","twitter","grupo","link","clique","acesse","leia",
            "voicexpress","am","br","https","http","www","feed","rss",
            "o","a","os","as","lá","cá","tão","bem","sim","não",
        ])
        nuvem = WordCloud(width=900, height=300, background_color="white", stopwords=stopwords).generate(texto_unico)
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.imshow(nuvem, interpolation='bilinear')
        ax.axis("off")
        return fig

    st.pyplot(gerar_nuvem(df_filtrado['trecho']))

    st.divider()
    st.subheader(" Auditoria de Trechos")

    df_base_auditoria = df_filtrado
    if _politico_clique and _sentimento_clique:
        df_base_auditoria = df_filtrado[
            (df_filtrado['politico'] == _politico_clique) &
            (df_filtrado['classificacao'] == _sentimento_clique)
        ]
        cores_map = {"Positivo": "green", "Negativo": "red", "Neutro": "gray"}
        cor_label = cores_map.get(_sentimento_clique, "gray")
        st.info(
            f"Filtro ativo: **{_politico_clique}** · :{cor_label}[**{_sentimento_clique}**] "
            f"— clique na mesma barra ou fora do gráfico para limpar"
        )

    busca = st.text_input("Buscar palavra nos trechos (Ex: Saúde, Chuva):")
    df_auditoria = (
        df_base_auditoria[df_base_auditoria['trecho'].str.contains(busca, case=False, na=False)]
        if busca else df_base_auditoria
    )
    st.dataframe(
        df_auditoria[['data', 'portal', 'politico', 'trecho', 'motivo', 'classificacao', 'link']],
        column_config={
            "link": st.column_config.LinkColumn("Notícia", display_text="Abrir")
        },
        use_container_width=True, height=350
    )


# ==========================================
# TELA 4: LABORATÓRIO IA
# ==========================================
elif tela_atual == " Laboratório IA":
    st.title(" Laboratório Expresso")
    st.markdown("Área de demonstração livre. Avalie qualquer texto em tempo real sem salvar no banco de dados.")
    st.divider()

    col1, col2 = st.columns([1, 2])
    with col1:
        nome_teste = st.text_input("Nome da Pessoa ou Tema Alvo:", value="João da Silva")
        st.info("A IA buscará o sentimento estritamente focado no nome que você digitar acima.")
    with col2:
        trecho_teste = st.text_area(
            "Cole a notícia ou frase:", height=165,
            value="O prefeito João da Silva inaugurou a nova escola hoje, o que agradou muitos moradores da região. "
                    "Contudo, a oposição apontou atrasos severos na obra."
        )

    campos_ok = bool(nome_teste.strip()) and bool(trecho_teste.strip())
    if not nome_teste.strip():
        st.warning(" Preencha o **nome do político** antes de analisar.")
    elif not trecho_teste.strip():
        st.warning(" Cole o **texto da notícia** antes de analisar.")

    if st.button(" Analisar Texto Agora", disabled=not campos_ok):
        with st.spinner("Processando..."):
            nome_modelo_lab = st.session_state.get("modelo_global", list(MODELOS_DISPONIVEIS.keys())[0])
            cfg_lab = MODELOS_DISPONIVEIS[nome_modelo_lab]
            tipo_lab = cfg_lab["tipo"]

            try:
                resposta = None
                if tipo_lab == "decoder":
                    pipe = carregar_modelo_llama()
                    messages = [
                        {"role": "system", "content": PROMPT_SISTEMA},
                        {"role": "user",   "content": f"Político alvo: {nome_teste}\nTrecho: \"{trecho_teste}\""}
                    ]
                    outputs = pipe(messages, max_new_tokens=150)
                    resposta = outputs[0]["generated_text"][-1]["content"].strip()
                    motivo, classificacao = parsear_resposta_llama(resposta)
                else:
                    enc_tok, enc_model = carregar_modelo_encoder(cfg_lab["model_id"], tipo_lab)
                    motivo, classificacao = classificar_com_encoder(trecho_teste, enc_tok, enc_model, tipo_lab)

                st.success("Análise Concluída com Sucesso!")
                cores_map = {"Positivo": "green", "Negativo": "red", "Neutro": "gray"}
                cor = cores_map.get(classificacao, "gray")
                st.markdown(f"### Classificação: :{cor}[**{classificacao}**]")
                st.markdown(f"**Justificativa:** {motivo}")

                if resposta is not None:
                    with st.expander("Ver Resposta Bruta do Modelo (Log de Debug)"):
                        st.code(resposta)

            except Exception as e:
                st.error(f"Erro ao processar o texto na placa de vídeo: {e}")