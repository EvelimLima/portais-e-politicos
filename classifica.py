"""
classificar.py — Roda fora do Streamlit, salva incrementalmente (1 por vez).
Se interrompido, retoma de onde parou na próxima execução.

Uso:
    python classificar.py
"""
import torch
import config
import re
import pandas as pd
from sqlalchemy import create_engine, text
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
from coletor_sbbd import ColetorParaMySQL, parsear_resposta_llama as parsear_resposta

PROMPT_SISTEMA = """Você é um analista político objetivo. Avalie o sentimento do trecho em relação ao político.
Responda APENAS no formato abaixo com quebra de linha:
Motivo: [Explique em 1 a 2 frases por que o trecho é positivo, negativo ou neutro em relação ao político]
Classificacao: [Positivo, Negativo ou Neutro]"""

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"


def carregar_modelo():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()
    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer,
        max_new_tokens=100, do_sample=False,
        temperature=None, top_p=None, repetition_penalty=1.1,
    )
    print("Modelo carregado.")
    return pipe


def extrair_trecho(texto, nome_politico, janela=300):
    # Limpa o texto antes de extrair — reutiliza pipeline do coletor
    _c = ColetorParaMySQL.__new__(ColetorParaMySQL)
    texto = _c._pipeline_de_limpeza(texto)

    texto_lower = texto.lower()
    nome_lower = nome_politico.lower()
    posicoes, inicio = [], 0
    while True:
        pos = texto_lower.find(nome_lower, inicio)
        if pos == -1: break
        posicoes.append(pos)
        inicio = pos + 1

    if posicoes:
        fragmentos, fim_ant = [], -1
        for pos in posicoes:
            ini_f = max(0, pos - janela)
            fim_f = min(len(texto), pos + len(nome_politico) + janela)

            # Não corta palavra no início
            if ini_f > 0:
                espaco = texto.rfind(' ', ini_f, pos)
                if espaco != -1:
                    ini_f = espaco + 1

            # Não corta palavra no fim
            if fim_f < len(texto):
                espaco = texto.find(' ', fim_f)
                if espaco != -1:
                    fim_f = espaco

            if fragmentos and ini_f <= fim_ant:
                fragmentos[-1] = (fragmentos[-1][0], fim_f)
            else:
                fragmentos.append((ini_f, fim_f))
            fim_ant = fim_f

        trecho = " [...] ".join(texto[i:f] for i, f in fragmentos)
        return trecho[:1500] + "..." if len(trecho) > 1500 else trecho

    return texto[:800]


MODELOS_CLI = {
    "llama": {
        "tipo": "decoder",
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "nome_banco": "Llama-3.2-3B-PyTorch",
    },
    "roberta": {
        "tipo": "encoder_roberta",
        "model_id": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        "nome_banco": "twitter-xlm-roberta-base-sentiment",
    },
}


def carregar_encoder(model_id, tipo):
    from transformers import AutoTokenizer as AT, AutoModelForSequenceClassification
    import torch
    print(f"Carregando encoder: {model_id}")
    tok = AT.from_pretrained(model_id, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_id).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.eval()
    print("Encoder carregado.")
    return tok, model


def classificar_encoder(trecho, tok, model, tipo):
    import torch
    inputs = tok(trecho, return_tensors="pt", truncation=True,
                    max_length=512, padding=True).to(next(model.parameters()).device)
    with torch.no_grad():
        logits = model(**inputs).logits
    idx = logits.argmax(-1).item()
    label = model.config.id2label[idx].lower()
    cls = {"negative": "Negativo", "neutral": "Neutro", "positive": "Positivo"}.get(label, "Neutro")
    motivo = f"RoBERTa: {label}"
    return motivo, cls


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Classifica notícias políticas.")
    parser.add_argument(
        "--modelo", choices=list(MODELOS_CLI.keys()), default="llama",
        help="Modelo a usar: llama (padrão), roberta"
    )
    args = parser.parse_args()

    cfg = MODELOS_CLI[args.modelo]
    tipo = cfg["tipo"]
    nome_banco = cfg["nome_banco"]
    print(f"Modelo selecionado: {nome_banco}")

    engine = create_engine(f"mysql+pymysql://{config.USUARIO_MYSQL}:{config.SENHA_MYSQL}@{config.HOST_MYSQL}:{config.PORTA_MYSQL}/{config.BANCO_MYSQL}")

    try:
        tabelas = pd.read_sql("SHOW TABLES LIKE 'analises_llama'", engine)
        if tabelas.empty:
            query = "SELECT id, titulo, politico_busca, conteudo_limpo FROM noticias_processadas"
            df = pd.read_sql(query, engine)
        else:
            # Pendente = não classificado por ESTE modelo
            query = """
                SELECT id, titulo, politico_busca, conteudo_limpo
                FROM noticias_processadas
                WHERE id NOT IN (
                    SELECT DISTINCT noticia_id FROM analises_llama
                    WHERE modelo = %(modelo)s
                )
            """
            df = pd.read_sql(query, engine, params={"modelo": nome_banco})
    except Exception as e:
        print(f"Erro ao buscar pendentes: {e}")
        return

    total = len(df)
    if total == 0:
        print(f"Nenhuma notícia pendente para o modelo {nome_banco}.")
        return

    print(f"{total} notícias para classificar com {nome_banco}.")

    # Carrega modelo
    if tipo == "decoder":
        pipe = carregar_modelo()
        tok, enc_model = None, None
    else:
        pipe = None
        tok, enc_model = carregar_encoder(cfg["model_id"], tipo)

    for i, (_, row) in enumerate(df.iterrows(), 1):
        nome = row['politico_busca']
        trecho = extrair_trecho(str(row['conteudo_limpo']), nome)

        try:
            if tipo == "decoder":
                messages = [
                    {"role": "system", "content": PROMPT_SISTEMA},
                    {"role": "user",   "content": f"Político alvo: {nome}\nTrecho: \"{trecho}\""}
                ]
                outputs = pipe(messages, max_new_tokens=100)
                resposta = outputs[0]["generated_text"][-1]["content"].strip()
                motivo, classificacao = parsear_resposta(resposta)
            else:
                motivo, classificacao = classificar_encoder(trecho, tok, enc_model, tipo)
        except Exception as e:
            motivo, classificacao = f"Erro: {e}", "Neutro"

        # Salva — verifica duplicata por noticia_id + modelo
        try:
            with engine.connect() as conn:
                ja_existe = conn.execute(text(
                    "SELECT COUNT(*) FROM analises_llama WHERE noticia_id = :nid AND modelo = :mod"
                ), {"nid": int(row['id']), "mod": nome_banco}).scalar()
                if not ja_existe:
                    conn.execute(text("""
                        INSERT INTO analises_llama
                            (noticia_id, politico, trecho, classificacao, motivo, modelo)
                        VALUES (:nid, :pol, :trecho, :classif, :motivo, :modelo)
                    """), {
                        "nid": int(row['id']), "pol": nome,
                        "trecho": trecho, "classif": classificacao,
                        "motivo": motivo, "modelo": nome_banco
                    })
                    conn.commit()
        except Exception as e:
            print(f"  Erro ao salvar id={row['id']}: {e}")
            continue

        print(f"[{i}/{total}] {nome} → {classificacao} | {row['titulo'][:60]}")

    print("Classificação concluída.")


if __name__ == "__main__":
    main()