"""
classificar_validacao.py — Classifica os trechos anotados manualmente com o Llama
e calcula métricas de avaliação (acurácia, F1, matriz de confusão).

Uso:
    python classificar_validacao.py

Saídas:
    resultado_validacao.csv   — cruzamento manual vs. Llama
    matriz_confusao.png       — matriz de confusão
"""
import io
import re
import torch
import pandas as pd
import matplotlib.pyplot as plt
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)

CSV_PATH = r"C:\Users\eveli\PIBIC_Coleta\dash integrado\Classificação de Notícias Políticas em Português - Classificação de Notícias Políticas em Português.csv"
MODEL_ID  = "meta-llama/Llama-3.2-3B-Instruct"
SAIDA_CSV = "resultado_validacao.csv"

PROMPT_SISTEMA = """Você é um analista político objetivo. Avalie o sentimento do trecho em relação ao político.
Responda APENAS no formato abaixo com quebra de linha:
Motivo: [até 400 caracteres]
Classificacao: [Positivo, Negativo ou Neutro]"""


# ── 1. Carrega CSV ────────────────────────────────────────────────────────────
linhas = open(CSV_PATH, encoding='utf-8').readlines()
dados  = [l for l in linhas if l.strip() and not l.startswith('link,titulo')]
df = pd.read_csv(
    io.StringIO(''.join(dados)),
    names=['link', 'titulo', 'politico', 'trecho', 'motivo_da_classificacao', 'classificacao']
)
df = df.dropna(subset=['classificacao', 'trecho', 'politico'])
df['classificacao_manual'] = df['classificacao'].str.strip()
df['politico']             = df['politico'].str.strip()
df['trecho']               = df['trecho'].str.strip()

print(f"Registros carregados: {len(df)}")
print(df['classificacao'].value_counts().to_string())


# ── 2. Carrega modelo ─────────────────────────────────────────────────────────
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


def parsear_resposta(resposta):
    motivo_match  = re.search(r'Motivo:\s*(.*?)(?:Classifica[cç][aã]o:|$|\n)', resposta, re.IGNORECASE | re.DOTALL)
    classif_match = re.search(r'Classifica[cç][aã]o:\s*([A-Za-z]+)', resposta, re.IGNORECASE)
    motivo        = motivo_match.group(1).strip() if motivo_match else "Não extraído"
    classificacao = classif_match.group(1).strip().capitalize() if classif_match else "Neutro"
    if classificacao not in ["Positivo", "Negativo", "Neutro"]:
        if "positiv" in classificacao.lower():  classificacao = "Positivo"
        elif "negativ" in classificacao.lower(): classificacao = "Negativo"
        else:                                    classificacao = "Neutro"
    return motivo, classificacao


pipe = carregar_modelo()


# ── 3. Classifica cada trecho com o Llama ────────────────────────────────────
classificacoes_llama  = []
motivos_llama         = []
total = len(df)

for i, (_, row) in enumerate(df.iterrows(), 1):
    print(f"[{i}/{total}] {row['politico']}", end=" → ")

    messages = [
        {"role": "system", "content": PROMPT_SISTEMA},
        {"role": "user",   "content": f"Político alvo: {row['politico']}\nTrecho: \"{row['trecho']}\""}
    ]

    try:
        outputs   = pipe(messages, max_new_tokens=100)
        resposta  = outputs[0]["generated_text"][-1]["content"].strip()
        motivo, classificacao = parsear_resposta(resposta)
    except Exception as e:
        motivo, classificacao = f"Erro: {e}", "Neutro"

    print(classificacao)
    classificacoes_llama.append(classificacao)
    motivos_llama.append(motivo)

df['classificacao_llama'] = classificacoes_llama
df['motivo_llama']        = motivos_llama


# ── 4. Métricas ───────────────────────────────────────────────────────────────
classes = ['Positivo', 'Negativo', 'Neutro']
y_true  = df['classificacao_manual']
y_pred  = df['classificacao_llama']

acc      = accuracy_score(y_true, y_pred)
acc_bal  = balanced_accuracy_score(y_true, y_pred)
f1_macro = f1_score(y_true, y_pred, average='macro', labels=classes, zero_division=0)
f1_cls   = f1_score(y_true, y_pred, average=None,    labels=classes, zero_division=0)

print("\n" + "="*55)
print("  MÉTRICAS — Llama 3.2 vs. Anotação Manual")
print("="*55)
print(f"  Acurácia simples    : {acc:.4f}  ({acc*100:.1f}%)")
print(f"  Acurácia balanceada : {acc_bal:.4f}  ({acc_bal*100:.1f}%)")
print(f"  F1 macro            : {f1_macro:.4f}")
print("-"*55)
for cls, f1 in zip(classes, f1_cls):
    print(f"  F1 {cls:<12} : {f1:.4f}")
print("="*55)
print("\nRelatório completo:")
print(classification_report(y_true, y_pred, labels=classes, zero_division=0))

print("\nF1 macro por político:")
for pol in sorted(df['politico'].unique()):
    sub = df[df['politico'] == pol]
    if len(sub) < 3:
        print(f"  {pol:<20} — poucos exemplos ({len(sub)}), pulado")
        continue
    f1_pol  = f1_score(sub['classificacao_manual'], sub['classificacao_llama'],
                        average='macro', labels=classes, zero_division=0)
    acc_pol = accuracy_score(sub['classificacao_manual'], sub['classificacao_llama'])
    print(f"  {pol:<20} F1={f1_pol:.3f}  Acc={acc_pol:.3f}  (n={len(sub)})")

# Padrões de erro
df['correto'] = y_true.values == y_pred.values
erros = df[~df['correto']].copy()
erros['padrao_erro'] = erros['classificacao_manual'] + " → " + erros['classificacao_llama']
print(f"\nTotal de erros: {len(erros)}/{len(df)} ({len(erros)/len(df)*100:.1f}%)")
print("\nPadrões de erro:")
print(erros['padrao_erro'].value_counts().to_string())


# ── 5. Matriz de confusão ─────────────────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred, labels=classes)
fig, ax = plt.subplots(figsize=(7, 5))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes).plot(
    ax=ax, colorbar=True, cmap='Blues', values_format='d'
)
ax.set_title("Matriz de Confusão — Llama 3.2 vs. Anotação Manual", fontsize=13, pad=14)
ax.set_xlabel("Predito pelo Llama", fontsize=11)
ax.set_ylabel("Anotado manualmente", fontsize=11)
plt.tight_layout()
plt.savefig("matriz_confusao.png", dpi=150)
plt.close()
print("\nMatriz de confusão salva em: matriz_confusao.png")


# ── 6. Salva resultado completo ───────────────────────────────────────────────
df.to_csv(SAIDA_CSV, index=False, encoding='utf-8')
print(f"Resultado completo salvo em: {SAIDA_CSV}")