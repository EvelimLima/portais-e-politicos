"""
validar_roberta.py — Valida o Twitter-XLM-RoBERTa (cardiffnlp/twitter-xlm-roberta-base-sentiment)
contra as 426 anotações manuais usando apenas o trecho.

Uso:
    python validar_roberta.py

Saídas:
    resultado_validacao_roberta.csv
    matriz_confusao_roberta.png
"""
import io
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)

CSV_PATH  = "Classificação de Notícias Políticas em Português - Classificação de Notícias Políticas em Português.csv"
MODEL_ID  = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
SAIDA_CSV = "resultado_validacao_roberta.csv"


def mapear_label(label):
    """Mapeia saída Positive/Negative/Neutral para português."""
    mapa = {
        "positive": "Positivo",
        "negative": "Negativo",
        "neutral":  "Neutro",
    }
    return mapa.get(label.lower(), "Neutro")


# ── 1. Carrega CSV ────────────────────────────────────────────────────────────
linhas = open(CSV_PATH, encoding='utf-8').readlines()
dados  = [l for l in linhas if l.strip() and not l.startswith('link,titulo')]
df = pd.read_csv(
    io.StringIO(''.join(dados)),
    names=['link', 'titulo', 'politico', 'trecho', 'motivo_manual', 'classificacao_manual']
)
df = df.dropna(subset=['classificacao_manual', 'trecho', 'politico'])
df['classificacao_manual'] = df['classificacao_manual'].str.strip()
df['trecho']               = df['trecho'].str.strip()
df['politico']             = df['politico'].str.strip()

print(f"Registros carregados: {len(df)}")
print(df['classificacao_manual'].value_counts().to_string())


# ── 2. Carrega modelo ─────────────────────────────────────────────────────────
print(f"\nCarregando {MODEL_ID}...")
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).to("cuda")
model.eval()

def classificador(texto):
    inputs = tokenizer(texto, return_tensors="pt", truncation=True,
                       max_length=512, padding=True).to("cuda")
    with torch.no_grad():
        logits = model(**inputs).logits
    idx = logits.argmax(-1).item()
    label = model.config.id2label[idx]
    return [{"label": label}]
print("Modelo carregado.")


# ── 3. Classifica cada trecho ─────────────────────────────────────────────────
classificacoes_roberta = []
total = len(df)

for i, (_, row) in enumerate(df.iterrows(), 1):
    print(f"[{i}/{total}] {row['politico']}", end=" → ")
    try:
        resultado     = classificador(row['trecho'][:512])
        classificacao = mapear_label(resultado[0]['label'])
    except Exception as e:
        print(f"ERRO: {e}")
        classificacao = "Neutro"

    print(classificacao)
    classificacoes_roberta.append(classificacao)

df['classificacao_roberta'] = classificacoes_roberta


# ── 4. Métricas ───────────────────────────────────────────────────────────────
classes = ['Positivo', 'Negativo', 'Neutro']
y_true  = df['classificacao_manual']
y_pred  = df['classificacao_roberta']

acc      = accuracy_score(y_true, y_pred)
acc_bal  = balanced_accuracy_score(y_true, y_pred)
f1_macro = f1_score(y_true, y_pred, average='macro',  labels=classes, zero_division=0)
f1_cls   = f1_score(y_true, y_pred, average=None,     labels=classes, zero_division=0)

print("\n" + "="*55)
print("  MÉTRICAS — Twitter-XLM-RoBERTa vs. Anotação Manual")
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
    f1_pol  = f1_score(sub['classificacao_manual'], sub['classificacao_roberta'],
                        average='macro', labels=classes, zero_division=0)
    acc_pol = accuracy_score(sub['classificacao_manual'], sub['classificacao_roberta'])
    print(f"  {pol:<20} F1={f1_pol:.3f}  Acc={acc_pol:.3f}  (n={len(sub)})")

df['correto'] = y_true.values == y_pred.values
erros = df[~df['correto']].copy()
erros['padrao_erro'] = erros['classificacao_manual'] + " → " + erros['classificacao_roberta']
print(f"\nTotal de erros: {len(erros)}/{len(df)} ({len(erros)/len(df)*100:.1f}%)")
print("\nPadrões de erro:")
print(erros['padrao_erro'].value_counts().to_string())


# ── 5. Matriz de confusão ─────────────────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred, labels=classes)
fig, ax = plt.subplots(figsize=(7, 5))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes).plot(
    ax=ax, colorbar=True, cmap='Oranges', values_format='d'
)
ax.set_title("Matriz de Confusão — RoBERTa vs. Anotação Manual", fontsize=13, pad=14)
ax.set_xlabel("Predito pelo RoBERTa", fontsize=11)
ax.set_ylabel("Anotado manualmente", fontsize=11)
plt.tight_layout()
plt.savefig("matriz_confusao_roberta.png", dpi=150)
plt.close()
print("\nMatriz salva em: matriz_confusao_roberta.png")


# ── 6. Salva resultado ────────────────────────────────────────────────────────
df.to_csv(SAIDA_CSV, index=False, encoding='utf-8')
print(f"Resultado salvo em: {SAIDA_CSV}")