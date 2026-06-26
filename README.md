# Portais e Políticos

Ferramenta modular de código aberto para monitoramento e análise de sentimento de notícias políticas regionais via modelos de linguagem locais.

Desenvolvida no contexto do **Projeto PIBIC 2025/2026** — Instituto de Computação, Universidade Federal do Amazonas (IComp/UFAM).

> ⚠️ **Projeto em desenvolvimento.** As classificações de sentimento são geradas automaticamente por modelos de linguagem e podem conter erros. Consulte sempre os links das notícias originais para uma leitura completa e contextualizada.

---

## Demonstração

📹 Vídeo demonstrativo: [YouTube](https://youtu.be/20AWtBJaK8c)

---

## Funcionalidades

* Cadastro de portais de notícias com descoberta automática de feed RSS
* Registro de políticos com variações nominais (alcunhas, títulos, abreviações)
* Pipeline ETL automatizado: extração RSS, limpeza textual e deduplicação
* Segmentação adaptativa de trechos ao redor de cada menção política
* Análise de sentimento local via **Llama 3.2 3B** (decoder/prompt) e **Twitter-XLM-RoBERTa** (encoder)
* Dashboard interativo com evolução temporal, balanço por político, nuvem de palavras e auditoria com link para notícia original
* Processamento 100% local — sem APIs pagas, sem envio de dados externos

---

## Requisitos de Hardware


| Componente        | Mínimo recomendado              |
| ----------------- | -------------------------------- |
| GPU NVIDIA (CUDA) | 6 GB VRAM                        |
| RAM               | 8 GB                            |
| Armazenamento     | 10 GB livres (cache dos modelos) |

Testado com NVIDIA GeForce RTX 3050 6 GB Laptop GPU, CUDA 12.1.

---

## Instalação

**1. Clone o repositório**

```bash
git clone https://github.com/EvelimLima/portais-e-politicos
cd portais-e-politicos
```

**2. Crie e ative o ambiente conda**

```bash
conda create -n portais python=3.11
conda activate portais
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

**3. Instale as dependências**

```bash
pip install -r requirements.txt
```

**4. Configure as credenciais**

```bash
cp config_exemplo.py config.py
# Edite config.py com suas credenciais MySQL e token do HuggingFace
```

**5. Crie o banco de dados**

```bash
mysql -u root -p < banco/schema.sql
```

**6. Acesso ao Llama 3.2**

Acesse [huggingface.co/meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct), aceite os termos de uso e insira seu token HuggingFace no `config.py`.

---

## Como usar

**Interface web**

```bash
streamlit run app.py
```

**Classificação via terminal (incremental — retoma se interrompida)**

```bash
# Llama 3.2 3B (padrão)
python classifica.py

# Twitter-XLM-RoBERTa
python classifica.py --modelo roberta
```

---

## Estrutura do repositório

```
portais-e-politicos/
├── app.py                      # Interface Streamlit (4 telas)
├── classifica.py               # Classificação incremental via terminal
├── coletor_sbbd.py             # Pipeline ETL de coleta RSS
├── config_exemplo.py           # Template de configuração (copie para config.py)
├── requirements.txt
├── banco/
│   └── schema.sql              # Estrutura do banco MySQL
└── validacao/
    ├── classificar_llama.py    # Validação do Llama vs. anotações manuais
    ├── classificar_bert.py     # Validação do BERT Multilingual
    └── classificar_roberta.py  # Validação do Twitter-XLM-RoBERTa
```

---

## Resultados da Validação

Validação sobre 426 notícias políticas do Amazonas anotadas manualmente.


| Modelo              | Tipo    | Acurácia | F1 macro  | Erros graves (Pos↔Neg) |
| ------------------- | ------- | --------- | --------- | ----------------------- |
| **Llama 3.2 3B**    | Decoder | **69,2%** | **0,642** | **19**                  |
| Twitter-XLM-RoBERTa | Encoder | 58,0%     | 0,573     | 16                      |
| BERT Multilingual   | Encoder | 56,6%     | 0,495     | 58                      |

A principal dificuldade dos três modelos é a classe Neutro — comportamento esperado em textos jornalísticos formais. A confusão entre Positivo e Negativo (erro mais grave) é mínima no Llama e no RoBERTa.

## Licença

MIT License — veja [LICENSE](https://github.com/EvelimLima/portais-e-politicos/tree/main?tab=MIT-1-ov-file) para detalhes.
