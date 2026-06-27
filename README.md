# Reconhecimento de Alfabeto em Libras/ASL

Projeto de visão computacional para reconhecer gestos estáticos do alfabeto
de linguagem de sinais a partir de imagens, usando MediaPipe para extração de
landmarks e aprendizado de máquina para classificação.


> **Atenção:** as pastas `dataset/` e o arquivo `hand_landmarker.task` **não
> devem ser versionados**

---

## Dataset

Utilizamos o **ASL Alphabet** disponível no Kaggle:

- Link: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
- 87.000 imagens (200×200 px), 29 classes: A–Z + `del`, `nothing`, `space`

---

### Instale as dependências

```bash
pip install -r requirements.txt
```

---

## Como executar

Execute os scripts **na ordem abaixo**. Cada etapa depende da saída da anterior.

### 1 — Pré-processamento

Cria os pipelines de treino, validação e teste em memória.

```bash
python preprocessing.py
```

### 2 — Extração de Features

Percorre todas as imagens, extrai os 21 landmarks da mão com MediaPipe e salva
os vetores de features em `dataset/features/`.

```bash
python feature_extraction.py
```

Saídas geradas:

| Arquivo | Descrição |
|---|---|
| `landmarks.csv` | Features + label legível (para inspeção) |
| `X.npy` | Matriz `(N, 63)` float32 — entrada do modelo |
| `y.npy` | Vetor `(N,)` int32 — labels numéricas |
| `label_map.npy` | Dicionário `{'A': 0, 'B': 1, …}` |
| `extraction_report.txt` | Taxa de detecção por classe |

### 3 — Modelo de Classificação

Carregue os arquivos gerados no passo 2:

```python
import numpy as np

X         = np.load("dataset/features/X.npy")
y         = np.load("dataset/features/y.npy")
label_map = np.load("dataset/features/label_map.npy", allow_pickle=True).item()
```


### 4 — Aplicação em Tempo Real


### 5 — Avaliação e Documentação

---

## Observações técnicas

- **Python:** 3.10 ou superior (3.12+ recomendado).
- **MediaPipe:** versão 0.10+. A API `mp.solutions` foi removida; o projeto
  usa a Tasks API com o arquivo `.task`.
- **Normalização dos landmarks:** cada vetor é centralizado no pulso
  (landmark 0) e normalizado pela distância máxima entre os pontos, tornando
  as features invariantes a posição e escala da mão no frame.