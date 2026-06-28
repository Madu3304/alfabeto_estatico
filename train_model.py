import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#  Caminhos 
FEATURES_DIR = "dataset/features"
MODEL_DIR = "modelo"
os.makedirs(MODEL_DIR, exist_ok=True)

#  Carrega dados 
print("Carregando dados...")
X = np.load(os.path.join(FEATURES_DIR, "X.npy"))
y = np.load(os.path.join(FEATURES_DIR, "y.npy"))
label_map = np.load(os.path.join(FEATURES_DIR, "label_map.npy"), allow_pickle=True).item()

label_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]

#  70% treino / 15% val / 15% teste
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"Treino : {X_train.shape[0]} amostras")
print(f"Val    : {X_val.shape[0]} amostras")
print(f"Teste  : {X_test.shape[0]} amostras\n")

# Definição dos modelos 
modelos = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    ),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=10, gamma="scale")),
    ]),
    "MLP": MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        max_iter=100,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
    ),
}

# Treino e avaliação
resultados = {}
report_lines = ["=" * 60, "Comparação de Modelos", "=" * 60]

for nome, modelo in modelos.items():
    print(f"Treinando {nome}...")
    modelo.fit(X_train, y_train)

    acc_val  = accuracy_score(y_val,  modelo.predict(X_val))
    y_pred   = modelo.predict(X_test)
    acc_test = accuracy_score(y_test, y_pred)

    resultados[nome] = (acc_val, acc_test, modelo, y_pred)

    linha = f"  {nome:<15}  val={acc_val:.4f}  teste={acc_test:.4f}"
    print(linha)
    report_lines.append(linha)

# Seleciona melhor modelo
melhor_nome = max(resultados, key=lambda k: resultados[k][1])
_, melhor_acc, melhor_modelo, melhor_pred = resultados[melhor_nome]

report_lines += [
    "",
    f"Melhor modelo: {melhor_nome} (acurácia no teste={melhor_acc:.4f})",
    "",
    "=" * 60,
    f"Classification Report — {melhor_nome}",
    "=" * 60,
    classification_report(y_test, melhor_pred, target_names=label_names),
]

# Salva relatório de métricas
report_path = os.path.join(MODEL_DIR, "metrics_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

# Matriz de confusão
cm = confusion_matrix(y_test, melhor_pred)

fig, ax = plt.subplots(figsize=(16, 14))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_names,
    yticklabels=label_names,
    linewidths=0.4,
    ax=ax,
)
ax.set_title(f"Matriz de Confusão — {melhor_nome} (teste={melhor_acc:.4f})", fontsize=14)
ax.set_xlabel("Predito", fontsize=12)
ax.set_ylabel("Real", fontsize=12)
plt.tight_layout()

cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150)
plt.close()

# Salva modelo e label_map
model_path = os.path.join(MODEL_DIR, "classificador.pkl")
joblib.dump(melhor_modelo, model_path)
np.save(os.path.join(MODEL_DIR, "label_map.npy"), label_map)
