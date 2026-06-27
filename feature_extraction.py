"""
Pessoa 2 — Extração de Features com MediaPipe Hands
====================================================
Transforma cada imagem do dataset ASL num vetor de 63 features
(21 landmarks × 3 coordenadas x, y, z) e salva o resultado em
CSV e numpy, prontos para o treinamento (Pessoa 3).

Pré-requisitos:
    pip install mediapipe opencv-python pandas numpy tqdm

    Modelo necessário (MediaPipe 0.10+):
        Baixe o arquivo hand_landmarker.task e coloque na raiz do projeto:
        https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

Estrutura esperada do dataset (gerada pelo preprocessing.py):
    dataset/asl_alphabet_train/asl_alphabet_train/
        A/  B/  C/ ... Z/  del/  nothing/  space/

Saídas geradas em dataset/features/:
    landmarks.csv         — features + label em texto (para inspeção)
    X.npy                 — matriz (N, 63) de float32
    y.npy                 — vetor  (N,)   de int32  (índice da classe)
    label_map.npy         — dict   classe → índice  (para decodificar y)
    extraction_report.txt — relatório de detecção por classe
"""

import os
import time
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from tqdm import tqdm

# ── Configurações ──────────────────────────────────────────────────────────────
RAW_DIR    = "dataset/asl_alphabet_train/asl_alphabet_train"
OUT_DIR    = "dataset/features"
MODEL_PATH = "hand_landmarker.task"   # baixe o arquivo e coloque na raiz do projeto

CSV_PATH       = os.path.join(OUT_DIR, "landmarks.csv")
NPY_X_PATH     = os.path.join(OUT_DIR, "X.npy")
NPY_Y_PATH     = os.path.join(OUT_DIR, "y.npy")
LABEL_MAP_PATH = os.path.join(OUT_DIR, "label_map.npy")
REPORT_PATH    = os.path.join(OUT_DIR, "extraction_report.txt")

NUM_LANDMARKS    = 21
COORDS_PER_POINT = 3          # x, y, z
FEATURE_SIZE     = NUM_LANDMARKS * COORDS_PER_POINT  # 63

MIN_DETECTION_CONF = 0.3      # tolerante para imagens recortadas
SUPPORTED_EXT      = {".jpg", ".jpeg", ".png"}

os.makedirs(OUT_DIR, exist_ok=True)

# ── Inicializa MediaPipe (Tasks API — mediapipe 0.10+) ─────────────────────────
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Modelo não encontrado: '{MODEL_PATH}'\n"
        "Baixe em: https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task\n"
        "e coloque na raiz do projeto."
    )

base_options     = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
detector_options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=MIN_DETECTION_CONF,
    min_hand_presence_confidence=MIN_DETECTION_CONF,
)
hands_detector = mp_vision.HandLandmarker.create_from_options(detector_options)


# ── Transformações de retry ────────────────────────────────────────────────────
# Aplicadas em sequência quando o MediaPipe não detecta a mão na imagem original.
# Cada função recebe e retorna um array BGR (formato padrão do OpenCV).
# Útil especialmente para M, N, del e space, que têm gestos de difícil detecção.

def _flip(img: np.ndarray) -> np.ndarray:
    """Espelha horizontalmente — inverte lateralidade da mão."""
    return cv2.flip(img, 1)

def _increase_contrast(img: np.ndarray) -> np.ndarray:
    """CLAHE no canal L do espaço LAB — realça bordas sem saturar brilho."""
    lab   = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def _upscale(img: np.ndarray) -> np.ndarray:
    """Dobra a resolução — ajuda o detector em imagens pequenas ou borradas."""
    h, w = img.shape[:2]
    return cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

def _flip_and_contrast(img: np.ndarray) -> np.ndarray:
    """Combina espelhamento + contraste para casos mais difíceis."""
    return _increase_contrast(_flip(img))

RETRY_TRANSFORMS = [_flip, _increase_contrast, _upscale, _flip_and_contrast]


# ── Detecção com MediaPipe ─────────────────────────────────────────────────────

def _detect(img_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Tenta detectar landmarks em um array BGR.
    Retorna o array (63,) normalizado, ou None se não detectar.
    """
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result   = hands_detector.detect(mp_image)

    if not result.hand_landmarks:
        return None

    hand   = result.hand_landmarks[0]
    coords = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand],
        dtype=np.float32,
    )  # shape (21, 3)

    wrist   = coords[0].copy()
    coords -= wrist                                   # centraliza no pulso
    scale   = np.max(np.linalg.norm(coords, axis=1)) # distância máxima
    if scale > 1e-8:
        coords /= scale                               # normaliza escala

    return coords.flatten()  # (63,)


# ── Extração de landmarks ──────────────────────────────────────────────────────

def extract_landmarks(image_path: str) -> Optional[np.ndarray]:
    """
    Extrai e normaliza os 21 landmarks de uma imagem de mão.

    Estratégia:
        1. Tenta a imagem original.
        2. Se falhar, aplica transformações em sequência (flip, contraste,
           upscale, flip+contraste) e tenta novamente a cada passo.
        3. Descarta a imagem apenas se todas as tentativas falharem.

    Normalização aplicada em cada tentativa:
        - Centralização no pulso (landmark 0): invariante à posição no frame.
        - Escala pela distância máxima: invariante ao tamanho/distância da mão.

    Retorna:
        np.ndarray shape (63,) float32, ou None se nenhuma tentativa detectar.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None  # arquivo corrompido ou inexistente

    # Tentativa 1: imagem original
    vec = _detect(img_bgr)
    if vec is not None:
        return vec

    # Tentativas 2–5: transformações progressivas
    for transform in RETRY_TRANSFORMS:
        vec = _detect(transform(img_bgr))
        if vec is not None:
            return vec

    return None  # descartada após todas as tentativas


# ── Iteração sobre o dataset ───────────────────────────────────────────────────

def build_dataset(raw_dir: str):
    """
    Percorre todas as classes do dataset e extrai landmarks imagem a imagem.

    Retorna:
        features    : list[np.ndarray] — vetores de landmarks detectados
        labels      : list[str]        — classe correspondente a cada vetor
        per_class   : dict[str, dict]  — estatísticas de detecção por classe
    """
    class_dirs = sorted([
        d for d in os.listdir(raw_dir)
        if os.path.isdir(os.path.join(raw_dir, d))
    ])

    features: list  = []
    labels:   list  = []
    per_class: dict = {}

    for class_name in class_dirs:
        class_path  = os.path.join(raw_dir, class_name)
        image_files = [
            f for f in os.listdir(class_path)
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXT
        ]

        detected = 0
        skipped  = 0

        for fname in tqdm(image_files, desc=f"  {class_name:<8}", leave=False):
            vec = extract_landmarks(os.path.join(class_path, fname))

            if vec is not None:
                features.append(vec)
                labels.append(class_name)
                detected += 1
            else:
                skipped += 1

        per_class[class_name] = {
            "total":    len(image_files),
            "detected": detected,
            "skipped":  skipped,
        }

    return features, labels, per_class


# ── Execução principal ─────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Extração de Features — MediaPipe Hands")
    print("=" * 60)
    print(f"  Dataset : {RAW_DIR}")
    print(f"  Saída   : {OUT_DIR}")
    print(f"  Features por imagem: {FEATURE_SIZE} ({NUM_LANDMARKS} landmarks × {COORDS_PER_POINT} coords)")
    print(f"  Tentativas por imagem: 1 original + {len(RETRY_TRANSFORMS)} transformações")
    print()

    t0 = time.time()
    features, labels, per_class = build_dataset(RAW_DIR)
    elapsed = time.time() - t0

    if not features:
        print("❌  Nenhuma mão detectada. Verifique o caminho do dataset.")
        return

    # ── Converte para numpy ────────────────────────────────────────────────────
    X           = np.array(features, dtype=np.float32)        # (N, 63)
    label_names = sorted(set(labels))
    label_map   = {name: idx for idx, name in enumerate(label_names)}
    y           = np.array([label_map[l] for l in labels], dtype=np.int32)

    # ── Salva CSV (features + label em texto) ─────────────────────────────────
    col_names = [
        f"lm{i}_{axis}"
        for i in range(NUM_LANDMARKS)
        for axis in ("x", "y", "z")
    ]
    df = pd.DataFrame(X, columns=col_names)
    df.insert(0, "label", labels)
    df.to_csv(CSV_PATH, index=False)

    # ── Salva numpy ───────────────────────────────────────────────────────────
    np.save(NPY_X_PATH,     X)
    np.save(NPY_Y_PATH,     y)
    np.save(LABEL_MAP_PATH, label_map)

    # ── Relatório por classe ───────────────────────────────────────────────────
    total_imgs = sum(v["total"]    for v in per_class.values())
    total_det  = sum(v["detected"] for v in per_class.values())
    total_skip = sum(v["skipped"]  for v in per_class.values())
    det_rate   = 100 * total_det / total_imgs if total_imgs else 0

    report_lines = [
        "=" * 60,
        "Relatório de Extração — MediaPipe Hands",
        "=" * 60,
        f"{'Classe':<10} {'Total':>6} {'Detectadas':>11} {'Descartadas':>12} {'Taxa':>7}",
        "-" * 60,
    ]
    for cls, stats in per_class.items():
        t   = stats["total"]
        d   = stats["detected"]
        s   = stats["skipped"]
        pct = 100 * d / t if t else 0
        flag = "  ⚠" if pct < 70 else ""
        report_lines.append(
            f"{cls:<10} {t:>6} {d:>11} {s:>12} {pct:>6.1f}%{flag}"
        )
    report_lines += [
        "-" * 60,
        f"{'TOTAL':<10} {total_imgs:>6} {total_det:>11} {total_skip:>12} {det_rate:>6.1f}%",
        "",
        f"Shape de X   : {X.shape}",
        f"Shape de y   : {y.shape}",
        f"Nº de classes: {len(label_names)}",
        f"Classes      : {label_names}",
        f"Tempo total  : {elapsed:.1f}s",
        "",
        "Observações:",
        "  • Classe 'nothing' tende a ter baixa detecção (sem mão na imagem).",
        "  • Classes com taxa < 70% estão marcadas com ⚠ — podem precisar",
        "    de ajuste no min_detection_confidence ou revisão das imagens.",
        "  • Cada imagem é tentada até 5 vezes (original + 4 transformações)",
        "    antes de ser descartada: flip, contraste (CLAHE), upscale, flip+contraste.",
        "",
        "Arquivos gerados:",
        f"  {CSV_PATH}",
        f"  {NPY_X_PATH}",
        f"  {NPY_Y_PATH}",
        f"  {LABEL_MAP_PATH}",
        f"  {REPORT_PATH}",
        "=" * 60,
    ]

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")

    print("\n✓ Extração concluída com sucesso!")


if __name__ == "__main__":
    main()