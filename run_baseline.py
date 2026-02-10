import os
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression

import spacy
from datasets import load_dataset


SEED = 42


@dataclass
class RunConfig:
    dataset_name: str = "holistic-ai/EMGSD"
    text_col: str = "text"
    category_col: str = "category"
    test_size: float = 0.2
    seed: int = SEED
    spacy_model: str = "en_core_web_lg"
    max_iter: int = 1000
    C: float = 1.0
    penalty: str = "l2"  # or "none"
    solver: str = "lbfgs"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def to_binary_label(category: str) -> int:
    # stereotype -> 1, neutral/unrelated -> 0
    return 1 if category.strip().lower() == "stereotype" else 0


def embed_texts(nlp, texts, batch_size=128):
    """Compute sentence embeddings using spaCy doc.vector."""
    vectors = []
    for doc in nlp.pipe(texts, batch_size=batch_size):
        vectors.append(doc.vector)
    return np.vstack(vectors)


def main():
    cfg = RunConfig()
    set_seed(cfg.seed)

    os.makedirs("outputs", exist_ok=True)

    print(f"[INFO] Loading dataset: {cfg.dataset_name}")
    ds = load_dataset(cfg.dataset_name)

    # Some HF datasets expose 'train' only; handle both cases.
    if "train" in ds:
        data = ds["train"]
    else:
        # fallback: pick first split
        split_name = list(ds.keys())[0]
        data = ds[split_name]

    texts = data[cfg.text_col]
    y = [to_binary_label(c) for c in data[cfg.category_col]]

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts, y,
        test_size=cfg.test_size,
        random_state=cfg.seed,
        stratify=y
    )

    print(f"[INFO] Loading spaCy model: {cfg.spacy_model}")
    try:
        nlp = spacy.load(cfg.spacy_model)
    except OSError:
        raise RuntimeError(
            f"spaCy model '{cfg.spacy_model}' not found. Run:\n"
            f"python -m spacy download {cfg.spacy_model}"
        )

    print("[INFO] Computing embeddings...")
    X_train = embed_texts(nlp, X_train_text)
    X_test = embed_texts(nlp, X_test_text)

    print("[INFO] Training Logistic Regression...")
    # sklearn uses penalty='none' instead of None
    penalty = "none" if cfg.penalty.lower() in ["none", "no", "null"] else cfg.penalty

    # solver compatibility: 'lbfgs' supports l2/none; 'liblinear' supports l1/l2
    clf = LogisticRegression(
        C=cfg.C,
        penalty=penalty,
        solver=cfg.solver,
        max_iter=cfg.max_iter,
        random_state=cfg.seed,
        n_jobs=1
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print(f"[RESULT] Macro-F1 (EMGSD, binary, stratified 80/20): {macro_f1:.4f}")

    out = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "macro_f1": macro_f1,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "config": asdict(cfg),
    }
    with open("outputs/results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("[INFO] Saved outputs/results.json")


if __name__ == "__main__":
    main()