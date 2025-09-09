import os
import time
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, hamming_loss, accuracy_score

from transformers import AutoTokenizer

from sentence_transformers import SentenceTransformer

from bil import (
    BertBiLSTMCNN,
    LABEL_COLS,
    TEXT_COL,
    MAX_LEN,
    DEVICE,
    THRESH,
)
# Offline/SSL settings (corporate environments)
OFFLINE = os.environ.get("HF_OFFLINE", "0") == "1"
try:
    import certifi  # type: ignore
    ca = certifi.where()
    os.environ.setdefault("SSL_CERT_FILE", ca)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", ca)
except Exception:
    pass



@dataclass
class BenchmarkResult:
    name: str
    train_seconds: float
    infer_seconds_per_sample: float
    val_f1_macro: float
    val_hamming: float
    val_accuracy: float


def load_dataset(csv_path: str, test_size: float = 0.15, seed: int = 42):
    df = pd.read_csv(csv_path)
    texts = df[TEXT_COL].fillna("").astype(str).tolist()
    labels = df[LABEL_COLS].astype(float).values
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=test_size, random_state=seed, shuffle=True
    )
    return X_train, X_val, y_train, y_val


class SimpleTextDataset(Dataset):
    def __init__(self, texts: List[str], labels: np.ndarray, tokenizer, max_len: int):
        self.texts = texts
        self.labels = labels.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        item = {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }
        return item


def run_bert_bilstm_cnn(csv_path: str, model_name: str, epochs: int = 1, batch_size: int = 8, lr: float = 1e-5) -> BenchmarkResult:
    X_train, X_val, y_train, y_val = load_dataset(csv_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=OFFLINE)

    train_ds = SimpleTextDataset(X_train, y_train, tokenizer, MAX_LEN)
    val_ds = SimpleTextDataset(X_val, y_val, tokenizer, MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = BertBiLSTMCNN(bert_model_name=model_name, num_labels=len(LABEL_COLS)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    t0 = time.perf_counter()
    model.train()
    for _ in range(epochs):
        for batch in train_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    train_seconds = time.perf_counter() - t0

    # Eval
    model.eval()
    all_preds, all_labels = [], []
    n_tokens = 0
    with torch.no_grad():
        # warmup
        for batch in val_loader:
            model(batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE))
            break
        t1 = time.perf_counter()
        n_samples = 0
        for batch in val_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].cpu().numpy()
            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= THRESH).astype(int)
            all_preds.append(preds)
            all_labels.append(labels)
            n_samples += input_ids.size(0)
        t2 = time.perf_counter()
    infer_seconds_per_sample = (t2 - t1) / max(1, n_samples)

    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_labels)
    f1m = f1_score(y_true, y_pred, average='macro', zero_division=0)
    ham = hamming_loss(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    return BenchmarkResult(
        name=f"BERT+BiLSTM+CNN ({model_name})",
        train_seconds=train_seconds,
        infer_seconds_per_sample=infer_seconds_per_sample,
        val_f1_macro=f1m,
        val_hamming=ham,
        val_accuracy=acc,
    )


def run_st_embedding_logreg(csv_path: str, st_model_name: str, threshold: float = 0.5) -> BenchmarkResult:
    X_train, X_val, y_train, y_val = load_dataset(csv_path)
    # If OFFLINE, st_model_name must be a local directory
    if OFFLINE and not os.path.isdir(st_model_name):
        raise RuntimeError(f"OFFLINE=1: SentenceTransformer path not found: {st_model_name}")
    st_model = SentenceTransformer(st_model_name)

    # Encode
    t0 = time.perf_counter()
    emb_train = st_model.encode(X_train, batch_size=64, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
    emb_val = st_model.encode(X_val, batch_size=64, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
    t1 = time.perf_counter()

    # Train One-vs-Rest Logistic Regression
    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, n_jobs=-1))
    clf.fit(emb_train, y_train)
    t2 = time.perf_counter()

    # Inference
    t3 = time.perf_counter()
    probs = clf.predict_proba(emb_val)
    preds = (probs >= threshold).astype(int)
    t4 = time.perf_counter()

    f1m = f1_score(y_val, preds, average='macro', zero_division=0)
    ham = hamming_loss(y_val, preds)
    acc = accuracy_score(y_val, preds)

    train_seconds = (t1 - t0) + (t2 - t1)
    infer_seconds_per_sample = (t4 - t3) / max(1, len(X_val))

    return BenchmarkResult(
        name=f"ST+LogReg ({st_model_name})",
        train_seconds=train_seconds,
        infer_seconds_per_sample=infer_seconds_per_sample,
        val_f1_macro=f1m,
        val_hamming=ham,
        val_accuracy=acc,
    )


def main():
    csv_path = os.environ.get("CSV_PATH", "/workspace/yeni.csv")
    bert_model = os.environ.get("BERT_MODEL", "dbmdz/bert-base-turkish-uncased")
    # ST models: allow comma/semicolon separated local paths via ST_MODELS
    st_env = os.environ.get("ST_MODELS", "")
    if st_env.strip():
        sep = ";" if ";" in st_env else ","
        candidates_st = [s.strip() for s in st_env.split(sep) if s.strip()]
    else:
        candidates_st = [
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "intfloat/multilingual-e5-small",
            "intfloat/multilingual-e5-base",
            "sentence-transformers/LaBSE",
        ]

    results: List[BenchmarkResult] = []

    print("Running BERT+BiLSTM+CNN fine-tune (1 epoch) ...")
    try:
        res_bert = run_bert_bilstm_cnn(csv_path, bert_model, epochs=int(os.environ.get("EPOCHS", "1")))
        results.append(res_bert)
    except Exception as e:
        print("BERT+BiLSTM+CNN failed:", e)

    print("\nRunning Sentence-Transformers + LogisticRegression ...")
    for st_name in candidates_st:
        try:
            res_st = run_st_embedding_logreg(csv_path, st_name)
            results.append(res_st)
        except Exception as e:
            print(f"{st_name} failed:", e)

    if results:
        print("\n=== Benchmark Results ===")
        for r in results:
            print(f"- {r.name}")
            print(f"  train_s: {r.train_seconds:.2f} | infer_s/sample: {r.infer_seconds_per_sample*1000:.2f} ms | F1_macro: {r.val_f1_macro:.4f} | Hamming: {r.val_hamming:.4f} | Acc: {r.val_accuracy:.4f}")
    else:
        print("No results produced.")


if __name__ == "__main__":
    main()

