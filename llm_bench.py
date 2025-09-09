import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from bil import (
    BertBiLSTMCNN,
    LABEL_COLS,
    TEXT_COL,
    MAX_LEN,
    DEVICE,
    THRESH,
)

# Optional llama.cpp backend
try:
    from llama_cpp import Llama  # type: ignore
    HAS_LLAMA = True
except Exception:
    HAS_LLAMA = False

# Offline/SSL settings for corporate environments
OFFLINE = os.environ.get("HF_OFFLINE", "0") == "1"
try:
    import certifi  # type: ignore
    ca = certifi.where()
    os.environ.setdefault("SSL_CERT_FILE", ca)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", ca)
except Exception:
    pass


@dataclass
class LlmBenchResult:
    name: str
    count: int
    repair_rate: float
    delta_hamming: float
    infer_ms_per_sample: float


@torch.no_grad()
def load_classifier_and_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name, local_files_only=os.environ.get("HF_OFFLINE", "0") == "1")
    clf = BertBiLSTMCNN(bert_model_name=model_name, num_labels=len(LABEL_COLS)).to(DEVICE).eval()
    if os.path.exists("best_model.pt"):
        clf.load_state_dict(torch.load("best_model.pt", map_location=DEVICE), strict=False)
    return clf, tok


def predict_labels_batch(model, tokenizer, texts: List[str], max_len: int, thresh: float) -> np.ndarray:
    preds = []
    bs = 16
    for i in range(0, len(texts), bs):
        chunk = texts[i:i+bs]
        enc = tokenizer(chunk, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
        with torch.no_grad():
            logits = model(enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE))
            probs = torch.sigmoid(logits).cpu().numpy()
            pred = (probs >= thresh).astype(int)
        preds.append(pred)
    return np.vstack(preds) if preds else np.zeros((0, len(LABEL_COLS)), dtype=int)


def build_prompt(requirement: str, missing: List[str]) -> str:
    return (
        "Yalnızca TÜRKÇE yaz. Girişi tekrar etme.\n"
        "Belirsizliği kaldır, doğrulanabilir ve ölçülebilir hale getir.\n"
        f"Eksikleri gider: {', '.join(missing)}\n\n"
        f"Gereksinim: {requirement}\n"
        "ÇIKTI: Yalnızca tek satır 'İyileştirilmiş gereksinim: <cümle>' yaz."
    )


def run_llama_cpp(model_path: str, prompts: List[str], max_new_tokens: int = 128) -> Tuple[List[str], float]:
    if not HAS_LLAMA:
        raise RuntimeError("llama-cpp-python not installed")
    llm = Llama(model_path=model_path, n_ctx=2048, n_threads=int(os.environ.get("LLAMA_THREADS", "6")))
    outputs = []
    # warmup
    _ = llm("test")
    t0 = time.perf_counter()
    for p in prompts:
        out = llm.create_completion(prompt=p, max_tokens=max_new_tokens, temperature=0.2, top_p=0.9)
        text = out["choices"][0]["text"].strip()
        outputs.append(text)
    t1 = time.perf_counter()
    ms = (t1 - t0) * 1000 / max(1, len(prompts))
    return outputs, ms


def run_hf_pipeline(model_id: str, prompts: List[str], max_new_tokens: int = 128) -> Tuple[List[str], float]:
    from transformers import pipeline
    # Offline-friendly: if OFFLINE and model_id is a local directory, preload objects
    if OFFLINE and os.path.isdir(model_id):
        tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        mdl = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=True)
        gen = pipeline("text-generation", model=mdl, tokenizer=tok, device=0 if torch.cuda.is_available() else -1, return_full_text=False)
    else:
        gen = pipeline("text-generation", model=model_id, device=0 if torch.cuda.is_available() else -1, return_full_text=False)
    outputs = []
    # warmup
    _ = gen("test", max_new_tokens=8)
    t0 = time.perf_counter()
    for p in prompts:
        o = gen(p, max_new_tokens=max_new_tokens, do_sample=False, num_beams=4)
        outputs.append(o[0].get("generated_text", "").strip())
    t1 = time.perf_counter()
    ms = (t1 - t0) * 1000 / max(1, len(prompts))
    return outputs, ms


def extract_sentence(text: str) -> str:
    low = text.lower()
    if "iyileştirilmiş gereksinim" in low:
        try:
            start = low.index("iyileştirilmiş gereksinim")
            text = text[start:].split("\n", 1)[0]
            if ":" in text:
                text = text.split(":", 1)[1].strip()
        except Exception:
            pass
    return text.split("\n", 1)[0].strip().strip('"').strip("'")


def llm_bench(csv_path: str, classifier_model_name: str, llm_specs: List[Tuple[str, str]]) -> List[LlmBenchResult]:
    df = pd.read_csv(csv_path)
    texts = df[TEXT_COL].fillna("").astype(str).tolist()
    clf, tok = load_classifier_and_tokenizer(classifier_model_name)
    base_preds = predict_labels_batch(clf, tok, texts, MAX_LEN, THRESH)

    results: List[LlmBenchResult] = []
    for name, kind in llm_specs:
        # Eksik etiketleri çıkar ve promptları hazırla
        miss_list = []
        prompts = []
        for i, t in enumerate(texts):
            missing = [LABEL_COLS[j] for j, v in enumerate(base_preds[i]) if v == 0]
            miss_list.append(missing)
            prompts.append(build_prompt(t, missing))

        # Çıktıları üret
        if kind == "gguf":
            outputs, ms = run_llama_cpp(name, prompts)
        else:
            outputs, ms = run_hf_pipeline(name, prompts)

        improved = [extract_sentence(o) for o in outputs]
        new_preds = predict_labels_batch(clf, tok, improved, MAX_LEN, THRESH)

        # Repair rate: eksik olan etiketlerden kaçını 1'e çevirdi
        repaired = []
        delta_h = []
        for i in range(len(texts)):
            base = base_preds[i]
            new = new_preds[i]
            missing_idx = [j for j, v in enumerate(base) if v == 0]
            if missing_idx:
                r = np.mean([int(new[j] == 1) for j in missing_idx])
                repaired.append(r)
            # Hamming delta: 1 - exact match proportion difference
            delta_h.append(np.mean(base != new) - np.mean(base != base))  # simplifies to mean(base!=new)

        results.append(LlmBenchResult(
            name=name,
            count=len(texts),
            repair_rate=float(np.mean(repaired) if repaired else 0.0),
            delta_hamming=float(np.mean(delta_h)),
            infer_ms_per_sample=ms,
        ))

    return results


def main():
    csv_path = os.environ.get("CSV_PATH", "/workspace/yeni.csv")
    clf_model = os.environ.get("BERT_MODEL", "dbmdz/bert-base-turkish-uncased")
    # LLM listesi: GGUF yolunu "gguf", HF model id'yi "hf" ile işaretle
    # Örnek: LLM_SPECS=C:\\models\\gemma-2b.Q4.gguf:gguf;google/gemma-2b-it:hf
    specs_env = os.environ.get("LLM_SPECS", "")
    specs: List[Tuple[str, str]] = []
    if specs_env.strip():
        parts = specs_env.split(";")
        for p in parts:
            if not p.strip():
                continue
            # Windows path contains a drive letter 'C:\\' -> split from the right
            name, kind = p.rsplit(":", 1)
            specs.append((name.strip(), kind.strip()))
    else:
        specs = []

    if not specs:
        print("No LLM specs provided. Set LLM_SPECS to run (e.g., C:\\models\\gemma.gguf:gguf;google/gemma-2b-it:hf)")
        return

    results = llm_bench(csv_path, clf_model, specs)
    print("\n=== LLM Bench Results ===")
    for r in results:
        print(f"- {r.name} | repair_rate: {r.repair_rate:.3f} | Δhamming: {r.delta_hamming:.3f} | {r.infer_ms_per_sample:.1f} ms/sample")


if __name__ == "__main__":
    main()

