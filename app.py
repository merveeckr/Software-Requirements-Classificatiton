import os
import sys
import time
import difflib
from typing import List, Dict

import numpy as np
import pandas as pd
import streamlit as st

# Python 3.13 guard (PyTorch henüz desteklemiyor olabilir)
if sys.version_info >= (3, 13):
    st.error("Python 3.13 ile PyTorch/Transformers henüz stabil değil. Lütfen Python 3.10–3.12 kullanın.")
    st.stop()

# Lazy import: torch/transformers/bil yalnızca guard sonrası
try:
    from transformers import AutoTokenizer
    from bil import (
        BertBiLSTMCNN,
        LABEL_COLS,
        TEXT_COL,
        MODEL_NAME,
        MAX_LEN,
        DEVICE,
        THRESH,
        load_gemma_model,
        generate_ai_suggestion,
    )
except Exception as import_err:
    st.error(f"Kütüphane import hatası: {import_err}. Lütfen Python sürümünüzü ve paket kurulumlarınızı kontrol edin.")
    st.stop()


@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BertBiLSTMCNN(bert_model_name=model_name, num_labels=len(LABEL_COLS))
    import torch  # local import to avoid top-level import on unsupported envs
    model.to(DEVICE)
    # Eğer eğitilmiş ağırlıklar varsa yükle
    if os.path.exists("best_model.pt"):
        state = torch.load("best_model.pt", map_location=DEVICE)
        model.load_state_dict(state, strict=False)
    model.eval()
    return model, tokenizer


def batch_predict(model: BertBiLSTMCNN, tokenizer, texts: List[str], max_len: int, threshold: float) -> np.ndarray:
    predictions = []
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        enc = tokenizer(
            chunk,
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt",
        )
        import torch  # local import
        with torch.no_grad():
            logits = model(enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE))
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= threshold).astype(int)
        predictions.append(preds)
    return np.vstack(predictions) if predictions else np.zeros((0, len(LABEL_COLS)), dtype=int)


def build_editable_frame(df: pd.DataFrame, preds: np.ndarray) -> pd.DataFrame:
    pred_df = pd.DataFrame(preds, columns=[f"AI_{c}" for c in LABEL_COLS])
    merged = pd.concat([df.reset_index(drop=True), pred_df], axis=1)
    # Kullanıcı işaretlemesi için sütunlar
    for c in LABEL_COLS:
        merged[f"User_{c}"] = merged.get(c, 0)
    return merged


st.set_page_config(page_title="Gereksinim Analizi", layout="wide")

def build_llm_prompt(requirement: str, missing: List[str]) -> str:
    missing_str = ", ".join(missing)
    return (
        "Aşağıdaki gereksinimi, eksik yönlerini gidererek TEK CÜMLE hâlinde yeniden yaz.\n"
        "- Girişi tekrar etme, alıntılama yapma.\n"
        "- Yalnızca şu formatta dön: İyileştirilmiş gereksinim: <cümle>\n"
        "- Türkçe yaz. Belirsizlikten kaçın (net, ölçülebilir, doğrulanabilir).\n"
        "- Gerektiğinde kabul ölçütlerini cümle içine açık ve sayısal geçir.\n"
        "- Madde işareti, liste, açıklama ekleme.\n"
        f"Eksikler: {missing_str}\n"
        f"Girdi: {requirement}"
    )

def _normalize_text(s: str) -> str:
    return str(s).strip().lower().rstrip('.').replace('\n', ' ')

CLAUSE_BY_LABEL = {
    'Unambiguous': "belirsiz terimler kullanılmadan net olarak",
    'Verifiable': "ölçülebilir ve doğrulanabilir kabul kriterleri ile",
    'Complete': "girdi, işlem ve çıktıları tanımlanmış olarak",
    'Conforming': "kurumsal standart ve yönergelere uygun şekilde",
    'Correct': "alan tanımlarına ve iş kurallarına uygun olarak",
    'Feasible': "mevcut sistem kısıtları içinde uygulanabilir düzeyde",
    'Necessary': "iş hedefleri açısından gerekli olan kapsamda",
    'Singular': "tek bir amacı ifade edecek şekilde",
    'Appropriate': "hedef kullanıcı kitlesine uygun biçimde",
}

def rule_based_improvement(requirement: str, missing: List[str]) -> str:
    base = _normalize_text(requirement)
    clauses = [CLAUSE_BY_LABEL.get(m, "") for m in missing]
    clauses = [c for c in clauses if c]
    if clauses:
        clause_text = ", ".join(clauses)
        improved = f"İyileştirilmiş gereksinim: {base} ve {clause_text} olacaktır."
    else:
        improved = f"İyileştirilmiş gereksinim: {base} olacaktır."
    return improved

def enforce_improvement(requirement: str, generated: str, missing: List[str]) -> str:
    text = generated.strip()
    low = text.lower()
    if "iyileştirilmiş gereksinim" in low:
        try:
            start = low.index("iyileştirilmiş gereksinim")
            text = text[start:].split("\n",1)[0]
            if ':' in text:
                text = text.split(':',1)[1].strip()
        except Exception:
            pass
    text = text.split("\n",1)[0].strip().strip('"').strip("'")
    r_base = _normalize_text(requirement)
    r_out = _normalize_text(text)
    sim = difflib.SequenceMatcher(None, r_base, r_out).ratio()
    if sim >= 0.9 or len(r_out) <= len(r_base):
        return rule_based_improvement(requirement, missing)
    return f"İyileştirilmiş gereksinim: {text.rstrip('.')}." if not text.lower().startswith("iyileştirilmiş gereksinim") else text
st.title("Gereksinim Analizi: BERTürk + Gemma Öneri")

with st.sidebar:
    st.markdown("**Model Ayarları**")
    model_name = st.text_input("BERT Model", value=MODEL_NAME)
    threshold = st.slider("Eşik (sigmoid)", min_value=0.05, max_value=0.95, value=float(THRESH), step=0.05)
    st.divider()
    st.markdown("**LLM Ayarları**")
    llm_backend = st.selectbox("LLM Backend", options=["HF", "GGUF"], index=0)
    llm_offline = st.checkbox("HF offline", value=True)
    llm_models_input = st.text_input("LLM modelleri (; ile)", value="")
    llm_models = [m.strip() for m in llm_models_input.split(";") if m.strip()]
    active_llm = st.selectbox("Kullanılacak LLM", llm_models) if llm_models else None

uploaded = st.file_uploader("CSV yükleyin", type=["csv"])

if uploaded is None:
    st.write("CSV yükleyin ve analiz edin.")
    st.stop()

if uploaded is not None:
    df = pd.read_csv(uploaded)
    if TEXT_COL not in df.columns:
        st.error(f"CSV içinde '{TEXT_COL}' kolonu bulunamadı.")
        st.stop()
    # Eksikse label kolonları ekle
    for c in LABEL_COLS:
        if c not in df.columns:
            df[c] = 0

    st.success(f"Yüklendi: {uploaded.name}, satır: {len(df)}")

    with st.spinner("Model yükleniyor..."):
        model, tokenizer = load_model_and_tokenizer(model_name)

    texts = df[TEXT_COL].fillna("").astype(str).tolist()
    with st.spinner("Tahmin yapılıyor..."):
        preds = batch_predict(model, tokenizer, texts, MAX_LEN, threshold)

    work = build_editable_frame(df[[TEXT_COL] + LABEL_COLS], preds)
    # Ensure checkbox-compatible dtypes (bool) for editor
    try:
        work[TEXT_COL] = work[TEXT_COL].astype(str)
        for c in LABEL_COLS:
            if c in work.columns:
                work[c] = work[c].fillna(0).astype(int).astype(bool)
            ai_c = f"AI_{c}"
            if ai_c in work.columns:
                work[ai_c] = work[ai_c].fillna(0).astype(int).astype(bool)
            user_c = f"User_{c}"
            if user_c in work.columns:
                work[user_c] = work[user_c].fillna(0).astype(int).astype(bool)
    except Exception:
        pass

    st.subheader("Sonuçlar")
    ai_cols = [f"AI_{c}" for c in LABEL_COLS]
    user_cols = [f"User_{c}" for c in LABEL_COLS]

    left, right = st.columns(2)
    with left:
        st.markdown("**AI tahminleri**")
        st.dataframe(
            work[[TEXT_COL] + ai_cols],
            use_container_width=True,
            height=500,
        )
    with right:
        st.markdown("**Kullanıcı işaretlemeleri**")
        user_view = work[[TEXT_COL] + user_cols].copy()
        edited_user = st.data_editor(
            user_view,
            num_rows="dynamic",
            use_container_width=True,
            height=500,
            column_config={
                TEXT_COL: st.column_config.TextColumn("Gereksinim", width=400),
                **{uc: st.column_config.CheckboxColumn(uc.replace("User_", "Kullanıcı ")) for uc in user_cols},
            },
        )

    # Per-row agreement
    try:
        # Coerce user-edited values to booleans robustly
        for uc in user_cols:
            if uc in edited_user.columns:
                edited_user[uc] = edited_user[uc].apply(
                    lambda v: bool(v) if isinstance(v, (bool, np.bool_, int, np.integer)) else str(v).strip().lower() in ("1","true","yes","on","x")
                )
        ai_mat = work[ai_cols].astype(bool).to_numpy()
        user_mat = edited_user[user_cols].astype(bool).to_numpy()
        agree_ratio = (ai_mat == user_mat).mean(axis=1)
        summary = pd.DataFrame({
            TEXT_COL: work[TEXT_COL].astype(str).str.slice(0, 80) + "...",
            "Uyum_orani": np.round(agree_ratio, 3),
            "Uyum_sayisi": (ai_mat == user_mat).sum(axis=1),
            "Toplam_label": ai_mat.shape[1]
        })
        st.markdown("**AI vs Kullanıcı Uyum Özeti**")
        st.dataframe(summary, use_container_width=True, height=240)
    except Exception as e:
        st.warning(f"Uyum hesabı yapılamadı: {e}")

    # Birleştirilmiş çerçeve (AI + User) diğer adımlar için
    merged = work[[TEXT_COL] + ai_cols].copy()
    for uc in user_cols:
        merged[uc] = edited_user[uc].astype(bool)

    # Eksik etiketlerin çıkarımı
    def row_missing(row) -> List[str]:
        return [c for c in LABEL_COLS if int(row.get(f"AI_{c}", 0)) == 0]

    merged["Eksikler_AI"] = merged.apply(row_missing, axis=1)

    st.download_button(
        label="CSV indir",
        data=merged.to_csv(index=False).encode("utf-8"),
        file_name="gereksinim_sonuclari.csv",
        mime="text/csv",
    )

    st.subheader("LLM ile Öneri Üretimi")
    with st.expander("Satır seçerek öneri üret"):
        sel_idx = st.number_input("Satır index", min_value=0, max_value=len(merged)-1, value=0, step=1)
        if st.button("Seçili satıra öneri üret"):
            req_text = merged.iloc[sel_idx][TEXT_COL]
            miss = [c for c in LABEL_COLS if int(merged.iloc[sel_idx].get(f"AI_{c}", 0)) == 0]
            if not miss:
                st.warning("AI'ya göre eksik yok.")
            else:
                try:
                    prompt = build_llm_prompt(req_text, miss)
                    if active_llm is None:
                        st.error("LLM modeli seçin.")
                    else:
                        if llm_backend == "HF":
                            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
                            import torch
                            cache = st.session_state.setdefault("_hf_gen_cache", {})
                            key = (active_llm, llm_offline)
                            if key not in cache:
                                tok = AutoTokenizer.from_pretrained(active_llm, local_files_only=llm_offline)
                                mdl = AutoModelForCausalLM.from_pretrained(active_llm, local_files_only=llm_offline)
                                cache[key] = pipeline("text-generation", model=mdl, tokenizer=tok, device=0 if torch.cuda.is_available() else -1, return_full_text=False)
                            gen = cache[key]
                            t0 = time.perf_counter()
                            out = gen(prompt, max_new_tokens=96, do_sample=False, num_beams=4)
                            dt = time.perf_counter() - t0
                            text = out[0].get('generated_text','').strip()
                        else:
                            try:
                                from llama_cpp import Llama
                            except Exception:
                                st.error("llama-cpp-python yüklü değil.")
                                raise
                            cache = st.session_state.setdefault("_llama_cache", {})
                            if active_llm not in cache:
                                cache[active_llm] = Llama(model_path=active_llm, n_ctx=2048)
                            llm = cache[active_llm]
                            t0 = time.perf_counter()
                            out = llm.create_completion(prompt=prompt, max_tokens=96, temperature=0.2, top_p=0.9)
                            dt = time.perf_counter() - t0
                            text = out["choices"][0]["text"].strip()
                        final_text = enforce_improvement(req_text, text, miss)
                        st.text_area("AI Önerisi", value=final_text, height=200)
                        st.info(f"Süre: {dt:.2f} sn")
                except Exception as e:
                    st.error(f"LLM çalıştırılamadı: {e}")

    st.subheader("Tekil Gereksinim Analizi ve Öneri")
    single_req = st.text_area("Gereksinimi girin", placeholder="Gereksinimi buraya yapıştırın", height=120)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Eksikleri Analiz Et") and single_req.strip():
            enc = tokenizer(
                [single_req],
                truncation=True,
                padding=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            )
            import torch  # local import
            with torch.no_grad():
                logits = model(enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE))
                probs = torch.sigmoid(logits).cpu().numpy()[0]
                single_preds = (probs >= threshold).astype(int)
            single_missing = [LABEL_COLS[i] for i, v in enumerate(single_preds) if v == 0]
            st.session_state["single_missing"] = single_missing
            st.write("Eksikler:", single_missing if single_missing else "Yok")
    with col2:
        if st.button("LLM ile Öneri Üret") and single_req.strip():
            miss = st.session_state.get("single_missing", [])
            if not miss:
                st.warning("Önce eksikleri analiz edin veya eksik yok.")
            else:
                try:
                    prompt = build_llm_prompt(single_req, miss)
                    if active_llm is None:
                        st.error("LLM modeli seçin.")
                    else:
                        if llm_backend == "HF":
                            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
                            import torch
                            cache = st.session_state.setdefault("_hf_gen_cache", {})
                            key = (active_llm, llm_offline)
                            if key not in cache:
                                tok = AutoTokenizer.from_pretrained(active_llm, local_files_only=llm_offline)
                                mdl = AutoModelForCausalLM.from_pretrained(active_llm, local_files_only=llm_offline)
                                cache[key] = pipeline("text-generation", model=mdl, tokenizer=tok, device=0 if torch.cuda.is_available() else -1, return_full_text=False)
                            gen = cache[key]
                            t0 = time.perf_counter()
                            out = gen(prompt, max_new_tokens=96, do_sample=False, num_beams=4)
                            dt = time.perf_counter() - t0
                            text = out[0].get('generated_text','').strip()
                        else:
                            try:
                                from llama_cpp import Llama
                            except Exception:
                                st.error("llama-cpp-python yüklü değil.")
                                raise
                            cache = st.session_state.setdefault("_llama_cache", {})
                            if active_llm not in cache:
                                cache[active_llm] = Llama(model_path=active_llm, n_ctx=2048)
                            llm = cache[active_llm]
                            t0 = time.perf_counter()
                            out = llm.create_completion(prompt=prompt, max_tokens=96, temperature=0.2, top_p=0.9)
                            dt = time.perf_counter() - t0
                            text = out["choices"][0]["text"].strip()
                        final_text = enforce_improvement(single_req, text, miss)
                        st.text_area("İyileştirilmiş Gereksinim", value=final_text, height=180)
                        st.info(f"Süre: {dt:.2f} sn")
    # end uploaded path

