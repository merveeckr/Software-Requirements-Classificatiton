import os
import sys
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
st.title("Gereksinim Analizi: BERTürk + Gemma Öneri")

with st.sidebar:
    st.markdown("**Model Ayarları**")
    model_name = st.text_input("BERT Model", value=MODEL_NAME)
    threshold = st.slider("Eşik (sigmoid)", min_value=0.05, max_value=0.95, value=float(THRESH), step=0.05)
    st.divider()
    st.markdown("**Gemma Ayarları** (opsiyonel)")
    gemma_path = st.text_input("Gemma model yolu", value="C:\\pyy\\models\\gemma")

uploaded = st.file_uploader("CSV yükleyin", type=["csv"])

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

    st.subheader("Gemma ile Öneri Üretimi")
    with st.expander("Satır seçerek öneri üret"):
        sel_idx = st.number_input("Satır index", min_value=0, max_value=len(merged)-1, value=0, step=1)
        if st.button("Seçili satıra öneri üret"):
            req_text = merged.iloc[sel_idx][TEXT_COL]
            miss = [c for c in LABEL_COLS if int(merged.iloc[sel_idx].get(f"AI_{c}", 0)) == 0]
            if not miss:
                st.warning("AI'ya göre eksik yok.")
            else:
                try:
                    gen = load_gemma_model(gemma_path)
                    sug = generate_ai_suggestion(gen, req_text, miss)
                    st.text_area("AI Önerisi", value=sug, height=200)
                except Exception as e:
                    st.error(f"Gemma çalıştırılamadı: {e}")

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
        if st.button("Gemma ile Öneri Üret") and single_req.strip():
            miss = st.session_state.get("single_missing", [])
            if not miss:
                st.warning("Önce eksikleri analiz edin veya eksik yok.")
            else:
                try:
                    gen = load_gemma_model(gemma_path)
                    sug = generate_ai_suggestion(gen, single_req, miss)
                    st.text_area("İyileştirilmiş Gereksinim", value=sug, height=180)
                except Exception as e:
                    st.error(f"Gemma çalıştırılamadı: {e}")
else:
    st.write("CSV yükleyin ve analiz edin.")
