"""
BERT + BiLSTM + CNN multi-label classifier + Gemma AI öneri entegrasyonu
Requirements:
- transformers
- torch
- sklearn
- pandas
- numpy
- tqdm
- tiktoken
- protobuf

torch>=2.0.0
transformers>=4.35.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
tqdm>=4.65.0
tiktoken>=0.4.0
protobuf>=4.24.0

"""

import os
import random
from typing import List, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline, get_linear_schedule_with_warmup
from torch.optim import AdamW

from sklearn.metrics import f1_score, accuracy_score, hamming_loss
from sklearn.model_selection import train_test_split


# SETTINGS


MODEL_NAME = r"I:\\BERTürk"   
CSV_PATH = r"I:\\proje\\yeni.csv"  
TEXT_COL = "Requirement"
LABEL_COLS = ['Appropriate','Complete','Conforming','Correct','Feasible','Necessary','Singular','Unambiguous','Verifiable']

RANDOM_SEED = 42
BATCH_SIZE = 8          
EPOCHS = 3
LR = 1e-5
WEIGHT_DECAY = 1e-2
MAX_LEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FINE_TUNE_BERT = True     
GRAD_CLIP = 1.0
THRESH = 0.5              


# REPRODUCIBILITY


def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()


# DATASET


class ReqDataset(Dataset):
    def __init__(self, texts: List[str], labels: np.ndarray, tokenizer, max_len=128):
        self.texts = texts
        # NaN/Inf temizliği ve aralık güvenliği
        clean = np.nan_to_num(labels, nan=0.0, posinf=1.0, neginf=0.0)
        self.labels = clean.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float)
        }
        return item

def collate_fn(batch):
    input_ids = torch.stack([b['input_ids'] for b in batch])
    attention_mask = torch.stack([b['attention_mask'] for b in batch])
    labels = torch.stack([b['labels'] for b in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# MODEL: BERT -> BiLSTM -> Multi-CNN -> Classifier


class BertBiLSTMCNN(nn.Module):
    def __init__(self, bert_model_name, lstm_hidden=256, lstm_layers=1, cnn_out_channels=128, kernel_sizes=(2,3,4), dropout=0.3, num_labels=9):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        bert_hidden = self.bert.config.hidden_size  

        self.lstm = nn.LSTM(input_size=bert_hidden, hidden_size=lstm_hidden, num_layers=lstm_layers,
                            batch_first=True, bidirectional=True, dropout=0.0 if lstm_layers==1 else dropout)
     
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=2*lstm_hidden, out_channels=cnn_out_channels, kernel_size=k)
            for k in kernel_sizes
        ])
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        total_conv_out = cnn_out_channels * len(kernel_sizes)

        self.classifier = nn.Sequential(
            nn.Linear(total_conv_out, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden_states = outputs.last_hidden_state  
        lstm_out, _ = self.lstm(hidden_states)  
        x = lstm_out.permute(0, 2, 1)
        conv_outs = []
        for conv in self.convs:
            c = conv(x)
            p = self.global_pool(c)
            conv_outs.append(p.squeeze(-1))
        cat = torch.cat(conv_outs, dim=1)
        logits = self.classifier(cat)
        return logits


# METRICS


def evaluate_model(model, dataloader, device, threshold=THRESH):
    model.eval()
    all_labels, all_preds = [], []
    total_loss = 0.0
    loss_fn = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device).float()
            labels = torch.nan_to_num(labels, nan=0.0, posinf=1.0, neginf=0.0)
            labels = labels.clamp(0.0, 1.0)

            logits = model(input_ids, attention_mask)
            if not torch.isfinite(logits).all():
                print("[WARN] Non-finite logits detected during evaluation. Skipping batch.")
                continue
            loss = loss_fn(logits, labels)
            if not torch.isfinite(loss):
                print("[WARN] Non-finite loss detected during evaluation. Skipping batch.")
                continue
            total_loss += loss.item() * input_ids.size(0)

            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= threshold).astype(int)
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)
    ham_loss = hamming_loss(all_labels, all_preds)
    avg_loss = total_loss / len(dataloader.dataset)
    return {"loss": avg_loss, "f1_macro": f1_macro, "accuracy": accuracy, "hamming": ham_loss}


# TRAINING


def train_loop(model, train_loader, val_loader, device, epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY):
    model.to(device)
    no_decay = ["bias", "LayerNorm.weight"]

    if not FINE_TUNE_BERT:
        for p in model.bert.parameters():
            p.requires_grad = False

    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(1, int(0.10*total_steps)), num_training_steps=total_steps)
    loss_fn = nn.BCEWithLogitsLoss()

    best_f1 = -1.0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for step, batch in enumerate(loop):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device).float()
            labels = torch.nan_to_num(labels, nan=0.0, posinf=1.0, neginf=0.0)
            labels = labels.clamp(0.0, 1.0)

            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids, attention_mask)
            if not torch.isfinite(logits).all():
                print("[WARN] Non-finite logits detected during training. Skipping batch.")
                continue
            loss = loss_fn(logits, labels)
            if not torch.isfinite(loss):
                with torch.no_grad():
                    l_min = float(torch.min(logits).item())
                    l_max = float(torch.max(logits).item())
                    y_min = float(torch.min(labels).item())
                    y_max = float(torch.max(labels).item())
                print(f"[WARN] Non-finite loss. logits[min,max]=[{l_min:.4f},{l_max:.4f}] labels[min,max]=[{y_min:.4f},{y_max:.4f}] -> skipping batch")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item() * input_ids.size(0)
            loop.set_postfix(loss=loss.item())

        val_metrics = evaluate_model(model, val_loader, device)
        print(f"Epoch {epoch+1} - TrainLoss: {epoch_loss/len(train_loader.dataset):.4f} | "
              f"ValLoss: {val_metrics['loss']:.4f} | F1_macro: {val_metrics['f1_macro']:.4f} | "
              f"Hamming: {val_metrics['hamming']:.4f}")
        if val_metrics['f1_macro'] > best_f1:
            best_f1 = val_metrics['f1_macro']
            torch.save(model.state_dict(), "best_model.pt")
            print("Saved best model.")

    return model


# GEMMA AI SUGGESTION


def load_gemma_model(gemma_model_path):
    tokenizer = AutoTokenizer.from_pretrained(gemma_model_path)
    model = AutoModelForCausalLM.from_pretrained(gemma_model_path)
    device = 0 if torch.cuda.is_available() else -1
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        return_full_text=False,
        clean_up_tokenization_spaces=True,
    )
    return generator

def generate_ai_suggestion(generator, requirement, missing_features):
    missing_str = ", ".join(missing_features)
    prompt = (
        "Yalnızca TÜRKÇE yaz. Giriş cümlesini tekrar etme.\n"
        "Belirsizliği kaldır, doğrulanabilir ve ölçülebilir hale getir.\n"
        "Eksikleri gider: " + missing_str + "\n\n"
        "Gereksinim: " + requirement + "\n"
        "ÇIKTI: Yalnızca tek satır 'İyileştirilmiş gereksinim: <cümle>' yaz. Başka hiçbir şey ekleme.\n"
    )
    response = generator(
        prompt,
        max_new_tokens=96,
        num_return_sequences=1,
        do_sample=False,
        num_beams=4,
        repetition_penalty=1.3,
    )
    # transformers >=4.35 text-generation returns list of dicts with 'generated_text'
    text = response[0].get('generated_text', '').strip() if isinstance(response, list) else str(response)
    # 'İyileştirilmiş gereksinim:' sonrası ilk satırı al
    lower = text.lower()
    if "iyileştirilmiş gereksinim" in lower:
        try:
            start = lower.index("iyileştirilmiş gereksinim")
            text = text[start:]
            text = text.split("\n", 1)[0]
        except Exception:
            text = text.split("\n", 1)[0]
    else:
        text = text.split("\n", 1)[0]
    # Liste/numara/alıntı temizliği
    for prefix in ['- ', '* ', '• ', '1) ', '1. ', '"', "'"]:
        if text.strip().startswith(prefix):
            text = text.strip()[len(prefix):].strip()
    # Etiket başlığı varsa ayıkla
    if ':' in text:
        head, tail = text.split(':', 1)
        if head.lower().strip().startswith('iyileştirilmiş gereksinim'):
            text = tail.strip()
    # Aynı cümleyi döndürdüyse küçük bir revizyon ekle
    if text.strip().rstrip('.') == requirement.strip().rstrip('.'):
        text = text.strip().rstrip('.') + ". Ölçülebilir kabul kriterleri tanımlıdır"
    return text


# MAIN


def main():
    # CSV load
    df = pd.read_csv(CSV_PATH)
    texts = df[TEXT_COL].fillna("").tolist()
    labels = df[LABEL_COLS].values 

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.15, random_state=RANDOM_SEED, shuffle=True)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Datasets & Loaders
    train_dataset = ReqDataset(X_train, np.array(y_train), tokenizer, max_len=MAX_LEN)
    val_dataset   = ReqDataset(X_val, np.array(y_val), tokenizer, max_len=MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Model
    model = BertBiLSTMCNN(bert_model_name=MODEL_NAME, num_labels=len(LABEL_COLS))
    model.to(DEVICE)

    # Train
    trained_model = train_loop(model, train_loader, val_loader, DEVICE, epochs=EPOCHS, lr=LR)

    # Evaluate
    metrics = evaluate_model(trained_model, val_loader, DEVICE)
    print("Final metrics:", metrics)

    # Satır bazlı eksik özelliklerin yazdırılması
    trained_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    sample_count = min(5, len(X_val))
    print("\nÖrnek eksik özellikler (ilk", sample_count, "kayıt):")
    with torch.no_grad():
        for i in range(sample_count):
            text = X_val[i]
            true_labels = y_val[i]
            enc = tokenizer(text, truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors='pt')
            logits = trained_model(enc['input_ids'].to(DEVICE), enc['attention_mask'].to(DEVICE))
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            preds = (probs >= THRESH).astype(int)
            missing = [LABEL_COLS[j] for j, v in enumerate(preds) if v == 0]
            print(f"- Gereksinim: {text[:80]}...")
            print(f"  AI Eksikler: {missing}")

    
    # Gemma AI kullanımı
    
    gemma_path = r"I:\gemma"
    generator = load_gemma_model(gemma_path)

    # Örnek kullanım
    example_req = "Bu ürünün görünümü, profesyonel ve hatasız olmalıdır."
    prediction = {
        'Appropriate': 1, 'Complete': 1, 'Conforming': 1, 'Correct': 1,
        'Feasible': 1, 'Necessary': 1, 'Singular': 1, 'Unambiguous': 0, 'Verifiable': 0
    }
    missing = [k for k,v in prediction.items() if v==0]

    if missing:
        suggestion = generate_ai_suggestion(generator, example_req, missing)
        print("\nAI Önerisi:\n", suggestion)

if __name__ == "__main__":
    main()
