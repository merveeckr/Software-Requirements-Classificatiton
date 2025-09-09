# Software-Requirements-Classificatiton

Bu proje, BERTürk tabanlı çoklu etiket sınıflandırıcı (BERT → BiLSTM → CNN) ve Gemma AI ile öneri üretimini içerir. Streamlit arayüzü ile CSV yükleme, eksik etiketlerin toplu görülmesi, kullanıcı işaretlemeleri ve AI önerilerinin düzenlenmesi desteklenir.

## Çalıştırma

1) Gerekli paketleri kurun:
```
pip install -r requirements.txt
```

2) Streamlit uygulamasını başlatın:
```
streamlit run app.py
```

3) Uygulamada CSV dosyanızı yükleyin, eşiği ayarlayın ve önerileri üretin.

## Benchmark (karşılaştırma)

Çeşitli gömme (embedding) ve mimarileri kıyaslamak için:

```
python bench.py
```

Ortam değişkenleri:
- `CSV_PATH=/workspace/yeni.csv`
- `BERT_MODEL=dbmdz/bert-base-turkish-uncased`
- `EPOCHS=1`

Çıktı; eğitim süresi, örnek başına tahmin süresi, F1-macro, Hamming ve accuracy verir. Sentence-Transformers adayları: `paraphrase-multilingual-MiniLM-L12-v2`, `intfloat/multilingual-e5-small`, `intfloat/multilingual-e5-base`, `LaBSE`.

### LLM öneri kıyaslaması

LLM’lerin öneri kalitesini (eksik onarım oranı, gecikme) ölçmek için:

```
set CSV_PATH=C:\path\to\yeni.csv
set BERT_MODEL=C:\models\dbmdz-bert-turkish  (veya HF id)
set LLM_SPECS=C:\models\gemma-2b.Q4.gguf:gguf;google/gemma-2b-it:hf

python llm_bench.py
```

Notlar:
- GGUF için `llama-cpp-python` gerekir. HF id için internet veya kurum sertifikası gerekir; offline için GGUF önerilir.
- Rapor: repair_rate (eksik etiketlerin 1’e dönme oranı), Δhamming, ms/sample.
