# AI Integration Service

## Genel Bakış

Bu servis, Spring Boot HR Backend ile MCP AI Pipeline arasındaki köprüdür. Mülakat tamamlandığında backend bu servisi tetikler, servis 4 AI stream'i çalıştırır ve sonuçları XAI destekli raporlarla birlikte döndürür.

```
Spring Boot Backend  ──REST──▶  AI Integration Service  ──MCP──▶  AI Pipeline
                                        │
                                        ├── XAI Explainer (feature contributions)
                                        ├── NLG HR Report (detaylı analiz)
                                        └── NLG Candidate Feedback (gelişim önerileri)
```

## Mimari

```
ai_integration_service/
├── app.py                          # FastAPI ana uygulama (REST API)
├── Dockerfile
├── requirements.txt
├── xai/
│   └── explainer.py                # XAI: Feature contribution, counterfactual, confidence
├── reports/
│   └── nlg_generator.py            # NLG: HR raporu + Aday geri bildirimi + Red geri bildirimi
├── models/
│   └── result_store.py             # JSON tabanlı sonuç depolama (MongoDB-uyumlu)
└── mcp_pipeline/                   # AI Pipeline (4 stream)
    ├── config.yaml
    ├── agent/
    │   ├── orchestrator.py         # Pipeline orkestrasyon
    │   └── fusion.py               # Ağırlıklı skor birleştirme
    ├── mcp_servers/
    │   ├── resume_server.py        # Stream 1: Gemini 2.5 Flash CV analizi
    │   ├── speech_recognition_server.py  # Stream 2: Whisper STT
    │   ├── speech_emotion_server.py      # Stream 3: HuBERT SER
    │   └── facial_emotion_server.py      # Stream 4: EfficientNetV2-S FER
    └── utils/
        ├── schemas.py              # Pydantic veri şemaları
        └── media_splitter.py       # Video → Audio ayırıcı
```

## API Endpoint'leri

| Method | Endpoint | Açıklama |
|--------|----------|----------|
| `POST` | `/api/v1/ai/analyze` | AI pipeline'ı tetikle (async, anında döner) |
| `GET` | `/api/v1/ai/results/{id}` | Birleşik analiz sonuçları |
| `GET` | `/api/v1/ai/results/{id}/xai` | XAI açıklamaları (feature contributions, counterfactual) |
| `GET` | `/api/v1/ai/results/{id}/feedback` | Aday geri bildirimi (NLG) |
| `GET` | `/api/v1/ai/results/{id}/hr-report` | HR detaylı raporu (XAI destekli) |
| `POST` | `/api/v1/ai/results/{id}/decision` | HR kabul/red kararı (red → gelişim önerileri üretir) |
| `GET` | `/api/v1/ai/health` | Sağlık kontrolü |

## Kurulum

### 1. Bağımlılıkları Yükle

```bash
cd ai_integration_service
pip install -r requirements.txt
pip install -r mcp_pipeline/requirements.txt
```

### 2. Model Dosyalarını Yerleştir

```
models/
├── HuBERT_SER/                    # HuBERT SER modeli (klasör)
├── efficientnetv2s_hybrid_model.pt # FER modeli
└── yolov11n-face.pt               # YOLO yüz modeli
```

### 3. Config'i Düzenle

`mcp_pipeline/config.yaml` dosyasında model yollarını ve Gemini API key'i ayarlayın.

### 4. Çalıştır

```bash
# Geliştirme
python app.py

# veya uvicorn ile
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

API dökümantasyonu: `http://localhost:8080/api/v1/ai/docs`

### 5. Docker ile Çalıştır

```bash
docker build -t ai-integration-service .
docker run -p 8080:8080 -v ./models:/app/models ai-integration-service
```

## Spring Boot Entegrasyonu

### Interview Tamamlandığında (Backend → AI Service)

```java
// InterviewService.java - Interview COMPLETED olduğunda
@EventListener
public void onInterviewCompleted(InterviewCompletedEvent event) {
    AnalyzeRequest request = new AnalyzeRequest();
    request.setApplicationId(event.getApplicationId());
    request.setCandidateId(event.getCandidateId());
    request.setVideoUrl(event.getVideoUrl());
    request.setResumeUrl(event.getResumeUrl());
    
    restTemplate.postForObject(
        "http://ai-service:8080/api/v1/ai/analyze", 
        request, AnalyzeResponse.class
    );
}
```

### HR Analiz Sonuçlarını Getirme

```java
// HrController.java
@GetMapping("/applications/{appId}/analysis")
public ResponseEntity<?> getAnalysis(@PathVariable UUID appId) {
    return restTemplate.getForEntity(
        "http://ai-service:8080/api/v1/ai/results/" + analysisId + "/hr-report",
        Map.class
    );
}
```

### Aday Geri Bildirimi

```java
// CandidateController.java
@GetMapping("/applications/{appId}/feedback")
public ResponseEntity<?> getFeedback(@PathVariable UUID appId) {
    return restTemplate.getForEntity(
        "http://ai-service:8080/api/v1/ai/results/" + analysisId + "/feedback",
        FeedbackResponse.class
    );
}
```

## XAI (Explainable AI) Yapısı

### Feature Contribution Analizi
Her stream'in nihai karara ne kadar katkı sağladığını gösterir:
```json
{
  "stream": "Facial Emotion",
  "contribution_percentage": 32.5,
  "impact": "positive",
  "impact_magnitude": "high"
}
```

### Counterfactual Analizi
"Eğer X farklı olsaydı sonuç ne olurdu?" sorusuna cevap verir:
```json
{
  "stream": "Resume Analysis",
  "current_score": 0.45,
  "if_perfect": 0.82,
  "description": "If Resume Analysis scored 1.0, final score would be 0.82"
}
```

### NLG Raporları
- **HR Raporu**: Detaylı teknik analiz, XAI açıklamaları, stream kırılımı, counterfactual senaryolar
- **Aday Geri Bildirimi**: Yapıcı, profesyonel dil, güçlü yönler + gelişim alanları
- **Red Geri Bildirimi**: Gelişim odaklı, counterfactual insight ile en etkili iyileşme alanı önerisi

## MongoDB Entegrasyonu (Production)

Mevcut `ResultStore` JSON dosyası kullanır. Production'da `result_store.py` dosyasındaki file işlemlerini MongoDB çağrılarıyla değiştirin:

```python
# result_store.py → MongoDB versiyonu
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017")
db = client["recruiter_db"]
collection = db["ai_analysis_results"]

def save(self, analysis_id, data):
    collection.update_one({"_id": analysis_id}, {"$set": data}, upsert=True)

def get(self, analysis_id):
    return collection.find_one({"_id": analysis_id})
```

Document yapısı zaten `AiAnalysisResult` MongoDB şemasıyla uyumludur.

## Kafka Entegrasyonu (Opsiyonel)

Interview tamamlandığında REST yerine Kafka event'i dinlemek için:

```python
# kafka_consumer.py (opsiyonel eklenti)
from aiokafka import AIOKafkaConsumer

async def consume_interview_events():
    consumer = AIOKafkaConsumer('interview.completed', bootstrap_servers='kafka:9092')
    await consumer.start()
    async for msg in consumer:
        event = json.loads(msg.value)
        # Trigger pipeline...
```
