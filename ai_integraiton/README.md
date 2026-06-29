# AI Integration Service — Backend Entegrasyon Kılavuzu

## Genel Bakış

Bu paket iki bölümden oluşur:

1. **`backend_additions/`** — Spring Boot backend'e eklenecek Java dosyaları
2. **`ai_integration_service/`** — Python FastAPI servisi (MCP AI Pipeline + XAI + NLG)

```
Spring Boot (8081)                    AI Integration Service (8080)
┌──────────────────┐                  ┌─────────────────────────────┐
│ InterviewController                 │  FastAPI REST API            │
│   POST /complete ──Event──────────▶│  POST /api/v1/ai/analyze    │
│                   │                 │         │                    │
│ HrController      │                 │    MCP Pipeline (4 stream)  │
│   GET /analysis ──REST GET────────▶│    ├─ CV Analysis            │
│                   │                 │    ├─ Whisper STT            │
│ CandidateCtrl     │                 │    ├─ HuBERT SER            │
│   GET /feedback ──REST GET────────▶│    └─ FER (EfficientNetV2)  │
│                   │                 │         │                    │
│ MongoDB ◀─────────┼──store──────── │    XAI Explainer             │
│ (AiAnalysisResult)│                 │    NLG Report Generator      │
└──────────────────┘                  └─────────────────────────────┘
```

## Kurulum Adımları

### Adım 1: Spring Boot Backend'e Java Dosyalarını Ekle

Aşağıdaki dosyaları mevcut backend projenize kopyalayın:

```
backend_additions/ dosyası → backend projenizin src/main/java/com/example/AiPoweredRecruitmentSystem/ altına

Kopyalanacak dosyalar:
├── config/
│   ├── AiIntegrationConfig.java    → config/ klasörüne
│   └── AsyncConfig.java            → config/ klasörüne
├── controller/
│   └── InterviewController.java    → controller/ klasörüne (YENİ)
├── event/
│   └── InterviewCompletedEvent.java → event/ klasörü oluşturup içine (YENİ)
├── service/
│   ├── AiIntegrationService.java    → service/ klasörüne (YENİ)
│   ├── InterviewService.java        → service/ klasörüne (YENİ)
│   └── impl/
│       ├── AiIntegrationServiceImpl.java → service/impl/ klasörüne (YENİ)
│       └── InterviewServiceImpl.java     → service/impl/ klasörüne (YENİ)
```

### Adım 2: Mevcut Dosyalarda Küçük Değişiklikler

**a) `application.yml`'a ekle** (dosyanın sonuna):
```yaml
ai-service:
  url: ${AI_SERVICE_URL:http://localhost:8080}
```

**b) `InterviewRepository.java`'ya ekle:**
```java
Optional<Interview> findByApplicationId(UUID applicationId);
```

**c) `User.java`'da `fullName` alanı yoksa ekle:**
```java
private String fullName;
```

### Adım 3: AI Integration Service'i Kur

```bash
cd ai_integration_service
pip install -r requirements.txt
pip install -r mcp_pipeline/requirements.txt
```

### Adım 4: Model Dosyalarını Yerleştir

```
ai_integration_service/models/
├── HuBERT_SER/                    # HuBERT klasörü
├── efficientnetv2s_hybrid_model.pt # FER modeli
└── yolov11n-face.pt               # YOLO yüz modeli
```

`mcp_pipeline/config.yaml`'da yolları güncelleyin.

### Adım 5: Servisleri Başlat

```bash
# Terminal 1: Spring Boot backend
cd ai-supported-recruitment-system-dev
./mvnw spring-boot:run

# Terminal 2: AI Integration Service
cd ai_integration_service
python app.py
```

## Akış Diyagramı

```
1. Aday mülakatı tamamlar
   └─▶ POST /api/v1/interviews/{id}/complete

2. InterviewServiceImpl.completeInterview()
   └─▶ InterviewCompletedEvent yayınlar

3. AiIntegrationServiceImpl.onInterviewCompleted() (async)
   └─▶ Python AI servise REST çağrısı: POST /api/v1/ai/analyze
   └─▶ Sonuçları bekler (polling)
   └─▶ MongoDB'ye kaydeder (AiAnalysisResult)
   └─▶ Interview durumunu ANALYZED yapar

4. HR analiz sonuçlarını görür
   └─▶ GET /api/v1/hr/applications/{id}/analysis
   └─▶ HrServiceImpl → MongoDB'den AiAnalysisResult okur

5. HR karar verir
   └─▶ POST /api/v1/hr/applications/{id}/decision?decision=ACCEPT
   └─▶ Red ise → XAI ile gelişim önerileri üretilir

6. Aday geri bildirimini görür
   └─▶ GET /api/v1/candidate/applications/{id}/feedback
   └─▶ MongoDB'deki NLG raporunu döndürür
```

## XAI Yapısı

Üç katmanlı açıklanabilir AI:

| Katman | Ne üretir | Kime gösterilir |
|--------|-----------|-----------------|
| Feature Contributions | Her stream'in karara katkı yüzdesi | HR |
| Counterfactual Analysis | "X farklı olsaydı sonuç ne olurdu" | HR |
| NLG Feedback | Yapıcı dilde güçlü yönler + gelişim alanları | Aday |

## API Referansı

### AI Integration Service (Python - port 8080)

| Method | Endpoint | Açıklama |
|--------|----------|----------|
| POST | `/api/v1/ai/analyze` | Pipeline'ı tetikle |
| GET | `/api/v1/ai/results/{id}` | Sonuçları getir |
| GET | `/api/v1/ai/results/{id}/xai` | XAI açıklamaları |
| GET | `/api/v1/ai/results/{id}/feedback` | Aday raporu |
| GET | `/api/v1/ai/results/{id}/hr-report` | HR raporu |
| POST | `/api/v1/ai/results/{id}/decision` | HR kararı kaydet |

### Spring Boot Backend (Java - port 8081)

| Mevcut Endpoint | AI Bağlantısı |
|----------------|---------------|
| `GET /hr/applications/{id}/analysis` | → MongoDB AiAnalysisResult |
| `GET /candidate/applications/{id}/feedback` | → MongoDB AiAnalysisResult |
| `POST /hr/applications/{id}/decision` | → AI servise karar bildirimi |
| `POST /interviews/{id}/complete` | → InterviewCompletedEvent → AI pipeline |
