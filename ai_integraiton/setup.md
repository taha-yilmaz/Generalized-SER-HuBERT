# AI Integration Service — Kurulum Rehberi (venv)

Bu rehber, `ai_integration_service`'i yerel makinede Python sanal ortamı (venv) ile nasıl kurup çalıştıracağınızı adım adım açıklar.

---

## Ön Gereksinimler

| Araç | Minimum Sürüm | Notlar |
|------|--------------|-------|
| **Python** | 3.10+ | `python --version` ile kontrol edin |
| **pip** | 23.0+ | `pip --version` ile kontrol edin |
| **ffmpeg** | Herhangi | Whisper için gerekli |
| **Git** | — | Opsiyonel |

### ffmpeg Kurulumu (Windows)

```powershell
# winget ile (önerilen):
winget install Gyan.FFmpeg

# veya chocolatey ile:
choco install ffmpeg
```

ffmpeg'u kurduktan sonra terminali yeniden başlatın ve `ffmpeg -version` ile test edin.

---

## Adım 1 — Proje Dizinine Gidin

```powershell
cd ai_integraiton\ai_integration_service
```

---

## Adım 2 — Sanal Ortam (venv) Oluşturun

```powershell
python -m venv venv
```

Bu komut, `ai_integration_service/venv/` klasörünü oluşturur.

---

## Adım 3 — Sanal Ortamı Etkinleştirin

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
venv\Scripts\activate.bat
```

> Terminalde `(venv)` öneki görünüyorsa ortam başarıyla etkinleştirilmiştir.

---

## Adım 4 — Bağımlılıkları Yükleyin

Servisin iki ayrı `requirements.txt` dosyası vardır; her ikisini de yükleyin:

```powershell
pip install --upgrade pip

# FastAPI servisi bağımlılıkları
pip install -r requirements.txt

# MCP Pipeline (AI modeller) bağımlılıkları
pip install -r mcp_pipeline/requirements.txt
```

> **Not:** `torch`, `whisper` ve `ultralytics` büyük paketlerdir. İlk kurulum birkaç dakika sürebilir.

### GPU Desteği (İsteğe Bağlı, CUDA)

Eğer NVIDIA GPU'nuz varsa PyTorch'u CUDA sürümüyle yükleyin:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## Adım 5 — Ortam Değişkenlerini Ayarlayın

`mcp_pipeline/config.yaml` dosyasını açıp aşağıdaki alanları doldurun:

```yaml
resume:
  gemini_api_key: "BURAYA_GOOGLE_AI_STUDIO_API_KEY"   # https://aistudio.google.com/
  gemini_model: "gemini-2.5-flash"                    # veya tercih ettiğiniz model

ollama:
  url: "http://localhost:11434/api/generate"
  model: "gemma4:e4b"   # Ollama'da yüklü modelin adı
```

---

## Adım 6 — AI Modellerini Yerleştirin

`mcp_pipeline/config.yaml` içindeki model yollarına göre dosyaları konumlandırın:

```
ai_integration_service/
└── models/
    ├── HuBERT_SER/                      # Speech Emotion modeli
    │   ├── config.json
    │   ├── preprocessor_config.json
    │   └── model.safetensors
    ├── effnetv2s_hybrid_mouth_occluded.pt   # Yüz ifadesi (FER) modeli
    └── yolov11n-face.pt                     # YOLO yüz tespiti modeli
```

> Model dosyaları yoksa `config.yaml`'daki ilgili yolları güncellemeniz gerekir.

---

## Adım 7 — Servisi Başlatın

```powershell
python app.py
```

Alternatif olarak doğrudan uvicorn ile:

```powershell
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

---

## Adım 8 — Servisin Çalıştığını Doğrulayın

Tarayıcıda veya `curl` ile kontrol edin:

```powershell
# Health check
curl http://localhost:8080/api/v1/ai/health

# Swagger UI (API dokümantasyonu)
# Tarayıcıda açın: http://localhost:8080/api/v1/ai/docs
```

Başarılı yanıt:
```json
{
  "status": "healthy",
  "service": "ai-integration-service",
  "version": "1.0.0"
}
```

---

## Sanal Ortamı Devre Dışı Bırakma

```powershell
deactivate
```

---

## Sık Karşılaşılan Hatalar

| Hata | Çözüm |
|------|-------|
| `ModuleNotFoundError: No module named 'cv2'` | `pip install opencv-python` |
| `ffmpeg not found` | ffmpeg'u kurun ve PATH'e ekleyin |
| `CUDA out of memory` | `config.yaml`'da `device: "cpu"` yapın |
| `Gemini API key invalid` | `config.yaml`'daki `gemini_api_key`'i kontrol edin |
| PowerShell script execution policy hatası | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `whisper` yüklenmiyor | `pip install openai-whisper` ayrıca çalıştırın |

---

## Proje Yapısı (AI Kısmı)

```
ai_integraiton/
└── ai_integration_service/
    ├── app.py                  # FastAPI ana uygulama
    ├── requirements.txt        # Servis bağımlılıkları
    ├── Dockerfile              # Docker alternatifi
    ├── mcp_pipeline/
    │   ├── config.yaml         # ⚙️  Tüm ayarlar burada
    │   ├── requirements.txt    # Pipeline bağımlılıkları
    │   ├── agent/              # MCP orchestrator
    │   ├── mcp_servers/        # 4 AI stream sunucusu
    │   └── utils/
    ├── models/                 # AI model dosyaları (manuel eklenecek)
    ├── xai/                    # XAI açıklama modülü
    ├── reports/                # NLG rapor üretici
    ├── interview_session/      # Canlı mülakat yönetimi
    └── pipeline_output/        # Analiz sonuçları (otomatik oluşur)
```

---

## Docker ile Alternatif Kurulum

Eğer Docker kullanmak isterseniz:

```powershell
# ai_integration_service dizininde:
docker build -t ai-integration-service .
docker run -p 8080:8080 -v ${PWD}/models:/app/models ai-integration-service
```
