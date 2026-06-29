FROM python:3.10-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
COPY mcp_pipeline/requirements.txt mcp_pipeline_requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -r mcp_pipeline_requirements.txt

# Application code
COPY . .

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
