# Enterprise Multi-Tenant Knowledge Intelligence

Subcriber: https://aiengineeringinsider.substack.com/subscribe

![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-FF4B4B.svg?logo=streamlit)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector_Database-dc2c55.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

[![Demo the NexusRAG Multi-Tenant System](https://img.youtube.com/vi/zxpY_I5cWmI/maxresdefault.jpg)](https://www.youtube.com/watch?v=zxpY_I5cWmI)

NexusRAG is a production-grade **Multi-Tenant Retrieval-Augmented Generation (RAG) System** built with FastAPI, Qdrant, and Streamlit. This platform provides secure, isolated RAG capabilities for multiple organizations with advanced document processing, scalable vector search, and seamless LLM integration.

---

## 📖 Table of Contents
- [Architecture](#-architecture)
- [Key Features](#-key-features)
- [Quick Start](#-quick-start)
- [Usage Guide](#-usage-guide)
- [Development Setup](#-development-setup)
- [Configuration](#-configuration)
- [Deployment & Security](#-deployment--security)
- [API Documentation](#-api-documentation)

---

## 🏗 Architecture

```mermaid
graph LR
    A[NexusRAG Streamlit Frontend] -->|REST API| B(FastAPI Backend)
    B -->|SQL| C[(PostgreSQL)]
    B -->|Embeddings| D[(Qdrant Vector DB)]
    B -->|Cache| E[(Redis)]
```

---

## ✨ Key Features

### 🏢 Enterprise Core
- **True Multi-Tenancy**: Complete data isolation across different organizations.
- **Advanced Vector Search**: Powered by Qdrant for blazing-fast semantic retrieval.
- **Model Agnostic**: Supports OpenAI, Anthropic, and local LLM models.
- **Document Processing Pipeline**: Handles PDF, TXT, and DOCX files automatically.

### 🎨 Modern Frontend (NexusRAG)
- **Minimalist Aesthetic**: Ultra-clean UI built with Streamlit featuring a light-mode, monochromatic design system.
- **Interactive Chat Interface**: ChatGPT-style bubbles, source citation grounding, latency tracking, and token usage metrics.
- **Document Hub**: Drag-and-drop file processing with visual tagging, format badges, and animated status pills.
- **Analytics Dashboard**: Real-time KPI metrics for query throughput, latency, and success rates.

### 🔒 Enterprise Security
- Strict tenant isolation enforced at the PostgreSQL row level and Qdrant metadata level.
- Robust JWT-based authentication with role-based access control.
- Input validation and secure environment-based configuration management.

---

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.11+ (for local development)
- OpenAI / Anthropic API Key (optional, for LLM functionality)

**1. Setup Environment**
```bash
cp env.example .env
# Edit .env with your API keys and secure configuration
```

**2. Start the Stack**
```bash
docker-compose up -d
```

**3. Access the Application**
- **Frontend (NexusRAG)**: http://localhost:8501
- **API Swagger Docs**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/health

**4. Load Sample Data (Optional)**
```bash
python scripts/setup_sample_data.py
```

---

## 💡 Usage Guide

1. **Sign In**: Navigate to `http://localhost:8501`. Create a new organization workspace or login with existing credentials.
   - *Sample Credentials (if you ran the setup script)*: 
     - Email: `user@acme.com`
     - Password: `user123`
2. **Upload Knowledge**: Go to the **Document Library** section, upload your PDFs/DOCX, and wait for the backend to chunk and index the vectors.
3. **Query Intelligence**: Navigate to the **Chat Assistant** to ask questions grounded exclusively in your tenant's document space.

---

## 🛠 Development Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Local Infrastructure
```bash
# Start only the required databases
docker-compose up -d postgres redis qdrant
```

### 3. Run Backend (FastAPI)
```bash
export DATABASE_URL="postgresql://postgres:password@localhost:5432/multi_tenant_rag"
export QDRANT_HOST="localhost"
export REDIS_URL="redis://localhost:6379"

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Run Frontend (Streamlit)
```bash
streamlit run frontend/streamlit_app.py --server.port 8501
```

### Testing & Code Quality
```bash
# Run tests with coverage
pip install pytest pytest-asyncio
pytest tests/ --cov=app --cov-report=html

# Format & Lint
black app/ tests/
isort app/ tests/
flake8 app/ tests/
```

---

## ⚙️ Configuration

Control the application via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | *Required* |
| `QDRANT_HOST` | Qdrant server host | `localhost` |
| `QDRANT_PORT` | Qdrant server port | `6333` |
| `OPENAI_API_KEY` | OpenAI API key | *Optional* |
| `ANTHROPIC_API_KEY` | Anthropic API key | *Optional* |
| `JWT_SECRET_KEY` | JWT signing secret | *Required* |
| `MAX_FILE_SIZE_MB` | Maximum upload file size | `10` |
| `EMBEDDING_MODEL` | Sentence transformer model | `all-MiniLM-L6-v2` |

---

## 🌍 Deployment & Security

### Production Deployment
- **Docker Swarm**: Use `docker stack deploy -c docker-compose.prod.yml rag-stack`
- **Kubernetes**: Apply manifests using `kubectl apply -f k8s/ -n multi-tenant-rag`

### Security Checklist
- [ ] Rotate default `JWT_SECRET_KEY`.
- [ ] Enforce strong PostgreSQL passwords.
- [ ] Configure reverse proxy (Nginx/Traefik) with HTTPS/TLS.
- [ ] Restrict Qdrant port access to internal backend networks only.

---

## 📚 API Documentation

Once the backend is running, full interactive API documentation is available at `/docs` (Swagger UI). 

### Core Endpoints:
- **Authentication**: `POST /api/v1/auth/login`, `POST /api/v1/auth/register`
- **Documents**: `POST /api/v1/documents/upload`, `GET /api/v1/documents/`
- **Queries**: `POST /api/v1/queries/rag`, `POST /api/v1/queries/rag/stream`

---

## 🔧 Troubleshooting

- **Database Errors**: `docker-compose logs postgres`
- **Qdrant Connection**: Ensure vector DB is healthy with `curl http://localhost:6333/health`
- **LLM Failures**: Check `.env` API keys and verify provider rate limits.
