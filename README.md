# Nextbrick Agentic AI — Setup Guide

> **Stack:** FastAPI backend · React/Vite frontend · LangChain ReAct Agent · Elasticsearch · Kafka · BGE-M3 embeddings (runs on-device, no API key)

---

## Prerequisites

Install these once on the new machine:

| Tool | Install |
|---|---|
| **Python 3.11+** | `brew install python` |
| **Node 18+** | `brew install node` |
| **Docker Desktop** | [download](https://www.docker.com/products/docker-desktop) |
| **Git** | pre-installed on macOS |
| **Make** | pre-installed on macOS |

---

## 1 — Clone the repo

```bash
git clone <your-repo-url> dcx
cd dcx
```

---
{"CRAWLABLE":"false","COUNTRIES":"CA,US","CREATED_DATE":"2019-08-12T15:10:45.000Z","ZENDESK_TICKET":"3067504","ASSET_PATH":"https://stgwww.keysight.com/content/dam/keysight/en/doc/gate/nww/training-materials/9018-06265.pdf","CONTENT_TYPE_NAME":"training-materials","PUB_SUFFIX":"EN","AUTHOR":"","CONTENT_TYPE_TITLE":"Training Materials","ABSTRACT":"","NWW":"true","DOC_TYPE":"Asset","PUBKEY":"9018-06265","OVERRIDE_DISPLAY_PUB_NUMBER":"false","DESCRIPTION":"Slides","DISPLAY_PUB_NUMBER":"9018-06265","TITLE":"Connect Design & Test","PUB_DATE":"2019-08-13T00:00:00.000Z","LANGUAGE_TITLE":"","TAGS":"","KEYWORDS":""}
## 2 — Configure environment variables

```bash
cp backend/.env.example backend/.env   # if you have one, otherwise:
nano backend/.env
```

Minimum required values:

```env
# ── LLM (local Ollama) ────────────────────────────
ONPREM_MODEL_URL=http://localhost:11434
ONPREM_MODEL_NAME=qwen2.5-coder:latest
ONPREM_MODEL_API_KEY=EMPTY

# ── Elasticsearch ─────────────────────────────────
ES_HOST=http://localhost:9200
ES_USERNAME=elastic
ES_PASSWORD=changeme
ES_VECTOR_INDEX=nextbrick-vectors

# ── Embeddings (BGE-M3 — runs locally, no API key) ─
EMBEDDING_MODEL=BAAI/bge-m3

# ── CORS ──────────────────────────────────────────
FRONTEND_URL=http://localhost:8080

# ── Kafka (optional) ──────────────────────────────
KAFKA_ENABLED=false              # set true when Kafka is running
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
```

---

## 3 — Start Elasticsearch

```bash
docker run -d \
  --name elasticsearch \
  -p 9200:9200 \
  -e discovery.type=single-node \
  -e ELASTIC_PASSWORD=changeme \
  -e xpack.security.enabled=true \
  -e ES_JAVA_OPTS="-Xms1g -Xmx1g" \
  docker.elastic.co/elasticsearch/elasticsearch:9.3.1

# Verify it's up (takes ~30 seconds):
curl -u elastic:changeme http://localhost:9200
```

The `nextbrick-vectors` index is **created automatically** when the first search request hits the backend.

---

## 4 — Install Ollama + LLM

```bash
# Install Ollama
brew install ollama

# Pull the model (runs locally, ~5 GB)
ollama pull qwen2.5-coder:latest

# Start the Ollama server (keep this running)
ollama serve
```

> **On Apple Silicon (M1/M2/M3):** BGE-M3 and Ollama both use the Metal GPU automatically — no extra config needed.

---

## 5 — Install Python dependencies

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd ..
```

> **First `elasticsearch_semantic_search` call** will download BGE-M3 from HuggingFace (~570 MB) and cache it at `~/.cache/huggingface/`. This only happens once.

---

## 6 — Install frontend dependencies

```bash
cd nextbrick-ai-assistant-main
npm install
cd ..
```

---

## 7 — Run everything

```bash
# From project root — starts backend (port 8000) + frontend (port 8080)
make run
```

| Service | URL |
|---|---|
| Frontend | http://localhost:8080 |
| Backend API | http://localhost:8000 |
| Swagger docs | http://localhost:8000/api/docs |
| Health check | http://localhost:8000/api/health |

---

## 8 — (Optional) Start Kafka + Spark pipeline

Kafka streams every chat and agent response for real-time analytics.

### Start / Stop

```bash
# Start Kafka broker + Kafdrop UI (Spark not included by default)
make pipeline-up
# or directly:
docker compose -f data-pipeline/docker-compose.kafka.yml up -d

# Stop everything
make pipeline-down
# or:
docker compose -f data-pipeline/docker-compose.kafka.yml down

# Stop and delete all data volumes
docker compose -f data-pipeline/docker-compose.kafka.yml down -v
```

### Check status

```bash
# Are containers running?
docker ps --filter name=dcx-kafka --filter name=dcx-kafdrop

# Kafka broker health
docker inspect --format='{{.State.Health.Status}}' dcx-kafka
```

### Kafka UI (Kafdrop)

```
http://localhost:9000
```

Kafdrop shows all topics, partitions, consumer groups, and lets you browse messages live.

### Topics

```bash
# List all topics
docker exec dcx-kafka /opt/kafka/bin/kafka-topics.sh \
  --list --bootstrap-server localhost:9092

# Create a topic manually (auto-created after first API call)
docker exec dcx-kafka /opt/kafka/bin/kafka-topics.sh \
  --create --topic chat.events --partitions 3 --replication-factor 1 \
  --bootstrap-server localhost:9092

# Describe a topic (partitions, offsets)
docker exec dcx-kafka /opt/kafka/bin/kafka-topics.sh \
  --describe --topic chat.events --bootstrap-server localhost:9092
```

### Consume messages (live tail)

```bash
# Watch chat events in real time
docker exec dcx-kafka /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic chat.events --from-beginning

# Watch agent events (tool traces)
docker exec dcx-kafka /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic agent.events --from-beginning
```

### Tail container logs

```bash
make pipeline-logs          # Spark consumer logs
docker logs -f dcx-kafka    # Kafka broker logs
docker logs -f dcx-kafdrop  # Kafdrop logs
```

### Start Spark consumer (optional)

```bash
# Spark reads from chat.events and writes aggregated JSON to ./output/
docker compose -f data-pipeline/docker-compose.kafka.yml \
  --profile spark up -d spark-consumer

docker logs -f dcx-spark-consumer
```

### .env settings for Kafka

```env
KAFKA_ENABLED=true                         # false = silent no-op, never breaks API
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPIC_CHAT_EVENTS=chat.events
KAFKA_TOPIC_AGENT_EVENTS=agent.events
```

> **Fire-and-forget:** If Kafka is down and `KAFKA_ENABLED=true`, the API still works — events are silently skipped and a warning is logged.

---

## Make commands reference

```bash
make run           # start backend + frontend (no Docker)
make compose-up    # same but via docker-compose
make build         # build Docker image
make pipeline-up   # start Kafka + Spark pipeline
make pipeline-down # stop Kafka
make logs          # tail Docker logs
make clean         # remove containers and image
```

---

## Apple Silicon GPU notes

| Component | GPU usage |
|---|---|
| **BGE-M3** (embeddings) | Uses Metal GPU via `torch` — automatic on Apple Silicon |
| **Ollama** (LLM) | Uses Metal GPU — automatic |
| **Elasticsearch** | CPU only (Java-based) |

To explicitly force BGE-M3 to use MPS (Metal):
```python
# In elasticsearch_tool.py, change:
model_kwargs={"device": "mps"}   # instead of "cpu"
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `401 Unauthorized` from ES | Check `ES_USERNAME` / `ES_PASSWORD` in `.env` |
| `BGE-M3 download slow` | Normal on first boot — 570 MB, cached after |
| `Port 8000 in use` | `kill -9 $(lsof -ti :8000)` |
| `Kafka unavailable` | Set `KAFKA_ENABLED=false` to disable safely |
| `Ollama connection refused` | Run `ollama serve` in a terminal |
| `pip externally-managed` | Always use the venv: `source backend/venv/bin/activate` |

---

## Project structure

```
dcx/
├── backend/                    # FastAPI backend
│   ├── app/
│   │   ├── routers/
│   │   │   ├── chat.py         # POST /api/chat
│   │   │   └── agent.py        # POST /api/agent  ← LangChain ReAct agent
│   │   ├── services/
│   │   │   ├── agent_service.py
│   │   │   ├── kafka_service.py
│   │   │   └── llm_service.py
│   │   └── tools/
│   │       ├── elasticsearch_tool.py  # BGE-M3 + ES kNN
│   │       ├── salesforce_tool.py
│   │       └── confluence_tool.py
│   ├── requirements.txt
│   └── .env                    # your local config (not committed)
├── nextbrick-ai-assistant-main/ # React/Vite frontend
├── data-pipeline/
│   ├── producer/kafka_producer.py
│   ├── consumer/spark_consumer.py
│   └── docker-compose.kafka.yml
├── Makefile
└── docker-compose.yml
```
