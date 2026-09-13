# Deployment & Production Serving Guide

**Project:** AI-Based Stock Market Prediction & Predictive Sequence Analytics  
**Author:** Kaleab Mezgebe

---

## 1. Local Environment Execution

### Run Training & Benchmark Evaluation
```bash
python3 scripts/generate_data.py
python3 src/train_benchmark.py
```

### Run Master Test Suite
```bash
python3 tests/run_all_tests.py
```

### Launch Interactive Web Dashboard
```bash
# Direct browser opening
open dashboard/index.html

# Or via static HTTP server
python3 -m http.server 3000
# Access via http://localhost:3000/dashboard/index.html
```

---

## 2. GitHub Pages Deployment

The repository is pre-configured for GitHub Pages:
1. Push repository to `https://github.com/kaleabmezgebe/stock-trend-lstm`.
2. In repository settings, navigate to **Pages** -> **Branch:** `main` / `root`.
3. The root `index.html` and `.nojekyll` automatically serve the dashboard at:
   `https://kaleabmezgebe.github.io/stock-trend-lstm/`

---

## 3. REST API Microservice Deployment

### Run Standalone HTTP Server
```bash
python3 backend/app/main.py 8000
```

### Docker Containerization (Optional)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . /app
EXPOSE 8000
CMD ["python3", "backend/app/main.py", "8000"]
```

Build and run:
```bash
docker build -t stock-trend-lstm:v1 .
docker run -p 8000:8000 stock-trend-lstm:v1
```
