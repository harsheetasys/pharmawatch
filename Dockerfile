FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Build the synthetic world + model + index at image build time, so the container
# starts instantly and a free-tier host never needs a separate database.
RUN python -m scripts.build_all --source synthetic
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"
CMD ["sh", "-c", "uvicorn pharmawatch.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
