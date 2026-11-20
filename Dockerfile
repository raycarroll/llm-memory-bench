FROM python:3.11-slim

WORKDIR /app
ENV HOME=/tmp
COPY pyproject.toml README.md ./
COPY src ./src
COPY scenarios ./scenarios
COPY configs ./configs

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir '.[ui,memoryhub]'

EXPOSE 8501
CMD ["streamlit", "run", "src/llm_memory_bench/ui/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--server.fileWatcherType=none"]
