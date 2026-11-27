FROM python:3.11-slim

WORKDIR /app
ENV HOME=/tmp

# The OpenShift demo verifier uses the cluster CLI for independent checks.
# Install the client in the image so verification does not depend on a
# separate executable being present in the runtime container.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && curl -fsSL https://mirror.openshift.com/pub/openshift-v4/clients/ocp/stable/openshift-client-linux.tar.gz \
      -o /tmp/openshift-client.tar.gz \
    && python -c "import tarfile; tarfile.open('/tmp/openshift-client.tar.gz').extract('oc', '/usr/local/bin')" \
    && oc version --client \
    && apt-get purge -y --auto-remove curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY scenarios ./scenarios
COPY configs ./configs

# OpenShift assigns a random non-root UID at runtime. Keep the application
# tree immutable while providing a writable location for UI run artifacts.
RUN mkdir -p /app/results \
    && chmod 0777 /app/results

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir '.[ui,memoryhub]'

EXPOSE 8501
CMD ["streamlit", "run", "src/llm_memory_bench/ui/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--server.fileWatcherType=none"]
