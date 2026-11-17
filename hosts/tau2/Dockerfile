ARG TAU2_REF=c06b07dd7c4d4ec61ba6ff9bcdf0b15dd7ef36d4

FROM python:3.13-slim

ARG TAU2_REF

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TAU2_DATA_DIR=/opt/tau2-data

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install the harness from the build context so the image contains the exact
# source being tested, including local uncommitted changes during development.
COPY pyproject.toml README.md ./
COPY src ./src
COPY prompts ./prompts
COPY scenarios ./scenarios

RUN python -m pip install -e ".[memoryhub]" \
    && python -m pip install audioop-lts \
    && python -m pip install \
        "git+https://github.com/sierra-research/tau2-bench.git@${TAU2_REF}"

# τ² keeps its benchmark data in the repository rather than in the Python
# wheel. Bake the pinned data into the image so runtime users do not need a
# second checkout or a local TAU2_DATA_DIR setting.
RUN git init /tmp/tau2-bench \
    && git -C /tmp/tau2-bench remote add origin https://github.com/sierra-research/tau2-bench.git \
    && git -C /tmp/tau2-bench fetch --depth 1 origin "${TAU2_REF}" \
    && git -C /tmp/tau2-bench checkout --detach FETCH_HEAD \
    && mkdir -p /opt/tau2-data \
    && cp -a /tmp/tau2-bench/data/. /opt/tau2-data/ \
    && rm -rf /tmp/tau2-bench

COPY hosts/tau2/entrypoint.sh /usr/local/bin/tau2-value-run
RUN chmod +x /usr/local/bin/tau2-value-run

ENTRYPOINT ["/usr/local/bin/tau2-value-run"]
