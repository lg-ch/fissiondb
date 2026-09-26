FROM python:3.11-slim-bookworm AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc make liburing-dev libroaring-dev libxxhash-dev libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY src/ src/
COPY scripts/ scripts/
COPY Makefile pyproject.toml setup.py MANIFEST.in README.md LICENSE NOTICE ./
RUN make product && pip wheel --no-deps --wheel-dir /wheels .

FROM python:3.11-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends \
    liburing2 libroaring-dev libxxhash0 libgomp1 libcurl4 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels
COPY --from=builder /src/mangrove-engine /usr/local/bin/mangrove-engine
ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ENTRYPOINT ["mangrove-serve"]
CMD ["--help"]
