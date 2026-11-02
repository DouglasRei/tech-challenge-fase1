# syntax=docker/dockerfile:1

# Tech Challenge FIAP - Fase 1
# Ambiente Linux compativel com o Python 3.13 usado no desenvolvimento.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true

WORKDIR /app

# Biblioteca de runtime utilizada por dependencias cientificas
# (por exemplo, implementacoes com OpenMP).
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Instala dependencias em uma camada separada, facilitando rebuilds.
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

# Executa o aplicativo como usuario sem privilegios de root.
RUN useradd --create-home --shell /usr/sbin/nologin appuser

# Inclui a interface, dados, modelo treinado e relatorios do repositorio.
# O .dockerignore exclui temporarios, segredos e backups.
COPY --chown=appuser:appuser . /app

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3).read()" || exit 1

CMD ["python", "-m", "streamlit", "run", "app/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true"]
