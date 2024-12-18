
# Use miniconda base image
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    postgresql-client \
    curl \
    tesseract-ocr \
    poppler-utils \
    libmagic1 \
    pandoc \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama CLI
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Copy environment.yml and requirements.txt
COPY environment.yml requirements.txt ./

# Create conda environment and install dependencies
RUN conda env create -f environment.yml

# Add debug commands to verify installation
RUN echo '#!/bin/bash\n\
source /opt/conda/etc/profile.d/conda.sh\n\
conda activate ollama_env\n\
echo "Python version:"\n\
exec "$@"' > /entrypoint.sh && \
chmod +x /entrypoint.sh

# Verify llama-index installation explicitly
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate ollama_env && \
    pip install llama-index-core llama-index-readers-file && \
    python -c 'from llama_index.core import SimpleDirectoryReader'"

# Copy the application code
COPY ./app ./app
COPY ./wait-for-postgres.sh .
RUN chmod +x ./wait-for-postgres.sh

RUN echo '#!/bin/bash' > /app/entrypoint.sh && \
    echo 'set -e' >> /app/entrypoint.sh && \
    echo 'source /opt/conda/etc/profile.d/conda.sh' >> /app/entrypoint.sh && \
    echo 'conda activate ollama_env' >> /app/entrypoint.sh && \
    echo '' >> /app/entrypoint.sh && \
    echo '# Wait for postgres' >> /app/entrypoint.sh && \
    echo 'until PGPASSWORD=postgres pg_isready -h db -p 5432 -U postgres; do' >> /app/entrypoint.sh && \
    echo '  >&2 echo "Postgres is unavailable - sleeping"' >> /app/entrypoint.sh && \
    echo '  sleep 1' >> /app/entrypoint.sh && \
    echo 'done' >> /app/entrypoint.sh && \
    echo '>&2 echo "Postgres is up - executing command"' >> /app/entrypoint.sh && \
    echo 'exec "$@"' >> /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# Remove any duplicate entrypoint scripts
RUN rm -f /entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]
