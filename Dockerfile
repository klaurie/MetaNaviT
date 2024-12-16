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
python --version\n\
echo "Installed packages:"\n\
pip list\n\
echo "Conda environment:"\n\
conda list\n\
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

ENTRYPOINT ["/entrypoint.sh"]
CMD ["./wait-for-postgres.sh", "db", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]