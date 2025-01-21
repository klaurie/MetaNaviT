FROM debian:bullseye-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    cmake \
    libfftw3-dev \
    libcurl4-openssl-dev \
    wget \
    ffmpeg \
    python3 \
    python3-pip

# Install Flask
RUN pip3 install flask

# Set the working directory
WORKDIR /app

# Clone the whisper.cpp repository
RUN git clone https://github.com/ggerganov/whisper.cpp.git

# Build the project
WORKDIR /app/whisper.cpp
RUN make build

# Copy the models directory
COPY models /app/whisper.cpp/models

# Copy the server script
COPY server.py /app/whisper.cpp/server.py

# Expose port 8080
EXPOSE 8080

# Start the Flask server
CMD ["python3", "server.py"]
