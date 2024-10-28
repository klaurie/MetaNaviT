# Use the official Ollama Docker image
FROM ollama/ollama:latest

# Expose port 11434
EXPOSE 11434

# Command to run Ollama and serve
ENTRYPOINT ["/bin/ollama"]
CMD ["serve"]