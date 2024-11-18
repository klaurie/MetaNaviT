#!/bin/bash
# Install PostgreSQL and PGVector if not already installed
sudo apt update
sudo apt install -y postgresql postgresql-contrib

# Create database and enable PGVector
sudo -u postgres psql -c "CREATE DATABASE rag_project;"
sudo -u postgres psql -d rag_project -c "CREATE EXTENSION IF NOT EXISTS vector;"

echo "PostgreSQL with PGVector setup is complete."
