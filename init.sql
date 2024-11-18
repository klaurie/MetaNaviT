-- Enable the vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Drop existing table if it exists
DROP TABLE IF EXISTS vector_store;

-- Create the vector_store table with the correct vector type
CREATE TABLE vector_store (
    id UUID PRIMARY KEY,
    metadata JSONB,
    snippet TEXT,
    embedding vector(768)  -- Specify the vector dimension
);

-- Create an index for vector similarity search
CREATE INDEX IF NOT EXISTS vector_store_embedding_idx ON vector_store 
USING ivfflat (embedding vector_cosine_ops);

-- Create the query_history table
CREATE TABLE IF NOT EXISTS query_history (
    query_id UUID PRIMARY KEY,
    query TEXT NOT NULL,
    response TEXT,
    context TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
