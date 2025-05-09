import sys
import os

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.database.db_base_manager import DatabaseManager

# Set environment variables for PostgreSQL connection
os.environ["PG_CONNECTION_STRING"] = "postgresql://postgres:password@localhost:5432/metanavit"
os.environ["PSYCOPG2_CONNECTION_STRING"] = "dbname=metanavit user=postgres password=password host=localhost port=5432"
os.environ["DB_NAME"] = "metanavit"

# Initialize the DatabaseManager
db = DatabaseManager()

# Create documents table (safe if it already exists)
db.create_documents_table()

# Insert sample documents
docs = [
    ("PostgreSQL Tutorial", "Learn PostgreSQL with full-text search and BM25 ranking."),
    ("Hybrid Search", "Combine semantic and keyword-based search with pgvector."),
    ("ParadeDB", "Use ParadeDB to enable BM25 on PostgreSQL."),
    ("BM25 Ranking", "This method ranks documents by term relevance and frequency."),
]

for title, content in docs:
    db.insert_document(title, content)

# Create BM25 index
db.create_bm25_index(table="documents", fields="title, content", key_field="id")

# Run BM25-ranked search
results = db.run_bm25_search(table="documents", search_column="content", query_text="postgresql search tutorial", limit=5)

# Print results
for row in results:
    print(f"ID: {row[0]}, Title: {row[1]}, Score: {row[-1]}")
