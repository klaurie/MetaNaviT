import os
from datetime import datetime
import psycopg2
import json

def get_file_path():
    while True:
        file_path = input("Enter the file path to set root dir: ")
        if os.path.exists(file_path):
            return file_path
        else:
            print("Invalid file path.")

file_path = get_file_path()
print("This is the file_path: ", file_path)

# Collection of all things in file (metadata only)
parsedfile = []

for root, dirs, files in os.walk(file_path):
    for filename in files:
        file_path = os.path.join(root, filename)

        metaData = {
            "file_name": filename,
            "file_type": os.path.splitext(filename)[1],  # Get file extension
            "file_size": os.path.getsize(file_path),      # File size in bytes
            "last_modified": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),  # Format as ISO string
            "file_path": file_path  # Add file_path to metadata
        }
        parsedfile.append(metaData)

try:
    connection = psycopg2.connect(
        database="metanavit",
        user="postgres",
        host="localhost",
        port="5432"  # Default PostgreSQL port; change if different
    )
    print("Connection established successfully.")
except Exception as e:
    print("Connection error: ", e)

cursor = connection.cursor()

# Insert metadata into file_contents
insert_file_contents_query = """
INSERT INTO file_contents (file_name, file_type, file_size, last_modified, created_at, metadata)
VALUES (%s, %s, %s, %s, NOW(), %s)
RETURNING id;
"""

insert_embeddings_query = """
INSERT INTO embeddings (embedding, file_content_id, file_name, file_path)
VALUES (NULL, %s, %s, %s);
"""

for file_metadata in parsedfile:
    try:
        # Delete existing records for this file
        cursor.execute("DELETE FROM embeddings WHERE file_name = %s", (file_metadata["file_name"],))
        cursor.execute("DELETE FROM file_contents WHERE file_name = %s", (file_metadata["file_name"],))
        connection.commit()
        print(f"Existing records for {file_metadata['file_name']} deleted.")
    except psycopg2.Error as e:
        print(f"Error deleting existing data for {file_metadata['file_name']}: {e}")
        connection.rollback()
        continue

    # Convert metadata to JSON format for storage
    metadata_json = json.dumps(file_metadata)
    # Prepare data for file_contents table
    values_file_contents = (
        file_metadata["file_name"],
        file_metadata["file_type"],
        file_metadata["file_size"],
        file_metadata["last_modified"],
        metadata_json
    )

    try:
        # Insert into file_contents and get the ID
        cursor.execute(insert_file_contents_query, values_file_contents)
        file_content_id = cursor.fetchone()[0]

        # Insert into embeddings with file_content_id and file_path as a reference
        values_embeddings = (
            file_content_id,
            file_metadata["file_name"],
            file_metadata["file_path"]  # Include file_path in embeddings
        )

        cursor.execute(insert_embeddings_query, values_embeddings)
        connection.commit()
        print(f"Data for {file_metadata['file_name']} inserted.")
    except psycopg2.Error as e:
        print(f"Error inserting data for {file_metadata['file_name']}: {e}")
        connection.rollback()

cursor.close()
connection.close()

print("Database update and insertion complete, and connection closed.")
