import os
from datetime import datetime
import psycopg2
import json

def get_file_path():
    # get file path from user

    while True:
        file_path = input("Enter the file path to set root dir: ")

        if os.path.exists(file_path):
            return file_path
        else:
            print("Invalid file path.")

file_path = get_file_path()
print("this is the file_path: ", file_path)

# collection of all things in file 

parsedfile = []

for root, dirs, files in os.walk(file_path):
    for filename in files:

        file_path = os.path.join(root, filename)

        metaData = {
            "file_path": file_path,
            "file_name": filename,
            "file_type": os.path.splitext(filename)[1],  # Get file extension
            "file_size": os.path.getsize(file_path),      # File size in bytes
            "last_modified": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()  # Format as ISO string
        }
        parsedfile.append(metaData)
for file in parsedfile:
    print(file)

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

insert_query = """
INSERT INTO embeddings (embedding, metadata, file_path, file_name, file_type, file_size, last_modified, created_at)
VALUES (NULL, %s, %s, %s, %s, %s, %s, NOW())
"""


for file_metadata in parsedfile:
    metadata_json = json.dumps(file_metadata)  # Convert metadata to JSON if needed
    values = (
        metadata_json,                      # metadata as JSON
        file_metadata["file_path"],         # file path
        file_metadata["file_name"],         # file name
        file_metadata["file_type"],         # file type
        file_metadata["file_size"],         # file size
        file_metadata["last_modified"],     # last modified date
    )
    try:
        cursor.execute(insert_query, values)
        connection.commit() 
    except psycopg2.Error as e:
        print(f"Error inserting data for {file_metadata['file_path']}: {e}")

        
        
cursor.close()
connection.close()

print("Database insertion complete and connection closed.")

