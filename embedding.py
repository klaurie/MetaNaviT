import psycopg2
import requests

OLLAMA_URL = "http://localhost:11434"

try:
    
    connection = psycopg2.connect(
        database="metanavit",
        user="postgres",
        host="localhost",
        port="5432"  
    )
    print("Connection established successfully.")

    cursor = connection.cursor()

    query = """
        SELECT e.id, fc.metadata
        FROM embeddings e
        JOIN file_contents fc ON e.file_content_id = fc.id
        WHERE e.embedding IS NULL;
        """

    cursor.execute(query)
    records = cursor.fetchall()

    for row in records:
        embedding_id = row[0]
        metadata = row[1]

        text_to_embed = (
            f"File name: {metadata.get('file_name', 'unknown')}, "
            f"File type: {metadata.get('file_type', 'unknown')}, "
            f"File size: {metadata.get('file_size', 'unknown')} bytes, "
            f"Last modified on: {metadata.get('last_modified', 'unknown')}."
        )

        try:

            response = requests.post(
                f"{OLLAMA_URL}/api/embed",  
                json={
                    "model": "mxbai-embed-large", 
                    "input": text_to_embed
                }
            )

            if response.status_code == 200:
                response_data = response.json()
                print(f"Full response for ID {embedding_id}: {response_data}")

                embedding = response_data.get("embeddings")

                if isinstance(embedding, list):
                    if len(embedding) > 0 and isinstance(embedding[0], list):
                        embedding = [item for sublist in embedding for item in sublist]
                    embedding = [float(x) for x in embedding]

                print(f"Embedding for ID {embedding_id}: {embedding}")

                if embedding:
                    update_query = """
                    UPDATE embeddings
                    SET embedding = %s
                    WHERE id = %s;
                    """
                    cursor.execute(update_query, (embedding, embedding_id))
                    connection.commit()
            else:
                print(f"Failed to get embedding for ID {embedding_id}: {response.status_code} - {response.text}")

        except requests.exceptions.RequestException as e:
            print(f"Error connecting to Ollama: {e}")

        except psycopg2.Error as db_error:
            print(f"Database error for ID {embedding_id}: {db_error}")
            connection.rollback() 

        except Exception as error:
            print(f"Error while processing embedding for ID {embedding_id}: {error}")
finally:
    if connection:
        cursor.close()
        connection.close()
        print("PostgreSQL connection is closed")
