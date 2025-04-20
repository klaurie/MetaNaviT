# Example Dockerfile for PostgreSQL 15 with pg_search
# Use an appropriate base image for PostgreSQL 15
# Example: FROM postgres:14-bookworm 
# Or potentially: FROM pgvector/pgvector:pg14 (if it's Debian-based)
FROM postgres:15 

# Add installation steps for pg_search
RUN apt-get update && \
    # Install curl and ca-certificates needed for downloading
    apt-get install -y --no-install-recommends curl ca-certificates && \
    \
    # Download the correct .deb file for PG15 / Bookworm
    # !!! Make sure this URL points to the PG 15 version of the .deb file !!!
    curl -L "https://github.com/paradedb/paradedb/releases/download/v0.15.16/postgresql-15-pg-search_0.15.16-1PARADEDB-bookworm_amd64.deb" -o /tmp/pg_search.deb && \
    \
    # Install the downloaded .deb package
    # Using apt-get install handles dependencies better than dpkg -i
    apt-get install -y /tmp/pg_search.deb && \
    \
    # Clean up downloaded file and apt cache
    rm /tmp/pg_search.deb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Your other PostgreSQL Dockerfile configurations (e.g., COPY init scripts, EXPOSE port)
# ...

# Standard command to run PostgreSQL
CMD ["postgres"]
