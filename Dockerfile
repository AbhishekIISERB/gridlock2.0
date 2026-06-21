FROM python:3.12-slim

WORKDIR /app

# Install system dependencies (for building some python packages if needed)
RUN apt-get update && apt-get install -y gcc sqlite3 && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Ensure start script is executable
RUN chmod +x start_server.sh

# Expose port 7860 (Hugging Face Spaces default, or configurable via cloud providers)
EXPOSE 7860

# Run the unified start script
CMD ["./start_server.sh"]
