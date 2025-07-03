# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application
COPY . .

# Create a non-root user
RUN useradd -m -u 1000 streamlit && chown -R streamlit:streamlit /app
USER streamlit

# Expose the port Streamlit runs on
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the Streamlit app
CMD ["streamlit", "run", "web_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
