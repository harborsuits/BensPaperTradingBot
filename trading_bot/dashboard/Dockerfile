FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create directory structure for the entire trading_bot package
RUN mkdir -p /trading_bot

# Set PYTHONPATH to include both app and project root
ENV PYTHONPATH="/:/app:$PYTHONPATH"

# Copy the dashboard files
COPY . /app/

# Expose the Streamlit port
EXPOSE 8501

# Command to run the application with proper Python path
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
