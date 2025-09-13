#!/bin/bash

# Gemma Embedder Service Startup Script
# Starts the semantic similarity service for trading decisions

echo "🧠 Starting Gemma Embedder Service..."
echo "====================================="

# Set working directory
cd "$(dirname "$0")/api/python"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python first."
    exit 1
fi

# Install dependencies if needed
if [ ! -f "requirements_embedder.txt" ]; then
    echo "❌ requirements_embedder.txt not found"
    exit 1
fi

echo "📦 Installing dependencies..."
pip3 install -r requirements_embedder.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Create data directory if it doesn't exist
mkdir -p ../data/embeddings

# Start the embedder service
echo "🚀 Starting embedder service on port 8002..."
echo "🔗 API available at: http://localhost:8002"
echo "📊 Health check: http://localhost:8002/health"
echo ""
echo "📋 Available endpoints:"
echo "   POST /embed - Generate text embeddings"
echo "   POST /store - Store headline with embedding"
echo "   POST /similar - Find similar historical cases"
echo "   GET /health - Service health status"
echo "   DELETE /clear - Clear stored embeddings"
echo ""
echo "⚙️  Configuration:"
echo "   Model: EmbeddingGemma-300M (256-dim)"
echo "   Vector Store: FAISS index"
echo "   Cache: Local pickle file"
echo ""
echo "🧠 Embedder service is running!"
echo "   Press Ctrl+C to stop"

python3 embedder_service.py
