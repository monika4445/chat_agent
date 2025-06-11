#!/bin/bash
# Helper script to install Ollama if missing (for MacOS/Linux)
if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama..."
    brew install ollama || curl -fsSL https://ollama.com/install.sh | sh
    ollama pull mistral
fi