#!/bin/bash

set -e

read -p "Vuoi installare Tesseract-OCR (ultima versione)? (s/n) " ansTesseract
if [[ "$ansTesseract" =~ ^[Ss]$ ]]; then
    echo "Installazione di Tesseract-OCR..."
    sudo apt update
    sudo apt install -y tesseract-ocr
    echo "Tesseract installato."
else
    echo "Salto installazione Tesseract."
fi

echo

read -p "Vuoi installare Ollama? (s/n) " ansOllama
if [[ "$ansOllama" =~ ^[Ss]$ ]]; then
    sudo apt update
    sudo apt install -y curl
    echo "Installazione di Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo "Ollama installato."

    read -p "Vuoi installare il modello llama3.2? (s/n) " ansLlama
    if [[ "$ansLlama" =~ ^[Ss]$ ]]; then
        echo "Avvio dell'installazione del modello llama3.2..."
        ollama run llama3.2:latest
    else
        echo "Salto installazione modello llama3.2."
    fi
else
    echo "Salto installazione Ollama."
fi

echo

read -p "Vuoi installare FFmpeg? (s/n) " ansFFmpeg
if [[ "$ansFFmpeg" =~ ^[Ss]$ ]]; then
    echo "Installazione di FFmpeg..."
    sudo apt update
    sudo apt install -y ffmpeg
    echo "FFmpeg installato."
else
    echo "Salto installazione FFmpeg."
fi

echo
echo "Operazione completata."
