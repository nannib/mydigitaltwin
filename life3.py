import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import subprocess
import requests
import torch
import gradio as gr
import speech_recognition as sr
from TTS.api import TTS
from lipsync import LipSync
import time
import whisper
import numpy as np
from transformers import pipeline
from transformers import WhisperForConditionalGeneration
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
import json
import faiss
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pytesseract
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModel, BlipProcessor, BlipForConditionalGeneration

# Configurazioni RAG
WORKSPACES_DIR = "defaultws"
DOCUMENTS_DIR = "documents"
DIMENSION = 768  # Dimensione embedding 
OLLAMA_BASE_URL = "http://localhost:11434"
embedder = "dbmdz/bert-base-italian-uncased" 
AUDIO_OUTPUT_PATH = "output_audio.wav"
VIDEO_OUTPUT_PATH = "output_video.mp4"

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else:
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
	
# Modelli Whisper per trascrizione
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="italian", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
forced_decoder_ids = tokenizer.get_decoder_prompt_ids(language="italian", task="transcribe")
transcriber = pipeline("automatic-speech-recognition", model=model, feature_extractor=feature_extractor,
                      tokenizer=tokenizer, chunk_length_s=10, stride_length_s=(4, 2), device=device)

# Inizializzazione TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Funzioni RAG
def get_ollama_models():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()
        return sorted([model['name'] for model in response.json().get('models', [])], key=lambda x: x.lower())
    except Exception as e:
        print(f"Errore recupero modelli: {str(e)}")
        return [""]

def initialize_workspace():
    os.makedirs(WORKSPACES_DIR, exist_ok=True)
    os.makedirs(DOCUMENTS_DIR, exist_ok=True)
    
    config_path = os.path.join(WORKSPACES_DIR, "config.json")
    if not os.path.exists(config_path):
        default_config = {
            "model": "",
            "embedder": "dbmdz/bert-base-italian-uncased",
            "temperature": 0.5,
            "chunk_size": 512,
            "top_k": 5,
            "system_prompt": "Sei un assistente esperto. Rispondi basandoti sul contesto fornito."
        }
        with open(config_path, 'w') as f:
            json.dump(default_config, f)
    
    # Crea strutture dati iniziali
    index_path = os.path.join(WORKSPACES_DIR, "vector.index")
    metadata_path = os.path.join(WORKSPACES_DIR, "metadata.pkl")
    log_path = os.path.join(WORKSPACES_DIR, "processed_files.log")
    
    if not os.path.exists(index_path):
        index = faiss.IndexFlatL2(DIMENSION)
        faiss.write_index(index, index_path)
        open(log_path, 'w').close()
        with open(metadata_path, 'wb') as f:
            pickle.dump([], f)

# Funzioni di indicizzazione
def get_processed_files(log_file):
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            return set(f.read().splitlines())
    return set()
	
def update_config(model, temperature, chunk_size, top_k, system_prompt):
    try:
        config_path = os.path.join(WORKSPACES_DIR, "config.json")
        ws_path = WORKSPACES_DIR
        index_file = os.path.join(ws_path, "vector.index")
        metadata_file = os.path.join(ws_path, "metadata.pkl")
        log_file = os.path.join(ws_path, "processed_files.log")
        index_config_file = os.path.join(ws_path, "config.json")

        # Carica la vecchia configurazione
        with open(config_path, 'r') as f:
            old_config = json.load(f)

        new_config = {
            "model": model,
            "embedder": "dbmdz/bert-base-italian-uncased",
            "temperature": float(temperature),
            "chunk_size": int(chunk_size),
            "top_k": int(top_k),
            "system_prompt": system_prompt
        }

        # Controlla se è necessario ricostruire l'indice
        rebuild = False
        if os.path.exists(index_config_file):
            with open(index_config_file, "r") as f:
                saved_config = json.load(f)
                rebuild = (saved_config['chunk_size'] != new_config['chunk_size'] or 
                          saved_config['top_k'] != new_config['top_k'])

        if rebuild:
            yield "🔄 Ricostruzione indice... (Questa operazione potrebbe richiedere alcuni minuti)"
            # Ricrea le strutture dati
            for f in [log_file, index_file, metadata_file, index_config_file]:
                if os.path.exists(f):
                    os.remove(f)
            initialize_workspace()

        # Processa i documenti
        processed_files = get_processed_files(log_file)
        current_files = set()
        
        for root, _, files in os.walk(DOCUMENTS_DIR):
            for file in files:
                current_files.add(os.path.join(root, file))
        
        new_files = current_files - processed_files
        removed_files = processed_files - current_files

        if new_files or removed_files:
            # Aggiorna l'indice
            index = faiss.read_index(index_file) if os.path.exists(index_file) else faiss.IndexFlatL2(DIMENSION)
            metadata = []
            
            if os.path.exists(metadata_file):
                with open(metadata_file, 'rb') as f:
                    metadata = pickle.load(f)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=new_config['chunk_size'],
                chunk_overlap=new_config['chunk_size'] // 4
            )

            # Processa nuovi file
            for path in new_files:
                try:
                    text = extract_text(path)
                    if text:
                        chunks = text_splitter.split_text(text)
                        for chunk in chunks:
                            embedding = generate_embedding(chunk, new_config['embedder'])
                            index.add(np.array([embedding]))
                            metadata.append({
                                'path': path,
                                'content': chunk[:1000],
                                'embedding': embedding
                            })
                        yield f"📄 Processato: {os.path.basename(path)}"
                except Exception as e:
                    yield f"❌ Errore processing {os.path.basename(path)}: {str(e)}"

            # Rimuovi file eliminati
            metadata = [m for m in metadata if m['path'] in current_files]

            # Salva lo stato
            faiss.write_index(index, index_file)
            with open(metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
            with open(log_file, 'w') as f:
                f.write("\n".join(current_files))

        # Salva nuova configurazione
        with open(config_path, 'w') as f:
            json.dump(new_config, f)
        with open(index_config_file, 'w') as f:
            json.dump(new_config, f)

        yield "✅ Configurazione aggiornata con successo! Documenti indicizzati: {}".format(len(current_files))

    except Exception as e:
        yield f"❌ Errore critico: {str(e)}"
        raise
    
    
    # Salva la configurazione usata per l'indice
    with open(index_config_file, "w") as f:
        json.dump({"model":model,
            "embedder": embedder,
            "chunk_size": chunk_size,
            "top_k": top_k, "temperature":temperature,"system_prompt":system_prompt
        }, f)
    return status_output
# Inizializza il modello BERT all'avvio
try:
    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-uncased")
    bert_model = AutoModel.from_pretrained("dbmdz/bert-base-italian-uncased")
except Exception as e:
    print(f"Errore caricamento modello BERT: {str(e)}")
    raise

def extract_text(path):
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == '.pdf':
            
            with open(path, 'rb') as f:
                return ''.join([page.extract_text() for page in PdfReader(f).pages])
        elif ext == '.txt':
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        elif ext in ['.jpg', '.png', '.jpeg']:
            
            return pytesseract.image_to_string(Image.open(path), lang='ita')
        elif ext in ['.mp3', '.wav']:
            return whisper.load_model('base').transcribe(path, language='it')['text']
        else:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
    except Exception as e:
        print(f"Errore estrazione testo: {str(e)}")
        return ""

# funzione generate_embedding
def generate_embedding(text, embedder):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()[0]

def process_documents(config):
    index_path = os.path.join(WORKSPACES_DIR, "vector.index")
    metadata_path = os.path.join(WORKSPACES_DIR, "metadata.pkl")
    log_path = os.path.join(WORKSPACES_DIR, "processed_files.log")

    stats = {"processed": 0, "errors": 0}
    
    try:
        processed_files = set(open(log_path).read().splitlines()) if os.path.exists(log_path) else set()
        current_files = set()
        
        for root, _, files in os.walk(DOCUMENTS_DIR):
            for file in files:
                current_files.add(os.path.join(root, file))
        
        new_files = current_files - processed_files
        stats["total"] = len(new_files)

        if not new_files:
            return stats

        index = faiss.read_index(index_path)
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config['chunk_size'],
            top_k=config['top_k']
        )

        for path in new_files:
            try:
                text = extract_text(path)
                if text:
                    chunks = text_splitter.split_text(text)
                    for chunk in chunks:
                        embedding = generate_embedding(chunk, config['embedder'])
                        embedding = np.array(embedding).astype('float32')
                        index.add(np.array([embedding]))
                        metadata.append({
                            'path': path,
                            'content': chunk,
                            'embedding': embedding
                        })
                    stats["processed"] += 1
            except Exception as e:
                print(f"Errore processing {path}: {str(e)}")
                stats["errors"] += 1

        faiss.write_index(index, index_path)
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        with open(log_path, 'a') as f:
            f.write('\n'.join(new_files) + '\n')

    except Exception as e:
        print(f"Errore generale processamento: {str(e)}")
        stats["errors"] += 1
    
    return stats
	
def rag_search(query, config):
    index_path = os.path.join(WORKSPACES_DIR, "vector.index")
    metadata_path = os.path.join(WORKSPACES_DIR, "metadata.pkl")
    
    try:
        index = faiss.read_index(index_path)
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        query_embedding = generate_embedding(query, config['embedder'])
        
        # Controllo dimensione embedding
        if query_embedding.shape[0] != DIMENSION:
            raise ValueError(f"Dimensione embedding non valida: {query_embedding.shape}")
        
        # Ricerca con controllo risultati
        _, indices = index.search(np.array([query_embedding]).astype("float32"), config['top_k'])
        
        context = []
        for idx in indices[0]:
            if 0 <= idx < len(metadata):
                context.append(metadata[idx]['content'][:1000])  # Limita lunghezza
        
        return "\n\n".join(context) if context else "Nessun contesto rilevante trovato"
    
    except Exception as e:
        print(f"Errore ricerca RAG: {str(e)}")
        return ""

# Funzioni core applicazione
# Correzione della funzione query_ollama
def query_ollama(prompt: str) -> str:
    try:
        with open(os.path.join(WORKSPACES_DIR, "config.json"), 'r') as f:
            config = json.load(f)
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": config['model'],
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": config['temperature'],
                    "num_ctx": 8192
                }
            }
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    
    except Exception as e:
        return f"Errore nella generazione della risposta: {str(e)}"

def synthesize_voice(text: str, speaker_wav: str, output_path: str = "output.wav") -> str:
    """Genera audio dalla text-to-speech con controllo degli input"""
    try:
        # Verifica presenza file voce campione
        if not os.path.exists(speaker_wav):
            raise FileNotFoundError(f"File voce campione non trovato: {speaker_wav}")
        
        # Verifica testo valido
        if not text or not isinstance(text, str):
            raise ValueError("Testo non valido per la sintesi vocale")
        
        # Genera audio
        tts.tts_to_file(
            text=text,
            speaker_wav=speaker_wav,
            file_path=output_path,
            language="it"
        )
        return output_path
    
    except Exception as e:
        raise RuntimeError(f"Errore sintesi vocale: {str(e)}") from e

def lip_sync(audio_path: str, video_sample: str, output_video: str) -> str:
    lip = LipSync(
        model='wav2lip',
        checkpoint_path='weights/wav2lip_gan.pth',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    lip.sync(video_sample, audio_path, output_video)
    return output_video

def transcribe_audio(audio):
    sr, y = audio
    y = y.astype(np.float32) / np.max(np.abs(y))
    return transcriber({"sampling_rate": sr, "raw": y.flatten()})["text"]

def run_pipeline(text_prompt, voice_input, history):
    history = history or []
    
    # Percorsi fissi dei file campione
    voice_sample = "sample_voice.wav"
    video_sample = "sample_face.mp4"
    
    try:
        # Carica configurazione
        with open(os.path.join(WORKSPACES_DIR, "config.json"), 'r') as f:
            config = json.load(f)
        
        # Gestione input vocale
		# Gestione input testuale
        if text_prompt:
            history.append(("👤", text_prompt))
        if not text_prompt and voice_input:
            text_prompt = transcribe_audio(voice_input)
            history.append(("👤", text_prompt))
        
        if not text_prompt:
            return history, "Nessun prompt fornito.", None

        # Verifica file campione
        if not os.path.exists(voice_sample):
            return history, "File voce campione mancante!", None
        if not os.path.exists(video_sample):
            return history, "File video campione mancante!", None

        # Costruisci la cronologia della chat
        chat_history_str = "\n".join([f"{role} {msg}" for role, msg in history])
		# Ricerca contestuale
        context = rag_search(text_prompt, config)
        # Combina la cronologia della chat con il contesto RAG e il prompt
        full_prompt = f"{config['system_prompt']}\nContesto:\n{context}\nChat:\n{chat_history_str}\nDomanda: {text_prompt}\nRisposta:"

        # Generazione risposta
        response_text = query_ollama(full_prompt)
		        # Aggiungi risposta alla cronologia PRIMA di restituire
        history.append(("🤖", response_text))
        
        # Generazione output audio/video
        output_audio = synthesize_voice(text=response_text,speaker_wav=voice_sample,output_path=AUDIO_OUTPUT_PATH)
        output_video = lip_sync(output_audio, video_sample, VIDEO_OUTPUT_PATH)
        
        return history, "", output_video
    
    except Exception as e:
        return history, f"Errore: {str(e)}", None

# Interfaccia Gradio
css = """
.chat-history {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 15px;
    height: 200px;
    overflow-y: auto;
    background: #f9f9f9;
}
.chat-message { 
    margin: 10px 0;
    padding: 8px 12px;
    border-radius: 15px;
    max-width: 80%;
}
.user-message {
    background: #e3f2fd;
    margin-left: auto;
	color: #c62828;
}
.bot-message {
    background: #f5f5f5;
	color: #c62828;
}
video {
    max-width: 100% !important;
    height: auto !important;
}
"""

autoplay_js = """
<script>
document.addEventListener("DOMContentLoaded", function() {
    const observer = new MutationObserver((mutations) => {
        const videoElement = document.querySelector('video');
        if (videoElement && !videoElement.autoplay) {
            videoElement.autoplay = true;
            videoElement.muted = true;
            videoElement.play();
        }
    });
    observer.observe(document.body, { childList: true, subtree: true });
});
</script>
"""
css += """
.status-box {
    padding: 10px;
    border-radius: 5px;
    margin: 10px 0;
}
.status-success {
    background: #e8f5e9;
    color: #2e7d32;
    border: 1px solid #a5d6a7;
}
.status-error {
    background: #ffebee;
    color: #c62828;
    border: 1px solid #ef9a9a;
}
"""

def cleanup_and_exit():
    """Pulisce i file e termina il programma"""
    try:
        if os.path.exists(AUDIO_OUTPUT_PATH):
            os.remove(AUDIO_OUTPUT_PATH)
        if os.path.exists(VIDEO_OUTPUT_PATH):
            os.remove(VIDEO_OUTPUT_PATH)
    except Exception as e:
        print(f"Errore durante la pulizia: {str(e)}")
    
    os._exit(0)  # Termina forzatamente il processo


with gr.Blocks(css=css, title="AI Assistant") as app:
    gr.HTML("<h1 style='text-align: center'>DigitalTwin con RAG Integrato</h1>")
    gr.HTML(autoplay_js)
    
    with gr.Row():
        with gr.Column(scale=3):
            chat_history = gr.HTML(elem_classes="chat-history",
                                  value="<div style='text-align: center'>Inizia la conversazione...</div>")
            
            text_input = gr.Textbox(label="Prompt Testuale", placeholder="Scrivi qui...", lines=2)
            audio_input = gr.Audio(sources="microphone", type="numpy", label="Registra Audio")
            
            with gr.Row():
                clear_btn = gr.Button("Pulisci", variant="secondary")
                submit_btn = gr.Button("Invia", variant="primary")
                exit_btn = gr.Button("EXIT", variant="stop") 

        with gr.Column(scale=2):
            video_output = gr.Video(label="Video Output", autoplay=True, format="mp4")
            
            with gr.Row():
                with gr.Column(scale=4):
                    with gr.Accordion("Configurazione Avanzata RAG", open=False):
                        initialize_workspace()
                        with open(os.path.join(WORKSPACES_DIR, "config.json"), 'r') as f:
                            current_config = json.load(f)
                
                        ollama_models = get_ollama_models()
                        ordered_models = sorted(
                            ollama_models,
                            key=lambda x: x == current_config.get('model', '---'),
                            reverse=True
                        )
                
                        with gr.Row():
                            model_selector = gr.Dropdown(
                                choices=ordered_models,
                                label="Modello LLM",
                                value=current_config.get('model', '---'),allow_custom_value=True
                            )
                            temperature = gr.Slider(0.0, 1.0, 
                                                  value=current_config.get('temperature', 0.7), 
                                                  label="Temperature")
                
                        with gr.Row():
                            chunk_size = gr.Number(value=current_config.get('chunk_size', 512), 
                                                label="Chunk Size")
                            top_k = gr.Number(value=current_config.get('top_k', 5), 
                                            label="Top K Documents")
                
                        system_prompt = gr.Textbox(value=current_config.get('system_prompt', ''), 
                                                 label="System Prompt", 
                                                 lines=3)
                        status_output = gr.Textbox(label="Stato Configurazione", interactive=False)
                        config_submit = gr.Button("Applica Configurazione", variant="primary")

    history_state = gr.State()
    
    # Event handlers
    config_submit.click(
    update_config,
    [model_selector, temperature, chunk_size, top_k, system_prompt],
    status_output,
    api_name="update_config"
)  # Collegamento all'output di stato

    
    submit_btn.click(
        run_pipeline,
        [text_input, audio_input, history_state],
        [history_state, text_input, video_output]
    ).then(
        lambda history: gr.update(value=format_chat_history(history)),
        inputs=history_state,
        outputs=chat_history
    )
    
    clear_btn.click(
        lambda: ([], "", None),
        outputs=[history_state, text_input, video_output]
    ).then(
        lambda: gr.update(value="<div style='text-align: center'>Conversazione resettata...</div>"),
        outputs=chat_history
    )
    exit_btn.click(
        cleanup_and_exit,
        inputs=None,
        outputs=None,
        queue=False
    )

def format_chat_history(history):
    if not history:
        return "<div style='text-align: center'>Inizia la conversazione...</div>"
    
    html = []
    for role, msg in history:
        css_class = "user-message" if role == "👤" else "bot-message"
        html.append(f"<div class='chat-message {css_class}'>{role} {msg}</div>")
    
    return "<div class='chat-history'>" + "\n".join(html) + "</div>"

if __name__ == "__main__":
    # Pulisci file residui all'avvio
    if os.path.exists(AUDIO_OUTPUT_PATH):
        os.remove(AUDIO_OUTPUT_PATH)
    if os.path.exists(VIDEO_OUTPUT_PATH):
        os.remove(VIDEO_OUTPUT_PATH)
    initialize_workspace()
    app.launch()
