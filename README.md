[![Watch the video](https://img.youtube.com/vi/AqPgEY65E0E/maxresdefault.jpg)](https://youtu.be/AqPgEY65E0E)
<p align="center">by <a href="https://nannibassetti.com" target="_blank">Nanni Bassetti</a></p>   
VIDEO: <a href="https://youtu.be/AqPgEY65E0E" target="_blank"> https://youtu.be/AqPgEY65E0E </a>   
  
.
  
[ITALIAN](#ITALIANO)  -  [ENGLISH](#ENGLISH)  

Based on **PYTHON 3.12.4**   
###  ---------------- I tuoi dati rimangono sul tuo computer ---------------   
<a name="ITALIANO"></a>  
# mydigitaltwin  
MyDigitalTwin: è un programma che può creare il tuo gemello digitale utilizzando un campione video e audio di te, con cui potrai interagire tramite microfono o chat di testo.
Ti vedrai parlare con te, con la tua voce e il tuo viso in sincronizzazione labiale.  
È chiaramente un ## PROTOTIPO ## che potrebbe funzionare meglio ed essere ampliato, quindi tutti sono invitati a contribuire!  

Utilizza un modello di intelligenza artificiale e un sistema RAG per risposte personalizzate, così potrai indicizzare la tua vita, i tuoi pensieri, così da poter vivere per sempre in modalità vita digitale.
Funziona in locale sul tuo computer, senza API, senza server esterni.

### ---------------- I tuoi dati rimangono sul tuo computer ---------------

Questo framework ti permette di chattare con i tuoi documenti in RAG, inclusi contenuti multimediali (audio, video, immagini e riconoscimento ottico dei caratteri).
Il framework è un'interfaccia grafica per chattare con un modello GPT scaricato da OLLAMA; consigliamo LLAMA 3.2 (2 GB), che funziona perfettamente anche su computer di medie dimensioni.
Inoltre, è necessario installare il software Tesseract; per il riconoscimento OCR, consigliamo di scegliere italiano e inglese durante l'installazione.

---------------- I tuoi dati rimangono sul tuo computer ---------------

È un framework italiano che ti permette di chattare con i tuoi documenti in RAG, inclusi i contenuti multimediali (audio, video, immagini e OCR).
Il framework è un'interfaccia grafica per chattare con un modello GPT scaricato da OLLAMA; consigliamo LLAMA 3.2 (2 GB) che funziona perfettamente anche su computer di medie dimensioni.
Inoltre, è necessario installare il software Tesseract; per il riconoscimento OCR, consigliamo di scegliere italiano e inglese durante l'installazione.
MyDigitalTwin ti permette di:
1) Chattare con il modello senza RAG.
2) Chattare utilizzando la casella di testo o il microfono.
3) Indicizzare una cartella di documenti di vario tipo per il RAG.
4) Interrogare il sistema, che trascriverà l'audio e il video nei documenti, eseguirà l'OCR sulle immagini e descriverà anche 10 fotogrammi equamente distribuiti nel video.
5) Devi usare un video di esempio (sample_face.mp4) e un audio di esempio della tua voce (sample_voice.wav).
6) Il sistema necessita di una connessione Internet solo all'avvio per scaricare i modelli da HuggingFace, ecc. Dopodiché puoi anche scollegare il computer.
7) Se nel sistema RAG sono presenti molti documenti su di te, puoi ottenere il tuo gemello digitale.

## ISTRUZIONI PER SISTEMI WINDOWS

1) Esegui il file install.bat (installa Tesseract, Ollama, il modello LLama3.2 e FFMpeg)
2) ## Scarica il checkpoint wav2lip_gan.pth come indicato nel file nella cartella weights.
3) Nel framework segui le istruzioni (ad esempio, scarica un modello).
4) L'area di lavoro si trova nella cartella documents, dove andrai a mettere i tuoi documenti da indicizzare nel RAG.
5) Scegli un embedder (di default è bert-base-italian-uncased per l'italiano).
6) Aggiorna l'indice.
7) CHAT
8) Il programma scarica i file in C:\Users\YOUR_USER_NAME\.cache\huggingface\hub: models--dbmdz--bert-base-italian-uncased, models--Salesforce--blip-image-captioning-base, whisper, coqui-tts.

## Come eseguirlo:
1) creare la cartella mydigitaltwin
2) copiare tutto il contenuto di questo repository.
3) creare un ambiente Python: python -m venv nbmultirag
4) attivare l'ambiente (per Windows: mydigitaltwin\Scripts\activate)
5) pip install -r requirements.txt
6) python life3.py
7) indicizzare una cartella di documenti di vario tipo per il RAG.
8) interrogare il sistema, che trascriverà l'audio e il video nei documenti, eseguirà l'OCR sulle immagini e descriverà anche 10 fotogrammi equamente distribuiti nel video. 9) Il sistema necessita di una connessione Internet solo all'avvio per scaricare i modelli da HuggingFace, dopodiché è possibile anche scollegare il computer.

  
<a name="ENGLISH"></a>
# mydigitaltwin  
My digital twin - it is a program can make your digital twin using a video and audio sample of you, then you can interact by microphone or text chatting.  
You will see yourself speaking with you, with your voice and your face in lip sync.  
This is clearly a ## PROTOTYPE ## that could work better and be expanded, so everyone is welcome to contribute!  

It uses an AI model and a RAG system for custom answers, you could index your life, your thoughts, so you can live forever in a digital life mode.  
It runs local on your computer, no API, no external servers.  
  
### ---------------- Your data remains on your computer ---------------  

This framework allows you to chat with your documents in RAG, including multimedia (audio, video, images and OCR).
The framework is a GUI to chat with a GPT model downloaded from OLLAMA, we recommend LLAMA 3.2 (2Gb) which performs perfectly even on medium-sized machines.
In addition, you need to install the Tesseract software, for OCR recognition, we recommend choosing Italian and English during installation.

---------------- Your data remains on your computer ---------------  

It is an Italian, which allows you to chat with your documents in RAG, including multimedia (audio, video, images and OCR).
The framework is a GUI to chat with a GPT model downloaded from OLLAMA, we recommend LLAMA 3.2 (2Gb) which performs perfectly even on medium-sized machines.
In addition, you need to install the Tesseract software, for OCR recognition, we recommend choosing Italian and English during installation.
MyDigitalTwin, allows you to:
1) Chat with the model without RAG.
2) Chat using textbox or microphone.
3) Index a folder of documents of various types for the RAG.
4) Query the system, which will transcribe the audio and video in the documents, perform OCR on the images and also describe 10 frames equally distributed in the video.
5) You need to use a sample video of you (sample_face.mp4) and a sample audio of your voice (sample_voice.wav).
6) The system only needs an Internet connection at launch to download the models from HuggingFace, etc.. Then you can also disconnect the computer.
7) If there are many documents about you in the RAG system, you can have your digital twin.

## INSTRUCTIONS FOR WINDOWS SYSTEMS

1) Run the install.bat file (it installs Tesseract, Ollama, LLama3.2 model and FFMpeg)
2) In the framework follow the prompts (e.g. download a template).
3) ## Download the checkpoint wav2lip_gan.pth following the link into the folder named "weights".
4) The workspace is in the "documents" folder, where you put your documents for the RAG indexing.
5) Choose an embedder (by default there is bert-base-italian-uncased for Italian.
6) Update the index.
7) CHAT
8) The program downloads the files to C:\Users\YOUR_USER_NAME\.cache\huggingface\hub: models--dbmdz--bert-base-italian-uncased, models--Salesforce--blip-image-captioning-base, whisper, coqui-tts.

## How to run:
1) create the mydigitaltwin folder
2) copy all the contents of this repository.
3) create a Python environment: python -m venv nbmultirag
4) Activate the environment (for Windows: mydigitaltwin\Scripts\activate)
5) pip install -r requirements.txt
6) python life3.py 
7) Index a folder of documents of various types for the RAG.
8) Query the system, which will transcribe the audio and video in the documents, perform OCR on the images and also describe 10 frames equally distributed in the video.
9) The system only needs an Internet connection at launch to download the models from HuggingFace, then you can also disconnect the computer.
 
# ** How to run **  
python life3.py  
 
