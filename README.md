[![Watch the video](https://img.youtube.com/vi/AqPgEY65E0E/maxresdefault.jpg)](https://youtu.be/AqPgEY65E0E)

# mydigitaltwin  
My digital twin - it is a program can make your digital twin using a video and audio sample of you, then you can interact by microphone or text chatting.  
You will see yourself speaking with you, with your voice and your face in lip sync.  

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
3) The workspace is in the documents folder.
4) Choose an embedder (by default there is bert-base-italian-uncased for Italian.
5) Update the index.
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

This is the notebook for trying it:  
https://www.kaggle.com/code/nannib/notebook193e2ba556  
