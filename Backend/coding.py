from fastapi import FastAPI, HTTPException,UploadFile, File
import requests
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import base64
from diffusers import StableDiffusionPipeline
import torch
import base64
from io import BytesIO
from PIL import Image
from speach_to_text import convertion
import os
from fastapi.responses import JSONResponse
import PyPDF2
from RAG.llama_index_rag import LlamaIndexRAGProcessor
from RAG.Calcul_similarite import hybrid_retrieval
from text_utils import remove_quotes


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
SD_API_URL = "http://127.0.0.1:7860/sdapi/v1/txt2img"



@app.post("/ask")
def ask_ollama(message: str):
    
     chunks, embeddings =processor.get_chunks_and_embeddings()
     context=processor.get_context_conversation()
     context_text = "\n".join(
        [f"Utilisateur : {c['question']}\nAssistant : {c['answer']}" for c in context]
    )

     if not chunks:
            Templet=f"tu une expert en QA tu repondait au question de utilisateur : {message} en recpectant la language de le utilisateur et voici le context de conversation : {context_text}  "
            payload = {
                "model": "llama3.2:latest",
                "prompt": Templet,
                "stream": False  # JSON complet
            }
            try:
                r = requests.post(OLLAMA_URL, json=payload, timeout=180)
                r.raise_for_status()
            except requests.RequestException as e:
                raise HTTPException(status_code=500, detail=f"Erreur en appelant Ollama: {e}")

            data = r.json()
            processor.enregistrer_OA_in_Base_donnes(message,data.get("response"))
            text_net=remove_quotes(data.get("response"))
            return {"answer": text_net}
     else : 
         chunks=hybrid_retrieval(message)
         Templet=f"""
Tu es un assistant intelligent spécialisé dans la recherche documentaire (QA RAG System).
Ta tâche est de répondre **uniquement** en te basant sur le contenu suivant extrait de documents PDF.
Tu ne dois pas inventer d'informations, ni utiliser des connaissances externes.

---------------- CONTEXTE (documents) ----------------
{chunks}
---------------- derniere message de utilisateur  ----------------
{context_text}
-----------------------------------------------------

Question de l'utilisateur :
"{message}"

Règles strictes :
1. Si la réponse n'existe pas explicitement dans le contexte, dis calmement :
   "Je ne trouve pas d'information à ce sujet dans les documents disponibles."
2. Réponds dans la même langue que la question de l'utilisateur.
3. Fournis une réponse **claire, structurée et complète**, tout en restant fidèle au contexte.
4. Ne mentionne pas que tu utilises un contexte ou des documents.
5. Ne fais pas d'hypothèses.

"""
         payload = {
                "model": "llama3.2:latest",
                "prompt": Templet,
                "stream": False  # JSON complet
            }
         try:
                r = requests.post(OLLAMA_URL, json=payload, timeout=180)
                r.raise_for_status()
         except requests.RequestException as e:
                raise HTTPException(status_code=500, detail=f"Erreur en appelant Ollama: {e}")

         data = r.json()
         processor.enregistrer_OA_in_Base_donnes(message,data.get("response"))
         text_net=remove_quotes(data.get("response"))
         return {"answer": text_net}




class ImgRequest(BaseModel):
    idea: str
    width: int = 512
    height: int = 512
    steps: int = 20
    samples: int = 1
    cfg_scale: float = 7.5

class ImageRequest(BaseModel):
    message: str



model_id = "stabilityai/sd-turbo"
device = "cpu"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype= torch.float32)
pipe = pipe.to(device)

# Schéma de la requête
class Prompt(BaseModel):
    text: str

@app.post("/generate")
def generate_image(prompt: Prompt):
    try:
        # Générer l’image
        image = pipe(prompt.text,
                     num_inference_steps=10,   # <= accélère énormément
                      guidance_scale=0.0 
                     ).images[0]

        # Convertir en base64 pour l’API
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {"image_base64": img_str}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/speech-to-text")
async def speech_to_text(file: UploadFile = File(...)):
    try:
        # Lire le contenu du fichier audio
        audio_bytes = await file.read()
        
        os.makedirs("enregistrer", exist_ok=True)

        # Sauvegarder temporairement
        with open("enregistrer/temp.wav", "wb") as f:
            f.write(audio_bytes)
        
        # Transcrire
        conv=convertion()
        result = conv.speech_to_text_bytes("enregistrer/temp.wav")
        print(result)
        os.remove("enregistrer/temp.wav")
        return {"text": result}
        
    except Exception as e:
        print("❌ Erreur dans speech_to_text :", e)
        raise HTTPException(status_code=500, detail=str(e))
    

processor = LlamaIndexRAGProcessor()
@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Vérifier le type de fichier
        if not file.filename.endswith(".pdf"):
            return JSONResponse(content={"success": False, "error": "Seuls les fichiers PDF sont supportés."})

        # Lire le contenu du PDF
        pdf_reader = PyPDF2.PdfReader(file.file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"

        # Chunker et créer embeddings
        chunks, embeddings = processor.process_document(text)

        # Sauvegarder en base
        processor.save_document_to_db(pdf_name=file.filename, pdf_url=file.filename, chunks=chunks, embeddings=embeddings)

        return {"success": True}
    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)})