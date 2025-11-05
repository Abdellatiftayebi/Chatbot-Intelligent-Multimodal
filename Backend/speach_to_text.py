import whisper

# Charger le modèle (tiny, base, small, medium, large)

model = whisper.load_model("base")

class convertion : 
    def speech_to_text_bytes(self,file_path:str):
        result = model.transcribe(file_path)
        return result["text"]