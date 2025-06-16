import os
import re
import uuid
import shutil
import textwrap
import requests  # Añadido
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.compositing.concatenate import concatenate_videoclips
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.video.VideoClip import TextClip
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# ---------------------------------------------------
# CONFIGURACIÓN INICIAL
# ---------------------------------------------------

# Cargar variables de entorno
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-85086d8f4aba478a9190eb489d395764")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"  # URL de la API

# ---------------------------------------------------
# CONFIGURACIÓN FASTAPI
# ---------------------------------------------------

try:
    from moviepy.config import change_settings
    change_settings({"IMAGEMAGICK_BINARY": r"C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\magick.exe"})
except Exception as e:
    print(f"Advertencia: Error configurando ImageMagick - {str(e)}")

app = FastAPI(
    title="Video Merger API con OpenAI",
    description="API para unir videos con subtítulos generados por IA",
    version="1.0.0"
)

# Configuración CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"]
)

# ---------------------------------------------------
# CONFIGURACIÓN DE DIRECTORIOS
# ---------------------------------------------------

UPLOAD_DIR = "uploads"
RESULT_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ---------------------------------------------------
# MODELOS PYDANTIC
# ---------------------------------------------------

class MergeRequest(BaseModel):
    file_paths: List[str]
    product_name: str

# ---------------------------------------------------
# FUNCIONES PRINCIPALES (Actualizadas para OpenAI v1.0+)
# ---------------------------------------------------

async def generate_product_hooks(product_name: str, num_hooks: int = 3) -> List[str]:
    """
    Genera hooks de marketing usando la API de DeepSeek directamente
    """
    try:
        prompt = (
            f"Genera exactamente {num_hooks} frases promocionales muy cortas (3-5 palabras) "
            f"para el producto '{product_name}'. Cada frase debe ser llamativa y directa, "
            "con emojis relevantes. Separa las frases con el caracter '|'. "
            "Ejemplo: '¡Oferta especial! 🔥|Solo hoy ⏳|No te lo pierdas 🚀'"
        )
        
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100,
            "temperature": 0.7
        }
        
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        hooks = [hook.strip() for hook in content.split("|") if hook.strip()]
        
        if not hooks or len(hooks) < num_hooks:
            hooks = [
                f"¡{product_name} increíble! 🚀",
                f"Oferta especial en {product_name} 🔥",
                f"Compra ahora {product_name} 💯"
            ]
            
        return hooks[:num_hooks]
    
    except Exception as e:
        print(f"Error generando hooks: {str(e)}")
        return [
            f"¡Descubre {product_name}! ✨",
            f"Lo mejor en {product_name.split()[0]} 🏆",
            f"Oferta limitada ⏳"
        ]

def create_subtitle_clip(
    text: str,
    duration: float,
    video_size: tuple,
    fontsize=42,
    color='white',
    stroke_color='black',
    stroke_width=2
) -> TextClip:
    """
    Crea un clip de subtítulo estilo TikTok
    """
    wrapped_text = textwrap.fill(text, width=20)
    
    return TextClip(
        txt=wrapped_text,
        fontsize=fontsize,
        color=color,
        font='Arial-Bold',
        stroke_color=stroke_color,
        stroke_width=stroke_width,
        method='caption',
        align='center',
        size=(video_size[0] * 0.9, None)
    ).set_position(('center', 0.15), relative=True).set_duration(duration)

# ---------------------------------------------------
# ENDPOINTS (Se mantienen igual)
# ---------------------------------------------------

@app.post("/upload/")
async def upload_videos(files: List[UploadFile] = File(...)):
    """Sube uno o múltiples archivos de video"""
    if not files:
        raise HTTPException(status_code=400, detail="Debes subir al menos un video")
    
    saved_files = []
    for file in files:
        if not file.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail=f"El archivo {file.filename} no es un video válido")
        
        file_ext = os.path.splitext(file.filename)[1]
        file_name = f"{uuid.uuid4()}{file_ext}"
        file_path = os.path.join(UPLOAD_DIR, file_name)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        saved_files.append(file_path)
    
    return {
        "message": "Videos subidos exitosamente",
        "files": saved_files,
        "count": len(saved_files)
    }

@app.post("/merge/")
async def merge_videos(request: MergeRequest):
    """Procesa videos con subtítulos generados por OpenAI"""
    normalized_paths = []
    for path in request.file_paths:
        abs_path = os.path.abspath(path.replace('\\', '/'))
        if not os.path.exists(abs_path):
            raise HTTPException(status_code=404, detail=f"Archivo no encontrado: {abs_path}")
        normalized_paths.append(abs_path)
    
    clips = []
    try:
        hooks = await generate_product_hooks(request.product_name, len(normalized_paths))
        clips = [VideoFileClip(path) for path in normalized_paths]
        output_name = f"merged_{uuid.uuid4()}.mp4"
        output_path = os.path.join(RESULT_DIR, output_name)
        
        if len(clips) == 1:
            clip = clips[0]
            subtitle = create_subtitle_clip(hooks[0], clip.duration, clip.size)
            final_clip = CompositeVideoClip([clip, subtitle])
        else:
            clips_with_subs = [
                CompositeVideoClip([clip, create_subtitle_clip(hooks[i % len(hooks)], clip.duration, clip.size)])
                for i, clip in enumerate(clips)
            ]
            final_clip = concatenate_videoclips(clips_with_subs)
        
        final_clip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            threads=4,
            preset='ultrafast',
            ffmpeg_params=['-crf', '22'],
            logger=None
        )
        final_clip.close()
        
        return {
            "message": "✅ Videos procesados exitosamente",
            "output_filename": output_name,
            "hooks_used": hooks,
            "video_duration": sum(c.duration for c in clips)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar videos: {str(e)}")
    finally:
        for clip in clips:
            try:
                if clip:
                    clip.close()
            except:
                pass

@app.get("/download/{filename}")
async def download_video(filename: str):
    try:
        if not re.match(r'^merged_[a-f0-9-]+\.mp4$', filename):
            raise HTTPException(status_code=400, detail="Nombre de archivo inválido")
        
        file_path = os.path.join(RESULT_DIR, filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Archivo no encontrado")
        
        return FileResponse(
            path=file_path,
            media_type="video/mp4",
            filename=filename,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al descargar: {str(e)}")

@app.get("/clean/")
async def clean_directories():
    try:
        deleted_files = 0
        for folder in [UPLOAD_DIR, RESULT_DIR]:
            if os.path.exists(folder):
                for filename in os.listdir(folder):
                    file_path = os.path.join(folder, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                            deleted_files += 1
                    except Exception as e:
                        print(f"Error eliminando {file_path}: {str(e)}")
        
        return {
            "message": "Directorios limpiados",
            "deleted_files": deleted_files
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al limpiar: {str(e)}")

# @app.get("/")
# async def root():
#     return {
#         "message": "🚀 Video Merger API con OpenAI funcionando",
#         "endpoints": {
#             "upload": "POST /upload/ - Subir videos",
#             "merge": "POST /merge/ - Procesar videos",
#             "download": "GET /download/{filename} - Descargar video",
#             "clean": "GET /clean/ - Limpiar temporales"
#         },
#         "openai_status": "activo" if OPENAI_API_KEY else "no configurado"
#     }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)