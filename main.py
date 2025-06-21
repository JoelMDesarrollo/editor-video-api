import os
import re
import uuid
import shutil
import textwrap
import requests
import asyncio
from typing import List, Optional
import logging
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, status, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

from moviepy.editor import VideoFileClip, concatenate_videoclips, CompositeVideoClip, TextClip
from PIL import Image
import io

# ---------------------------------------------------
# CONFIGURACIÓN DE LOGGING
# ---------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------
# CARGA DE VARIABLES DE ENTORNO
# ---------------------------------------------------
load_dotenv()

# Configuración de MoviePy
from moviepy.config import change_settings
change_settings({"IMAGEMAGICK_BINARY": r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe"})

# Configuración de la API
API_KEY = os.getenv("API_KEY", "your-default-api-key")
if not API_KEY:
    raise RuntimeError("Falta definir API_KEY en .env")

# Configuración para IA local
AI_PROVIDER = os.getenv("AI_PROVIDER", "ollama")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

# ---------------------------------------------------
# DIRECTORIOS
# ---------------------------------------------------
UPLOAD_DIR = "uploads"
RESULT_DIR = "results"
THUMBNAIL_DIR = "thumbnails"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(THUMBNAIL_DIR, exist_ok=True)

# ---------------------------------------------------
# MODELOS PYDANTIC
# ---------------------------------------------------
class MergeRequest(BaseModel):
    file_paths: List[str]
    product_name: str
    combinations: int = 1  # Número de combinaciones diferentes a generar

class UploadResponse(BaseModel):
    message: str
    files: List[str]
    count: int

class VideoFile(BaseModel):
    filename: str
    path: str
    size: int
    created_at: str
class Combination(BaseModel):
    video_url: str
    thumbnail: str

class ProjectResponse(BaseModel):
    id: str
    name: str
    product: str
    status: str
    views: str
    engagement: str
    videos: int
    thumbnail: str
    created_at: str
    last_updated: str
    merged_filename: str
    video_url: str
    combinations: List[Combination]
# ---------------------------------------------------
# CONFIGURACIÓN FASTAPI
# ---------------------------------------------------
app = FastAPI(
    title="Video Merger API",
    description="API para unir videos con subtítulos generados por IA",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"]
)

# Montar directorios estáticos
app.mount("/static/videos", StaticFiles(directory=RESULT_DIR), name="videos")
app.mount("/static/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/static/thumbnails", StaticFiles(directory=THUMBNAIL_DIR), name="thumbnails")

# ---------------------------------------------------
# SEGURIDAD
# ---------------------------------------------------
X_API_KEY = APIKeyHeader(name="X-API-Key", auto_error=True)

def api_key_auth(x_api_key: str = Depends(X_API_KEY)):
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key inválida o faltante"
        )

# ---------------------------------------------------
# FUNCIONES AUXILIARES
# ---------------------------------------------------
def generate_fallback_hooks(product_name: str, num_hooks: int = 3) -> List[str]:
    templates = [
        f"¿Conoces otros beneficios de {product_name} aparte de su 20% de descuento esta semana?",
        f"5 beneficios que {product_name} añade a tu vida diaria al instante",
        f"{product_name} que transforma tu espacio con energía positiva",
        f"¿Sabías que {product_name} puede mejorar tu bienestar en un 30%?",
        f"Oferta exclusiva: {product_name} premium con 15% de descuento hoy",
        f"La forma inteligente de usar {product_name} para resultados visibles",
        f"{product_name} de calidad premium - ¿Por qué esperar para probarlo?"
    ]
    import random
    return random.sample(templates, min(num_hooks, len(templates)))

async def generate_product_hooks(product_name: str, num_hooks: int = 3) -> List[str]:
    if AI_PROVIDER == "ollama":
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": f"""Genera {num_hooks} frases promocionales descriptivas para '{product_name}' que:
1. Resalten beneficios específicos del producto
2. Incluyan porcentajes de descuento cuando sea apropiado
3. Usen lenguaje persuasivo
4. Sean preguntas o afirmaciones impactantes
5. Tengan entre 10-15 palabras máximo

Ejemplos:
- ¿Conoces otros beneficios del polen de abeja aparte de su 20% de descuento esta semana?
- 5 láminas que añaden 10 puntos de aura a tu habitación al instante.
- Arte que hace que tu pared se sienta como la energía de un protagonista.

Formato: frase1|frase2|frase3""",
                    "stream": False,
                    "options": {"temperature": 0.7}
                },
                timeout=30
            )
            if response.status_code == 200:
                content = response.json().get("response", "").strip()
                hooks = [hook.strip() for hook in content.split("|") if hook.strip()]
                return hooks[:num_hooks] if len(hooks) >= num_hooks else generate_fallback_hooks(product_name, num_hooks)
        except Exception as e:
            logger.error(f"Error con Ollama: {str(e)}")
    
    return generate_fallback_hooks(product_name, num_hooks)

def create_subtitle_clip(text: str, duration: float, video_size: tuple) -> TextClip:
    try:
        # No envolvemos el texto para mantener la frase completa
        txt_clip = TextClip(
            text,
            fontsize=36,  # Un poco más pequeño para frases más largas
            color='white',
            font='Arial-Bold',
            stroke_color='black',
            stroke_width=2,
            method='caption',
            align='center',
            size=(video_size[0] * 0.9, None)  # Usamos 90% del ancho
        )
        
        # Posicionamos en el 25% desde arriba (75% superior)
        txt_clip = txt_clip.set_position(('center', 0.25), relative=True)
        txt_clip = txt_clip.set_duration(duration)
        
        # Añadimos margen inferior para mejor legibilidad
        txt_clip = txt_clip.margin(bottom=20, opacity=0)
        
        return txt_clip
    except Exception as e:
        logger.error(f"Error creating subtitle: {str(e)}")
        raise

def generate_thumbnail(video_path: str, output_path: str):
    try:
        clip = VideoFileClip(video_path)
        frame = clip.get_frame(clip.duration / 2)
        img = Image.fromarray(frame)
        img.save(output_path)
        clip.close()
        return True
    except Exception as e:
        logger.error(f"Error generando thumbnail: {str(e)}")
        return False

def generate_unique_filename(product_name: str) -> str:
    clean_name = re.sub(r'[\\/*?:"<>|]', "", product_name).replace(" ", "_")[:50]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{clean_name}_{timestamp}.mp4"

def get_video_metadata(file_path: str) -> dict:
    try:
        clip = VideoFileClip(file_path)
        duration = clip.duration
        width, height = clip.size
        clip.close()
        return {
            "duration": duration,
            "resolution": f"{width}x{height}",
            "size": os.path.getsize(file_path)
        }
    except Exception as e:
        logger.error(f"Error obteniendo metadata: {str(e)}")
        return {}

# ---------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------
@app.post("/upload/", response_model=UploadResponse, dependencies=[Depends(api_key_auth)])
async def upload_videos(files: List[UploadFile] = File(...)):
    saved_files = []
    
    for file in files:
        if not file.content_type or not file.content_type.startswith("video/"):
            raise HTTPException(400, f"{file.filename} no es un video válido")
        
        ext = os.path.splitext(file.filename)[1]
        unique_name = f"{uuid.uuid4()}{ext}"
        file_path = os.path.join(UPLOAD_DIR, unique_name)
        
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(file_path)
        except Exception as e:
            logger.error(f"Error guardando {file.filename}: {str(e)}")
            raise HTTPException(500, f"Error guardando {file.filename}")
    
    return {
        "message": "Videos subidos exitosamente",
        "files": saved_files,
        "count": len(saved_files)
    }

@app.post("/merge/", response_model=List[ProjectResponse], dependencies=[Depends(api_key_auth)])
async def merge_videos(req: MergeRequest):
    if not 1 <= req.combinations <= 10:
        raise HTTPException(400, "El número de combinaciones debe estar entre 1 y 10")
    
    normalized_paths = []
    for path in req.file_paths:
        normalized_path = os.path.abspath(path.replace("\\", "/"))
        if not os.path.exists(normalized_path):
            raise HTTPException(404, f"Archivo no encontrado: {path}")
        normalized_paths.append(normalized_path)
    
    hooks = await generate_product_hooks(req.product_name, len(normalized_paths))
    results = []
    project_id = str(uuid.uuid4())
    base_filename = generate_unique_filename(req.product_name).replace(".mp4", "")
    
    combinations = []
    
    for combo in range(req.combinations):
        try:
            # Mezclar los videos para crear combinaciones diferentes
            import random
            if combo > 0:
                random.shuffle(normalized_paths)
            
            video_clips = []
            for i, path in enumerate(normalized_paths):
                clip = VideoFileClip(path)
                hook_index = i % len(hooks)
                subtitle_clip = create_subtitle_clip(hooks[hook_index], clip.duration, clip.size)
                composite_clip = CompositeVideoClip([clip, subtitle_clip])
                video_clips.append(composite_clip)
            
            final_clip = concatenate_videoclips(video_clips) if len(video_clips) > 1 else video_clips[0]
            
            output_filename = f"{base_filename}_{combo+1}.mp4"
            output_path = os.path.join(RESULT_DIR, output_filename)
            
            final_clip.write_videofile(
                output_path,
                codec="libx264",
                audio_codec="aac",
                threads=4,
                preset="medium",
                ffmpeg_params=["-crf", "20"]
            )
            
            # Generar thumbnail
            thumbnail_filename = f"{base_filename}_{combo+1}.jpg"
            thumbnail_path = os.path.join(THUMBNAIL_DIR, thumbnail_filename)
            generate_thumbnail(output_path, thumbnail_path)
            
            combinations.append({
                "video_url": f"/static/videos/{output_filename}",
                "thumbnail": f"/static/thumbnails/{thumbnail_filename}"
            })
            
        except Exception as e:
            logger.error(f"Error en combinación {combo+1}: {str(e)}")
            continue
        finally:
            for clip in video_clips:
                clip.close()
    
    if not combinations:
        raise HTTPException(500, "Error al generar todas las combinaciones")
    
    # Creamos un solo proyecto con todas las combinaciones
    main_filename = f"{base_filename}_1.mp4"
    main_thumbnail = f"{base_filename}_1.jpg"
    
    results.append({
        "id": project_id,
        "name": f"Video de {req.product_name}",
        "product": req.product_name,
        "status": "Published",
        "views": "0",
        "engagement": "0%",
        "videos": len(normalized_paths),
        "thumbnail": f"/static/thumbnails/{main_thumbnail}",
        "created_at": datetime.now().isoformat(),
        "last_updated": "Recently",
        "merged_filename": main_filename,
        "video_url": f"/static/videos/{main_filename}",
        "combinations": combinations
    })
    
    return results

@app.get("/projects/", response_model=List[ProjectResponse], dependencies=[Depends(api_key_auth)])
async def list_projects():
    projects = {}
    
    # Primero agrupamos todos los archivos por proyecto base
    for filename in os.listdir(RESULT_DIR):
        if filename.endswith(".mp4"):
            try:
                # Extraemos el nombre base (sin _X.mp4)
                base_name = re.sub(r'_\d+\.mp4$', '', filename)
                project_key = base_name.split('_')[:-2]  # Eliminamos fecha/hora
                project_key = '_'.join(project_key)
                
                if project_key not in projects:
                    projects[project_key] = {
                        "files": [],
                        "created_at": None,
                        "product_name": ' '.join(base_name.split('_')[:-2])
                    }
                
                file_path = os.path.join(RESULT_DIR, filename)
                created_at = datetime.fromtimestamp(os.path.getctime(file_path))
                
                projects[project_key]["files"].append({
                    "filename": filename,
                    "created_at": created_at
                })
                
                if not projects[project_key]["created_at"] or created_at > projects[project_key]["created_at"]:
                    projects[project_key]["created_at"] = created_at
                    
            except Exception as e:
                logger.error(f"Error procesando {filename}: {str(e)}")
    
    # Ahora construimos la respuesta
    response = []
    for project_key, data in projects.items():
        # Ordenamos los archivos por fecha de creación
        sorted_files = sorted(data["files"], key=lambda x: x["created_at"])
        
        combinations = []
        for i, file in enumerate(sorted_files):
            base_name = file["filename"].replace(".mp4", "")
            combinations.append({
                "video_url": f"/static/videos/{file['filename']}",
                "thumbnail": f"/static/thumbnails/{base_name}.jpg"
            })
        
        # Usamos el primer archivo como principal
        main_file = sorted_files[0]
        base_name = main_file["filename"].replace(".mp4", "")
        
        response.append({
            "id": str(uuid.uuid4()),
            "name": f"Video de {data['product_name']}",
            "product": data["product_name"],
            "status": "Published",
            "views": "0",
            "engagement": "0%",
            "videos": len(combinations),  # Número de combinaciones
            "thumbnail": f"/static/thumbnails/{base_name}.jpg",
            "created_at": data["created_at"].isoformat(),
            "last_updated": "Recently",
            "merged_filename": main_file["filename"],
            "video_url": f"/static/videos/{main_file['filename']}",
            "combinations": combinations
        })
    
    return sorted(response, key=lambda x: x["created_at"], reverse=True)

@app.get("/uploads/", response_model=List[VideoFile], dependencies=[Depends(api_key_auth)])
async def list_uploads():
    uploads = []
    
    for filename in os.listdir(UPLOAD_DIR):
        file_path = os.path.join(UPLOAD_DIR, filename)
        if os.path.isfile(file_path):
            created_at = datetime.fromtimestamp(os.path.getctime(file_path))
            uploads.append({
                "filename": filename,
                "path": f"/static/uploads/{filename}",
                "size": os.path.getsize(file_path),
                "created_at": created_at.isoformat()
            })
    
    return sorted(uploads, key=lambda x: x["created_at"], reverse=True)

@app.get("/download/{filename}", dependencies=[Depends(api_key_auth)])
async def download_video(filename: str):
    if not re.match(r"^[a-zA-Z0-9_\-]+\.mp4$", filename):
        raise HTTPException(400, "Nombre de archivo inválido")
    
    file_path = os.path.join(RESULT_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(404, "Archivo no encontrado")
    
    return FileResponse(
        file_path,
        media_type="video/mp4",
        filename=filename,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@app.get("/health/")
async def health_check():
    return {
        "status": "healthy",
        "api_version": "3.0.0",
        "storage": {
            "uploads": len(os.listdir(UPLOAD_DIR)),
            "results": len(os.listdir(RESULT_DIR))
        }
    }

# Generar thumbnails al iniciar
@app.on_event("startup")
async def startup_event():
    logger.info("Generando thumbnails para videos existentes...")
    for filename in os.listdir(RESULT_DIR):
        if filename.endswith(".mp4"):
            thumbnail_path = os.path.join(THUMBNAIL_DIR, f"{os.path.splitext(filename)[0]}.jpg")
            if not os.path.exists(thumbnail_path):
                generate_thumbnail(
                    os.path.join(RESULT_DIR, filename),
                    thumbnail_path
                )

# ---------------------------------------------------
# RUN
# ---------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")