import os
import sys
import re
import uuid
import shutil
import requests
import logging
import random

from datetime import datetime
from pathlib import Path
from typing import List, Dict
from typing import Optional

from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, status, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

from moviepy.editor import VideoFileClip, concatenate_videoclips, CompositeVideoClip, TextClip
from PIL import Image
from moviepy.config import change_settings

# ---------------------------------------------------
# CONFIGURACIÓN INICIAL
# ---------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Configuración de MoviePy
# Cambia esta parte de la configuración de MoviePy
change_settings({
    "IMAGEMAGICK_BINARY": "/usr/bin/convert",
    "FFMPEG_BINARY": "/usr/bin/ffmpeg"
})
try:
    from moviepy.editor import VideoFileClip, concatenate_videoclips, CompositeVideoClip
    from PIL import Image
    import numpy as np
except ImportError as e:
    print(f"Error de importación: {e}")
    sys.exit(1)

# Configuración de la API
API_KEY = os.getenv("API_KEY", "12345678")
if not API_KEY:
    raise RuntimeError("API_KEY no definida en .env")

# Configuración para OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY no definida en .env")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

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
    combinations: int = 1

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

class ProjectStatus(BaseModel):
    project_id: str
    status: str
    progress: int
    message: str
    error: Optional[str] = None
    result: Optional[dict] = None

# ---------------------------------------------------
# ALMACENAMIENTO DE ESTADO
# ---------------------------------------------------
project_states = {}

# ---------------------------------------------------
# CONFIGURACIÓN FASTAPI
# ---------------------------------------------------
app = FastAPI(
    title="Video Merger API",
    description="API para unir videos con subtítulos generados por IA",
    version="3.0.0",
    redirect_slashes=False,
    max_upload_size=100 * 1024 * 1024  # 100MB
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
def update_project_state(project_id: str, status: str, progress: int, message: str, error: str = None, result: dict = None):
    if project_id not in project_states:
        project_states[project_id] = {
            "status": status,
            "progress": progress,
            "message": message,
            "error": error,
            "result": result,
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }
    else:
        project_states[project_id].update({
            "status": status,
            "progress": progress,
            "message": message,
            "error": error,
            "result": result,
            "updated_at": datetime.now()
        })

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
    return random.sample(templates, min(num_hooks, len(templates)))

async def generate_product_hooks(product_name: str, num_hooks: int = 3) -> List[str]:
    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        prompt = f"""
        Genera {num_hooks} frases promocionales creativas para el producto '{product_name}'.
        Cada frase debe ser atractiva, concisa (máximo 10 palabras) y destacar beneficios únicos.
        Devuelve las frases separadas por el carácter '|'.
        
        Ejemplo:
        Descubre {product_name} y cambia tu vida|{product_name} - calidad que se nota|Oferta especial: {product_name} con 30% de descuento
        """
        
        data = {
            "model": OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": "Eres un experto en marketing digital que genera frases promocionales efectivas."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 150
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"].strip()
            hooks = [hook.strip() for hook in content.split("|") if hook.strip()]
            return hooks[:num_hooks] if len(hooks) >= num_hooks else generate_fallback_hooks(product_name, num_hooks)
        
        logger.error(f"Error con OpenAI: {response.status_code} - {response.text}")
        return generate_fallback_hooks(product_name, num_hooks)
        
    except Exception as e:
        logger.error(f"Error con OpenAI: {str(e)}")
        return generate_fallback_hooks(product_name, num_hooks)

def create_subtitle_clip(text: str, duration: float, video_size: tuple) -> TextClip:
    try:
        txt_clip = TextClip(
            text,
            fontsize=36,
            color='white',
            font='Arial-Bold',
            stroke_color='black',
            stroke_width=2,
            method='caption',
            align='center',
            size=(video_size[0] * 0.9, None))
        
        txt_clip = txt_clip.set_position(('center', 0.25), relative=True)
        txt_clip = txt_clip.set_duration(duration)
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

# ---------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------
@app.post("/upload", response_model=UploadResponse, dependencies=[Depends(api_key_auth)])
async def upload_videos(files: List[UploadFile] = File(...)):
    saved_files = []
    
    for file in files:
        if not file.content_type.startswith("video/"):
            raise HTTPException(400, f"{file.filename} no es un video válido")
        
        # Guardar temporalmente para validar
        temp_path = f"temp_{file.filename}"
        try:
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Validar integridad del video
            try:
                clip = VideoFileClip(temp_path)
                clip.close()
            except Exception as e:
                raise HTTPException(400, f"Video corrupto: {file.filename}")
            
            # Mover al directorio final
            unique_name = f"{uuid.uuid4()}.mp4"
            final_path = os.path.join(UPLOAD_DIR, unique_name)
            shutil.move(temp_path, final_path)
            saved_files.append(unique_name)
            
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise HTTPException(500, f"Error guardando {file.filename}")
    
    return {
        "message": "Videos subidos exitosamente",
        "files": saved_files,
        "count": len(saved_files)
    }

@app.post("/merge", response_model=ProjectStatus, dependencies=[Depends(api_key_auth)])
async def merge_videos(req: MergeRequest, background_tasks: BackgroundTasks):
    """Endpoint para iniciar el proceso de mezcla de videos"""
    if len(req.file_paths) < 2:
        raise HTTPException(400, "Se necesitan al menos 2 videos para combinaciones")
    
    project_id = str(uuid.uuid4())
    
    # Iniciar estado del proyecto
    update_project_state(project_id, "queued", 0, "Proyecto en cola para procesamiento")
    
    # Procesar en segundo plano
    background_tasks.add_task(
        process_all_videos_combinations, 
        req.file_paths, 
        req.product_name, 
        req.combinations,
        project_id
    )
    
    return {
        "project_id": project_id,
        "status": "queued",
        "progress": 0,
        "message": "Proyecto creado y en cola para procesamiento",
        "error": None,
        "result": None
    }

async def process_all_videos_combinations(file_paths: List[str], product_name: str, combinations: int, project_id: str):
    """Procesa combinaciones que incluyen todos los videos en diferentes órdenes"""
    try:
        update_project_state(project_id, "validating", 5, "Validando archivos...")

        # Normaliza rutas y verifica archivos
        normalized_paths = []
        for path in file_paths:
            if path.startswith('uploads/'):
                path = path[8:]
            full_path = os.path.join(UPLOAD_DIR, path)
            if not os.path.exists(full_path):
                logger.error(f"Archivo no encontrado: {full_path}")
                continue
            normalized_paths.append(full_path)

        if len(normalized_paths) < 2:
            raise Exception("Se necesitan al menos 2 videos válidos para combinaciones")

        # Genera hooks promocionales (uno para cada video)
        hooks = await generate_product_hooks(product_name, len(normalized_paths))
        base_filename = generate_unique_filename(product_name).replace(".mp4", "")
        results = []
        used_orders = []

        # Generar órdenes únicos para las combinaciones
        for combo_num in range(combinations):
            try:
                progress = 20 + (combo_num * 70 // combinations)
                
                # Generar un orden único que no se haya usado antes
                while True:
                    if combo_num == 0:
                        # Primera combinación siempre en orden original
                        video_order = list(range(len(normalized_paths)))
                        break
                    else:
                        video_order = random.sample(range(len(normalized_paths)), len(normalized_paths))
                        if video_order not in used_orders:
                            break
                
                used_orders.append(video_order)
                update_msg = f"Procesando combinación {combo_num+1}/{combinations} (orden: {[x+1 for x in video_order]})"
                update_project_state(
                    project_id,
                    "processing",
                    progress,
                    update_msg
                )

                # Procesar todos los videos en el orden generado
                video_clips = []
                for i, video_idx in enumerate(video_order):
                    clip = VideoFileClip(normalized_paths[video_idx])
                    subtitle = create_subtitle_clip(
                        hooks[i % len(hooks)],
                        clip.duration,
                        clip.size
                    )
                    composite = CompositeVideoClip([clip, subtitle])
                    video_clips.append(composite)
                
                # Concatenar todos los videos
                final_clip = concatenate_videoclips(video_clips)
                
                # Nombre del archivo
                output_filename = f"{base_filename}_combo{combo_num+1}.mp4"
                output_path = os.path.join(RESULT_DIR, output_filename)
                
                final_clip.write_videofile(
                    output_path,
                    codec="libx264",
                    audio_codec="aac",
                    threads=4,
                    preset="slow",
                    ffmpeg_params=["-crf", "18"],
                    verbose=False
                )

                # Generar thumbnail
                thumbnail_filename = f"{base_filename}_combo{combo_num+1}.jpg"
                thumbnail_path = os.path.join(THUMBNAIL_DIR, thumbnail_filename)
                generate_thumbnail(output_path, thumbnail_path)
                
                results.append({
                    "filename": output_filename,
                    "thumbnail": thumbnail_filename,
                    "order": [x+1 for x in video_order]  # Para referencia
                })

                # Liberar recursos
                for clip in video_clips:
                    clip.close()
                final_clip.close()

            except Exception as e:
                logger.error(f"Error en combinación {combo_num+1}: {str(e)}")
                continue

        update_project_state(
            project_id,
            "completed",
            100,
            f"Generadas {len(results)} combinaciones con todos los videos",
            result={
                "output_files": [item["filename"] for item in results],
                "thumbnails": [item["thumbnail"] for item in results],
                "details": results
            }
        )

    except Exception as e:
        logger.error(f"Error crítico: {str(e)}")
        update_project_state(
            project_id,
            "error",
            0,
            "Falló la generación de combinaciones",
            error=str(e)
        )

@app.get("/project/{project_id}/status", response_model=ProjectStatus)
async def get_project_status(project_id: str):
    if project_id not in project_states:
        raise HTTPException(404, "Proyecto no encontrado")
    
    state = project_states[project_id]
    return {
        "project_id": project_id,
        "status": state["status"],
        "progress": state["progress"],
        "message": state["message"],
        "error": state.get("error"),
        "result": state.get("result")
    }

@app.get("/projects", response_model=List[ProjectResponse], dependencies=[Depends(api_key_auth)])
async def list_projects():
    projects = {}
    
    for filename in os.listdir(RESULT_DIR):
        if filename.endswith(".mp4"):
            try:
                base_name = re.sub(r'_\d+\.mp4$', '', filename)
                project_key = base_name.split('_')[:-2]
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
    
    response = []
    for project_key, data in projects.items():
        sorted_files = sorted(data["files"], key=lambda x: x["created_at"])
        
        combinations = []
        for i, file in enumerate(sorted_files):
            base_name = file["filename"].replace(".mp4", "")
            combinations.append({
                "video_url": f"/static/videos/{file['filename']}",
                "thumbnail": f"/static/thumbnails/{base_name}.jpg"
            })
        
        main_file = sorted_files[0]
        base_name = main_file["filename"].replace(".mp4", "")
        
        response.append({
            "id": str(uuid.uuid4()),
            "name": f"Video de {data['product_name']}",
            "product": data["product_name"],
            "status": "Published",
            "views": "0",
            "engagement": "0%",
            "videos": len(combinations),
            "thumbnail": f"/static/thumbnails/{base_name}.jpg",
            "created_at": data["created_at"].isoformat(),
            "last_updated": "Recently",
            "merged_filename": main_file["filename"],
            "video_url": f"/static/videos/{main_file['filename']}",
            "combinations": combinations
        })
    
    return sorted(response, key=lambda x: x["created_at"], reverse=True)

@app.get("/uploads", response_model=List[VideoFile], dependencies=[Depends(api_key_auth)])
async def list_uploads():
    uploads = []
    
    for filename in os.listdir(UPLOAD_DIR):
        file_path = os.path.join(UPLOAD_DIR, filename)
        if os.path.isfile(file_path):
            stat = os.stat(file_path)
            created_at = datetime.fromtimestamp(stat.st_ctime)
            uploads.append({
                "filename": filename,
                "path": f"/static/uploads/{filename}",
                "size": stat.st_size,
                "created_at": created_at.isoformat()
            })
    
    return sorted(uploads, key=lambda x: x["created_at"], reverse=True)

@app.delete("/uploads/{filename}", dependencies=[Depends(api_key_auth)])
async def delete_uploaded_video(filename: str):
    if not re.match(r"^[a-f0-9]{8}-([a-f0-9]{4}-){3}[a-f0-9]{12}\.\w+$", filename):
        raise HTTPException(status_code=400, detail="Nombre de archivo inválido")
    
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Video no encontrado")
    
    try:
        os.remove(file_path)
        return {"message": f"Video {filename} eliminado correctamente"}
    except Exception as e:
        logger.error(f"Error eliminando {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error eliminando el video: {str(e)}")

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

@app.get("/health")
async def health_check():
    return {
        "status": "ok", 
        "active_projects": len(project_states)
    }

@app.get("/media/videos/{filename}")
async def serve_video(filename: str):
    video_path = Path(f"results/{filename}")
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    
    return FileResponse(
        video_path,
        media_type="video/mp4",
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-store",
            "Content-Disposition": f'inline; filename="{filename}"'
        }
    )

@app.get("/media/uploads/{filename}")
async def serve_upload(filename: str):
    video_path = Path(f"uploads/{filename}")
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    
    return FileResponse(
        video_path,
        media_type="video/mp4",
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-store",
            "Content-Disposition": f'inline; filename="{filename}"'
        }
    )

@app.get("/media/thumbnails/{filename}")
async def serve_thumbnail(filename: str):
    thumbnail_path = Path(f"thumbnails/{filename}")
    if not thumbnail_path.exists():
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    
    return FileResponse(
        thumbnail_path,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store",
            "Content-Disposition": f'inline; filename="{filename}"'
        }
    )

@app.on_event("startup")
async def startup_event():
    logger.info("Iniciando servidor de Video Merger API...")
    logger.info("Generando thumbnails para videos existentes...")
    for filename in os.listdir(RESULT_DIR):
        if filename.endswith(".mp4"):
            thumbnail_path = os.path.join(THUMBNAIL_DIR, f"{os.path.splitext(filename)[0]}.jpg")
            if not os.path.exists(thumbnail_path):
                generate_thumbnail(
                    os.path.join(RESULT_DIR, filename),
                    thumbnail_path
                )
    logger.info("Servidor iniciado correctamente")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")