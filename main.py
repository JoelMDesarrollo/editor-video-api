import os
import re
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.compositing.concatenate import concatenate_videoclips
from typing import List
import uuid
import shutil
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Video Merger API", description="API para unir videos")

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"]
)

# Configuración de directorios
UPLOAD_DIR = "uploads"
RESULT_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# Modelo Pydantic para la solicitud de merge
class MergeRequest(BaseModel):
    file_paths: List[str]

@app.post("/upload/")
async def upload_videos(files: List[UploadFile] = File(...)):
    """Endpoint para subir videos (ahora acepta 1 o más videos)"""
    if len(files) == 0:
        raise HTTPException(
            status_code=400, 
            detail="Debes subir al menos un video",
            headers={"Access-Control-Allow-Origin": "*"}
        )
    
    saved_files = []
    for file in files:
        # Validar que sea un video
        if not file.content_type.startswith('video/'):
            raise HTTPException(
                status_code=400, 
                detail=f"El archivo {file.filename} no es un video",
                headers={"Access-Control-Allow-Origin": "*"}
            )
        
        # Generar nombre único para el archivo
        file_ext = os.path.splitext(file.filename)[1]
        file_name = f"{uuid.uuid4()}{file_ext}"
        file_path = os.path.join(UPLOAD_DIR, file_name)
        
        # Guardar el archivo
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        saved_files.append(file_path)
    
    return {"message": "Videos subidos exitosamente", "files": saved_files}

@app.post("/merge/")
async def merge_videos(request: MergeRequest):
    file_paths = request.file_paths
    
    # Verificar y normalizar rutas
    normalized_paths = []
    for path in file_paths:
        # Convertir rutas relativas a absolutas y normalizar separadores
        abs_path = os.path.abspath(path.replace('\\', '/'))
        if not os.path.exists(abs_path):
            raise HTTPException(status_code=404, detail=f"Archivo no encontrado: {abs_path}")
        normalized_paths.append(abs_path)
    
    clips = []
    try:
        clips = [VideoFileClip(path) for path in normalized_paths]
        
        # Asegurar que el directorio results existe
        os.makedirs(RESULT_DIR, exist_ok=True)
        
        output_name = f"merged_{uuid.uuid4()}.mp4"
        output_path = os.path.abspath(os.path.join(RESULT_DIR, output_name))
        
        if len(clips) == 1:
            # Si es solo un video, lo copiamos directamente
            shutil.copy2(normalized_paths[0], output_path)
        else:
            # Si son múltiples videos, los concatenamos
            final_clip = concatenate_videoclips(clips)
            final_clip.write_videofile(
                output_path, 
                codec="libx264", 
                audio_codec="aac",
                temp_audiofile="temp-audio.m4a",
                remove_temp=True
            )
            final_clip.close()
        
        # Retornar solo el nombre del archivo, no la ruta completa
        return {
            "message": "Video(s) procesado(s) exitosamente",
            "output_filename": output_name
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar videos: {str(e)}")
    finally:
        # Cerrar clips solo si fueron creados exitosamente
        for clip in clips:
            if clip:
                try:
                    clip.close()
                except:
                    pass

@app.get("/download/{filename}")
async def download_video(filename: str, request: Request):
    """Endpoint corregido para descargar videos"""
    try:
        # Seguridad: validar el nombre del archivo
        if not re.match(r'^merged_[a-f0-9-]+\.mp4$', filename):
            raise HTTPException(status_code=400, detail="Nombre de archivo inválido")
        
        file_path = os.path.join(RESULT_DIR, filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Archivo no encontrado")
        
        # Determinar el tipo de respuesta basado en el encabezado Accept
        accept_header = request.headers.get("Accept", "")
        
        if "application/json" in accept_header:
            return JSONResponse({
                "url": f"{str(request.base_url)}download/{filename}",
                "filename": filename,
                "size": os.path.getsize(file_path)
            })
        else:
            # Retornar el archivo directamente
            return FileResponse(
                path=file_path,
                media_type="video/mp4",
                filename=filename,
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Expose-Headers": "Content-Disposition",
                    "Content-Disposition": f"attachment; filename={filename}",
                    "Cross-Origin-Resource-Policy": "cross-origin",
                    "Cache-Control": "public, max-age=3600"
                }
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al servir archivo: {str(e)}")

@app.get("/stream/{filename}")
async def stream_video(filename: str, request: Request):
    """Endpoint adicional para streaming de video (para visualización)"""
    try:
        # Seguridad: validar el nombre del archivo
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
                "Accept-Ranges": "bytes",
                "Content-Disposition": f"inline; filename={filename}",
                "Cross-Origin-Resource-Policy": "cross-origin",
                "Cache-Control": "public, max-age=3600"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al servir archivo: {str(e)}")

@app.get("/clean/")
async def clean_directories():
    """Endpoint para limpiar los directorios (opcional, para desarrollo)"""
    try:
        for folder in [UPLOAD_DIR, RESULT_DIR]:
            if os.path.exists(folder):
                for filename in os.listdir(folder):
                    file_path = os.path.join(folder, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                    except Exception as e:
                        print(f"Error eliminando {file_path}: {str(e)}")
        
        return {"message": "Directorios limpiados exitosamente"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al limpiar: {str(e)}")

@app.get("/list-uploads/")
async def list_uploaded_videos():
    """Endpoint para listar los videos subidos"""
    try:
        if not os.path.exists(UPLOAD_DIR):
            return {"files": []}
        
        files = []
        for f in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, f)
            if os.path.isfile(file_path):
                files.append(file_path)
        
        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listando archivos: {str(e)}")

@app.get("/test-cors")
async def test_cors():
    return {"message": "CORS está funcionando!"}

@app.get("/")
async def root():
    return {"message": "Video Merger API funcionando correctamente"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)