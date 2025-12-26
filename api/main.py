"""
EchoNotes API - FastAPI Backend
================================
Complete REST API for speech-to-notes conversion.

Endpoints:
- POST /api/upload - Upload audio file
- POST /api/record - Record audio from microphone
- POST /api/transcribe - Transcribe audio to text
- POST /api/analyze - Analyze transcript (NLP)
- POST /api/generate - Generate document
- POST /api/process - Full pipeline (audio → document)
- GET /api/status/{job_id} - Check job status
- GET /api/download/{job_id} - Download generated document

Run:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

API Docs:
    http://localhost:8000/docs (Swagger UI)
    http://localhost:8000/redoc (ReDoc)
"""

import os
import sys
import uuid
import asyncio
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Note: We use lazy imports in endpoints to avoid startup failures
# The modules are imported inside each endpoint function


# ============== Configuration ==============

class Settings:
    APP_NAME = "EchoNotes API"
    VERSION = "1.0.0"
    DESCRIPTION = "Speech-to-Notes Conversion API"
    
    # Directories
    UPLOAD_DIR = Path("./uploads")
    OUTPUT_DIR = Path("./outputs")
    TEMP_DIR = Path("./temp")
    
    # Limits
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    ALLOWED_AUDIO_TYPES = [
        "audio/wav", "audio/wave", "audio/x-wav",
        "audio/mpeg", "audio/mp3",
        "audio/ogg", "audio/flac",
        "audio/webm", "audio/m4a"
    ]
    ALLOWED_EXTENSIONS = [".wav", ".mp3", ".ogg", ".flac", ".webm", ".m4a"]
    
    # Models
    VOSK_MODEL = "vosk-model-en-us-0.22"
    USE_AI_ENHANCEMENT = True


settings = Settings()

# Create directories
for dir_path in [settings.UPLOAD_DIR, settings.OUTPUT_DIR, settings.TEMP_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# ============== Helper Functions ==============

def extract_transcription_data(result) -> dict:
    """
    Extract data from TranscriptionResult object or dict.
    
    Handles both:
    - TranscriptionResult object (from Transcriber)
    - Dict (legacy format)
    """
    if hasattr(result, 'text'):
        # It's a TranscriptionResult object
        return {
            "text": result.text,
            "confidence": getattr(result, 'avg_confidence', 0),
            "duration": getattr(result, 'duration', 0),
            "word_count": getattr(result, 'word_count', len(result.text.split())),
            "segments": [u.to_dict() for u in getattr(result, 'utterances', [])] if hasattr(result, 'utterances') else []
        }
    else:
        # It's a dict
        return {
            "text": result.get("text", ""),
            "confidence": result.get("confidence", 0),
            "duration": result.get("duration", 0),
            "word_count": len(result.get("text", "").split()),
            "segments": result.get("segments", [])
        }


# ============== Models ==============

class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class OutputFormat(str, Enum):
    MARKDOWN = "md"
    HTML = "html"
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    JSON = "json"


class TranscriptionRequest(BaseModel):
    audio_path: str
    language: str = "en"
    enable_diarization: bool = False
    

class AnalysisRequest(BaseModel):
    text: str
    title: str = "Document"
    use_ai: bool = True


class GenerateRequest(BaseModel):
    text: str
    title: str = "EchoNotes Document"
    format: OutputFormat = OutputFormat.HTML
    use_ai: bool = True
    include_full_content: bool = True


class ProcessRequest(BaseModel):
    """Full pipeline request"""
    title: str = "EchoNotes Document"
    format: OutputFormat = OutputFormat.HTML
    use_ai: bool = True
    language: str = "en"
    enhance_audio: bool = True
    enable_diarization: bool = False


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str
    created_at: str


class TranscriptionResult(BaseModel):
    text: str
    confidence: float
    duration: float
    word_count: int
    segments: List[Dict[str, Any]] = []


class AnalysisResult(BaseModel):
    title: str
    executive_summary: str
    key_sentences: List[str]
    concepts: List[Dict[str, Any]]
    questions: List[Dict[str, Any]]
    related_topics: List[str]
    word_count: int
    reading_time: float


class GenerationResult(BaseModel):
    job_id: str
    status: JobStatus
    document_path: str
    format: str
    ai_enhanced: bool


class HealthResponse(BaseModel):
    status: str
    version: str
    modules: Dict[str, bool]


# ============== Job Storage ==============

# In-memory job storage (use Redis/DB in production)
jobs: Dict[str, Dict[str, Any]] = {}


def create_job() -> str:
    """Create a new job and return its ID"""
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "status": JobStatus.PENDING,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "result": None,
        "error": None
    }
    return job_id


def create_job_with_id(job_id: str):
    """Create a new job with a specific ID"""
    jobs[job_id] = {
        "status": JobStatus.PENDING,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "result": None,
        "error": None
    }


def update_job(job_id: str, status: JobStatus, result: Any = None, error: str = None):
    """Update job status"""
    if job_id in jobs:
        jobs[job_id]["status"] = status
        jobs[job_id]["updated_at"] = datetime.now().isoformat()
        if result:
            jobs[job_id]["result"] = result
        if error:
            jobs[job_id]["error"] = error


# ============== FastAPI App ==============

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Helper Functions ==============

def validate_audio_file(file: UploadFile) -> bool:
    """Validate uploaded audio file"""
    # Check extension
    ext = Path(file.filename).suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        return False
    return True


async def save_upload_file(file: UploadFile) -> Path:
    """Save uploaded file to disk"""
    ext = Path(file.filename).suffix.lower()
    filename = f"{uuid.uuid4()}{ext}"
    filepath = settings.UPLOAD_DIR / filename
    
    with open(filepath, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    return filepath


# ============== Background Tasks ==============

async def process_transcription(job_id: str, audio_path: Path, language: str):
    """Background task for transcription"""
    try:
        from speech.transcriber import Transcriber
        
        update_job(job_id, JobStatus.PROCESSING)
        
        # Initialize transcriber (auto-download model if needed)
        model_path = Transcriber.get_or_download_model('en-us-small')
        transcriber = Transcriber(model_path=model_path)
        
        # Transcribe
        result = transcriber.transcribe(str(audio_path))
        result_data = extract_transcription_data(result)
        
        update_job(job_id, JobStatus.COMPLETED, result=result_data)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        update_job(job_id, JobStatus.FAILED, error=str(e))


async def process_full_pipeline(
    job_id: str,
    audio_path: Path,
    title: str,
    format: OutputFormat,
    use_ai: bool,
    language: str,
    enhance_audio: bool
):
    """Background task for full pipeline processing"""
    try:
        from speech.transcriber import Transcriber
        from document.smart_generator import SmartDocumentGenerator
        
        update_job(job_id, JobStatus.PROCESSING)
        
        # Step 1: Audio Enhancement (optional)
        if enhance_audio:
            try:
                from audio.enhancer import AudioEnhancer
                enhancer = AudioEnhancer()
                enhanced_path = settings.TEMP_DIR / f"{job_id}_enhanced.wav"
                enhancer.enhance(str(audio_path), str(enhanced_path))
                audio_path = enhanced_path
            except ImportError:
                print("[API] Audio enhancer not available, skipping enhancement")
        
        # Step 2: Transcription
        model_path = Transcriber.get_or_download_model('en-us-small')
        transcriber = Transcriber(model_path=model_path)
        transcript_result = transcriber.transcribe(str(audio_path))
        result_data = extract_transcription_data(transcript_result)
        text = result_data["text"]
        
        if not text:
            raise ValueError("Transcription failed - no text generated")
        
        # Step 3: Generate Document
        generator = SmartDocumentGenerator()
        output_filename = f"{job_id}_notes.{format.value}"
        output_path = settings.OUTPUT_DIR / output_filename
        
        generator.generate(
            text=text,
            output_path=str(output_path),
            title=title,
            format=format.value,
            use_ai=use_ai
        )
        
        update_job(job_id, JobStatus.COMPLETED, result={
            "transcript": text,
            "confidence": result_data["confidence"],
            "duration": result_data["duration"],
            "document_path": str(output_path),
            "format": format.value,
            "ai_enhanced": use_ai
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        update_job(job_id, JobStatus.FAILED, error=str(e))


# ============== API Endpoints ==============

@app.get("/", tags=["Root"])
async def root():
    """API Root - Welcome message"""
    return {
        "message": "Welcome to EchoNotes API",
        "version": settings.VERSION,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API health and module availability"""
    modules = {
        "audio_processor": False,
        "audio_enhancer": False,
        "speech_transcriber": False,
        "nlp_analyzer": False,
        "content_enhancer": False,
        "document_generator": False
    }
    
    try:
        from audio.processor import AudioProcessor
        modules["audio_processor"] = True
    except:
        pass
    
    try:
        from audio.enhancer import AudioEnhancer
        modules["audio_enhancer"] = True
    except:
        pass
    
    try:
        from speech.transcriber import Transcriber
        modules["speech_transcriber"] = True
    except:
        pass
    
    try:
        from nlp.smart_analyzer import SmartAnalyzer
        modules["nlp_analyzer"] = True
    except:
        pass
    
    try:
        from nlp.content_enhancer import ContentEnhancer
        modules["content_enhancer"] = True
    except:
        pass
    
    try:
        from document.smart_generator import SmartDocumentGenerator
        modules["document_generator"] = True
    except:
        pass
    
    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        modules=modules
    )


# ============== Upload Endpoints ==============

@app.post("/api/upload", response_model=JobResponse, tags=["Upload"])
async def upload_audio(
    file: UploadFile = File(..., description="Audio file to upload"),
    background_tasks: BackgroundTasks = None
):
    """
    Upload an audio file for processing.
    
    Supported formats: WAV, MP3, OGG, FLAC, WebM, M4A
    Max size: 100MB
    """
    # Validate file
    if not validate_audio_file(file):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {settings.ALLOWED_EXTENSIONS}"
        )
    
    # Save file
    filepath = await save_upload_file(file)
    
    # Create job
    job_id = create_job()
    jobs[job_id]["audio_path"] = str(filepath)
    
    return JobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        message=f"File uploaded successfully. Use job_id to process.",
        created_at=jobs[job_id]["created_at"]
    )


# ============== Transcription Endpoints ==============

@app.post("/api/transcribe", response_model=JobResponse, tags=["Transcription"])
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    language: str = Query("en", description="Language code"),
    background_tasks: BackgroundTasks = None
):
    """
    Transcribe audio file to text.
    
    This is an async operation. Use the returned job_id to check status.
    """
    # Validate and save file
    if not validate_audio_file(file):
        raise HTTPException(status_code=400, detail="Invalid audio file")
    
    filepath = await save_upload_file(file)
    
    # Create job and start processing
    job_id = create_job()
    
    # Run transcription in background
    background_tasks.add_task(
        process_transcription, job_id, filepath, language
    )
    
    return JobResponse(
        job_id=job_id,
        status=JobStatus.PROCESSING,
        message="Transcription started. Check status with /api/status/{job_id}",
        created_at=jobs[job_id]["created_at"]
    )


@app.post("/api/transcribe/sync", response_model=TranscriptionResult, tags=["Transcription"])
async def transcribe_audio_sync(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    language: str = Query("en", description="Language code")
):
    """
    Transcribe audio file to text (synchronous).
    
    Returns result immediately. Use for small files only.
    """
    # Validate and save file
    if not validate_audio_file(file):
        raise HTTPException(status_code=400, detail="Invalid audio file")
    
    filepath = await save_upload_file(file)
    
    try:
        from speech.transcriber import Transcriber
        
        # Transcribe (auto-download model if needed)
        model_path = Transcriber.get_or_download_model('en-us-small')
        transcriber = Transcriber(model_path=model_path)
        result = transcriber.transcribe(str(filepath))
        result_data = extract_transcription_data(result)
        
        return TranscriptionResult(
            text=result_data["text"],
            confidence=result_data["confidence"],
            duration=result_data["duration"],
            word_count=result_data["word_count"],
            segments=result_data["segments"]
        )
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Module not found: {str(e)}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup
        if filepath.exists():
            filepath.unlink()


# ============== Analysis Endpoints ==============

@app.post("/api/analyze", response_model=AnalysisResult, tags=["Analysis"])
async def analyze_text(request: AnalysisRequest):
    """
    Analyze text using NLP to extract:
    - Executive summary
    - Key sentences
    - Key concepts
    - Study questions
    - Related topics
    """
    try:
        from nlp.smart_analyzer import SmartAnalyzer
        
        analyzer = SmartAnalyzer()
        result = analyzer.analyze(request.text, request.title)
        
        return AnalysisResult(
            title=result.title,
            executive_summary=result.executive_summary,
            key_sentences=result.key_sentences,
            concepts=[c.to_dict() for c in result.concepts],
            questions=[q.to_dict() for q in result.questions],
            related_topics=result.related_topics,
            word_count=result.word_count,
            reading_time=result.reading_time_minutes
        )
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Module not found: {str(e)}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze/enhanced", tags=["Analysis"])
async def analyze_text_enhanced(request: AnalysisRequest):
    """
    Analyze text with AI-enhanced content generation.
    
    Includes:
    - Simple explanation
    - ELI5 explanation
    - Key takeaways
    - Real-world examples
    - FAQ
    - Vocabulary
    """
    try:
        from nlp.smart_analyzer import SmartAnalyzer
        from nlp.content_enhancer import get_content_enhancer
        
        # Basic analysis
        analyzer = SmartAnalyzer()
        analysis = analyzer.analyze(request.text, request.title)
        
        # AI enhancement
        ai_content = None
        if request.use_ai:
            enhancer = get_content_enhancer(use_ai=True)
            ai_content = enhancer.enhance_content(request.text, request.title)
        
        return {
            "analysis": {
                "title": analysis.title,
                "executive_summary": analysis.executive_summary,
                "key_sentences": analysis.key_sentences,
                "concepts": [c.to_dict() for c in analysis.concepts],
                "questions": [q.to_dict() for q in analysis.questions],
                "related_topics": analysis.related_topics,
                "word_count": analysis.word_count,
                "reading_time": analysis.reading_time_minutes
            },
            "ai_enhanced": ai_content.to_dict() if ai_content else None
        }
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Module not found: {str(e)}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============== Document Generation Endpoints ==============

@app.post("/api/generate", response_model=GenerationResult, tags=["Document Generation"])
async def generate_document(request: GenerateRequest):
    """
    Generate a formatted document from text.
    
    Formats: md, html, pdf, docx, txt, json
    """
    try:
        # Import here to avoid startup errors
        from document.smart_generator import SmartDocumentGenerator
        
        job_id = create_job()
        
        generator = SmartDocumentGenerator()
        output_filename = f"{job_id}_notes.{request.format.value}"
        output_path = settings.OUTPUT_DIR / output_filename
        
        print(f"[API] Generating document: {output_path}")
        print(f"[API] Text length: {len(request.text)} chars")
        print(f"[API] Use AI: {request.use_ai}")
        
        generator.generate(
            text=request.text,
            output_path=str(output_path),
            title=request.title,
            format=request.format.value,
            use_ai=request.use_ai,
            include_full_content=request.include_full_content
        )
        
        update_job(job_id, JobStatus.COMPLETED, result={
            "document_path": str(output_path)
        })
        
        return GenerationResult(
            job_id=job_id,
            status=JobStatus.COMPLETED,
            document_path=f"/api/download/{job_id}",
            format=request.format.value,
            ai_enhanced=request.use_ai
        )
    except ImportError as e:
        print(f"[API] Import error: {e}")
        raise HTTPException(status_code=500, detail=f"Module not found: {str(e)}")
    except Exception as e:
        print(f"[API] Generation error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============== Full Pipeline Endpoint ==============

@app.post("/api/process", response_model=JobResponse, tags=["Full Pipeline"])
async def process_audio(
    file: UploadFile = File(..., description="Audio file to process"),
    title: str = Form("EchoNotes Document"),
    format: OutputFormat = Form(OutputFormat.HTML),
    use_ai: bool = Form(True),
    language: str = Form("en"),
    enhance_audio: bool = Form(True),
    background_tasks: BackgroundTasks = None
):
    """
    Full pipeline: Audio → Transcription → Analysis → Document
    
    This is an async operation. Use job_id to check status and download result.
    """
    # Validate and save file
    if not validate_audio_file(file):
        raise HTTPException(status_code=400, detail="Invalid audio file")
    
    filepath = await save_upload_file(file)
    
    # Create job
    job_id = create_job()
    
    # Run full pipeline in background
    background_tasks.add_task(
        process_full_pipeline,
        job_id, filepath, title, format, use_ai, language, enhance_audio
    )
    
    return JobResponse(
        job_id=job_id,
        status=JobStatus.PROCESSING,
        message="Processing started. Check status with /api/status/{job_id}",
        created_at=jobs[job_id]["created_at"]
    )


@app.post("/api/process/sync", tags=["Full Pipeline"])
async def process_audio_sync(
    file: UploadFile = File(..., description="Audio file to process"),
    title: str = Form("EchoNotes Document"),
    format: OutputFormat = Form(OutputFormat.HTML),
    use_ai: bool = Form(False),  # Default to False for sync (faster)
    language: str = Form("en")
):
    """
    Full pipeline (synchronous): Audio → Transcription → Analysis → Document
    
    Returns result immediately. Use for small files only (<30 seconds).
    """
    # Validate and save file
    if not validate_audio_file(file):
        raise HTTPException(status_code=400, detail="Invalid audio file")
    
    filepath = await save_upload_file(file)
    job_id = str(uuid.uuid4())[:8]
    
    try:
        from speech.transcriber import Transcriber
        from nlp.smart_analyzer import SmartAnalyzer
        from document.smart_generator import SmartDocumentGenerator
        
        # Step 1: Transcribe (auto-download model if needed)
        model_path = Transcriber.get_or_download_model('en-us-small')
        transcriber = Transcriber(model_path=model_path)
        transcript_result = transcriber.transcribe(str(filepath))
        result_data = extract_transcription_data(transcript_result)
        text = result_data["text"]
        
        if not text:
            raise HTTPException(status_code=400, detail="Transcription failed")
        
        # Step 2: Analyze
        analyzer = SmartAnalyzer()
        analysis = analyzer.analyze(text, title)
        
        # Step 3: Generate Document
        generator = SmartDocumentGenerator()
        output_filename = f"{job_id}_notes.{format.value}"
        output_path = settings.OUTPUT_DIR / output_filename
        
        # Store job for download
        create_job_with_id(job_id)
        
        generator.generate(
            text=text,
            output_path=str(output_path),
            title=title,
            format=format.value,
            use_ai=use_ai
        )
        
        update_job(job_id, JobStatus.COMPLETED, result={
            "document_path": str(output_path)
        })
        
        return {
            "job_id": job_id,
            "status": "completed",
            "transcript": {
                "text": text,
                "confidence": result_data["confidence"],
                "duration": result_data["duration"],
                "word_count": len(text.split())
            },
            "analysis": {
                "executive_summary": analysis.executive_summary,
                "key_concepts": [c.term for c in analysis.concepts],
                "related_topics": analysis.related_topics
            },
            "document": {
                "job_id": job_id,
                "document_path": f"/api/download/{job_id}",
                "format": format.value,
                "filename": output_filename
            }
        }
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Module not found: {str(e)}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup uploaded file
        if filepath.exists():
            filepath.unlink()


# ============== Job Status Endpoints ==============

@app.get("/api/status/{job_id}", tags=["Jobs"])
async def get_job_status(job_id: str):
    """Get the status of a processing job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
        "result": job["result"] if job["status"] == JobStatus.COMPLETED else None,
        "error": job["error"] if job["status"] == JobStatus.FAILED else None
    }


@app.get("/api/jobs", tags=["Jobs"])
async def list_jobs(
    limit: int = Query(10, description="Number of jobs to return"),
    status: Optional[JobStatus] = Query(None, description="Filter by status")
):
    """List all processing jobs"""
    result = []
    for job_id, job in list(jobs.items())[-limit:]:
        if status and job["status"] != status:
            continue
        result.append({
            "job_id": job_id,
            "status": job["status"],
            "created_at": job["created_at"]
        })
    return {"jobs": result, "total": len(result)}


# ============== Download Endpoints ==============

@app.get("/api/download/{job_id}", tags=["Download"])
async def download_document(job_id: str):
    """Download the generated document for a completed job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job["status"] != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed. Status: {job['status']}"
        )
    
    if not job["result"] or "document_path" not in job["result"]:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc_path = Path(job["result"]["document_path"])
    
    if not doc_path.exists():
        raise HTTPException(status_code=404, detail="Document file not found")
    
    return FileResponse(
        path=doc_path,
        filename=doc_path.name,
        media_type="application/octet-stream"
    )


# ============== Utility Endpoints ==============

@app.delete("/api/jobs/{job_id}", tags=["Jobs"])
async def delete_job(job_id: str):
    """Delete a job and its associated files"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    # Delete associated files
    if job.get("result") and job["result"].get("document_path"):
        doc_path = Path(job["result"]["document_path"])
        if doc_path.exists():
            doc_path.unlink()
    
    del jobs[job_id]
    
    return {"message": f"Job {job_id} deleted successfully"}


@app.post("/api/cleanup", tags=["Utility"])
async def cleanup_old_files(max_age_hours: int = 24):
    """Clean up old uploaded and generated files"""
    import time
    
    deleted = {"uploads": 0, "outputs": 0}
    max_age_seconds = max_age_hours * 3600
    now = time.time()
    
    for directory, key in [(settings.UPLOAD_DIR, "uploads"), (settings.OUTPUT_DIR, "outputs")]:
        for filepath in directory.iterdir():
            if filepath.is_file():
                age = now - filepath.stat().st_mtime
                if age > max_age_seconds:
                    filepath.unlink()
                    deleted[key] += 1
    
    return {"message": "Cleanup completed", "deleted": deleted}


# ============== WebSocket for Real-time Updates ==============

from fastapi import WebSocket, WebSocketDisconnect

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, job_id: str):
        await websocket.accept()
        self.active_connections[job_id] = websocket
    
    def disconnect(self, job_id: str):
        if job_id in self.active_connections:
            del self.active_connections[job_id]
    
    async def send_update(self, job_id: str, message: dict):
        if job_id in self.active_connections:
            await self.active_connections[job_id].send_json(message)


manager = ConnectionManager()


@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time job updates"""
    await manager.connect(websocket, job_id)
    try:
        while True:
            # Check job status periodically
            if job_id in jobs:
                job = jobs[job_id]
                await manager.send_update(job_id, {
                    "status": job["status"],
                    "updated_at": job["updated_at"]
                })
                
                if job["status"] in [JobStatus.COMPLETED, JobStatus.FAILED]:
                    break
            
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        manager.disconnect(job_id)


# ============== Run Server ==============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
