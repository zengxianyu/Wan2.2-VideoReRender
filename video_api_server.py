"""
Video Re-rendering API Server

This FastAPI server provides endpoints for video processing using the WAN 2.2 video re-rendering model.
The models are loaded once at startup and kept in memory for efficient processing of multiple requests.
"""

import os
import sys
import asyncio
import logging
from typing import Optional, Dict, Any
from datetime import datetime
import tempfile
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add ComfyUI to path
sys.path.insert(0, "ComfyUI")

# Import our video processor
from wan22_style import VideoProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Video Re-rendering API",
    description="API for AI-powered video style transfer and re-rendering using WAN 2.2",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global video processor instance
video_processor: Optional[VideoProcessor] = None

# Job storage for tracking processing status
processing_jobs: Dict[str, Dict[str, Any]] = {}

# Pydantic models for API requests/responses
class ProcessingRequest(BaseModel):
    positive_prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    style_prompt: Optional[str] = None
    fps: int = 16
    num_frames: int = 121
    seed: int = -1
    preprocess_option: str = "Intensity"
    lowvram: bool = True

class JobStatus(BaseModel):
    job_id: str
    status: str  # "queued", "processing", "completed", "failed"
    progress: float = 0.0
    message: str = ""
    result_path: Optional[str] = None
    created_at: str
    updated_at: str

class JobResponse(BaseModel):
    job_id: str
    status: str
    message: str

# Create directories for temporary files
TEMP_DIR = Path("temp")
OUTPUT_DIR = Path("api_outputs")
TEMP_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

@app.on_event("startup")
async def startup_event():
    """Initialize the video processor on startup."""
    global video_processor
    
    logger.info("Starting Video Re-rendering API server...")
    logger.info("Deferring video processor initialization to first request...")
    
    # Don't initialize here - do it lazily on first request to avoid event loop conflicts
    video_processor = None
    logger.info("API server startup complete. VideoProcessor will initialize on first request.")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down API server...")
    
    # Clean up temporary files
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    
    logger.info("API server shutdown complete.")

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "Video Re-rendering API is running",
        "status": "healthy",
        "processor_ready": video_processor is not None
    }

@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "processor_initialized": video_processor is not None,
        "lazy_initialization": video_processor is None,
        "active_jobs": len([job for job in processing_jobs.values() if job["status"] == "processing"]),
        "total_jobs": len(processing_jobs)
    }

@app.post("/process", response_model=JobResponse)
async def process_video(
    background_tasks: BackgroundTasks,
    video_file: UploadFile = File(...),
    positive_prompt: Optional[str] = Form(None),
    negative_prompt: Optional[str] = Form(None),
    style_prompt: Optional[str] = Form(None),
    fps: int = Form(16),
    num_frames: int = Form(121),
    seed: int = Form(-1),
    preprocess_option: str = Form("Intensity"),
):
    """
    Process a video file with AI re-rendering using Flux-generated reference image.
    
    - **video_file**: The video file to process (mp4, avi, mov, etc.)
    - **positive_prompt**: Positive text prompt for style guidance
    - **negative_prompt**: Negative text prompt (what to avoid)
    - **style_prompt**: Style-specific prompt that will be combined with positive prompt
    - **fps**: Output video frame rate (default: 16)
    - **num_frames**: Number of frames to process per iteration (default: 121)
    - **seed**: Random seed for reproducible results (-1 for random)
    - **preprocess_option**: Control method - "Intensity", "Canny", or "None"
    """
    
    # Video processor will be initialized lazily in the background task
    
    # Validate file type
    if not video_file.content_type or not video_file.content_type.startswith('video/'):
        # If content_type is not set, check file extension
        if not video_file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
            raise HTTPException(status_code=400, detail="File must be a video")

    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded file temporarily
    temp_input_path = TEMP_DIR / f"{job_id}_input_{video_file.filename}"
    temp_output_path = OUTPUT_DIR / f"{job_id}_output"
    
    try:
        # Save uploaded file
        with open(temp_input_path, "wb") as buffer:
            content = await video_file.read()
            buffer.write(content)
        
        # Create job record
        job_record = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0.0,
            "message": "Video uploaded, processing queued",
            "result_path": None,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "input_path": str(temp_input_path),
            "reference_path": None,  # No reference image for this endpoint
            "output_path": str(temp_output_path),
            "processing_type": "with_flux",  # Use Flux to generate reference
            "params": {
                "positive_prompt": positive_prompt,
                "negative_prompt": negative_prompt,
                "style_prompt": style_prompt,
                "fps": fps,
                "num_frames": num_frames,
                "seed": seed,
                "preprocess_option": preprocess_option,
            }
        }
        
        processing_jobs[job_id] = job_record
        
        # Start processing in background
        background_tasks.add_task(process_video_background, job_id)
        
        return JobResponse(
            job_id=job_id,
            status="queued",
            message="Video processing started"
        )
    
    except Exception as e:
        # Cleanup on error
        if temp_input_path.exists():
            temp_input_path.unlink()
        
        logger.error(f"Error starting video processing: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start processing: {str(e)}")

@app.post("/process_with_reference", response_model=JobResponse)
async def process_video_with_reference(
    background_tasks: BackgroundTasks,
    video_file: UploadFile = File(...),
    reference_image: UploadFile = File(...),
    positive_prompt: Optional[str] = Form(None),
    negative_prompt: Optional[str] = Form(None),
    fps: int = Form(16),
    num_frames: int = Form(121),
    seed: int = Form(-1),
    preprocess_option: str = Form("Intensity"),
):
    """
    Process a video file with AI re-rendering using a provided reference image.
    
    - **video_file**: The video file to process (mp4, avi, mov, etc.)
    - **reference_image**: Reference image file (jpg, png, etc.) to use instead of Flux generation
    - **positive_prompt**: Positive text prompt for WAN model
    - **negative_prompt**: Negative text prompt (what to avoid)
    - **fps**: Output video frame rate (default: 16)
    - **num_frames**: Number of frames to process per iteration (default: 121)
    - **seed**: Random seed for reproducible results (-1 for random)
    - **preprocess_option**: Control method - "Intensity", "Canny", or "None"
    """
    
    # Video processor will be initialized lazily in the background task
    
    # Validate video file type
    if not video_file.content_type or not video_file.content_type.startswith('video/'):
        # If content_type is not set, check file extension
        if not video_file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
            raise HTTPException(status_code=400, detail="video_file must be a video")
    
    # Validate reference image file type
    if not reference_image.content_type or not reference_image.content_type.startswith('image/'):
        # If content_type is not set, check file extension
        if not reference_image.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            raise HTTPException(status_code=400, detail="reference_image must be an image")

    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded files temporarily
    temp_video_path = TEMP_DIR / f"{job_id}_input_{video_file.filename}"
    temp_reference_path = TEMP_DIR / f"{job_id}_reference_{reference_image.filename}"
    temp_output_path = OUTPUT_DIR / f"{job_id}_output"
    
    try:
        # Save uploaded video file
        with open(temp_video_path, "wb") as buffer:
            video_content = await video_file.read()
            buffer.write(video_content)
        
        # Save uploaded reference image
        with open(temp_reference_path, "wb") as buffer:
            image_content = await reference_image.read()
            buffer.write(image_content)
        
        # Create job record
        job_record = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0.0,
            "message": "Files uploaded, processing queued",
            "result_path": None,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "input_path": str(temp_video_path),
            "reference_path": str(temp_reference_path),
            "output_path": str(temp_output_path),
            "processing_type": "with_reference",
            "params": {
                "positive_prompt": positive_prompt,
                "negative_prompt": negative_prompt,
                "fps": fps,
                "num_frames": num_frames,
                "seed": seed,
                "preprocess_option": preprocess_option,
            }
        }
        
        processing_jobs[job_id] = job_record
        
        # Start processing in background
        background_tasks.add_task(process_video_background, job_id)
        
        return JobResponse(
            job_id=job_id,
            status="queued",
            message="Video and reference image processing started"
        )
    
    except Exception as e:
        # Cleanup on error
        if temp_video_path.exists():
            temp_video_path.unlink()
        if temp_reference_path.exists():
            temp_reference_path.unlink()
        
        logger.error(f"Error starting video processing with reference: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start processing: {str(e)}")
    """
    Process a video file with AI re-rendering.
    
    - **video_file**: The video file to process (mp4, avi, mov, etc.)
    - **positive_prompt**: Positive text prompt for style guidance
    - **negative_prompt**: Negative text prompt (what to avoid)
    - **style_prompt**: Style-specific prompt that will be combined with positive prompt
    - **fps**: Output video frame rate (default: 16)
    - **num_frames**: Number of frames to process per iteration (default: 121)
    - **seed**: Random seed for reproducible results (-1 for random)
    - **preprocess_option**: Control method - "Intensity", "Canny", or "None"
    """
    
    # Video processor will be initialized lazily in the background task
    
    # Validate file type
    if not video_file.content_type or not video_file.content_type.startswith('video/'):
        # If content_type is not set, check file extension
        if not video_file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
            raise HTTPException(status_code=400, detail="File must be a video")
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded file temporarily
    temp_input_path = TEMP_DIR / f"{job_id}_input_{video_file.filename}"
    temp_output_path = OUTPUT_DIR / f"{job_id}_output"
    
    try:
        # Save uploaded file
        with open(temp_input_path, "wb") as buffer:
            content = await video_file.read()
            buffer.write(content)
        
        # Create job record
        job_record = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0.0,
            "message": "Video uploaded, processing queued",
            "result_path": None,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "input_path": str(temp_input_path),
            "output_path": str(temp_output_path),
            "params": {
                "positive_prompt": positive_prompt,
                "negative_prompt": negative_prompt,
                "style_prompt": style_prompt,
                "fps": fps,
                "num_frames": num_frames,
                "seed": seed,
                "preprocess_option": preprocess_option,
            }
        }
        
        processing_jobs[job_id] = job_record
        
        # Start processing in background
        background_tasks.add_task(process_video_background, job_id)
        
        return JobResponse(
            job_id=job_id,
            status="queued",
            message="Video processing started"
        )
    
    except Exception as e:
        # Cleanup on error
        if temp_input_path.exists():
            temp_input_path.unlink()
        
        logger.error(f"Error starting video processing: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start processing: {str(e)}")

async def process_video_background(job_id: str):
    """Background task to process video."""
    job = processing_jobs.get(job_id)
    if not job:
        return
    
    global video_processor
    
    try:
        # Initialize video processor if not already done
        if video_processor is None:
            job["status"] = "processing"
            job["progress"] = 5.0
            job["message"] = "Initializing video processor and loading models..."
            job["updated_at"] = datetime.now().isoformat()
            
            # Initialize in a thread to avoid event loop issues
            import concurrent.futures
            
            def init_processor():
                return VideoProcessor(lowvram=True)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(init_processor)
                video_processor = future.result(timeout=180)  # 3 minute timeout
            
            logger.info("VideoProcessor initialized successfully!")
        
        # Update status to processing
        job["status"] = "processing"
        job["progress"] = 10.0
        job["message"] = "Starting video processing..."
        job["updated_at"] = datetime.now().isoformat()
        
        # Get parameters
        params = job["params"]
        input_path = job["input_path"]
        reference_path = job.get("reference_path")
        output_path = job["output_path"]
        processing_type = job.get("processing_type", "with_flux")
        
        # Use default prompts if not provided
        positive_prompt = params["positive_prompt"]
        if positive_prompt is None:
            positive_prompt = ("A video of a wide, multi-lane highway in a mountainous region. The road curves gently to the right, "
                              "with smooth asphalt and bright white dashed lane markings. A silver car drives slightly ahead in the left lane, "
                              "with glowing blue tail lights. On the right side, a tall concrete barrier with a blue fence section lines the edge "
                              "of the highway. Beyond it, a forest of tall evergreen trees rises against the base of mist-covered rocky mountains. "
                              "Streetlights stand along the road, casting a faint industrial presence, though the ambient light comes mainly from "
                              "the overcast sky. The air feels hazy, with muted visibility softening the distant trees and hills. "
                              "The camera moves steadily forward")
        
        negative_prompt = params["negative_prompt"]
        if negative_prompt is None:
            negative_prompt = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
                              "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，CG, game, cartoon, anime, "
                              "render, 渲染，游戏，卡通")
        
        style_prompt = params.get("style_prompt")
        if style_prompt is None and processing_type == "with_flux":
            style_prompt = "Turn it into a photorealistic picture as if it's from a movie. Keep the original lane markers and number of lanes."
        
        # Update progress
        job["progress"] = 20.0
        job["message"] = "Models loaded, starting video processing..."
        job["updated_at"] = datetime.now().isoformat()
        
        # Process the video based on processing type
        if processing_type == "with_reference" and reference_path:
            # Use the new method with reference image
            result_path = video_processor.process_video_with_reference_image(
                video_file_path=os.path.abspath(input_path),
                reference_image_path=os.path.abspath(reference_path),
                output_prefix=output_path,
                positive_prompt=positive_prompt,
                negative_prompt=negative_prompt,
                style_prompt=style_prompt,
                preprocess_option=params["preprocess_option"],
                num_frames=params["num_frames"],
                fps=params["fps"],
                seed=params["seed"]
            )
        else:
            # Use the original method with Flux generation
            result_path = video_processor._process_single_video(
                video_file_path=os.path.abspath(input_path),
                output_prefix=output_path,
                positive_prompt=positive_prompt,
                negative_prompt=negative_prompt,
                style_prompt=style_prompt,
                preprocess_option=params["preprocess_option"],
                num_frames=params["num_frames"],
                fps=params["fps"],
                seed=params["seed"]
            )
        
        # Update job as completed
        job["status"] = "completed"
        job["progress"] = 100.0
        job["message"] = "Video processing completed successfully"
        job["result_path"] = result_path
        job["updated_at"] = datetime.now().isoformat()
        
        logger.info(f"Job {job_id} completed successfully: {result_path}")
        
    except Exception as e:
        # Update job as failed
        job["status"] = "failed"
        job["message"] = f"Processing failed: {str(e)}"
        job["updated_at"] = datetime.now().isoformat()
        
        logger.error(f"Job {job_id} failed: {e}")
    
    finally:
        # Cleanup input files
        input_path_obj = Path(job["input_path"])
        if input_path_obj.exists():
            input_path_obj.unlink()
        
        # Cleanup reference image if it exists
        reference_path = job.get("reference_path")
        if reference_path:
            reference_path_obj = Path(reference_path)
            if reference_path_obj.exists():
                reference_path_obj.unlink()

@app.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get the status of a processing job."""
    job = processing_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobStatus(**job)

@app.get("/result/{job_id}")
async def get_result(job_id: str):
    """Download the processed video result."""
    job = processing_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    result_path = job["result_path"]
    if not result_path or not Path(result_path).exists():
        raise HTTPException(status_code=404, detail="Result file not found")
    
    return FileResponse(
        result_path,
        media_type='video/mp4',
        filename=f"processed_{job_id}.mp4"
    )

@app.get("/jobs")
async def list_jobs():
    """List all processing jobs."""
    return {
        "jobs": [
            {
                "job_id": job_id,
                "status": job["status"],
                "created_at": job["created_at"],
                "updated_at": job["updated_at"],
                "progress": job.get("progress", 0)
            }
            for job_id, job in processing_jobs.items()
        ]
    }

@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its associated files."""
    job = processing_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Remove result file if exists
    if job.get("result_path") and Path(job["result_path"]).exists():
        Path(job["result_path"]).unlink()
    
    # Remove input file if still exists
    if job.get("input_path") and Path(job["input_path"]).exists():
        Path(job["input_path"]).unlink()
    
    # Remove reference image if still exists
    if job.get("reference_path") and Path(job["reference_path"]).exists():
        Path(job["reference_path"]).unlink()
    
    # Remove job record
    del processing_jobs[job_id]
    
    return {"message": f"Job {job_id} deleted successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")