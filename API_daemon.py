import subprocess     # Used to execute external system commands (i.e., 'python run_model.py')
import threading      # Used to run the training worker loop in the background, non-blocking the web server
import queue          # Provides a thread-safe Queue for holding incoming training jobs
import time           # Used for timestamp generation and process cooldowns
import os             # Used for path manipulation
from contextlib import asynccontextmanager # NEW: Required for the lifespan manager
from fastapi import FastAPI, HTTPException # Core FastAPI classes for building the web API
from pydantic import BaseModel             # Used to define structured data schemas for requests
from typing import Optional, Literal       # Used for type hints
from collections import deque              # Used for the logs
import uvicorn        #server
# --- Configuration ---
XTRAIN_SCRIPT = "run_model.py"
LOG_MAX_LINES = 20

# --- Pydantic Data Schema for Requests ---
class TrainRequest(BaseModel):
    model: Literal["llm", "cnn", "multimodal"] 
    dataset: str = "wikitext"
    epochs: int = 1
    mode: str = "train"

# --- Global State Management ---
job_queue = queue.Queue()

current_job_status = {
    "state": "IDLE",
    "job_id": None,
    "job_config": None,
    "output_log": deque(maxlen=LOG_MAX_LINES) 
}

# --- Background Worker Thread Function ---
def training_worker():
    """
    The core worker function running in a separate thread.
    Pulls jobs from the queue and executes them via subprocess.
    """
    print(" [DAEMON] Worker thread started. Waiting for jobs...")

    while True:
        # 1. Blocking wait for a new job
        request_data: TrainRequest = job_queue.get() 
        
        try:
            # 2. Setup the job state
            job_id = f"{request_data.model}-{int(time.time())}"
            
            # Update the global status dictionary
            current_job_status["state"] = "TRAINING"
            current_job_status["job_id"] = job_id
            current_job_status["job_config"] = request_data.model_dump()
            current_job_status["output_log"].clear()
            current_job_status["output_log"].append(f"Starting job: {job_id}")
            
            # 3. Construct the command to execute
            cmd = [
                "python3", XTRAIN_SCRIPT,
                "--model", request_data.model,
                "--dataset", request_data.dataset,
                "--mode", request_data.mode,
                "--epochs", str(request_data.epochs)
            ]

            current_job_status["output_log"].append(f"Executing: {' '.join(cmd)}")
            
            # 4. Execute the XTRAIN script
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, 
                universal_newlines=True,
                cwd=os.getcwd()
            )
            
            # 5. Stream the output logs
            for line in process.stdout:
                line = line.strip()
                if line:
                    current_job_status["output_log"].append(line) 

            # 6. Wait for the process to finish
            process.wait() 
            
            # 7. Finalize status
            if process.returncode == 0:
                current_job_status["state"] = "COMPLETED"
                current_job_status["output_log"].append("Job finished successfully.")
            else:
                current_job_status["state"] = "ERROR"
                current_job_status["output_log"].append(f"Job failed with exit code {process.returncode}.")
        
        except Exception as e:
            current_job_status["state"] = "ERROR"
            current_job_status["output_log"].append(f"CRITICAL DAEMON ERROR: {e}")

        finally:
            # Signals the queue that this job is finished
            job_queue.task_done() 
            
            # Reset state if the queue is now empty
            if job_queue.empty():
                current_job_status["state"] = "IDLE"
                current_job_status["job_id"] = None
                current_job_status["job_config"] = None

            time.sleep(1)

# --- FastAPI Lifespan Manager ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Replaces @app.on_event("startup") and handles the application's lifecycle.
    """
    # --- STARTUP ---
    print(" [DAEMON] Worker thread requested to start.")
    # Create and start the training worker thread
    t = threading.Thread(target=training_worker, daemon=True)
    t.start()
    
    # Yield control back to FastAPI to start serving requests
    yield
    
    # --- SHUTDOWN ---
    # This section runs when the server is gracefully shutting down
    print(" [DAEMON] Application shutdown complete.")


# --- FastAPI Initialization (NOW includes the lifespan manager) ---
app = FastAPI(title="XTRAIN CPU Daemon", lifespan=lifespan)


# --- API Endpoints (UNMODIFIED) ---

@app.post("/api/train")
async def queue_training(job: TrainRequest):
    """
    Accepts a new training request and adds it to the job queue.
    """
    position = job_queue.qsize() + 1
    job_queue.put(job)
    
    return {
        "status": "queued",
        "queue_position": position,
        "job": job.model_dump()
    }

@app.get("/api/status")
async def get_status():
    """
    Returns the real-time status of the worker and the job queue.
    """
    recent_logs = list(current_job_status["output_log"])

    return {
        "daemon_state": current_job_status["state"],
        "current_job_id": current_job_status["job_id"],
        "jobs_in_queue": job_queue.qsize(),
        "current_config": current_job_status["job_config"],
        "live_logs": recent_logs
    }
if __name__ == "__main__":
    import uvicorn
    # The default port for Uvicorn is 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)