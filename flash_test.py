import time
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from fastapi import FastAPI, File, UploadFile
import soundfile as sf
import numpy as np
from pydub import AudioSegment
import os
import asyncio
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Constants
SAMPLE_RATE = 16000

# Set up PyTorch optimizations
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load model with Flash Attention support
model_id = "openai/whisper-large-v3-turbo"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, 
    torch_dtype=torch_dtype, 
    low_cpu_mem_usage=True,
    use_safetensors=True
).to(device)

# Enabling Flash Attention
if hasattr(model.config, "attn_implementation"):
    model.config.attn_implementation = "flash_attention_2"

processor = AutoProcessor.from_pretrained(model_id)

# Optimized ASR pipeline
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
    batch_size=1
)

def load_audio(file_path: str) -> Tuple[np.ndarray, int]:
    """Load audio file and return audio data with sample rate."""
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_channels(1)  # Convert to mono
    audio = audio.set_frame_rate(SAMPLE_RATE)  # Convert to 16kHz
    
    # Convert to numpy array
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    samples = samples / (1 << (8 * audio.sample_width - 1))  # Normalize
    
    return samples, SAMPLE_RATE

@torch.inference_mode()
@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    request_start_time = time.time()
    
    # Save the uploaded file
    input_file_path = f"/tmp/{file.filename}"
    try:
        with open(input_file_path, "wb") as f:
            f.write(await file.read())
        
        # Load and prepare audio
        samples, sample_rate = load_audio(input_file_path)
        
        # Perform transcription on the entire audio at once (leveraging Flash Attention)
        with torch.cuda.amp.autocast(enabled=True):
            result = pipe(
                inputs=samples,
                return_timestamps=True,
                generate_kwargs={
                    "max_new_tokens": 256,
                    "do_sample": False,
                    "use_cache": True
                }
            )
        
        final_text = result["text"]
        processing_time = time.time() - request_start_time
        logger.info(f"Total processing time: {processing_time:.2f} seconds")
        
        return {
            "transcription": final_text,
            "time_taken": processing_time
        }
        
    finally:
        if os.path.exists(input_file_path):
            os.remove(input_file_path)
