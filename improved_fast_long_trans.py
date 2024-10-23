import time
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from fastapi import FastAPI, File, UploadFile
import soundfile as sf
import numpy as np
from pydub import AudioSegment
import io
import subprocess
import os
import asyncio
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Constants for chunking
CHUNK_LENGTH_SEC = 29
OVERLAP_SEC = 5
SAMPLE_RATE = 16000

# Set up PyTorch optimizations
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load model with optimizations
model_id = "openai/whisper-large-v3-turbo"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, 
    torch_dtype=torch_dtype, 
    low_cpu_mem_usage=True,
    use_safetensors=True
).to(device)

model.generation_config.cache_implementation = "static"
model.generation_config.max_new_tokens = 256

if hasattr(model.config, "attn_implementation"):
    model.config.attn_implementation = "flash_attention_2"

processor = AutoProcessor.from_pretrained(model_id)

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

def split_audio(samples: np.ndarray, sample_rate: int) -> List[np.ndarray]:
    """Split audio into overlapping chunks."""
    chunk_length_samples = int(CHUNK_LENGTH_SEC * sample_rate)
    overlap_samples = int(OVERLAP_SEC * sample_rate)
    
    # Calculate chunks
    chunks = []
    start = 0
    while start < len(samples):
        end = min(start + chunk_length_samples, len(samples))
        chunk = samples[start:end]
        chunks.append(chunk)
        start += chunk_length_samples - overlap_samples
    
    return chunks

def merge_transcriptions(results: List[dict], chunk_length: float, overlap: float) -> str:
    """Merge transcriptions from multiple chunks, handling overlaps and avoiding repetition."""
    if not results:
        return ""
    
    merged_text = []
    previous_text = ""
    
    for i, result in enumerate(results):
        current_text = result["text"].strip()
        
        if i == 0:
            # For the first chunk, use the entire text
            merged_text.append(current_text)
            previous_text = current_text
        else:
            # For subsequent chunks, check overlap with the previous text
            prev_words = previous_text.split()
            current_words = current_text.split()

            # Define a window for overlap detection (let's use 50% of the overlap length)
            overlap_window = int((overlap / chunk_length) * len(prev_words))

            # Look for overlap at the end of the previous chunk
            overlap_found = False
            for j in range(overlap_window, 0, -1):
                overlap_phrase = " ".join(prev_words[-j:])
                if overlap_phrase in current_text:
                    # Found overlap, append only the new portion
                    overlap_idx = current_text.index(overlap_phrase) + len(overlap_phrase)
                    merged_text.append(current_text[overlap_idx:].strip())
                    overlap_found = True
                    break

            if not overlap_found:
                # No overlap found, add the entire text for this chunk
                merged_text.append(current_text)
            
            previous_text = current_text
    
    # Join all merged parts into the final transcription
    return " ".join(merged_text).strip()

async def process_chunk(chunk: np.ndarray, chunk_index: int) -> dict:
    """Process a single audio chunk."""
    try:
        with torch.cuda.amp.autocast(enabled=True):
            with sdpa_kernel(SDPBackend.MATH):
                result = pipe(
                    inputs=chunk,
                    return_timestamps=True,
                    generate_kwargs={
                        "max_new_tokens": 256,
                        "do_sample": False,
                        "use_cache": True
                    }
                )
        logger.info(f"Processed chunk {chunk_index}")
        return result
    except Exception as e:
        logger.error(f"Error processing chunk {chunk_index}: {str(e)}")
        raise

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
        
        # Split audio into chunks
        chunks = split_audio(samples, sample_rate)
        logger.info(f"Split audio into {len(chunks)} chunks")
        
        # Process chunks
        tasks = [process_chunk(chunk, i) for i, chunk in enumerate(chunks)]
        results = await asyncio.gather(*tasks)
        
        # Merge results
        final_text = merge_transcriptions(results, CHUNK_LENGTH_SEC, OVERLAP_SEC)
        
        processing_time = time.time() - request_start_time
        logger.info(f"Total processing time: {processing_time:.2f} seconds")
        
        return {
            "transcription": final_text,
            "chunks_processed": len(chunks),
            "time_taken": processing_time
        }
        
    finally:
        if os.path.exists(input_file_path):
            os.remove(input_file_path)
