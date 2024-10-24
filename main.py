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

# Constants for chunking
CHUNK_LENGTH_SEC = 29
OVERLAP_SEC = 1
SAMPLE_RATE = 16000

# Set up PyTorch optimizations
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"

def load_model():
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
    
    return pipe

pipe = load_model()

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
    """Merge transcriptions from multiple chunks, handling overlaps."""
    if not results:
        return ""
    
    # If there's only one chunk, return its text directly
    if len(results) == 1:
        return results[0]["text"].strip()
    
    merged_text = []
    for i, result in enumerate(results):
        current_text = result["text"].strip()
        
        if i == 0:
            # For first chunk, use the full text
            merged_text.append(current_text)
        else:
            # For subsequent chunks, try to find overlap with previous text
            # and merge appropriately
            prev_words = merged_text[-1].split()
            current_words = current_text.split()
            
            # Look for overlap in the last few words
            overlap_found = False
            for j in range(min(len(prev_words), 10)):  # Check last 10 words
                overlap_phrase = " ".join(prev_words[-j:])
                if overlap_phrase in current_text:
                    # Found overlap, add only the new part
                    overlap_idx = current_text.index(overlap_phrase) + len(overlap_phrase)
                    merged_text.append(current_text[overlap_idx:])
                    overlap_found = True
                    break
            
            if not overlap_found:
                # If no clear overlap found, just add new text with a space
                merged_text.append(current_text)
    
    return " ".join(merged_text).strip()

from torch.nn.attention import SDPBackend, sdpa_kernel
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

async def transcribe(file: UploadFile = File(...)):
    request_start_time = time.perf_counter()
    
    input_file_path = f"/tmp/{file.filename}"
    try:
        with open(input_file_path, "wb") as f:
            f.write(await file.read())
        
        samples, sample_rate = load_audio(input_file_path)
        chunks = split_audio(samples, sample_rate)
        logger.info(f"Split audio into {len(chunks)} chunks")
        
        tasks = [process_chunk(chunk, i) for i, chunk in enumerate(chunks)]
        results = await asyncio.gather(*tasks)
        
        final_text = merge_transcriptions(results, CHUNK_LENGTH_SEC, OVERLAP_SEC)
        
        processing_time = time.perf_counter() - request_start_time
        logger.info(f"Total processing time: {processing_time:.2f} seconds")

        return {
            "transcription": final_text,
            "chunks_processed": len(chunks),
            "time_taken": processing_time
        }
        
    finally:
        if os.path.exists(input_file_path):
            os.remove(input_file_path)

@torch.inference_mode()
@app.post("/transcribe/")
async def handle_transcription(file: UploadFile = File(...)):
    return await transcribe(file)
