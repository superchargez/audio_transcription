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
from typing import List, Tuple, Dict
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from queue import Queue
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Constants for chunking
CHUNK_LENGTH_SEC = 29
OVERLAP_SEC = 5
SAMPLE_RATE = 16000

@dataclass
class TranscriptionChunk:
    index: int
    start_time: float
    end_time: float
    text: str
    confidence: float = 1.0

class AudioTranscriber:
    def __init__(self):
        self.setup_model()
        
    def setup_model(self):
        # Set up PyTorch optimizations
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Load model with optimizations
        model_id = "openai/whisper-large-v3-turbo"
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, 
            torch_dtype=self.torch_dtype, 
            low_cpu_mem_usage=True,
            use_safetensors=True
        ).to(self.device)

        self.model.generation_config.cache_implementation = "static"
        self.model.generation_config.max_new_tokens = 256

        if hasattr(self.model.config, "attn_implementation"):
            self.model.config.attn_implementation = "flash_attention_2"

        self.processor = AutoProcessor.from_pretrained(model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
            batch_size=1
        )

    @staticmethod
    def load_audio(file_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file and return normalized audio data."""
        audio = AudioSegment.from_file(file_path)
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(SAMPLE_RATE)
        
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        samples = samples / (1 << (8 * audio.sample_width - 1))
        
        return samples, SAMPLE_RATE

    def get_chunks(self, samples: np.ndarray) -> List[Tuple[int, np.ndarray, float, float]]:
        """Split audio into chunks with metadata."""
        chunk_length_samples = int(CHUNK_LENGTH_SEC * SAMPLE_RATE)
        overlap_samples = int(OVERLAP_SEC * SAMPLE_RATE)
        
        chunks = []
        start_sample = 0
        chunk_index = 0
        
        while start_sample < len(samples):
            end_sample = min(start_sample + chunk_length_samples, len(samples))
            chunk = samples[start_sample:end_sample]
            
            # Calculate time boundaries
            start_time = start_sample / SAMPLE_RATE
            end_time = end_sample / SAMPLE_RATE
            
            chunks.append((chunk_index, chunk, start_time, end_time))
            start_sample += chunk_length_samples - overlap_samples
            chunk_index += 1
        
        return chunks

    def transcribe_chunk(self, chunk_data: Tuple[int, np.ndarray, float, float]) -> TranscriptionChunk:
        """Transcribe a single chunk."""
        chunk_index, audio_chunk, start_time, end_time = chunk_data
        
        with torch.cuda.amp.autocast(enabled=True):
            with sdpa_kernel(SDPBackend.MATH):
                result = self.pipe(
                    inputs=audio_chunk,
                    return_timestamps=True,
                    generate_kwargs={
                        "max_new_tokens": 256,
                        "do_sample": False,
                        "use_cache": True
                    }
                )
        
        return TranscriptionChunk(
            index=chunk_index,
            start_time=start_time,
            end_time=end_time,
            text=result["text"].strip()
        )

    def remove_duplicates(self, chunks: List[TranscriptionChunk]) -> str:
        """Remove duplicated content from overlapping chunks."""
        def normalize_text(text: str) -> str:
            return ' '.join(text.lower().split())

        # Sort chunks by start time
        chunks.sort(key=lambda x: x.start_time)
        
        if not chunks:
            return ""
        
        final_text = chunks[0].text
        
        for i in range(1, len(chunks)):
            current_chunk = chunks[i]
            current_text = current_chunk.text
            
            # Find potential overlap with previous text
            prev_text = final_text
            prev_words = normalize_text(prev_text).split()
            current_words = normalize_text(current_text).split()
            
            # Look for significant overlapping sequence
            max_overlap = 0
            overlap_index = 0
            
            for j in range(len(prev_words)):
                for k in range(len(current_words)):
                    overlap_length = 0
                    while (j + overlap_length < len(prev_words) and 
                           k + overlap_length < len(current_words) and 
                           prev_words[j + overlap_length] == current_words[k + overlap_length]):
                        overlap_length += 1
                    
                    if overlap_length > max_overlap:
                        max_overlap = overlap_length
                        overlap_index = k
            
            # If significant overlap found, merge accordingly
            if max_overlap >= 3:  # Require at least 3 words overlap
                merged_text = ' '.join(current_words[overlap_index + max_overlap:])
                if merged_text:
                    final_text += ' ' + merged_text
            else:
                # If no significant overlap, just append with a space
                final_text += ' ' + current_text
        
        return final_text.strip()

transcriber = AudioTranscriber()

@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    request_start_time = time.time()
    
    # Save the uploaded file
    input_file_path = f"/tmp/{file.filename}"
    try:
        with open(input_file_path, "wb") as f:
            f.write(await file.read())
        
        # Load and prepare audio
        samples, _ = transcriber.load_audio(input_file_path)
        chunks = transcriber.get_chunks(samples)
        
        # Process chunks in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            chunk_results = list(executor.map(transcriber.transcribe_chunk, chunks))
        
        # Merge results with improved duplicate removal
        final_text = transcriber.remove_duplicates(chunk_results)
        
        processing_time = time.time() - request_start_time
        
        return {
            "transcription": final_text,
            "chunks_processed": len(chunks),
            "time_taken": processing_time
        }
        
    finally:
        if os.path.exists(input_file_path):
            os.remove(input_file_path)
