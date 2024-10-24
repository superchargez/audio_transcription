import time
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from fastapi import FastAPI, File, UploadFile
import soundfile as sf
import numpy as np
from pydub import AudioSegment
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Constants
SAMPLE_RATE = 16000

# Set device to CPU
device = "cpu"
torch_dtype = torch.float32  # Use full precision for CPU

# Load model for CPU usage
model_id = "openai/whisper-large-v3-turbo"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch_dtype, 
    low_cpu_mem_usage=True,
    use_safetensors=True  # Ensure memory-efficient loading for CPU
).to(device)

# # Optional: enable Flash Attention (if supported for CPU, else you can comment this out)
# if hasattr(model.config, "attn_implementation"):
#     model.config.attn_implementation = "flash_attention_2"

processor = AutoProcessor.from_pretrained(model_id)

# Speech recognition pipeline (adjusted for CPU)
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch.float32,  # Use full precision for CPU
    device=-1  # -1 means CPU in Hugging Face pipelines
)

def load_audio(file_path: str):
    """Load audio file and return audio data with sample rate."""
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_channels(1)  # Convert to mono
    audio = audio.set_frame_rate(SAMPLE_RATE)  # Convert to 16kHz
    
    # Convert to numpy array
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    samples = samples / (1 << (8 * audio.sample_width - 1))  # Normalize
    
    return samples, SAMPLE_RATE

@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    request_start_time = time.perf_counter()
    
    input_file_path = f"/tmp/{file.filename}"
    try:
        with open(input_file_path, "wb") as f:
            f.write(await file.read())
        
        # Load and prepare audio
        start_load_audio = time.perf_counter()
        samples, sample_rate = load_audio(input_file_path)
        end_load_audio = time.perf_counter()
        loading_time = end_load_audio - start_load_audio
        
        # Perform transcription on the entire audio at once
        start_transcription = time.perf_counter()
        result = pipe(
            inputs=samples,
            return_timestamps=True,
            generate_kwargs={
                "max_new_tokens": 256,
                "do_sample": False,
                "use_cache": True
            }
        )
        end_transcription = time.perf_counter()
        processing_time = end_transcription - start_transcription
        
        final_text = result["text"]
        
        total_processing_time = end_transcription - request_start_time
        logger.info(f"Total processing time: {total_processing_time:.2f} seconds")
        logger.info(f"Loading audio time: {loading_time:.2f} seconds")
        logger.info(f"Transcription time: {processing_time:.2f} seconds")
        
        return {
            "transcription": final_text,
            "TOTAL TIME": total_processing_time,
            "LOADING TIME": loading_time,
            "TRANSCRIPTION TIME": processing_time
        }
    
    finally:
        if os.path.exists(input_file_path):
            os.remove(input_file_path)
