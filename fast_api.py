import time
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from fastapi import FastAPI, File, UploadFile
import soundfile as sf
import io
import subprocess
import os

app = FastAPI()

# Set up PyTorch optimizations
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for faster compute
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

# Enable static cache and optimization configs
model.generation_config.cache_implementation = "static"
model.generation_config.max_new_tokens = 256

# Optional: Enable Flash Attention 2 if available
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
    batch_size=1  # Adjust based on your GPU memory
)

def convert_to_wav_16khz(input_file: str) -> str:
    output_file = "converted.wav"
    command = [
        "ffmpeg", "-i", input_file, "-ar", "16000", "-ac", "1", output_file
    ]
    subprocess.run(command, check=True)
    return output_file

def get_audio_info(input_file: str):
    command = [
        "ffprobe", "-v", "error", "-show_entries",
        "stream=sample_rate,codec_name", "-of",
        "default=noprint_wrappers=1:nokey=1", input_file
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.stdout.split()

@torch.inference_mode()  # More efficient than no_grad
@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    request_start_time = time.time()
    
    # Save the uploaded file
    input_file_path = f"/tmp/{file.filename}"
    with open(input_file_path, "wb") as f:
        f.write(await file.read())

    try:
        # Get audio info
        audio_info = get_audio_info(input_file_path)
        codec_name, sample_rate = audio_info[0], int(audio_info[1])

        # Check if conversion is needed
        if codec_name != "pcm_s16le" or sample_rate != 16000:
            converted_file_path = convert_to_wav_16khz(input_file_path)
        else:
            converted_file_path = input_file_path

        # Configure CUDA for optimal performance
        with torch.cuda.amp.autocast(enabled=True):
            with sdpa_kernel(SDPBackend.MATH):
                result = pipe(
                    inputs=converted_file_path,
                    return_timestamps=True,
                    generate_kwargs={
                        "max_new_tokens": 256,
                        "do_sample": False,  # Disable sampling for faster inference
                        "use_cache": True
                    }
                )
        
        return {
            "transcription": result["text"],
            "time_taken": time.time() - request_start_time
        }

    finally:
        # Clean up temporary files
        if os.path.exists(input_file_path):
            os.remove(input_file_path)
        if 'converted_file_path' in locals() and converted_file_path != input_file_path:
            if os.path.exists(converted_file_path):
                os.remove(converted_file_path)
