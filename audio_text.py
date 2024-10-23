import time
start_time = time.time()

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from fastapi import FastAPI, File, UploadFile
import soundfile as sf
import io
import subprocess
import os

app = FastAPI()

# Load the model and processor once at startup
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
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

@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    request_start_time = time.time()
    
    # Save the uploaded file
    input_file_path = f"/tmp/{file.filename}"
    with open(input_file_path, "wb") as f:
        f.write(await file.read())

    # Get audio info
    audio_info = get_audio_info(input_file_path)
    codec_name, sample_rate = audio_info[0], int(audio_info[1])

    # Check if conversion is needed
    if codec_name != "pcm_s16le" or sample_rate != 16000:
        converted_file_path = convert_to_wav_16khz(input_file_path)
    else:
        converted_file_path = input_file_path

    # Perform speech recognition
    result = pipe(inputs=converted_file_path, return_timestamps=True)
    
    # Clean up temporary files
    os.remove(input_file_path)
    if converted_file_path != input_file_path:
        os.remove(converted_file_path)
    request_end_time = time.time()    
    return {
        "transcription": result["text"],
        "time_taken": request_end_time - start_time
    }