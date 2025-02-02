from fastapi import FastAPI, Form
from fastapi.responses import StreamingResponse
from transformers import VitsModel, AutoTokenizer
from pydantic import BaseModel
import torch
import io

app = FastAPI()

# Load the model and tokenizer once
model = VitsModel.from_pretrained("facebook/mms-tts-kor")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-kor")

class TextRequest(BaseModel):
    text: str

@app.post("/generate_audio/mms_man")
async def generate_audio(request: TextRequest):
    text = request.text
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt")

    # Generate audio waveform
    with torch.no_grad():
        output = model(**inputs).waveform

    # Convert the waveform to a NumPy array and normalize
    waveform = output.squeeze().cpu().numpy()
    waveform = (waveform * 32767).astype("int16")  # Normalize to int16

    # Create an in-memory buffer
    audio_buffer = io.BytesIO()

    # Write the waveform as a WAV file to the buffer
    from scipy.io.wavfile import write
    write(audio_buffer, rate=model.config.sampling_rate, data=waveform)

    # Reset the buffer's pointer to the start
    audio_buffer.seek(0)

    n = request.headers[1].index
    filename = 'generated_audio.wav'
    if isinstance(n, int) and n > 0:
        filename = f'generated_audio ({n}).wav'

    # Return the buffer as a streaming response
    return StreamingResponse(
        audio_buffer,
        media_type="audio/wav",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
