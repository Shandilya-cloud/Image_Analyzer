import base64
import binascii
import io
import secrets
from typing import List

import ollama
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from openai import OpenAI
from PIL import Image, UnidentifiedImageError
from starlette.requests import Request

from config import settings

app = FastAPI(title="Image Analyzer", version="1.0.0")
security = HTTPBasic()

ALLOWED_MIME = {"image/jpeg", "image/png"}
ALLOWED_FORMATS = {"JPEG", "PNG"}

HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Image Analyzer</title>
<style>
:root{font-family:Inter,system-ui,Arial,sans-serif;color:#e8edf5;background:#101722}
*{box-sizing:border-box}body{margin:0;padding:32px 16px}
main{max-width:1000px;margin:auto}.panel{background:#192433;border:1px solid #2b3b50;
border-radius:16px;padding:24px;margin:18px 0;box-shadow:0 12px 32px #0003}
h1{margin:0 0 8px;font-size:30px}.muted{color:#a8b6c8}
label{display:block;font-weight:650;margin:16px 0 8px}
textarea,select,input[type=text]{width:100%;border:1px solid #40536b;background:#0f1926;
color:#f4f7fb;border-radius:9px;padding:12px;font:inherit}
textarea{min-height:155px;resize:vertical}
.row{display:grid;grid-template-columns:1fr 1fr;gap:16px}
input[type=file]{width:100%;padding:14px;background:#0f1926;border:1px dashed #536780;border-radius:9px;color:#cbd5e1}
button{border:0;border-radius:9px;background:#4f8cff;color:white;padding:13px 22px;
font-size:15px;font-weight:700;cursor:pointer;margin-top:18px}
button:disabled{opacity:.5;cursor:wait}
#result{white-space:pre-wrap;line-height:1.65;overflow-wrap:anywhere;background:#0f1926;
border-radius:10px;padding:18px;min-height:80px}
#previews{display:flex;flex-wrap:wrap;gap:10px;margin-top:12px}
#previews img{width:120px;height:100px;object-fit:cover;border-radius:8px;border:1px solid #40536b}
.error{color:#ff9b9b}.success{color:#9ce6b0}
@media(max-width:650px){.row{grid-template-columns:1fr}body{padding:16px 10px}.panel{padding:17px}}
</style>
</head>
<body><main>
  <h1>Image Analyzer</h1>
  <div class="muted">Upload images, write a prompt, and analyze them with a vision model.</div>
  <section class="panel">
    <form id="analyzeForm">
      <div class="row">
        <div><label for="provider">AI provider</label>
          <select id="provider" name="provider">
            <option value="ollama">Ollama (local)</option>
            <option value="openai">OpenAI API</option>
          </select>
        </div>
        <div><label for="model">Model name (optional override)</label>
          <input id="model" name="model" type="text" placeholder="Use configured default">
        </div>
      </div>
      <label for="prompt">Your prompt</label>
      <textarea id="prompt" name="prompt">Analyze the uploaded image(s).

1. What is present in the image?
2. Identify all visible objects.
3. Describe the scene.
4. What is happening in the image?

Be specific. If something is unclear, say so rather than guessing.</textarea>
      <label for="images">Choose JPG or PNG images</label>
      <input id="images" name="images" type="file" accept=".jpg,.jpeg,.png,image/jpeg,image/png" multiple required>
      <div id="previews"></div>
      <button id="submitBtn" type="submit">Analyze image(s)</button>
    </form>
  </section>
  <section class="panel">
    <h2>Analysis result</h2>
    <div id="status" class="muted">Your result will appear here.</div>
    <div id="result"></div>
  </section>
  <div class="muted">Provider and model defaults are managed in config.py.</div>
</main>
<script>
const filesInput=document.getElementById('images');
const previews=document.getElementById('previews');
filesInput.addEventListener('change',()=>{
  previews.innerHTML='';
  [...filesInput.files].forEach(file=>{
    if(!file.type.startsWith('image/')) return;
    const img=document.createElement('img'); img.alt=file.name;
    img.src=URL.createObjectURL(file); previews.appendChild(img);
  });
});
document.getElementById('analyzeForm').addEventListener('submit',async e=>{
  e.preventDefault();
  const btn=document.getElementById('submitBtn');
  const statusEl=document.getElementById('status');
  const result=document.getElementById('result');
  const files=filesInput.files;
  if(!files.length){statusEl.textContent='Please select at least one image.';return;}
  const data=new FormData();
  data.append('provider',document.getElementById('provider').value);
  data.append('model',document.getElementById('model').value);
  data.append('prompt',document.getElementById('prompt').value);
  [...files].forEach(f=>data.append('images',f));
  btn.disabled=true; btn.textContent='Analyzing…';
  statusEl.textContent='Sending image(s) to the selected model…'; result.textContent='';
  try{
    const response=await fetch('/analyze',{method:'POST',body:data});
    const payload=await response.json();
    if(!response.ok) throw new Error(payload.detail||'Analysis failed');
    result.textContent=payload.response;
    statusEl.textContent=`Completed · ${payload.provider} · ${payload.model} · ${payload.image_count} image(s)`;
    statusEl.className='success';
  }catch(err){
    statusEl.textContent='Error: '+err.message; statusEl.className='error';
  }finally{btn.disabled=false;btn.textContent='Analyze image(s)';}
});
</script>
</body></html>
"""


def verify_login(credentials: HTTPBasicCredentials):
    user_ok = secrets.compare_digest(
        credentials.username.encode("utf-8"),
        settings.APP_USERNAME.encode("utf-8")
    )
    pass_ok = secrets.compare_digest(
        credentials.password.encode("utf-8"),
        settings.APP_PASSWORD.encode("utf-8")
    )
    if not (user_ok and pass_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )


@app.get("/", response_class=HTMLResponse)
async def home(credentials: HTTPBasicCredentials = Depends(security)
):
    verify_login(credentials)
    return HTML


async def read_and_validate_images(files: List[UploadFile]) -> List[tuple]:
    if not files:
        raise HTTPException(400, "Upload at least one JPG or PNG image.")
    if len(files) > settings.MAX_IMAGES:
        raise HTTPException(400, f"Maximum {settings.MAX_IMAGES} images are allowed.")

    images = []
    max_bytes = settings.MAX_IMAGE_MB * 1024 * 1024

    for file in files:
        raw = await file.read()
        if len(raw) > max_bytes:
            raise HTTPException(413, f"{file.filename}: exceeds {settings.MAX_IMAGE_MB} MB.")
        if file.content_type not in ALLOWED_MIME:
            raise HTTPException(400, f"{file.filename}: only JPG and PNG are supported.")
        try:
            with Image.open(io.BytesIO(raw)) as img:
                if img.format not in ALLOWED_FORMATS:
                    raise HTTPException(400, f"{file.filename}: invalid image format.")
                img.verify()
        except (UnidentifiedImageError, OSError):
            raise HTTPException(400, f"{file.filename}: file is not a valid image.")
        images.append((file.filename or "image", raw))
    return images


def ollama_chat(prompt: str, images: List[tuple], model: str) -> str:
    client = ollama.Client(
        host=settings.OLLAMA_HOST,
        auth=(settings.OLLAMA_USERNAME, settings.OLLAMA_PASSWORD)
        if settings.OLLAMA_USERNAME else None,
    )
    # The Python Ollama client accepts image bytes for each image.
    response = client.chat(
        model=model,
        messages=[{
            "role": "user",
            "content": prompt,
            "images": [raw for _, raw in images],
        }],
    )
    return response["message"]["content"]


def openai_chat(prompt: str, images: List[tuple], model: str) -> str:
    if not settings.OPENAI_API_KEY and not settings.OPENAI_BASE_URL:
        raise HTTPException(500, "Configure OPENAI_API_KEY (and optionally OPENAI_BASE_URL) in config.py/environment.")
    client_args = {"api_key": settings.OPENAI_API_KEY or "not-required"}
    if settings.OPENAI_BASE_URL:
        client_args["base_url"] = settings.OPENAI_BASE_URL
    client = OpenAI(**client_args)

    content = [{"type": "text", "text": prompt}]
    for filename, raw in images:
        # Normalize to a browser/API-friendly data URL.
        mime = "image/png" if filename.lower().endswith(".png") else "image/jpeg"
        encoded = base64.b64encode(raw).decode("ascii")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{encoded}"},
        })
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
    )
    return response.choices[0].message.content or ""


@app.post("/analyze")
async def analyze(
    prompt: str = Form(...),
    provider: str = Form("ollama"),
    model: str = Form(""),
    images: List[UploadFile] = File(...),
    credentials: HTTPBasicCredentials = Depends(security),
):
    verify_login(credentials)
    prompt = prompt.strip()
    if not prompt:
        raise HTTPException(400, "Prompt cannot be empty.")

    provider = provider.lower().strip()
    if provider not in {"ollama", "openai"}:
        raise HTTPException(400, "Provider must be ollama or openai.")

    validated_images = await read_and_validate_images(images)
    selected_model = model.strip() or (
        settings.OLLAMA_MODEL if provider == "ollama" else settings.OPENAI_MODEL
    )

    try:
        if provider == "ollama":
            answer = ollama_chat(prompt, validated_images, selected_model)
        else:
            answer = openai_chat(prompt, validated_images, selected_model)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, f"{provider} request failed: {exc}") from exc

    return JSONResponse({
        "provider": provider,
        "model": selected_model,
        "image_count": len(validated_images),
        "response": answer,
    })


@app.get("/health")
async def health():
    return {"status": "ok"}



if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        'app:app',
        reload=True
    )