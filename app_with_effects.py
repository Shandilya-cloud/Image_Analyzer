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
:root{
  --bg:#14040a;--bg2:#220810;
  --panel:rgba(58,14,26,.62);--panel-line:rgba(214,120,138,.22);
  --maroon:#800020;--maroon-2:#a3243f;--maroon-3:#d2506a;
  --gold:#e9cf9b;--text:#f6e9ec;--muted:#c9a3ab;
  --field:rgba(20,4,10,.72);--field-line:rgba(214,120,138,.34);
  font-family:Inter,system-ui,Arial,sans-serif;color:var(--text);background:var(--bg);
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;min-height:100vh;color:var(--text);
  background:linear-gradient(180deg,var(--bg2) 0%,var(--bg) 60%);overflow-x:hidden}

/* ---------- parallax backdrop ---------- */
.px-bg{position:fixed;inset:0;z-index:0;overflow:hidden;pointer-events:none}
.px-layer{position:absolute;inset:-18%;will-change:transform;transition:transform .14s linear}
.px-grid{background-image:
  linear-gradient(rgba(214,120,138,.07) 1px,transparent 1px),
  linear-gradient(90deg,rgba(214,120,138,.07) 1px,transparent 1px);
  background-size:68px 68px;
  -webkit-mask-image:radial-gradient(ellipse at 50% 40%,#000 25%,transparent 72%);
  mask-image:radial-gradient(ellipse at 50% 40%,#000 25%,transparent 72%)}
.px-orb{position:absolute;border-radius:50%;filter:blur(80px)}
.o1{width:48vmax;height:48vmax;left:-10%;top:-8%;background:radial-gradient(circle,#a3243f 0%,transparent 68%);opacity:.55}
.o2{width:40vmax;height:40vmax;right:-8%;top:20%;background:radial-gradient(circle,#6b0f2a 0%,transparent 70%);opacity:.7}
.o3{width:44vmax;height:44vmax;left:28%;bottom:-16%;background:radial-gradient(circle,#c2415c 0%,transparent 70%);opacity:.28}
.px-ring{position:absolute;border-radius:50%;border:1px solid rgba(233,207,155,.18)}
.r1{width:420px;height:420px;left:8%;top:32%}
.r2{width:260px;height:260px;right:12%;top:58%;border-color:rgba(210,80,106,.28)}
.r3{width:640px;height:640px;right:-6%;top:-10%;border-color:rgba(214,120,138,.12)}
.px-dot{position:absolute;width:6px;height:6px;border-radius:50%;background:var(--gold);
  box-shadow:0 0 14px 3px rgba(233,207,155,.55);opacity:.7}

/* ---------- layout ---------- */
main{position:relative;z-index:1;max-width:1000px;margin:auto;padding:0 16px 48px}
.hero{padding:84px 0 30px;text-align:center;will-change:transform,opacity}
.badge{display:inline-block;padding:6px 14px;border:1px solid var(--panel-line);border-radius:999px;
  color:var(--gold);font-size:12px;letter-spacing:.14em;text-transform:uppercase;
  background:rgba(128,0,32,.28);margin-bottom:18px}
h1{margin:0 0 12px;font-size:clamp(34px,6vw,56px);line-height:1.05;letter-spacing:-.02em;
  background:linear-gradient(120deg,#fff 10%,var(--gold) 55%,var(--maroon-3) 100%);
  -webkit-background-clip:text;background-clip:text;color:transparent}
h2{margin:0 0 14px;font-size:20px;color:var(--gold);letter-spacing:.01em}
.muted{color:var(--muted)}
.hero .muted{max-width:560px;margin:0 auto;font-size:16px;line-height:1.6}

.panel{position:relative;background:var(--panel);border:1px solid var(--panel-line);
  -webkit-backdrop-filter:blur(14px);backdrop-filter:blur(14px);
  border-radius:20px;padding:28px;margin:22px 0;
  box-shadow:0 18px 50px rgba(0,0,0,.45),inset 0 1px 0 rgba(255,255,255,.05)}
.panel::before{content:"";position:absolute;left:28px;right:28px;top:0;height:2px;
  background:linear-gradient(90deg,transparent,var(--maroon-3),var(--gold),transparent);opacity:.8}

label{display:block;font-weight:650;margin:18px 0 8px;color:#f2d9de;font-size:14px;letter-spacing:.02em}
textarea,select,input[type=text]{width:100%;border:1px solid var(--field-line);background:var(--field);
  color:var(--text);border-radius:12px;padding:13px 14px;font:inherit;transition:border-color .2s,box-shadow .2s}
textarea:focus,select:focus,input[type=text]:focus{outline:none;border-color:var(--maroon-3);
  box-shadow:0 0 0 3px rgba(210,80,106,.25)}
input::placeholder,textarea::placeholder{color:#8f6870}
select option{background:#220810;color:var(--text)}
textarea{min-height:170px;resize:vertical;line-height:1.55}
.row{display:grid;grid-template-columns:1fr 1fr;gap:16px}

input[type=file]{width:100%;padding:18px;background:var(--field);border:1.5px dashed rgba(233,207,155,.4);
  border-radius:12px;color:var(--muted);cursor:pointer;transition:border-color .2s,background .2s}
input[type=file]:hover{border-color:var(--gold);background:rgba(128,0,32,.22)}
input[type=file]::file-selector-button{border:0;border-radius:8px;padding:9px 16px;margin-right:14px;
  background:linear-gradient(135deg,var(--maroon),var(--maroon-2));color:#fff;font-weight:700;cursor:pointer}

button{border:0;border-radius:12px;color:#fff;padding:14px 28px;font-size:15px;font-weight:700;
  letter-spacing:.02em;cursor:pointer;margin-top:22px;
  background:linear-gradient(135deg,var(--maroon) 0%,var(--maroon-2) 55%,var(--maroon-3) 100%);
  box-shadow:0 10px 26px rgba(128,0,32,.55),inset 0 1px 0 rgba(255,255,255,.2);
  transition:transform .15s,box-shadow .2s,filter .2s}
button:hover:not(:disabled){transform:translateY(-2px);filter:brightness(1.1);
  box-shadow:0 14px 32px rgba(163,36,63,.6),inset 0 1px 0 rgba(255,255,255,.25)}
button:active:not(:disabled){transform:translateY(0)}
button:disabled{opacity:.55;cursor:wait}

#status{font-size:14px;margin-bottom:12px}
#result{white-space:pre-wrap;line-height:1.7;overflow-wrap:anywhere;background:var(--field);
  border:1px solid var(--panel-line);border-left:3px solid var(--maroon-3);
  border-radius:12px;padding:20px;min-height:90px}
#previews{display:flex;flex-wrap:wrap;gap:12px;margin-top:14px}
#previews img{width:120px;height:100px;object-fit:cover;border-radius:10px;
  border:1px solid var(--field-line);box-shadow:0 8px 20px rgba(0,0,0,.4);transition:transform .2s}
#previews img:hover{transform:scale(1.06) rotate(-1deg)}
.error{color:#ff9fae}.success{color:#a8e8bd}
.footnote{text-align:center;font-size:13px;margin-top:8px}

/* ---------- reveal on scroll ---------- */
.px-js .panel{opacity:0;transform:translateY(40px) scale(.985);
  transition:opacity .75s ease,transform .75s cubic-bezier(.2,.7,.2,1)}
.px-js .panel.in{opacity:1;transform:none}

@media(max-width:650px){.row{grid-template-columns:1fr}.panel{padding:20px}.hero{padding:56px 0 18px}}
@media(prefers-reduced-motion:reduce){
  .px-layer{transition:none}
  .px-js .panel{opacity:1;transform:none;transition:none}
}
</style>
<script>document.documentElement.classList.add('px-js');</script>
</head>
<body>
<div class="px-bg" aria-hidden="true">
  <div class="px-layer px-grid" data-depth="0.06"></div>
  <div class="px-layer" data-depth="0.12">
    <span class="px-orb o1"></span><span class="px-orb o2"></span><span class="px-ring r3"></span>
  </div>
  <div class="px-layer" data-depth="0.26">
    <span class="px-orb o3"></span><span class="px-ring r1"></span><span class="px-ring r2"></span>
    <span class="px-dot" style="left:14%;top:22%"></span><span class="px-dot" style="left:80%;top:16%"></span>
    <span class="px-dot" style="left:66%;top:60%"></span><span class="px-dot" style="left:26%;top:76%"></span>
    <span class="px-dot" style="left:90%;top:82%"></span>
  </div>
</div>
<main>
  <header class="hero">
    <span class="badge">Vision &middot; Ollama &amp; OpenAI</span>
    <h1>Image Analyzer</h1>
    <div class="muted">Upload images, write a prompt, and analyze them with a vision model.</div>
  </header>
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
  <div class="muted footnote">Provider and model defaults are managed in config.py.</div>
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
<script>
/* Parallax: visual only, no effect on app behaviour. */
(function(){
  var reduce=window.matchMedia&&window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var layers=[].slice.call(document.querySelectorAll('.px-layer'));
  var hero=document.querySelector('.hero');
  var mx=0,my=0,ticking=false;
  function frame(){
    ticking=false;
    var y=window.scrollY||0;
    layers.forEach(function(el){
      var d=parseFloat(el.getAttribute('data-depth'))||0.1;
      el.style.transform='translate3d('+(mx*d*90).toFixed(1)+'px,'+(-y*d+my*d*90).toFixed(1)+'px,0)';
    });
    if(hero){
      hero.style.transform='translate3d(0,'+(y*0.3).toFixed(1)+'px,0)';
      hero.style.opacity=Math.max(0,1-y/460).toFixed(2);
    }
  }
  function req(){if(!ticking){ticking=true;requestAnimationFrame(frame);}}
  if(!reduce){
    window.addEventListener('scroll',req,{passive:true});
    window.addEventListener('mousemove',function(e){
      mx=e.clientX/window.innerWidth-0.5;my=e.clientY/window.innerHeight-0.5;req();
    },{passive:true});
    req();
  }
  var items=[].slice.call(document.querySelectorAll('.panel'));
  if(reduce||!('IntersectionObserver' in window)){
    items.forEach(function(el){el.classList.add('in');});return;
  }
  var io=new IntersectionObserver(function(entries){
    entries.forEach(function(en){
      if(en.isIntersecting){en.target.classList.add('in');io.unobserve(en.target);}
    });
  },{threshold:0.08,rootMargin:'0px 0px -6% 0px'});
  items.forEach(function(el){io.observe(el);});
})();
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
        'app_with_effects:app',
        reload=True,
        host='0.0.0.0',
	port=8001
    )
