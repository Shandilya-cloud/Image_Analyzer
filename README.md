# FastAPI Image Analyzer

## Features
- Browser interface for custom prompts and multiple JPG/PNG uploads.
- Select Ollama or OpenAI-compatible API for each request.
- Configurable provider endpoints, model names, optional Ollama basic auth, and web login.
- Upload validation and size/count limits.

## Install

```bash
cd image_analyzer
python -m venv venv
# Linux:
source venv/bin/activate
# Windows PowerShell:
# .\venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set your credentials. `config.py` reads environment variables.
For example, on Linux/macOS:

```bash
export APP_USERNAME=admin
export APP_PASSWORD='use-a-long-unique-password'
export OLLAMA_HOST=http://127.0.0.1:11434
export OLLAMA_MODEL='pdurlej/qwythos-9b-claude-mythos-5-1m:latest'
```

On Windows PowerShell, use `$env:APP_USERNAME="admin"` style environment variables.

## Run

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000/`. The browser asks for the APP_USERNAME and APP_PASSWORD configured above.

## Provider configuration
- Ollama: ensure Ollama is running and the vision-capable model is available locally. Set `OLLAMA_HOST` and `OLLAMA_MODEL`.
- OpenAI: set `OPENAI_API_KEY` and `OPENAI_MODEL`. For an OpenAI-compatible server, set `OPENAI_BASE_URL` too.
- `OLLAMA_USERNAME` / `OLLAMA_PASSWORD` are optional and are passed as HTTP Basic auth to Ollama.

## Notes
- The model must support image/vision inputs. A text-only model cannot analyze images.
- The web login uses HTTP Basic authentication. Use HTTPS and a strong password if exposing the app beyond localhost or a trusted LAN.
- `config.py` uses environment variables so credentials need not be committed to source control.
