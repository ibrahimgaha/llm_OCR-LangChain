# Project 4 — OCR with LangChain and NVIDIA AI

Upload a PNG or JPEG to extract its text with NVIDIA-hosted Mistral Large 4.

## Setup on Windows

Use Python 3.14 and run these commands in PowerShell:

```powershell
cd "C:\Users\Aziz\Desktop\Agentic IA\04-OCR-LangChain"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

Set `NVIDIA_API_KEY` in `.env` and save it. If `.env` already exists,
skip the copy command and edit the existing file.

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Upload an image up to 5 MB and 25 megapixels, check its preview, then select
**Extract text**. Select the transcription and press **Ctrl+C** to copy it,
or select **Download text** to save a UTF-8 text file. The text area also
allows edits before download. Replacing the image clears the previous result.
The app sends the uploaded image to NVIDIA when you select **Extract text**.

For a live check, upload a simple image containing two known lines of text
and compare the transcription with the image.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The tests mock NVIDIA's HTTP responses and run without an API key.

## Files

| File | Purpose |
| --- | --- |
| `app.py` | Upload, preview, extraction button, text area, and download |
| `ocr.py` | Image validation, base64 encoding, prompt, and NVIDIA request |
| `requirements.txt` | Pinned Python dependencies |
| `.env.example` | Blank NVIDIA key template |
| `.gitignore` | Excludes local credentials, environments, and caches |
| `tests/test_ocr.py` | Tests for image validation, the NVIDIA payload, errors, and the interface |
| `rapport/rapport.md` | Short technical report in French |

## Request flow

Pillow validates the image and detects its format. A LangChain
`ChatPromptTemplate` combines the OCR instruction and an `image_url` containing
the original image bytes encoded in base64. `ChatNVIDIA` sends the message to
the NVIDIA chat completions endpoint using a 90-second client timeout and
a 4,096-token completion limit.

[NVIDIA Mistral Large 4 API example](https://build.nvidia.com/mistralai/mistral-large-4-1t-a52b)
