# PlantCare AI — Leaf Disease Screening Website

A Flask website with a real image-classification model connection, photo upload, top predictions, suggested follow-up steps, and browser-local scan history.

## Important model limitations

The default model is `VaigandlaHemanth/leaf-disease-clip-vit`, a 38-class model trained on the PlantVillage dataset. It does **not** reliably diagnose every plant, every crop, or every field condition. Confidence is not a guarantee of correctness. Treat results as screening suggestions, not a confirmed diagnosis, and ask a local agricultural expert before applying treatments.

The AI model downloads from Hugging Face the first time a scan is requested, so an internet connection is required. Model files can be large and the first scan can take several minutes. If the model repository changes, is unavailable, or your computer/server does not have enough memory, prediction will show an error rather than a fake result.

## Run locally on macOS

Use Python 3.11 for the best compatibility with the pinned ML packages. In VS Code, open this project folder, then run these commands in the terminal:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:8000` in your browser. Keep the terminal running while testing. Stop the server with `Control+C`.

If `python3.11` is not installed, install Python 3.11 first, then repeat the commands. On some Macs, PyTorch/model downloads can take a while and need several GB of free disk space.

## Deploy to Render

1. Create a GitHub repository and upload all files and folders from this project.
2. In Render, create a **Web Service** connected to the repository (not a Static Site; the Python backend is needed for inference).
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn --bind 0.0.0.0:$PORT --timeout 300 app:app`
5. Choose a service plan with enough RAM and disk for PyTorch and the model. A free instance may run out of memory or sleep, so successful deployment is not guaranteed on every free plan.
6. Open the deployed URL and try a clear leaf image. The first prediction may be slow while the model downloads.

## API

- `GET /api/health` — service status
- `POST /api/predict` — multipart form upload with field name `image`

## Privacy

Images are processed by the configured server and are not written to an image archive by this app. Scan metadata is saved only in the current browser's localStorage. For a public deployment, publish a privacy notice and review the model/dataset license and hosting terms before use.
