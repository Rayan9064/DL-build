# Egocentric-10K Video Labeling Prototype

This repository is a runnable prototype for video data labeling. It streams a small sample from `builddotai/Egocentric-10K`, imports the clips into Label Studio, and runs a custom ML backend that auto-labels each clip with a starter action label.

The backend uses TorchVision `r2plus1d_18` with Kinetics-400 pretrained weights. It reads each MP4, samples 16 frames, predicts one Kinetics action, maps that prediction into `Assemble`, `Inspect`, or `Idle`, and returns a Label Studio timeline annotation covering the whole clip.

## Repository Map

```text
.
|-- Dockerfile
|-- docker-compose.yml
|-- .dockerignore
|-- .env.example
|-- README.md
|-- notion.md
|-- requirements.txt
|-- export_egocentric_samples.py
|-- label_config.xml
`-- egocentric_backend/
    |-- __init__.py
    |-- _wsgi.py
    |-- model.py
    `-- server.py
```

## What Each File Does

`Dockerfile` builds the Python 3.11 runtime with Label Studio, PyTorch, TorchVision, Decord, Hugging Face datasets, and the custom backend.

`docker-compose.yml` starts three services:

- `label-studio`: the labeling UI at `http://localhost:8080`
- `ml-backend`: the auto-labeling API at `http://localhost:9090`
- `exporter`: a one-shot utility service that streams a small Hugging Face sample into `egocentric_samples/`

`export_egocentric_samples.py` streams `builddotai/Egocentric-10K` instead of downloading the full dataset. It writes local MP4 clips, metadata JSON files, and `egocentric_samples/label_studio_tasks.json` for import into Label Studio.

`label_config.xml` defines the video timeline labeling interface and the starter labels: `Assemble`, `Inspect`, and `Idle`.

`egocentric_backend/model.py` contains the actual model logic: video loading, frame sampling, TorchVision inference, Kinetics-to-factory-label mapping, and Label Studio prediction formatting.

`egocentric_backend/server.py` exposes the backend over HTTP with `/health`, `/setup`, and `/predict`. This is the recommended local backend server.

`egocentric_backend/_wsgi.py` is kept for compatibility with the Label Studio ML SDK wrapper, but `server.py` is the tested path for this repo.

`notion.md` is a non-technical explanation you can paste into Notion for operations, labeling, or management stakeholders.

## Requirements

For Docker usage:

- Docker Desktop or Docker Engine
- Docker Compose
- Hugging Face account with access accepted for `builddotai/Egocentric-10K`

For local usage without Docker:

- Python 3.10 or 3.11
- This repo was tested with Python `3.11.9`

## Quick Start With Docker

From the repo folder:

```powershell
cd D:\Projects\Data-labelling-egocentric
copy .env.example .env
```

Edit `.env` and set `HF_TOKEN` if Hugging Face requires authentication for the dataset.

Build the Docker image:

```powershell
docker compose build
```

If this fails with a Docker Desktop pipe or engine error, start Docker Desktop first and wait until it says the engine is running. On Windows, the Docker Desktop service may require administrator privileges to start.

Export a small sample of video clips:

```powershell
docker compose --profile tools run --rm exporter
```

Start Label Studio and the ML backend:

```powershell
docker compose up label-studio ml-backend
```

Open:

```text
http://localhost:8080
```

The ML backend will be available at:

```text
http://localhost:9090
```

## Label Studio Setup

1. Open `http://localhost:8080`.
2. Create or log into your Label Studio account.
3. Create a new project.
4. In the labeling setup, paste the contents of `label_config.xml`.
5. Import tasks from:

```text
egocentric_samples/label_studio_tasks.json
```

6. Open project settings, then **Machine Learning**.
7. Add this backend URL:

```text
http://ml-backend:9090
```

Use `http://localhost:9090` only when connecting from your host machine. Inside Docker Compose, Label Studio should use the service name `http://ml-backend:9090`.

## Testing The Docker Package

Check that the backend is alive:

```powershell
curl http://localhost:9090/health
```

Expected response:

```json
{"model_version":"r2plus1d_18_kinetics400_starter","status":"UP","v2":false}
```

Check the running containers:

```powershell
docker compose ps
```

Follow logs:

```powershell
docker compose logs -f ml-backend
docker compose logs -f label-studio
```

Test from Label Studio by opening a task and requesting predictions. The backend should return a full-video timeline prediction under one of the labels in `label_config.xml`.

## Local Setup Without Docker

Create and activate a virtual environment:

```powershell
cd D:\Projects\Data-labelling-egocentric
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Log into Hugging Face if needed:

```powershell
huggingface-cli login
```

Export sample clips:

```powershell
python export_egocentric_samples.py --count 5 --output-dir egocentric_samples --overwrite
```

Start Label Studio:

```powershell
$env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED="true"
$env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=(Get-Location).Path
label-studio start --host 0.0.0.0 --port 8080 --no-browser
```

Start the ML backend in a second terminal:

```powershell
cd D:\Projects\Data-labelling-egocentric
.\venv\Scripts\Activate.ps1
$env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=(Get-Location).Path
python -m egocentric_backend.server --host 127.0.0.1 --port 9090 --preload
```

Connect Label Studio to:

```text
http://127.0.0.1:9090
```

## Testing The Local Backend

Health check:

```powershell
curl http://127.0.0.1:9090/health
```

Python/package check:

```powershell
python -m pip check
python -c "import torch, torchvision, decord; print(torch.__version__, torchvision.__version__, decord.__version__)"
```

The current tested versions are:

```text
Python 3.11.9
torch 2.3.1+cpu
torchvision 0.18.1+cpu
decord 0.6.0
```

## How The Model Works

For every Label Studio task:

1. The task provides a local MP4 path or Label Studio local-files URL.
2. `decord` opens the video.
3. The backend samples 16 frames uniformly across the clip.
4. TorchVision preprocessing resizes/crops/normalizes the frames using the official pretrained weights transform.
5. `r2plus1d_18` predicts a Kinetics-400 class.
6. `sample_label_mapping` maps the Kinetics class into `Assemble`, `Inspect`, or `Idle`.
7. The backend returns a `timelinelabels` annotation covering the full clip duration.

## Customizing Labels

Edit `label_config.xml` first:

```xml
<Label value="Assemble" background="#1f77b4"/>
<Label value="Inspect" background="#ff7f0e"/>
<Label value="Idle" background="#2ca02c"/>
```

Then edit `egocentric_backend/model.py`:

```python
TARGET_LABELS = ("Assemble", "Inspect", "Idle")

sample_label_mapping = {
    "welding": "Assemble",
    "checking": "Inspect",
    "standing": "Idle",
}
```

Keep the labels in `label_config.xml`, `TARGET_LABELS`, and `sample_label_mapping` aligned.

## Data And Volumes

Docker Compose stores:

- Label Studio database/media in the `label_studio_data` Docker volume
- Torch model weights in the `torch_cache` Docker volume
- Hugging Face cache in the `hf_cache` Docker volume
- Exported sample clips on the host in `./egocentric_samples`

To reset Docker state:

```powershell
docker compose down
docker volume rm data-labelling-egocentric_label_studio_data
```

Do not remove `torch_cache` unless you want to re-download the pretrained model weights.

## Known Limitations

This is a starter auto-labeling model, not a fine-tuned factory-action model. Kinetics-400 was trained on general internet action videos, so the first predictions should be treated as suggestions for human review.

The current backend predicts one label for the full clip. For multi-step workflows inside a single video, the next step is to add segment-level inference or fine-tune on corrected Label Studio timeline annotations.

## Fine-Tuning Later

After annotators correct enough clips:

1. Export annotations from Label Studio.
2. Convert timeline labels into training clips or frame windows.
3. Replace or fine-tune the classifier head in `r2plus1d_18`.
4. Update `fit()` in `egocentric_backend/model.py` to train from accepted annotations.
5. Save model checkpoints and load them in `EgocentricActionBackend.__init__`.

## Useful Commands

Build Docker image:

```powershell
docker compose build
```

Export samples:

```powershell
docker compose --profile tools run --rm exporter
```

Run app:

```powershell
docker compose up label-studio ml-backend
```

Stop app:

```powershell
docker compose down
```

Run backend only locally:

```powershell
python -m egocentric_backend.server --host 127.0.0.1 --port 9090 --preload
```
