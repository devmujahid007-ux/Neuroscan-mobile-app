# NeuroScan Mobile App

NeuroScan is a research/demo application combining a Flutter front-end (`f_a_det_app`) with a Python backend (`backend/`) for medical image analysis (MRI) and report generation. This README covers quick setup, running locally, and useful scripts.

## Quick overview

- **Frontend:** `f_a_det_app` — Flutter app for mobile and web.
- **Backend:** `backend/` — API server and ML inference code.
- **Data & models:** `data/`, `uploads/`, `models/` contain scans, results, and pretrained models.
- **Helpers:** root-level scripts like `run_backend.ps1`, `run_flutter_web.ps1`, and `docker-compose.yml` simplify common tasks.

## Prerequisites

- Git
- Python 3.10+ and pip
- Flutter SDK (stable)
- Chrome (for Flutter web)
- (Optional) Docker & Docker Compose

## Repository layout (important paths)

- `backend/` — FastAPI app, ML code, DB helpers
- `f_a_det_app/` — Flutter project (app code under `lib/`)
- `data/`, `uploads/`, `models/` — dataset, stored results, pretrained models
- `docker-compose.yml` — container orchestration
- Helper scripts: `run_backend.ps1`, `run_flutter_web.ps1`, `run_project.ps1`, `Run NeuroScan AI.bat`

## Backend — Local setup and run

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r backend/requirements.txt
```

3. Initialize database (if needed):

```powershell
python backend/create_tables.py
```

4. Run the backend API (example):

```powershell
# from repo root
python backend/main.py
# or using uvicorn if available:
# uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Notes:
- Use `run_backend.ps1` or `setup_and_run_backend.ps1` to automate setup.

## Frontend — Run Flutter web (Chrome)

There are helper tasks and scripts included. The Flutter web run requires the backend API URL to be supplied via `dart-define`.

Example (direct):

```powershell
cd f_a_det_app
flutter run -d chrome --no-web-resources-cdn --dart-define=NEUROSCAN_API_URL=http://127.0.0.1:8000
```

Or use the provided VS Code task: "Flutter web: Chrome (no Google CDN)" which uses `--dart-define=NEUROSCAN_API_URL=http://192.168.1.50:8000` by default. Update the URL to match your backend host.

## Run everything with Docker

If you prefer containers, use the provided compose file:

```powershell
docker-compose up --build
```

Check `docker-compose.yml` for service names and exposed ports.

## Configuration

- Frontend: pass the backend API via `NEUROSCAN_API_URL` dart-define at run/build time.
- Backend: configuration (DB, secrets, model paths) lives in `backend/` modules and can be set via environment variables or the helper scripts.

## Troubleshooting

- Run `flutter doctor` if Flutter reports SDK/tooling problems.
- If backend complains about DB missing tables, run `python backend/create_tables.py`.
- Ensure `data/` and `uploads/` folders are writable by the backend process.

## Contributing

1. Fork, create a branch, open a PR.
2. Include tests for significant backend logic where appropriate.

## License

No license file is included. Add a `LICENSE` if you intend to publish.

---

If you want, I can also:

- add an `.env.example` and document env vars
- enhance `docker-compose.yml` notes with concrete ports
- add a one-command quickstart script that launches backend and frontend

