# BearingSense

FastAPI dashboard for the trained bearing-fault CNN in `src/models/bearing_cnn.pth`.

## Run locally

From the `sathya` directory:

```powershell
uv sync
uv run uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000>. The dashboard can analyze any window from
`data/output/X.npy` or an uploaded `.npy`, `.csv`, `.txt`, or `.mat` file.
Uploaded signals must contain exactly 1,024 numeric values.

## API

- `GET /api/health` - model status, labels, and expected window size.
- `GET /api/samples` - dataset sample indexes and source recording names.
- `POST /api/predict` - JSON body with either `{"sample_index": 0}` or
  `{"signal": [1024 numeric values]}`.
- `POST /api/predict-file` - multipart upload for a supported signal file.

Predictions include the winning class, per-class probabilities, signal
statistics, waveform values, and a plain-language explanation.