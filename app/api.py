# FastAPI app: upload a mammogram, run preprocessing + all 3 models, return predictions.
import asyncio
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse

# Getting to the project root
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

DEFAULT_WEIGHTS = {
    "a": PROJECT_ROOT / "PrototypeA" / "best_model.weights.h5",
    "b": PROJECT_ROOT / "PrototypeB" / "best_model.weights.h5",
    "c": PROJECT_ROOT / "PrototypeC" / "best_model.weights.h5",
}

# It may be noted that only Prototype C (i.e the ensemble model) uses a saved config for build_model kwargs (e.g. dropout_rate, dense_units).
# Prototype A and B use a fixed build_model() with no config file.

DEFAULT_CONFIG_C = PROJECT_ROOT / "PrototypeC" / "best_model_config.json"
EVAL_PROTOS = [
    ("PrototypeA", "Prototype A (EfficientNetB0)"),
    ("PrototypeB", "Prototype B (ResNet50)"),
    ("PrototypeC", "Prototype C (EfficientNetB0 + ResNet50)"),
]

app = FastAPI(
    title="Mammogram 3-Model Inference",
    description="Upload a mammogram; get predictions from Prototype A (EfficientNetB0), B (ResNet50), and C (EfficientNetB0 + ResNet50).",
)
# One worker runs the whole predict_single i.e (all 3 models run sequentially inside it). max_workers = how many concurrent requests can run inference at once.
# If there are multiple users then we can set max_workers accordingly
_executor = ThreadPoolExecutor(max_workers=1)


def get_weights_paths():
    # This will resolve the weight paths from env or defaults.
    return {
        "a": os.environ.get("WEIGHTS_A", str(DEFAULT_WEIGHTS["a"])),
        "b": os.environ.get("WEIGHTS_B", str(DEFAULT_WEIGHTS["b"])),
        "c": os.environ.get("WEIGHTS_C", str(DEFAULT_WEIGHTS["c"])),
    }


def get_config_c_path():
    p = os.environ.get("CONFIG_C", str(DEFAULT_CONFIG_C))
    return p if os.path.isfile(p) else None


@app.get("/", response_class=HTMLResponse)
async def index():
    # This will serve the upload form.
    html_path = APP_DIR / "templates" / "index.html"
    if not html_path.is_file():
        return HTMLResponse(
            "<h1>Mammogram Inference</h1>"
            "<p>Upload endpoint: <code>POST /predict</code> with form field <code>file</code></p>"
        )
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


def _form_bool(val, default=True):
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    return str(val).lower() in ("true", "1", "on", "yes")


async def _stream_predict_generator(tmp_path, paths, use_tta_b, use_tta_c):
    # This will yield NDJSON: progress events and then the result.
    from app.inference import predict_single
    q = Queue()

    def progress_callback(current, total):
        q.put(("progress", current, total))

    def run():
        r = predict_single(
            tmp_path,
            weights_a=paths["a"],
            weights_b=paths["b"],
            weights_c=paths["c"],
            use_tta_b=use_tta_b,
            use_tta_c=use_tta_c,
            num_tta_c=10,
            config_c_path=get_config_c_path(),
            progress_callback=progress_callback,
        )
        q.put(("result", r))

    loop = asyncio.get_running_loop()
    loop.run_in_executor(_executor, run)

    while True:
        item = await loop.run_in_executor(None, q.get)
        if item[0] == "progress":
            yield json.dumps({"type": "progress", "current": item[1], "total": item[2]}) + "\n"
        elif item[0] == "result":
            yield json.dumps({"type": "result", "data": item[1]}) + "\n"
            break
    if os.path.isfile(tmp_path):
        os.unlink(tmp_path)


@app.post("/predict")
async def predict(
    file: UploadFile = File(..., description="Mammogram image (JPEG/PNG)"),
    use_tta_b: str = Form("false", description="Use TTA for Prototype B"),
    use_tta_c: str = Form("false", description="Use TTA for Prototype C"),
):
    # Upload a mammogram; run preprocessing and all three models. Streams NDJSON: progress then result.
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")
    ext = Path(file.filename).suffix.lower()
    if ext not in (".jpg", ".jpeg", ".png", ".gif", ".bmp"):
        raise HTTPException(status_code=400, detail="Unsupported file type. Use JPEG or PNG.")

    
    
    paths = get_weights_paths()
    for k, p in paths.items():
        if not os.path.isfile(p):
            raise HTTPException(status_code=503, detail=f"Model weights not found: {k} at {p}. Train the prototypes or set WEIGHTS_A/B/C.")

    
    
    
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="File is empty")

    suffix = Path(file.filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    return StreamingResponse(
        _stream_predict_generator(tmp_path, paths, _form_bool(use_tta_b), _form_bool(use_tta_c)),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/evaluation")
async def evaluation():
    # Returns the evaluation metrics from each prototype's evaluation_metrics.json.
    results = {}
    for key, proto in EVAL_PROTOS:
        path = PROJECT_ROOT / key / "evaluation_metrics.json"
        if path.is_file():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["display_name"] = proto
            results[key] = data
    return {"models": results}


@app.get("/prototypes/{proto}/{filename:path}")
async def serve_prototype_file(proto: str, filename: str):
    # Provides the confusion_matrix.png or training_history.png from prototype dirs.
    if proto not in ("PrototypeA", "PrototypeB", "PrototypeC"):
        raise HTTPException(404)
    if filename not in ("confusion_matrix.png", "training_history.png"):
        raise HTTPException(404)
    path = PROJECT_ROOT / proto / filename
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, media_type="image/png")



@app.get("/health")
async def health():
    # This is for Halth check only
    return {"status": "ok"}
