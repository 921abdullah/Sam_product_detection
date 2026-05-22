from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes.drawer_change import router as drawer_change_router
from app.core.config import PipelineConfig
from app.core.model_registry import ModelRegistry
from app.pipelines.drawer_change_pipeline import DrawerChangePipeline
from app.services.alignment_service import AlignmentService
from app.services.rendering_service import RenderingService
from app.services.segmentation_service import SegmentationService
from app.services.verification_service import VerificationService
from app.utils.file_utils import ensure_dir
from fastapi.middleware.cors import CORSMiddleware

APP_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = APP_ROOT / "outputs"
TEMP_DIR = APP_ROOT / "temp"

def _build_pipeline(models: ModelRegistry) -> DrawerChangePipeline:
    alignment_service = AlignmentService(
        extractor=models.extractor,
        matcher=models.matcher,
        device=models.device,
    )
    segmentation_service = SegmentationService(
        predictor=models.sam_predictor,
        predictor_lock=models.sam_lock,
    )
    rendering_service = RenderingService()
    verification_service = VerificationService(
        extractor=models.extractor,
        matcher=models.matcher,
        device=models.device,
    )

    return DrawerChangePipeline(
        alignment_service=alignment_service,
        segmentation_service=segmentation_service,
        rendering_service=rendering_service,
        dino_service=models.dino_service,
        verification_service=verification_service,
        output_dir=OUTPUT_DIR,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_dir(OUTPUT_DIR)
    ensure_dir(TEMP_DIR)

    models = ModelRegistry()
    models.load_models(PipelineConfig())

    app.state.models = models
    app.state.pipeline = _build_pipeline(models)
    app.state.outputs_root = OUTPUT_DIR
    app.state.temp_dir = TEMP_DIR

    yield


app = FastAPI(
    title="Drawer Change Detection API",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(drawer_change_router)
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}
