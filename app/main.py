import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.indexing import ensure_record_manager_schema
from app.routers.v1 import router as v1_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
    logger.info("Data directory ensured: %s", settings.data_dir)

    ensure_record_manager_schema()
    logger.info("Application startup complete")

    yield

    logger.info("Application shutdown")


app = FastAPI(title="RAG API Server", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(v1_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
