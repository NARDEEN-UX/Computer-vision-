"""FastAPI entry point for the current Computer Vision scope."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes.filters import router as filters_router
from .api.routes.noise import router as noise_router
from .api.routes.edge import router as edge_router
from .api.routes.histogram import router as histogram_router
from .api.routes.transform import router as transform_router

app = FastAPI(title="Computer Vision Task 1 API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(noise_router)
app.include_router(filters_router)
app.include_router(edge_router)
app.include_router(histogram_router)
app.include_router(transform_router)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
