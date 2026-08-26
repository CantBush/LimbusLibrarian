from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from limbus_librarian import __version__
from limbus_librarian.config import Settings, get_settings
from limbus_librarian.config_loader import list_config_ids, load_named_config
from limbus_librarian.graph import run_ask
from limbus_librarian.llm import LLMError
from limbus_librarian.models import AskAnswer, Chunk
from limbus_librarian.runtime import load_chunks, searcher_from_disk

DISCLAIMER = (
    "Limbus Librarian is an independent fan-made project and is not affiliated "
    "with Project Moon."
)


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    config_id: str | None = None
    debug: bool = False


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    limiter = Limiter(key_func=get_remote_address)
    static_dir = Path(__file__).with_name("static")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.searcher = searcher_from_disk(settings)
        app.state.chunks = {c.chunk_id: c for c in load_chunks(settings.chunks_path)}
        yield

    app = FastAPI(
        title="Limbus Librarian",
        version=__version__,
        description=DISCLAIMER,
        lifespan=lifespan,
    )
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    def dashboard():
        return FileResponse(static_dir / "index.html")

    @app.get("/v1/health")
    def health():
        return {
            "status": "ok",
            "version": __version__,
            "disclaimer": DISCLAIMER,
            "llm_configured": bool(settings.openai_api_key.strip()),
        }

    @app.get("/v1/configs")
    def configs():
        return {"configs": list_config_ids(settings.configs_dir)}

    @app.post("/v1/ask")
    @limiter.limit("30/minute")
    def ask(request: Request, body: AskRequest) -> AskAnswer:
        if not hasattr(request.app.state, "searcher"):
            request.app.state.searcher = searcher_from_disk(settings)
            request.app.state.chunks = {c.chunk_id: c for c in load_chunks(settings.chunks_path)}
        name = body.config_id or settings.default_config
        try:
            config = load_named_config(settings.configs_dir, name)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            return run_ask(
                body.query,
                request.app.state.searcher,
                config,
                api_key=settings.openai_api_key,
                generate_model=settings.generate_model,
                debug=body.debug,
            )
        except LLMError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.get("/v1/sources/{chunk_id}")
    def source(chunk_id: str) -> Chunk:
        chunk = app.state.chunks.get(chunk_id)
        if chunk is None:
            # refresh from disk in case ingest ran after startup
            app.state.chunks = {c.chunk_id: c for c in load_chunks(settings.chunks_path)}
            chunk = app.state.chunks.get(chunk_id)
        if chunk is None:
            raise HTTPException(status_code=404, detail="Unknown chunk_id")
        return chunk

    return app


app = create_app()
