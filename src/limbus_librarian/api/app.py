from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from limbus_librarian import __version__
from limbus_librarian.config import Settings, get_settings
from limbus_librarian.config_loader import list_config_ids, load_named_config
from limbus_librarian.graph import run_ask
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

    @app.get("/v1/health")
    def health():
        return {
            "status": "ok",
            "version": __version__,
            "disclaimer": DISCLAIMER,
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
        return run_ask(
            body.query,
            request.app.state.searcher,
            config,
            api_key=settings.openai_api_key,
            generate_model=settings.generate_model,
            debug=body.debug,
        )

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
