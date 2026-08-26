from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, get_args

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from limbus_librarian import __version__
from limbus_librarian.catalog import CatalogStore
from limbus_librarian.config import Settings, get_settings
from limbus_librarian.config_loader import list_config_ids, load_named_config
from limbus_librarian.graph import run_ask
from limbus_librarian.llm import LLMError
from limbus_librarian.models import AskAnswer, Chunk, DocumentType
from limbus_librarian.runtime import load_chunks, searcher_from_disk

DISCLAIMER = (
    "Limbus Librarian is an independent fan-made project and is not affiliated "
    "with Project Moon."
)


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    config_id: str | None = None
    debug: bool = False
    document_types: list[DocumentType] = Field(default_factory=list, max_length=12)
    max_canto: int | str | None = None
    cantos: list[str] = Field(default_factory=list, max_length=12)
    history: list[Annotated[str, Field(min_length=1, max_length=2000)]] = Field(
        default_factory=list,
        max_length=4,
    )


DOCUMENT_TYPES = set(get_args(DocumentType))


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    limiter = Limiter(key_func=get_remote_address)
    static_dir = Path(__file__).with_name("static")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.searcher = searcher_from_disk(settings)
        app.state.chunks = {c.chunk_id: c for c in load_chunks(settings.chunks_path)}
        app.state.catalog = CatalogStore(settings.catalog_path, settings.documents_path)
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

    @app.post("/v1/reload")
    def reload_indexes(request: Request):
        request.app.state.searcher = searcher_from_disk(settings)
        chunks = load_chunks(settings.chunks_path)
        request.app.state.chunks = {chunk.chunk_id: chunk for chunk in chunks}
        request.app.state.catalog = CatalogStore(
            settings.catalog_path,
            settings.documents_path,
        )
        return {"status": "reloaded", "chunk_count": len(chunks)}

    @app.get("/v1/documents")
    def documents(
        request: Request,
        type: str = Query(default=""),
        q: str = Query(default="", max_length=200),
        canto: str = Query(default="", max_length=40),
        page: int = Query(default=1, ge=1),
        per_page: int = Query(default=24, ge=1, le=100),
    ):
        requested_types = {
            item.strip() for item in type.split(",") if item.strip()
        }
        unknown_types = requested_types - DOCUMENT_TYPES
        if unknown_types:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown document type: {min(unknown_types)}",
            )
        catalog: CatalogStore = request.app.state.catalog
        return catalog.list(
            document_types=requested_types or None,
            query=q,
            canto=canto,
            page=page,
            per_page=per_page,
        )

    @app.get("/v1/documents/{doc_id}")
    def document(request: Request, doc_id: str):
        catalog: CatalogStore = request.app.state.catalog
        item = catalog.get(doc_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Unknown document")
        return item

    @app.get("/v1/documents/{doc_id}/related")
    def related_documents(
        request: Request,
        doc_id: str,
        limit: int = Query(default=12, ge=1, le=50),
    ):
        catalog: CatalogStore = request.app.state.catalog
        if catalog.get(doc_id) is None:
            raise HTTPException(status_code=404, detail="Unknown document")
        return {"items": catalog.related(doc_id, limit=limit)}

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
                filters={
                    "document_types": body.document_types,
                    "cantos": body.cantos,
                    "max_canto": body.max_canto,
                },
                history=body.history,
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
