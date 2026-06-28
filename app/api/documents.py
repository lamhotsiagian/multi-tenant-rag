"""
Document management API routes.

All HTTP concerns (status codes, exception mapping) live here.
Domain exceptions from DocumentService are caught and translated to
the appropriate HTTPException responses.
"""
import json
import structlog
from typing import List, Optional

from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form, BackgroundTasks
from sqlalchemy.orm import Session

from app.schemas.document import (
    DocumentResponse,
    DocumentList,
    DocumentProcessResponse,
    DocumentChunkResponse,
    DocumentSearchRequest,
    DocumentSearchResponse,
)
from app.dependencies import (
    CurrentUserDep,
    CurrentTenantDep,
    DatabaseDep,
    DocumentServiceDep,
    VectorServiceDep,
    EmbeddingServiceDep,
)
from app.models.document import Document, DocumentChunk
from app.services.document_service import (
    DocumentNotFoundError,
    DocumentValidationError,
    DocumentProcessingError,
)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])


# ── Upload ────────────────────────────────────────────────────────────────────

@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    background_tasks: BackgroundTasks,
    current_user: CurrentUserDep,
    current_tenant: CurrentTenantDep,
    db: DatabaseDep,
    document_service: DocumentServiceDep,
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None),
):
    """
    Upload a document (PDF, DOCX, or TXT) for the authenticated tenant.

    The file is stored immediately and text extraction / embedding runs as a
    background task.  Poll ``GET /documents/{id}`` to track processing status.
    """
    parsed_metadata: dict = {}
    if metadata:
        try:
            parsed_metadata = json.loads(metadata)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Metadata must be valid JSON",
            )

    try:
        document = await document_service.upload_document(
            db=db,
            tenant_id=str(current_tenant.id),
            file=file,
            metadata=parsed_metadata,
        )
    except DocumentValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except DocumentProcessingError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        )

    # Kick off processing asynchronously
    background_tasks.add_task(
        document_service.process_document,
        db=db,
        document_id=str(document.id),
        tenant_id=str(current_tenant.id),
    )

    logger.info("Document uploaded", document_id=str(document.id), user=current_user.email)
    return document


# ── List ──────────────────────────────────────────────────────────────────────

@router.get("/", response_model=DocumentList)
async def list_documents(
    current_user: CurrentUserDep,
    current_tenant: CurrentTenantDep,
    db: DatabaseDep,
    document_service: DocumentServiceDep,
    skip: int = 0,
    limit: int = 20,
    status_filter: Optional[str] = None,
):
    """Return a paginated list of documents for the authenticated tenant."""
    documents = document_service.list_documents(
        db=db,
        tenant_id=str(current_tenant.id),
        skip=skip,
        limit=limit,
        status_filter=status_filter,
    )

    total_query = db.query(Document).filter(Document.tenant_id == current_tenant.id)
    if status_filter:
        total_query = total_query.filter(Document.status == status_filter)
    total = total_query.count()

    return DocumentList(
        documents=documents,
        total=total,
        page=skip // limit + 1 if limit else 1,
        size=limit,
        pages=(total + limit - 1) // limit if limit else 1,
    )


# ── Get single ────────────────────────────────────────────────────────────────

@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    current_user: CurrentUserDep,
    current_tenant: CurrentTenantDep,
    db: DatabaseDep,
    document_service: DocumentServiceDep,
):
    """Retrieve metadata for a single document by ID (tenant-scoped)."""
    document = document_service.get_document(
        db=db,
        document_id=document_id,
        tenant_id=str(current_tenant.id),
    )
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return document


# ── Reprocess ─────────────────────────────────────────────────────────────────

@router.post("/{document_id}/process", response_model=DocumentProcessResponse)
async def process_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    current_user: CurrentUserDep,
    current_tenant: CurrentTenantDep,
    db: DatabaseDep,
    document_service: DocumentServiceDep,
    force_reprocess: bool = False,
):
    """Trigger (re-)processing for a document that has already been uploaded."""
    document = document_service.get_document(
        db=db,
        document_id=document_id,
        tenant_id=str(current_tenant.id),
    )
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    if document.status == "processed" and not force_reprocess:
        return DocumentProcessResponse(
            document_id=document.id,
            status="already_processed",
            message="Document already processed. Pass force_reprocess=true to reprocess.",
        )

    background_tasks.add_task(
        document_service.process_document,
        db=db,
        document_id=document_id,
        tenant_id=str(current_tenant.id),
    )

    return DocumentProcessResponse(
        document_id=document.id,
        status="processing",
        message="Document processing started",
    )


# ── Delete ────────────────────────────────────────────────────────────────────

@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    current_user: CurrentUserDep,
    current_tenant: CurrentTenantDep,
    db: DatabaseDep,
    document_service: DocumentServiceDep,
):
    """Permanently delete a document, its vectors, and its file from disk."""
    success = await document_service.delete_document(
        db=db,
        document_id=document_id,
        tenant_id=str(current_tenant.id),
    )
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    logger.info("Document deleted", document_id=document_id, user=current_user.email)
    return {"message": "Document deleted successfully"}


# ── Chunks ────────────────────────────────────────────────────────────────────

@router.get("/{document_id}/chunks", response_model=List[DocumentChunkResponse])
async def get_document_chunks(
    document_id: str,
    current_user: CurrentUserDep,
    current_tenant: CurrentTenantDep,
    db: DatabaseDep,
    document_service: DocumentServiceDep,
    skip: int = 0,
    limit: int = 50,
):
    """Return the text chunks for a processed document."""
    document = document_service.get_document(
        db=db,
        document_id=document_id,
        tenant_id=str(current_tenant.id),
    )
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    chunks = (
        db.query(DocumentChunk)
        .filter(DocumentChunk.document_id == document_id)
        .order_by(DocumentChunk.chunk_index)
        .offset(skip)
        .limit(limit)
        .all()
    )
    return chunks


# ── Vector search ─────────────────────────────────────────────────────────────

@router.post("/search", response_model=DocumentSearchResponse)
async def search_documents(
    search_request: DocumentSearchRequest,
    current_user: CurrentUserDep,
    current_tenant: CurrentTenantDep,
    vector_service: VectorServiceDep,
    embedding_service: EmbeddingServiceDep,
):
    """
    Perform a semantic similarity search against the tenant's document corpus.

    Uses the singleton ``EmbeddingService`` (injected via ``app.state``) — no
    new model is loaded per request.
    """
    try:
        query_embedding = await embedding_service.embed_text(search_request.query)

        search_results = await vector_service.search_documents(
            tenant_id=str(current_tenant.id),
            query_embedding=query_embedding,
            limit=search_request.limit,
            score_threshold=search_request.score_threshold,
            filter_conditions={},
        )

        formatted_results = [
            {
                "chunk_id": r["id"],
                "document_id": r["document_id"],
                "score": r["score"],
                "text": r["text"],
                "source": r["source"],
                "page_number": r.get("page_number"),
                "chunk_index": r["chunk_index"],
                "doc_metadata": r["metadata"],
            }
            for r in search_results
        ]

        return DocumentSearchResponse(
            query=search_request.query,
            results=formatted_results,
            total_found=len(search_results),
            search_time_ms=0.0,
        )

    except Exception as exc:
        logger.error("Document search failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document search failed",
        )