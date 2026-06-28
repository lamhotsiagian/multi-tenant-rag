"""
Application service container — manages singleton instances of heavy services.

All services are initialized ONCE during FastAPI lifespan startup and shared
across all requests via app.state. This prevents the costly pattern of
recreating heavy objects (e.g., loading PyTorch models) on every request.
"""
import structlog
from dataclasses import dataclass, field

from app.services.embedding_service import EmbeddingService
from app.services.vector_service import QdrantVectorService
from app.services.llm_service import LLMService
from app.services.auth_service import AuthService
from app.services.tenant_service import TenantService
from app.services.document_service import DocumentService

logger = structlog.get_logger(__name__)


@dataclass
class ServiceContainer:
    """
    Holds singleton instances of all application services.

    Lifecycle:
        - Created once in the FastAPI lifespan context manager.
        - Stored on ``app.state.container``.
        - Injected into route handlers via FastAPI dependency functions.

    Example::

        container = ServiceContainer.create()
        app.state.container = container
    """

    embedding_service: EmbeddingService
    vector_service: QdrantVectorService
    llm_service: LLMService
    auth_service: AuthService
    tenant_service: TenantService
    document_service: DocumentService

    @classmethod
    async def create(cls) -> "ServiceContainer":
        """
        Asynchronously build and validate all service singletons.

        This is the single authoritative place where services are wired
        together with their dependencies.
        """
        logger.info("Initializing service container")

        embedding_service = EmbeddingService()
        vector_service = QdrantVectorService()

        # Initialize Qdrant collection (creates it if it does not exist)
        await vector_service.init_collection()

        llm_service = LLMService()
        auth_service = AuthService()
        tenant_service = TenantService()

        # DocumentService receives injected dependencies — no internal instantiation
        document_service = DocumentService(
            embedding_service=embedding_service,
            vector_service=vector_service,
        )

        logger.info(
            "Service container ready",
            providers=llm_service.get_available_providers(),
        )

        return cls(
            embedding_service=embedding_service,
            vector_service=vector_service,
            llm_service=llm_service,
            auth_service=auth_service,
            tenant_service=tenant_service,
            document_service=document_service,
        )

    async def health_check(self) -> dict:
        """Return health status of all service dependencies."""
        results: dict = {}

        # Vector store health
        try:
            healthy = await self.vector_service.health_check()
            results["vector_store"] = "healthy" if healthy else "unhealthy"
        except Exception as exc:
            results["vector_store"] = f"unhealthy: {exc}"

        # LLM providers
        try:
            providers = self.llm_service.get_available_providers()
            results["llm_providers"] = f"healthy: {', '.join(providers)}"
        except Exception as exc:
            results["llm_providers"] = f"unhealthy: {exc}"

        return results
