"""
FastAPI dependency injection functions.

Design principles:
  - All *heavy* services (EmbeddingService, LLMService, QdrantVectorService) are
    retrieved from the ``ServiceContainer`` stored on ``app.state``.  They are
    created ONCE at startup — never per-request.
  - Lightweight services (AuthService, TenantService, DocumentService) follow
    the same pattern, keeping construction cost at zero per-request.
  - Auth dependencies form a chain:
      Bearer token → TenantUser → active check → Tenant validation
  - Role-based guards (``require_admin_role``, ``require_user_or_admin_role``)
    are standalone callables usable as ``Depends`` arguments.
"""
from typing import Optional, Annotated

from fastapi import Depends, HTTPException, Request, status, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.tenant import TenantUser, Tenant
from app.services.auth_service import AuthService
from app.services.tenant_service import TenantService
from app.services.document_service import DocumentService
from app.services.vector_service import QdrantVectorService
from app.services.llm_service import LLMService
from app.services.embedding_service import EmbeddingService

# Bearer token extractor — does NOT raise 401 automatically (auto_error=False)
# so we can emit a more descriptive error message ourselves.
_security = HTTPBearer(auto_error=False)


# ── Container-backed service accessors ────────────────────────────────────────

def get_auth_service(request: Request) -> AuthService:
    """Return the AuthService singleton from the app-level container."""
    return request.app.state.container.auth_service


def get_tenant_service(request: Request) -> TenantService:
    """Return the TenantService singleton from the app-level container."""
    return request.app.state.container.tenant_service


def get_document_service(request: Request) -> DocumentService:
    """Return the DocumentService singleton from the app-level container."""
    return request.app.state.container.document_service


def get_vector_service(request: Request) -> QdrantVectorService:
    """Return the QdrantVectorService singleton from the app-level container."""
    return request.app.state.container.vector_service


def get_llm_service(request: Request) -> LLMService:
    """Return the LLMService singleton from the app-level container."""
    return request.app.state.container.llm_service


def get_embedding_service(request: Request) -> EmbeddingService:
    """Return the EmbeddingService singleton from the app-level container."""
    return request.app.state.container.embedding_service


# ── Authentication dependency chain ──────────────────────────────────────────

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_security),
    db: Session = Depends(get_db),
    auth_service: AuthService = Depends(get_auth_service),
) -> TenantUser:
    """
    Decode the Bearer JWT and return the corresponding TenantUser.

    Raises:
        401 Unauthorized: if the token is absent, expired, or invalid.
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        return auth_service.get_user_by_token(db, credentials.credentials)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_active_user(
    current_user: TenantUser = Depends(get_current_user),
) -> TenantUser:
    """
    Extend ``get_current_user`` by verifying the account is not disabled.

    Raises:
        400 Bad Request: if the user account is inactive.
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User account is inactive",
        )
    return current_user


async def get_current_tenant(
    current_user: TenantUser = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    tenant_service: TenantService = Depends(get_tenant_service),
) -> Tenant:
    """
    Resolve and validate the tenant that owns the authenticated user.

    Raises:
        404 Not Found: if the tenant record no longer exists.
        403 Forbidden: if the tenant has been deactivated.
    """
    tenant = tenant_service.get_tenant_by_id(db, str(current_user.tenant_id))
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found",
        )
    if not tenant.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tenant account is inactive",
        )
    return tenant


# ── Public tenant resolution (optional) ──────────────────────────────────────

async def resolve_tenant_from_header(
    x_tenant_id: Optional[str] = Header(None),
    db: Session = Depends(get_db),
    tenant_service: TenantService = Depends(get_tenant_service),
) -> Optional[Tenant]:
    """
    Optionally resolve a tenant from the ``X-Tenant-ID`` header.

    Used by public endpoints (e.g., login) that need tenant context but do
    not require authentication.  Returns ``None`` if the header is absent.
    """
    if not x_tenant_id:
        return None
    tenant = tenant_service.get_tenant_by_identifier(db, x_tenant_id)
    if not tenant or not tenant.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or inactive tenant identifier",
        )
    return tenant


async def resolve_tenant_from_subdomain(
    request: Request,
    db: Session = Depends(get_db),
    tenant_service: TenantService = Depends(get_tenant_service),
) -> Optional[Tenant]:
    """
    Optionally resolve a tenant from the request Host subdomain.

    Expects hosts in the form ``<subdomain>.yourdomain.com``.
    Returns ``None`` when the host does not contain a resolvable subdomain.
    """
    host = request.headers.get("host", "")
    parts = host.split(".")
    if len(parts) < 3:
        return None

    subdomain = parts[0]
    tenant = tenant_service.get_tenant_by_subdomain(db, subdomain)
    if not tenant or not tenant.is_active:
        return None
    return tenant


# ── Role-based access guards ──────────────────────────────────────────────────

def require_admin_role(
    current_user: TenantUser = Depends(get_current_active_user),
) -> TenantUser:
    """
    Guard that requires the caller to have the ``admin`` role.

    Raises:
        403 Forbidden: if the user is not an admin.
    """
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required for this operation",
        )
    return current_user


def require_user_or_admin_role(
    current_user: TenantUser = Depends(get_current_active_user),
) -> TenantUser:
    """
    Guard that requires the caller to have the ``user`` or ``admin`` role.

    Raises:
        403 Forbidden: if the user has only ``viewer`` or an unknown role.
    """
    if current_user.role not in {"user", "admin"}:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User or admin role required for this operation",
        )
    return current_user


# ── Annotated type-alias shortcuts ────────────────────────────────────────────
# Use these in route signatures for cleaner, self-documenting function headers.

AuthServiceDep = Annotated[AuthService, Depends(get_auth_service)]
TenantServiceDep = Annotated[TenantService, Depends(get_tenant_service)]
DocumentServiceDep = Annotated[DocumentService, Depends(get_document_service)]
VectorServiceDep = Annotated[QdrantVectorService, Depends(get_vector_service)]
LLMServiceDep = Annotated[LLMService, Depends(get_llm_service)]
EmbeddingServiceDep = Annotated[EmbeddingService, Depends(get_embedding_service)]

CurrentUserDep = Annotated[TenantUser, Depends(get_current_active_user)]
CurrentTenantDep = Annotated[Tenant, Depends(get_current_tenant)]
AdminUserDep = Annotated[TenantUser, Depends(require_admin_role)]
DatabaseDep = Annotated[Session, Depends(get_db)]