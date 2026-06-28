"""
Comprehensive API endpoint integration tests
"""
import pytest
from fastapi.testclient import TestClient


class TestHealthEndpoints:
    """Test health check endpoints"""
    
    def test_health_check(self, client: TestClient):
        """Test basic health check endpoint"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "app_name" in data
    
    def test_root_endpoint(self, client: TestClient):
        """Test root endpoint"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data


class TestAuthEndpoints:
    """Test authentication and tenant creation endpoints"""
    
    def test_register_user_without_tenant(self, client: TestClient, sample_user_data):
        """Test user registration fails without a valid tenant ID"""
        response = client.post(
            "/api/v1/auth/register",
            json=sample_user_data,
            params={"tenant_id": "nonexistent-tenant-id"}
        )
        
        # Should fail because tenant doesn't exist
        assert response.status_code == 404
    
    def test_login_with_invalid_credentials(self, client: TestClient):
        """Test login fails with invalid credentials"""
        response = client.post("/api/v1/auth/login", json={
            "email": "nonexistent@example.com",
            "password": "wrongpassword"
        })
        
        assert response.status_code == 401
    
    def test_access_protected_endpoint_without_auth(self, client: TestClient):
        """Test accessing protected endpoint fails without authentication"""
        response = client.get("/api/v1/auth/me")
        
        assert response.status_code == 401


class TestDocumentEndpoints:
    """Test document management endpoints with missing auth"""
    
    def test_list_documents_without_auth(self, client: TestClient):
        """Test listing documents without authentication"""
        response = client.get("/api/v1/documents/")
        assert response.status_code == 401
    
    def test_upload_document_without_auth(self, client: TestClient):
        """Test uploading document without authentication"""
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.txt", "test content", "text/plain")}
        )
        assert response.status_code == 401


class TestQueryEndpoints:
    """Test RAG query endpoints with missing auth"""
    
    def test_rag_query_without_auth(self, client: TestClient):
        """Test RAG query without authentication"""
        response = client.post("/api/v1/queries/rag", json={
            "query": "What is the meaning of life?",
            "max_chunks": 5
        })
        assert response.status_code == 401
    
    def test_query_history_without_auth(self, client: TestClient):
        """Test query history without authentication"""
        response = client.get("/api/v1/queries/history")
        assert response.status_code == 401


class TestIntegrationWorkflow:
    """Test complete unified workflow for tenants, users, and uploads"""
    
    def test_complete_organization_workflow(self, client: TestClient):
        """Test complete workflow: create tenant/admin, login, check access"""
        
        # 1. Organization Signup (Creates Tenant + Admin User)
        signup_payload = {
            "organization_name": "Integration Corp",
            "subdomain": "intcorp",
            "llm_provider": "openai",
            "llm_model": "gpt-4",
            "admin_email": "admin@intcorp.com",
            "admin_username": "intadmin",
            "admin_password": "securetestpassword"
        }
        
        response = client.post("/api/v1/auth/signup", json=signup_payload)
        
        # Verify successful creation
        assert response.status_code == 200, f"Signup failed: {response.text}"
        data = response.json()
        assert "access_token" in data
        assert "tenant" in data
        assert data["tenant"]["name"] == "Integration Corp"
        
        token = data["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # 2. Verify protected `/auth/me` route using the token
        me_response = client.get("/api/v1/auth/me", headers=headers)
        assert me_response.status_code == 200
        me_data = me_response.json()
        assert me_data["email"] == "admin@intcorp.com"
        assert me_data["role"] == "admin"
        
        # 3. Test logging in explicitly
        login_response = client.post("/api/v1/auth/login", json={
            "email": "admin@intcorp.com",
            "password": "securetestpassword"
        })
        assert login_response.status_code == 200
        login_data = login_response.json()
        assert "access_token" in login_data
        
        # 4. Access documents endpoint (should be empty for new tenant)
        docs_response = client.get("/api/v1/documents/", headers=headers)
        assert docs_response.status_code == 200
        assert docs_response.json()["documents"] == []