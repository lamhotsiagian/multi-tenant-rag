#!/usr/bin/env python3
"""
Setup script for creating sample tenants, users, and documents in NexusRAG.
Run this script after the backend API is up and running.
"""
import requests
import json
import sys
import time

API_BASE_URL = "http://localhost:8000/api/v1"

# We define sample organizations (tenants) along with their initial admin credentials
SAMPLE_ORGANIZATIONS = [
    {
        "organization_name": "Google",
        "subdomain": "google",
        "llm_provider": "gemini",
        "llm_model": "gemini-2.5-pro",
        "admin_email": "admin@google.com",
        "admin_username": "google_admin",
        "admin_password": "securepassword123"
    },
    {
        "organization_name": "TechStart Inc",
        "subdomain": "techstart", 
        "llm_provider": "anthropic",
        "llm_model": "claude-3-haiku",
        "admin_email": "admin@techstart.com",
        "admin_username": "tech_admin",
        "admin_password": "securepassword123"
    }
]

# Standard users to add after the tenant is created
SAMPLE_USERS = {
    "google": {
        "email": "employee@google.com",
        "username": "google_employee",
        "password": "user123",
        "role": "user"
    },
    "techstart": {
        "email": "engineer@techstart.com",
        "username": "tech_eng",
        "password": "user123",
        "role": "user"
    }
}

# Rich sample documents to test vector semantic search
SAMPLE_DOCUMENTS = [
    {
        "filename": "Employee_Handbook_2024.md",
        "title": "2024 Employee Handbook",
        "category": "HR",
        "tags": ["hr", "policy", "handbook"],
        "content": """# Google Employee Handbook 2024

## 1. Introduction
Welcome to Google! Our mission is to organize the world's information and make it universally accessible and useful.

## 2. Remote Work Policy
Employees are allowed to work remotely up to 3 days per week. Core collaboration hours are between 10:00 AM and 3:00 PM (PST), during which all employees must be available.

## 3. Paid Time Off (PTO)
We offer a flexible PTO policy. Employees are encouraged to take at least 3 weeks of vacation per year to recharge. Sick leave is fully covered up to 14 days per calendar year.

## 4. Hardware and Equipment
Every new hire receives a $2,000 stipend to equip their home office, in addition to a standard-issue MacBook Pro (M2 or M3) and noise-canceling headphones.
"""
    },
    {
        "filename": "Q3_Financial_Report.md",
        "title": "Q3 2024 Financial Report",
        "category": "Finance",
        "tags": ["finance", "earnings", "Q3"],
        "content": """# Q3 2024 Financial Performance Report

## Executive Summary
Google achieved a record-breaking Q3 with total revenue hitting $45.2 million, a 22% increase year-over-year.

## Key Metrics
- **Gross Margin**: Improved to 68% (up from 64% in Q2).
- **Customer Acquisition Cost (CAC)**: Decreased by 15% down to $350 per enterprise customer.
- **Monthly Recurring Revenue (MRR)**: Crossed the $10 million milestone in August.

## Future Outlook
Based on the current pipeline, we project Q4 revenue to exceed $50 million, driven by the launch of our new Enterprise AI product suite.
"""
    }
]

class SetupClient:
    """Client for automating the setup of sample data via the unified API"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()
    
    def wait_for_api(self, max_retries: int = 15):
        """Wait for API to be available"""
        for i in range(max_retries):
            try:
                response = self.session.get(f"{self.base_url.replace('/api/v1', '')}/health")
                if response.status_code == 200:
                    print("✅ Backend API is responsive")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            print(f"⏳ Waiting for backend API to initialize... (attempt {i+1}/{max_retries})")
            time.sleep(2)
        
        return False
    
    def register_organization(self, org_data: dict) -> dict:
        """Create a new tenant workspace and admin via /auth/signup"""
        try:
            print(f"🏢 Registering organization: {org_data['organization_name']}...")
            response = self.session.post(f"{self.base_url}/auth/signup", json=org_data)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Created workspace for {org_data['organization_name']} (Tenant ID: {data['tenant']['id']})")
                return data
            else:
                print(f"❌ Failed to create organization {org_data['organization_name']}: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Organization registration failed: {e}")
            return None

    def add_user_to_tenant(self, admin_token: str, tenant_id: str, user_data: dict) -> dict:
        """Add a regular user to a tenant using the admin's token"""
        try:
            headers = {"Authorization": f"Bearer {admin_token}"}
            response = requests.post(
                f"{self.base_url}/auth/register",
                json=user_data,
                params={"tenant_id": tenant_id},
                headers=headers
            )
            
            if response.status_code == 200:
                user = response.json()
                print(f"✅ Added user {user['email']} to workspace.")
                return user
            else:
                print(f"❌ Failed to add user: {response.text}")
                return None
        except Exception as e:
            print(f"❌ User creation failed: {e}")
            return None

    def upload_document(self, token: str, doc_data: dict):
        """Upload a sample document"""
        try:
            headers = {"Authorization": f"Bearer {token}"}
            files = {
                "file": (doc_data["filename"], doc_data["content"].encode('utf-8'), "text/markdown")
            }
            metadata = {
                "title": doc_data["title"],
                "category": doc_data["category"],
                "tags": doc_data["tags"]
            }
            data = {"metadata": json.dumps(metadata)}
            
            response = requests.post(
                f"{self.base_url}/documents/upload", 
                files=files, 
                data=data,
                headers=headers
            )
            
            if response.status_code == 200:
                doc = response.json()
                print(f"✅ Uploaded document: {doc['original_filename']}")
            else:
                print(f"❌ Failed to upload document {doc_data['filename']}: {response.text}")
                
        except Exception as e:
            print(f"❌ Document upload failed: {e}")


def main():
    """Main setup function"""
    print("🚀 Initializing NexusRAG Sample Workspaces")
    print("=" * 50)
    
    client = SetupClient(API_BASE_URL)
    
    if not client.wait_for_api():
        print("❌ Backend API is not available. Please ensure Docker is running.")
        sys.exit(1)
    
    print("\nStarting automated provisioning...")
    
    # 1. Provision each organization
    for org in SAMPLE_ORGANIZATIONS:
        # We catch exceptions locally to not block the second organization if the first fails
        org_result = client.register_organization(org)
        
        if org_result:
            admin_token = org_result["access_token"]
            tenant_id = org_result["tenant"]["id"]
            subdomain = org["subdomain"]
            
            # 2. Add an employee user
            if subdomain in SAMPLE_USERS:
                client.add_user_to_tenant(admin_token, tenant_id, SAMPLE_USERS[subdomain])
            
            # 3. Upload sample knowledge documents for the first tenant to demonstrate RAG
            if subdomain == "google":
                print(f"📄 Uploading knowledge base documents for {org['organization_name']}...")
                for doc in SAMPLE_DOCUMENTS:
                    client.upload_document(admin_token, doc)
    
    print("\n✅ Setup completed successfully!")
    print("\n📋 NexusRAG Access:")
    print(f"   Frontend URL: http://localhost:8501")
    print(f"   API Docs: http://localhost:8000/docs")
    
    print("\n🔑 Credentials for Testing:")
    for org in SAMPLE_ORGANIZATIONS:
        print(f"   Workspace: {org['organization_name']}")
        print(f"   Admin: {org['admin_email']} | Pass: {org['admin_password']}")
        if org['subdomain'] in SAMPLE_USERS:
            u = SAMPLE_USERS[org['subdomain']]
            print(f"   User:  {u['email']} | Pass: {u['password']}")
        print("-" * 30)


if __name__ == "__main__":
    main()