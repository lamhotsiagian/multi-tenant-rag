"""
Streamlit frontend for Multi-Tenant RAG System
A minimal, ultra-clean enterprise frontend interface for querying and managing multi-tenant document intelligence.
"""
import streamlit as st
import requests
import json
import time
from typing import Dict, List, Optional
from datetime import datetime
import io
import os

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

# Page configuration
st.set_page_config(
    page_title="AI Engineering Insider • Minimal Intelligence",
    page_icon="▪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Minimalist, Clean, Refined Aesthetic (Light Mode)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* CSS Variables - Monochromatic Minimal Palette (Light) */
:root {
    --bg-main: #ffffff;
    --bg-surface: #fafafa;
    --bg-element: #f4f4f5;
    --bg-hover: #e4e4e7;
    --border-subtle: #e4e4e7;
    --border-strong: #d4d4d8;
    --text-primary: #09090b;
    --text-secondary: #52525b;
    --text-muted: #71717a;
    --accent-dark: #09090b;
    --radius-main: 8px;
}

/* Global Typography & Reset */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--text-primary);
    letter-spacing: -0.01em;
}

/* Background */
.stApp {
    background-color: var(--bg-main) !important;
}

/* Layout spacing */
.main > div {
    padding-top: 1.2rem !important;
    padding-bottom: 2rem !important;
    max-width: 1300px;
}

header[data-testid="stHeader"] {
    background: transparent !important;
}

/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 4px;
    height: 4px;
}
::-webkit-scrollbar-track {
    background: var(--bg-main);
}
::-webkit-scrollbar-thumb {
    background: var(--border-strong);
    border-radius: 2px;
}
::-webkit-scrollbar-thumb:hover {
    background: var(--text-muted);
}

/* Minimal Panel Container */
.minimal-panel {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-main);
    padding: 24px;
    box-shadow: none;
}

/* Sidebar styling overhaul */
section[data-testid="stSidebar"] {
    background-color: var(--bg-surface) !important;
    border-right: 1px solid var(--border-subtle) !important;
}

.sidebar-section {
    color: var(--text-muted);
    padding: 14px 0 6px 0;
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.sidebar-card {
    background: var(--bg-main);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-main);
    padding: 12px 14px;
    margin-bottom: 12px;
}

.sidebar-label {
    color: var(--text-muted);
    font-size: 11px;
    font-weight: 500;
    margin-bottom: 2px;
}

.sidebar-value {
    color: var(--text-primary);
    font-weight: 500;
    font-size: 13px;
    margin-bottom: 6px;
    word-break: break-all;
}

.sidebar-value:last-child {
    margin-bottom: 0;
}

.badge-minimal {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 500;
    background: var(--bg-element);
    color: var(--text-secondary);
    border: 1px solid var(--border-strong);
}

/* Navigation Buttons in Sidebar */
.stSidebar .stButton > button {
    width: 100% !important;
    background: transparent !important;
    color: var(--text-secondary) !important;
    border: 1px solid transparent !important;
    border-radius: var(--radius-main) !important;
    padding: 10px 14px !important;
    font-weight: 500 !important;
    font-size: 13.5px !important;
    text-align: left !important;
    transition: all 0.15s ease !important;
    margin-bottom: 4px !important;
}

.stSidebar .stButton > button:hover {
    background: var(--bg-element) !important;
    color: var(--text-primary) !important;
    border-color: var(--border-subtle) !important;
}

/* Minimal Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: var(--bg-surface);
    padding: 4px;
    border-radius: var(--radius-main);
    border: 1px solid var(--border-subtle);
    margin-bottom: 20px;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-muted) !important;
    border-radius: 6px !important;
    border: none !important;
    padding: 8px 20px !important;
    font-weight: 500 !important;
    font-size: 13.5px !important;
    transition: all 0.15s ease !important;
}

.stTabs [aria-selected="true"] {
    background: var(--bg-main) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-subtle) !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
}

/* Input elements styling */
.stTextInput input, .stTextArea textarea, .stSelectbox select, div[data-baseweb="select"] {
    background-color: var(--bg-main) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: var(--radius-main) !important;
    color: var(--text-primary) !important;
    padding: 10px 14px !important;
    font-size: 13.5px !important;
    transition: all 0.15s ease !important;
}

.stTextInput input:focus, .stTextArea textarea:focus, div[data-baseweb="select"]:focus-within {
    border-color: var(--text-primary) !important;
    box-shadow: none !important;
}

/* Buttons */
.stButton > button {
    border-radius: var(--radius-main);
    font-weight: 500;
    font-size: 13.5px;
    transition: all 0.15s ease;
}

button[kind="primary"] {
    background: var(--accent-dark) !important;
    color: #ffffff !important;
    border: 1px solid var(--accent-dark) !important;
}

button[kind="primary"]:hover {
    background: #27272a !important;
    border-color: #27272a !important;
}

button[kind="secondary"], .stButton > button:not([kind="primary"]) {
    background: var(--bg-main) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-strong) !important;
}

button[kind="secondary"]:hover, .stButton > button:not([kind="primary"]):hover {
    background: var(--bg-surface) !important;
    border-color: var(--text-secondary) !important;
}

/* Minimal Chat Styling */
.chat-bubble-user {
    background: var(--bg-surface);
    border: 1px solid var(--border-strong);
    color: var(--text-primary);
    padding: 14px 18px;
    border-radius: 12px 12px 2px 12px;
    margin: 10px 0 10px auto;
    max-width: 78%;
    font-size: 14px;
    line-height: 1.5;
}

.chat-bubble-assistant {
    background: var(--bg-main);
    border: 1px solid var(--border-subtle);
    color: var(--text-primary);
    padding: 16px 20px;
    border-radius: 12px 12px 12px 2px;
    margin: 10px auto 10px 0;
    max-width: 86%;
    font-size: 14px;
    line-height: 1.5;
}

.chat-meta {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px solid var(--border-subtle);
    font-size: 11px;
    color: var(--text-muted);
}

.chat-chip {
    background: var(--bg-surface);
    padding: 2px 6px;
    border-radius: 4px;
    border: 1px solid var(--border-subtle);
}

.source-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-left: 2px solid var(--text-secondary);
    border-radius: 4px;
    padding: 10px 14px;
    margin-top: 8px;
    font-size: 12.5px;
}

.source-header {
    font-weight: 500;
    color: var(--text-primary);
    display: flex;
    justify-content: space-between;
    margin-bottom: 4px;
}

/* Metric KPI Tile */
.kpi-card {
    background: var(--bg-main);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-main);
    padding: 16px;
    text-align: center;
}

.kpi-title {
    color: var(--text-muted);
    font-size: 11px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
}

.kpi-value {
    color: var(--text-primary);
    font-size: 24px;
    font-weight: 600;
}

/* Minimal Status Badges */
.status-processed {
    background: #f0fdf4;
    color: #166534;
    border: 1px solid #bbf7d0;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 500;
}

.status-processing {
    background: #fefce8;
    color: #854d0e;
    border: 1px solid #fef08a;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 500;
}

.status-failed {
    background: #fef2f2;
    color: #991b1b;
    border: 1px solid #fecaca;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 500;
}

/* Header Banner */
.hero-header {
    margin-bottom: 1.5rem;
    text-align: left;
    border-bottom: 1px solid var(--border-subtle);
    padding-bottom: 1rem;
}

.hero-title {
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 0.2rem;
    letter-spacing: -0.02em;
}

.hero-subtitle {
    color: var(--text-muted);
    font-size: 0.875rem;
    font-weight: 400;
}
</style>
""", unsafe_allow_html=True)


class APIClient:
    """API client for the Multi-Tenant RAG system"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()
    
    def set_auth_token(self, token: str):
        """Set authentication bearer token"""
        self.session.headers.update({"Authorization": f"Bearer {token}"})
    
    def login(self, email: str, password: str, tenant_identifier: str = None) -> Dict:
        """Login user"""
        data = {
            "email": email,
            "password": password,
            "tenant_identifier": tenant_identifier
        }
        response = self.session.post(f"{self.base_url}/auth/login", json=data)
        response.raise_for_status()
        return response.json()
    
    def signup(self, organization_name: str, admin_email: str, admin_username: str, 
               admin_password: str, subdomain: str = None, llm_provider: str = "openai", 
               llm_model: str = "gpt-3.5-turbo") -> Dict:
        """Sign up for new organization"""
        data = {
            "organization_name": organization_name,
            "admin_email": admin_email,
            "admin_username": admin_username,
            "admin_password": admin_password,
            "subdomain": subdomain,
            "llm_provider": llm_provider,
            "llm_model": llm_model
        }
        response = self.session.post(f"{self.base_url}/auth/signup", json=data)
        response.raise_for_status()
        return response.json()
    
    def upload_document(self, file_data: bytes, filename: str, metadata: Dict = None) -> Dict:
        """Upload document"""
        files = {"file": (filename, io.BytesIO(file_data), "application/octet-stream")}
        data = {}
        if metadata:
            data["metadata"] = json.dumps(metadata)
        
        response = self.session.post(f"{self.base_url}/documents/upload", files=files, data=data)
        response.raise_for_status()
        return response.json()
    
    def list_documents(self, skip: int = 0, limit: int = 20) -> Dict:
        """List documents"""
        params = {"skip": skip, "limit": limit}
        response = self.session.get(f"{self.base_url}/documents/", params=params)
        response.raise_for_status()
        return response.json()
    
    def rag_query(self, query: str, max_chunks: int = 5, stream: bool = False, **kwargs) -> Dict:
        """Submit RAG query"""
        data = {
            "query": query,
            "max_chunks": max_chunks,
            "stream": stream,
            **kwargs
        }
        
        if stream:
            response = self.session.post(f"{self.base_url}/queries/rag/stream", json=data, stream=True)
        else:
            response = self.session.post(f"{self.base_url}/queries/rag", json=data)
        
        response.raise_for_status()
        return response.json() if not stream else response
    
    def debug_vector_status(self) -> Dict:
        """Check vector store status (debug)"""
        response = self.session.get(f"{self.base_url}/queries/debug/vector-status")
        response.raise_for_status()
        return response.json()
    
    def get_query_history(self, skip: int = 0, limit: int = 10) -> Dict:
        """Get query history"""
        params = {"skip": skip, "limit": limit}
        response = self.session.get(f"{self.base_url}/queries/history", params=params)
        response.raise_for_status()
        return response.json()
    
    def get_tenant_info(self) -> Dict:
        """Get current tenant info"""
        response = self.session.get(f"{self.base_url}/tenant/info")
        response.raise_for_status()
        return response.json()
    
    def delete_document(self, document_id: str) -> Dict:
        """Delete document by ID"""
        try:
            response = self.session.delete(f"{self.base_url}/documents/{document_id}")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                raise Exception("Document not found")
            elif e.response.status_code == 403:
                raise Exception("Permission denied")
            else:
                raise Exception(f"HTTP {e.response.status_code}: {e.response.text}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Connection error: {str(e)}")


def initialize_session_state():
    """Initialize Streamlit session state"""
    if "api_client" not in st.session_state:
        st.session_state.api_client = APIClient(API_BASE_URL)
    
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    if "user_info" not in st.session_state:
        st.session_state.user_info = None
    
    if "tenant_info" not in st.session_state:
        st.session_state.tenant_info = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None


def login_form():
    """Display minimal clean login and signup form"""
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">NexusRAG</div>
        <div class="hero-subtitle">Multi-Tenant Document Intelligence Engine</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="minimal-panel">', unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["Sign In", "Register Workspace"])
        
        with tab1:
            with st.form("login_form", clear_on_submit=False):
                email = st.text_input("Email", placeholder="name@company.com")
                password = st.text_input("Password", type="password", placeholder="••••••••")
                tenant_identifier = st.text_input(
                    "Workspace Identifier (Optional)", 
                    placeholder="subdomain or tenant-id"
                )
                
                st.markdown("<br>", unsafe_allow_html=True)
                submit = st.form_submit_button("Sign In", use_container_width=True, type="primary")
                
                if submit:
                    if not email or not password:
                        st.error("Please enter both email and password")
                        return
                    
                    try:
                        with st.spinner("Authenticating..."):
                            response = st.session_state.api_client.login(
                                email=email,
                                password=password,
                                tenant_identifier=tenant_identifier or None
                            )
                        
                        st.session_state.api_client.set_auth_token(response["access_token"])
                        st.session_state.authenticated = True
                        st.session_state.user_info = response["user"]
                        st.session_state.tenant_info = response["tenant"]
                        
                        st.rerun()
                        
                    except requests.exceptions.RequestException as e:
                        if hasattr(e, 'response') and e.response is not None:
                            error_detail = e.response.json().get("detail", "Login failed")
                            st.error(f"Authentication failed: {error_detail}")
                        else:
                            st.error("Unable to connect to backend API.")
                    except Exception as e:
                        st.error(f"Login error: {str(e)}")
        
        with tab2:
            with st.container():
                organization_name = st.text_input("Organization Name *", placeholder="Google")
                subdomain = st.text_input("Subdomain (Optional)", placeholder="google")
                
                admin_email = st.text_input("Admin Email *", placeholder="admin@google.com")
                admin_username = st.text_input("Admin Username *", placeholder="admin")
                
                c_pass1, c_pass2 = st.columns(2)
                with c_pass1:
                    admin_password = st.text_input("Password *", type="password", placeholder="••••••••")
                with c_pass2:
                    confirm_password = st.text_input("Confirm Password *", type="password", placeholder="••••••••")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    llm_provider = st.selectbox("LLM Provider", ["openai", "anthropic", "gemini"])
                with col_b:
                    if llm_provider == "openai":
                        llm_model = st.selectbox("Model", ["gpt-4.1-mini", "gpt-4.1-nano", "gpt-4.1"])
                    elif llm_provider == "anthropic":
                        llm_model = st.selectbox("Model", ["claude-3-haiku", "claude-3-sonnet", "claude-3-opus"])
                    else:
                        llm_model = st.selectbox("Model", ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro", "gemini-3.5-flash"])
                
                st.markdown("<br>", unsafe_allow_html=True)
                submit_signup = st.button("Create Workspace", use_container_width=True, type="primary")
                
                if submit_signup:
                    if not organization_name or not admin_email or not admin_username or not admin_password:
                        st.error("Please fill required fields (*)")
                        return
                    
                    if len(admin_password) < 8 or admin_password != confirm_password:
                        st.error("Invalid password or passwords do not match.")
                        return
                    
                    try:
                        with st.spinner("Creating organization workspace..."):
                            response = st.session_state.api_client.signup(
                                organization_name=organization_name,
                                admin_email=admin_email,
                                admin_username=admin_username,
                                admin_password=admin_password,
                                subdomain=subdomain if subdomain else None,
                                llm_provider=llm_provider,
                                llm_model=llm_model
                            )
                        
                        st.session_state.api_client.set_auth_token(response["access_token"])
                        st.session_state.authenticated = True
                        st.session_state.user_info = response["admin_user"]
                        st.session_state.tenant_info = response["tenant"]
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Registration error: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)


def sidebar():
    """Display minimal sidebar navigation"""
    with st.sidebar:
        if st.session_state.authenticated:
            if 'current_page' not in st.session_state:
                st.session_state.current_page = "Chat"
            
            # Minimal header
            st.markdown("""
            <div style="padding:4px 0 12px 0; border-bottom:1px solid var(--border-subtle); margin-bottom:12px;">
                <div style="font-weight:600; font-size:15px; color:var(--text-primary);">NexusRAG</div>
            </div>
            """, unsafe_allow_html=True)
            
            nav_items = [
                ("Chat", "Chat Assistant"),
                ("Documents", "Documents"),
                ("History", "Logs & Metrics")
            ]
            
            for key, label in nav_items:
                is_active = st.session_state.current_page == key
                btn_style = "primary" if is_active else "secondary"
                
                if st.button(label, key=f"nav_{key}", use_container_width=True, type=btn_style):
                    st.session_state.current_page = key
                    st.rerun()
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Session info
            user = st.session_state.user_info
            tenant = st.session_state.tenant_info
            
            st.markdown(f"""
            <div class="sidebar-card">
                <div class="sidebar-label">Account</div>
                <div class="sidebar-value">{user['email']}</div>
                <div class="sidebar-label" style="margin-top:8px;">Workspace</div>
                <div class="sidebar-value">{tenant['name']} ({tenant['llm_provider'].upper()})</div>
            </div>
            """, unsafe_allow_html=True)
            
            return st.session_state.current_page
        
        return "Chat"


def chat_interface():
    """Minimal Chat Interface"""
    st.markdown("""
    <div style="margin-bottom:16px;">
        <h3 style="margin:0; font-weight:600; font-size:1.3rem; color:var(--text-primary);">Chat Assistant</h3>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("Settings & Hyperparameters"):
        col1, col2, col3 = st.columns(3)
        with col1:
            max_chunks = st.slider("Max Context Chunks", 1, 10, 5)
            temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
        with col2:
            score_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.25, 0.05)
            max_tokens = st.slider("Max Response Tokens", 100, 2000, 1000, 100)
        with col3:
            include_sources = st.checkbox("Include Sources", value=True)
            stream_response = st.checkbox("Stream Response", value=False)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if not st.session_state.chat_history:
        st.markdown("""
        <div class="minimal-panel" style="text-align:center; padding:30px 20px; margin-bottom:20px;">
            <div style="font-size:14px; color:var(--text-secondary); margin-bottom:16px;">Query documents or ask questions across your knowledge base.</div>
        </div>
        """, unsafe_allow_html=True)
        
        sp_col1, sp_col2, sp_col3 = st.columns(3)
        with sp_col1:
            if st.button("Summarize documents", use_container_width=True):
                st.session_state.pending_prompt = "Summarize key findings in uploaded documents."
                st.rerun()
        with sp_col2:
            if st.button("List key metrics", use_container_width=True):
                st.session_state.pending_prompt = "What are the key metrics and figures?"
                st.rerun()
        with sp_col3:
            if st.button("Compliance requirements", use_container_width=True):
                st.session_state.pending_prompt = "What are the compliance or procedural requirements?"
                st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)

    # Render Messages
    for msg in st.session_state.chat_history:
        if msg["type"] == "user":
            st.markdown(f"""
            <div class="chat-bubble-user">
                <div style="font-weight:600; font-size:11px; color:var(--text-muted); margin-bottom:4px; text-transform:uppercase;">User</div>
                {msg['content']}
            </div>
            """, unsafe_allow_html=True)
        else:
            meta_chips = []
            if "processing_time" in msg:
                meta_chips.append(f'<span class="chat-chip">{msg["processing_time"]:.0f}ms</span>')
            if "tokens_used" in msg:
                meta_chips.append(f'<span class="chat-chip">{msg["tokens_used"]} tokens</span>')
            
            chips_html = " ".join(meta_chips)
            
            st.markdown(f"""
            <div class="chat-bubble-assistant">
                <div style="font-weight:600; font-size:11px; color:var(--text-muted); margin-bottom:6px; text-transform:uppercase;">Assistant</div>
                <div>{msg['content']}</div>
                <div class="chat-meta">
                    {chips_html}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if "context_documents" in msg and msg["context_documents"] and include_sources:
                with st.expander(f"Sources ({len(msg['context_documents'])} retrieved)"):
                    for j, doc in enumerate(msg["context_documents"], 1):
                        score = doc.get('score', 0.0)
                        score_percent = f"{score * 100:.1f}%" if isinstance(score, float) else str(score)
                        st.markdown(f"""
                        <div class="source-card">
                            <div class="source-header">
                                <span>{doc.get('source', 'Document')}</span>
                                <span class="badge-minimal">{score_percent}</span>
                            </div>
                            <div style="color:var(--text-secondary); font-size:12px; margin-top:4px;">
                                "{doc['text'][:300]}{'...' if len(doc['text']) > 300 else ''}"
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    default_text = ""
    if st.session_state.pending_prompt:
        default_text = st.session_state.pending_prompt
        st.session_state.pending_prompt = None

    with st.form("chat_input_form", clear_on_submit=True):
        query_input = st.text_area(
            "Question Input",
            value=default_text,
            placeholder="Ask a question...",
            height=90,
            label_visibility="collapsed"
        )
        
        c1, c2 = st.columns([4, 1])
        with c1:
            submit_query = st.form_submit_button("Submit", type="primary", use_container_width=True)
        with c2:
            clear_chat = st.form_submit_button("Clear", use_container_width=True)
            
    if clear_chat:
        st.session_state.chat_history = []
        st.rerun()
        
    if submit_query and query_input.strip():
        user_text = query_input.strip()
        st.session_state.chat_history.append({
            "type": "user",
            "content": user_text,
            "timestamp": datetime.now()
        })
        
        try:
            with st.spinner("Generating answer..."):
                response = st.session_state.api_client.rag_query(
                    query=user_text,
                    max_chunks=max_chunks,
                    score_threshold=score_threshold,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    include_sources=include_sources,
                    stream=stream_response
                )
            
            assistant_msg = {
                "type": "assistant",
                "content": response["response"],
                "timestamp": datetime.now(),
                "processing_time": response.get("processing_time_ms", 0),
                "context_documents": response.get("context_documents", []),
                "tokens_used": response.get("total_tokens", 0)
            }
            
            st.session_state.chat_history.append(assistant_msg)
            st.rerun()
            
        except Exception as e:
            st.error(f"Execution error: {str(e)}")


def document_management():
    """Document Library Interface"""
    st.markdown("""
    <div style="margin-bottom:16px;">
        <h3 style="margin:0; font-weight:600; font-size:1.3rem; color:var(--text-primary);">Document Library</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="minimal-panel" style="margin-bottom:20px;">', unsafe_allow_html=True)
    st.markdown("<div style='font-size:13px; font-weight:500; margin-bottom:8px;'>Upload Document</div>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose file",
        type=['pdf', 'txt', 'docx'],
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("Metadata (Optional)", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                title = st.text_input("Title", value=uploaded_file.name)
                tags = st.text_input("Tags", placeholder="report, 2024")
            with col2:
                category = st.selectbox("Category", ["General", "Finance", "Legal", "Technical", "HR"])
                description = st.text_area("Description", placeholder="Summary...", height=70)
        
        if st.button("Upload & Index", type="primary", use_container_width=True):
            try:
                metadata = {
                    "title": title,
                    "category": category,
                    "description": description,
                    "tags": [tag.strip() for tag in tags.split(",") if tag.strip()]
                }
                
                with st.spinner("Processing document..."):
                    file_data = uploaded_file.read()
                    response = st.session_state.api_client.upload_document(
                        file_data=file_data,
                        filename=uploaded_file.name,
                        metadata=metadata
                    )
                
                st.success("Document uploaded successfully.")
                time.sleep(0.5)
                st.rerun()
                
            except Exception as e:
                st.error(f"Upload error: {str(e)}")
                
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Inventory List
    ctrl_col1, ctrl_col2 = st.columns([3, 1])
    with ctrl_col1:
        status_filter = st.selectbox(
            "Filter Status",
            ["All", "pending", "processing", "processed", "failed"],
            label_visibility="collapsed"
        )
    with ctrl_col2:
        if st.button("Refresh", use_container_width=True):
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    try:
        with st.spinner("Loading documents..."):
            docs_response = st.session_state.api_client.list_documents()
        
        documents = docs_response.get("documents", [])
        if status_filter != "All":
            documents = [doc for doc in documents if doc['status'] == status_filter]
            
        if not documents:
            st.info("No documents found.")
        else:
            for doc in documents:
                file_ext = doc['original_filename'].split('.')[-1].upper() if '.' in doc['original_filename'] else 'FILE'
                
                with st.expander(f"{doc['original_filename']} • {doc['status']}"):
                    d_col1, d_col2, d_col3 = st.columns([2.5, 2, 1])
                    
                    with d_col1:
                        st.markdown(f"**ID:** `{doc['id']}`")
                        st.markdown(f"**Format:** `{file_ext}`")
                        st.markdown(f"**Size:** `{doc['file_size']:,} bytes`")
                    
                    with d_col2:
                        st.markdown(f"**Uploaded:** `{doc['uploaded_at'][:19]}`")
                        st.markdown(f"**Chunks:** `{doc['processed_chunks']} / {doc['total_chunks']}`")
                    
                    with d_col3:
                        st.markdown("<br>", unsafe_allow_html=True)
                        if st.button("Delete", key=f"del_{doc['id']}", use_container_width=True):
                            try:
                                st.session_state.api_client.delete_document(doc['id'])
                                st.success("Deleted.")
                                time.sleep(0.5)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Delete error: {e}")
                                
    except Exception as e:
        st.error(f"Loading error: {str(e)}")


def query_history():
    """Minimal Query History Interface"""
    st.markdown("""
    <div style="margin-bottom:16px;">
        <h3 style="margin:0; font-weight:600; font-size:1.3rem; color:var(--text-primary);">Logs & Metrics</h3>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        with st.spinner("Fetching logs..."):
            history_response = st.session_state.api_client.get_query_history(limit=25)
            
        queries = history_response.get("queries", [])
        
        if not queries:
            st.info("No query logs available.")
            return
            
        total_queries = len(queries)
        avg_latency = sum(q.get("processing_time_ms", 0) for q in queries) / total_queries if total_queries > 0 else 0
        total_tokens = sum(q.get("total_tokens", 0) for q in queries)
        successful = sum(1 for q in queries if q.get("status") == "completed")
        success_rate = (successful / total_queries * 100) if total_queries > 0 else 100
        
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Total Queries</div>
                <div class="kpi-value">{total_queries}</div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Avg Latency</div>
                <div class="kpi-value">{avg_latency:.0f}ms</div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Tokens</div>
                <div class="kpi-value">{total_tokens:,}</div>
            </div>
            """, unsafe_allow_html=True)
        with k4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Success Rate</div>
                <div class="kpi-value">{success_rate:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        for q in queries:
            query_text = q.get('query_text', '')
            created_at = q.get('created_at', '')[:19]
            
            with st.expander(f"{query_text[:80]}... ({created_at})"):
                qc1, qc2 = st.columns(2)
                with qc1:
                    st.markdown(f"**Status:** `{q.get('status')}`")
                    st.markdown(f"**Latency:** `{q.get('processing_time_ms', 0):.0f}ms`")
                with qc2:
                    st.markdown(f"**Provider:** `{q.get('llm_provider', 'N/A')}`")
                    st.markdown(f"**Tokens:** `{q.get('total_tokens', 0):,}`")
                
                if q.get("response"):
                    resp_text = q["response"].get("response_text", "")
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("**Response Preview:**")
                    st.markdown(f"""
                    <div style="background:var(--bg-element); padding:12px; border-radius:6px; border:1px solid var(--border-subtle); font-size:13px;">
                        {resp_text[:350]}{'...' if len(resp_text) > 350 else ''}
                    </div>
                    """, unsafe_allow_html=True)
                    
    except Exception as e:
        st.error(f"Metrics error: {str(e)}")


def logout():
    """Logout current user session"""
    st.session_state.authenticated = False
    st.session_state.user_info = None
    st.session_state.tenant_info = None
    st.session_state.chat_history = []
    st.session_state.api_client = APIClient(API_BASE_URL)
    st.rerun()


def main():
    """Main Application Entry Point"""
    initialize_session_state()
    
    if not st.session_state.authenticated:
        login_form()
        return
    
    page = sidebar()
    
    with st.sidebar:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Sign Out", use_container_width=True):
            logout()
    
    if page == "Chat":
        chat_interface()
    elif page == "Documents":
        document_management()
    elif page == "History":
        query_history()


if __name__ == "__main__":
    main()