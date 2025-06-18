# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from backend.api.routes import upload, documents, graph, query
# from backend.api.routes import intelligent

# app = FastAPI(title="Pharma RAG API", version="1.0.0")
# app.include_router(intelligent.router, prefix="/api/intelligent", tags=["Intelligent Graph"])
# # Add CORS middleware for Streamlit frontend
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # In production, specify your Streamlit URL
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Include routers with proper prefixes
# app.include_router(upload.router, prefix="/api/upload", tags=["Upload"])
# app.include_router(documents.router, prefix="/api", tags=["Documents"])  # Changed prefix
# app.include_router(graph.router, prefix="/api/graph", tags=["Graph"])
# app.include_router(query.router, prefix="/api/query", tags=["Query"])  # Added query router

# # Root endpoint
# @app.get("/")
# async def root():
#     return {"message": "Pharma RAG API is running"}

# # Health check endpoint
# @app.get("/health")
# async def health_check():
#     return {"status": "healthy"}



# Code from GOOGLE ><><><><><><<><><><><><><><><><><><><<><><<<
# backend/main.py

from dotenv import load_dotenv
load_dotenv() # Load .env file at the earliest point
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.routes import upload, documents, graph # query We'll review query.py
from backend.api.routes import intelligent # This is our new advanced querying

app = FastAPI(
    title="Pharma RAG API with Knowledge Graph", # Updated title
    version="1.1.0" # Incremented version
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your Streamlit app's URL (e.g., "http://localhost:8501")
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"], # Be specific if possible
    allow_headers=["*"], # Or specify allowed headers
)

# Include routers
# The intelligent router now provides advanced querying and its own health check
app.include_router(intelligent.router, prefix="/api/intelligent", tags=["Intelligent Search & KG Interaction"])

app.include_router(upload.router, prefix="/api/upload", tags=["File Upload & KG Processing"])

# We should review documents.py, graph.py, and query.py to see if they are still needed
# or if their functionality is covered/enhanced by intelligent.router or neo4j_ops
app.include_router(documents.router, prefix="/api/documents", tags=["Document Management"]) # Adjusted prefix for consistency
app.include_router(graph.router, prefix="/api/graph", tags=["Basic Graph Operations"]) # Renamed tag for clarity
# app.include_router(query.router, prefix="/api/query", tags=["Simple Query"]) # Renamed tag for clarity, assuming it's simple

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    return {"message": "Pharma RAG API with Knowledge Graph is running"}

# The main health check is now part of the intelligent.router at /api/intelligent/health/
# If you still want a very basic root health check, you can keep this, but it's less informative.
# For now, let's rely on the more comprehensive one.
@app.get("/health")
async def simple_health_check():
    return {"status": "API service is active"}

if __name__ == "__main__":
    import uvicorn
    # This allows running directly for development, e.g., python backend/main.py
    # Production deployment would typically use Gunicorn + Uvicorn workers.
    uvicorn.run(app, host="0.0.0.0", port=8000)