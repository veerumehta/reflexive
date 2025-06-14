
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, List, Any, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Import your Reflexive Composition framework
from reflexive_composition_agentic import (
    ProductionReflexiveComposition, 
    create_production_config,
    EnvironmentConfig
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    logger.info("Initializing Reflexive Composition framework...")
    
    try:
        # Load configuration
        env_config = EnvironmentConfig()
        config = create_production_config()
        
        # Override with environment variables
        config["llm"] = env_config.get_llm_config("openai")  # or "anthropic"
        config["hitl"] = env_config.get_hitl_config()
        
        # Initialize system
        system = ProductionReflexiveComposition(config)
        await system.initialize()
        
        app_state["system"] = system
        logger.info("✓ Reflexive Composition framework initialized")
        
    except Exception as e:
        logger.error(f"✗ Failed to initialize framework: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Reflexive Composition framework...")

app = FastAPI(
    title="Reflexive Composition API",
    description="API for bidirectional enhancement between LLMs and Knowledge Graphs",
    version="1.0.0",
    lifespan=lifespan
)

# =====================================================================
# Pydantic Models
# =====================================================================

class DocumentRequest(BaseModel):
    text: str
    query: Optional[str] = None
    domain: str = "general"
    
class QueryRequest(BaseModel):
    query: str
    max_context_triples: int = 10

class BatchDocumentRequest(BaseModel):
    documents: List[DocumentRequest]

class KnowledgeResponse(BaseModel):
    success: bool
    triples_added: int
    kg_updated: bool
    schema_evolved: bool
    response: Optional[str] = None
    knowledge_used: List[Dict[str, Any]] = []
    processing_time: float
    errors: List[str] = []
    warnings: List[str] = []

class QueryResponse(BaseModel):
    response: str
    knowledge_used: int
    context_triples: List[Dict[str, Any]]
    confidence: float

# =====================================================================
# API Endpoints
# =====================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if "system" not in app_state:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    system = app_state["system"]
    health_status = system.get_health_status()
    
    if health_status["status"] == "healthy":
        return JSONResponse(content=health_status, status_code=200)
    else:
        return JSONResponse(content=health_status, status_code=503)

@app.post("/extract", response_model=KnowledgeResponse)
async def extract_knowledge(request: DocumentRequest):
    """Extract knowledge from a document."""
    if "system" not in app_state:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    system = app_state["system"]
    
    try:
        import time
        start_time = time.time()
        
        result = await system.agent.process_document(
            document_text=request.text,
            query=request.query,
            domain=request.domain
        )
        
        processing_time = time.time() - start_time
        
        return KnowledgeResponse(
            success=result["success"],
            triples_added=result.get("triples_added", 0),
            kg_updated=result.get("kg_updated", False),
            schema_evolved=result.get("schema_evolved", False),
            response=result.get("response"),
            knowledge_used=result.get("knowledge_used", []),
            processing_time=processing_time,
            errors=result.get("errors", []),
            warnings=result.get("warnings", [])
        )
        
    except Exception as e:
        logger.error(f"Error in knowledge extraction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_knowledge(request: QueryRequest):
    """Query the knowledge graph."""
    if "system" not in app_state:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    system = app_state["system"]
    
    try:
        response_data = system.framework.generate_grounded_response(request.query)
        
        return QueryResponse(
            response=response_data["response"],
            knowledge_used=response_data["knowledge_used"],
            context_triples=response_data["context_triples"],
            confidence=0.8  # Could calculate based on knowledge quality
        )
        
    except Exception as e:
        logger.error(f"Error in query processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch", response_model=List[KnowledgeResponse])
async def process_batch(request: BatchDocumentRequest, background_tasks: BackgroundTasks):
    """Process a batch of documents."""
    if "system" not in app_state:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    system = app_state["system"]
    
    try:
        documents = [
            {
                "text": doc.text,
                "query": doc.query,
                "domain": doc.domain
            }
            for doc in request.documents
        ]
        
        results = await system.process_batch(documents)
        
        response_list = []
        for result in results:
            response_list.append(
                KnowledgeResponse(
                    success=result["success"],
                    triples_added=result.get("triples_added", 0),
                    kg_updated=result.get("kg_updated", False),
                    schema_evolved=result.get("schema_evolved", False),
                    response=result.get("response"),
                    knowledge_used=result.get("knowledge_used", []),
                    processing_time=0.0,  # Would need to track individually
                    errors=result.get("errors", []),
                    warnings=result.get("warnings", [])
                )
            )
        
        return response_list
        
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get system metrics and statistics."""
    if "system" not in app_state:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    system = app_state["system"]
    
    try:
        metrics = await system.api.get_metrics_endpoint()
        return JSONResponse(content=metrics)
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/schema")
async def get_schema():
    """Get the current knowledge graph schema."""
    if "system" not in app_state:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    system = app_state["system"]
    
    try:
        schema = system.framework.knowledge_graph.schema
        return JSONResponse(content=schema)
        
    except Exception as e:
        logger.error(f"Error getting schema: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_document(file: UploadFile = File(...), domain: str = "general"):
    """Upload and process a document file."""
    if "system" not in app_state:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        # Read file content
        content = await file.read()
        text = content.decode("utf-8")
        
        # Process document
        request = DocumentRequest(text=text, domain=domain)
        return await extract_knowledge(request)
        
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be text-based")
    except Exception as e:
        logger.error(f"Error processing uploaded file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================================
# Run Server
# =====================================================================

if __name__ == "__main__":
    import os
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    debug = os.getenv("API_DEBUG", "false").lower() == "true"
    
    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )
