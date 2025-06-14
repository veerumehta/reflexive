"""
FastAPI endpoints for Reflexive Composition framework.
Provides REST API interface for all framework operations.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Depends
from fastapi.responses import JSONResponse

# Import your existing components
from ..core.framework import ReflexiveComposition
from ..core.data_models import (
    ExtractionRequest, QueryRequest, ExtractionResponse, 
    QueryResponse, ValidationResponse, SystemHealth,
    Triple, KnowledgeExtraction
)
from ..integrations.langchain_adapter import ReflexiveCompositionChain, AdapterFactory

logger = logging.getLogger(__name__)

# =====================================================================
# Request/Response Models (Additional)
# =====================================================================

from pydantic import BaseModel, Field

class BatchExtractionRequest(BaseModel):
    """Request model for batch processing."""
    documents: List[ExtractionRequest] = Field(description="List of documents to process")
    parallel: bool = Field(default=True, description="Process documents in parallel")

class UploadRequest(BaseModel):
    """Request model for file uploads."""
    domain: str = Field(default="general", description="Domain context for the document")
    query: Optional[str] = Field(default=None, description="Optional query about the document")

class SchemaRequest(BaseModel):
    """Request model for schema operations."""
    schema_update: Dict[str, Any] = Field(description="Schema update to apply")
    validate: bool = Field(default=True, description="Whether to validate the update")

class MetricsResponse(BaseModel):
    """Response model for system metrics."""
    framework_stats: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    component_health: Dict[str, bool]
    timestamp: datetime

# =====================================================================
# Dependency Injection
# =====================================================================

class FrameworkDependency:
    """Dependency injection for the framework instance."""
    
    def __init__(self):
        self._framework: Optional[ReflexiveComposition] = None
        self._chain: Optional[ReflexiveCompositionChain] = None
    
    def set_framework(self, framework: ReflexiveComposition, chain: ReflexiveCompositionChain):
        """Set the framework and chain instances."""
        self._framework = framework
        self._chain = chain
    
    def get_framework(self) -> ReflexiveComposition:
        """Get the framework instance."""
        if self._framework is None:
            raise HTTPException(
                status_code=503, 
                detail="Framework not initialized"
            )
        return self._framework
    
    def get_chain(self) -> ReflexiveCompositionChain:
        """Get the LangChain wrapper."""
        if self._chain is None:
            raise HTTPException(
                status_code=503,
                detail="LangChain integration not initialized"
            )
        return self._chain

# Global dependency instance
framework_dependency = FrameworkDependency()

def get_framework() -> ReflexiveComposition:
    """Dependency function for getting framework."""
    return framework_dependency.get_framework()

def get_chain() -> ReflexiveCompositionChain:
    """Dependency function for getting chain."""
    return framework_dependency.get_chain()

# =====================================================================
# Router Setup
# =====================================================================

router = APIRouter(prefix="/api/v1", tags=["reflexive-composition"])

# =====================================================================
# Health and Status Endpoints
# =====================================================================

@router.get("/health", response_model=SystemHealth)
async def health_check(framework: ReflexiveComposition = Depends(get_framework)):
    """Health check endpoint."""
    try:
        # Check component health
        components = {
            "knowledge_graph": framework.knowledge_graph is not None,
            "kb_llm": framework.kb_llm is not None,
            "target_llm": framework.target_llm is not None,
            "validator": framework.validator is not None
        }
        
        # Get basic metrics
        kg_stats = framework.knowledge_graph.get_stats()
        hitl_stats = framework.validator.get_stats() if framework.validator else {}
        
        # Determine overall health
        all_healthy = all(components.values())
        status = "healthy" if all_healthy else "degraded"
        
        return SystemHealth(
            status=status,
            components=components,
            metrics={
                "knowledge_graph": kg_stats,
                "hitl": hitl_stats
            },
            last_check=datetime.utcnow(),
            uptime=time.time() - getattr(framework, '_start_time', time.time())
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Health check failed")

@router.get("/status")
async def get_status(framework: ReflexiveComposition = Depends(get_framework)):
    """Get detailed system status."""
    try:
        stats = framework.get_framework_stats() if hasattr(framework, 'get_framework_stats') else {}
        
        return JSONResponse(content={
            "status": "operational",
            "version": "1.0.0",
            "stats": stats,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail="Status check failed")

# =====================================================================
# Knowledge Extraction Endpoints
# =====================================================================

@router.post("/extract", response_model=ExtractionResponse)
async def extract_knowledge(
    request: ExtractionRequest,
    chain: ReflexiveCompositionChain = Depends(get_chain)
):
    """Extract knowledge from text."""
    start_time = time.time()
    
    try:
        # Prepare inputs for the chain
        chain_inputs = {
            "source_text": request.text,
            "domain": request.domain or "general",
            "extraction_type": request.extraction_type.value,
            "schema": request.schema or {}
        }
        
        # Execute the extraction chain
        result = chain.extraction_chain(chain_inputs)
        extraction = result["extraction_result"]
        
        processing_time = time.time() - start_time
        
        return ExtractionResponse(
            success=True,
            extraction=extraction,
            triples_extracted=len(extraction.triples),
            processing_time=processing_time,
            errors=extraction.errors,
            warnings=[]
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Knowledge extraction failed: {e}")
        
        return ExtractionResponse(
            success=False,
            extraction=None,
            triples_extracted=0,
            processing_time=processing_time,
            errors=[str(e)],
            warnings=[]
        )

@router.post("/extract-and-validate")
async def extract_and_validate(
    request: ExtractionRequest,
    chain: ReflexiveCompositionChain = Depends(get_chain)
):
    """Extract knowledge and run through validation."""
    start_time = time.time()
    
    try:
        # Use the complete chain for extraction + validation
        chain_inputs = {
            "source_text": request.text,
            "domain": request.domain or "general",
            "extraction_type": request.extraction_type.value,
            "schema": request.schema or {}
        }
        
        # Execute extraction
        extraction_result = chain.extraction_chain(chain_inputs)
        
        # Execute validation
        validation_inputs = {
            "triples": [t.dict() for t in extraction_result["extraction_result"].triples],
            "source_text": request.text,
            "domain": request.domain or "general"
        }
        validation_result = chain.validation_chain(validation_inputs)
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "extraction": extraction_result["extraction_result"],
            "validation": validation_result,
            "processing_time": processing_time
        }
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Extract and validate failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================================
# Query Endpoints
# =====================================================================

@router.post("/query", response_model=QueryResponse)
async def query_knowledge(
    request: QueryRequest,
    chain: ReflexiveCompositionChain = Depends(get_chain)
):
    """Query the knowledge graph."""
    start_time = time.time()
    
    try:
        # Prepare inputs for generation chain
        generation_inputs = {
            "query": request.query,
            "max_context_triples": request.max_context_triples
        }
        
        # Execute the generation chain
        result = chain.generation_chain(generation_inputs)
        
        processing_time = time.time() - start_time
        
        return QueryResponse(
            response=result["response"],
            knowledge_used=result["knowledge_used"],
            context_triples=result["context_triples"],
            confidence=0.8,  # Could calculate based on knowledge quality
            processing_time=processing_time,
            metadata={
                "domain": request.domain,
                "include_metadata": request.include_metadata
            }
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process-document")
async def process_document(
    request: ExtractionRequest,
    chain: ReflexiveCompositionChain = Depends(get_chain)
):
    """Complete document processing pipeline."""
    start_time = time.time()
    
    try:
        # Prepare inputs for complete chain
        chain_inputs = {
            "source_text": request.text,
            "query": None,  # No query for document processing
            "domain": request.domain or "general",
            "schema": request.schema or {}
        }
        
        # Execute complete pipeline
        result = chain(chain_inputs)
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "extraction": result["extraction_result"],
            "validation_results": result["validation_results"],
            "triples_added": result["triples_added"],
            "response": result["response"],
            "processing_time": processing_time
        }
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Document processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================================
# Batch Processing Endpoints
# =====================================================================

@router.post("/batch")
async def process_batch(
    request: BatchExtractionRequest,
    background_tasks: BackgroundTasks,
    chain: ReflexiveCompositionChain = Depends(get_chain)
):
    """Process multiple documents."""
    try:
        if request.parallel and len(request.documents) > 1:
            # Process in parallel
            tasks = []
            for doc in request.documents:
                chain_inputs = {
                    "source_text": doc.text,
                    "domain": doc.domain or "general",
                    "extraction_type": doc.extraction_type.value,
                    "schema": doc.schema or {}
                }
                tasks.append(chain(chain_inputs))
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
        else:
            # Process sequentially
            results = []
            for doc in request.documents:
                try:
                    chain_inputs = {
                        "source_text": doc.text,
                        "domain": doc.domain or "general",
                        "extraction_type": doc.extraction_type.value,
                        "schema": doc.schema or {}
                    }
                    result = chain(chain_inputs)
                    results.append(result)
                except Exception as e:
                    results.append({"error": str(e), "success": False})
        
        # Process results
        successful = sum(1 for r in results if not isinstance(r, Exception) and r.get("success", False))
        failed = len(results) - successful
        
        return {
            "success": True,
            "total_documents": len(request.documents),
            "successful": successful,
            "failed": failed,
            "results": [r if not isinstance(r, Exception) else {"error": str(r)} for r in results]
        }
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================================
# File Upload Endpoints
# =====================================================================

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    domain: str = "general",
    query: Optional[str] = None,
    chain: ReflexiveCompositionChain = Depends(get_chain)
):
    """Upload and process a document file."""
    try:
        # Read file content
        content = await file.read()
        
        # Try to decode as text
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = content.decode("latin-1")
            except UnicodeDecodeError:
                raise HTTPException(
                    status_code=400, 
                    detail="File must be text-based and UTF-8 or Latin-1 encoded"
                )
        
        # Create extraction request
        request = ExtractionRequest(
            text=text,
            domain=domain
        )
        
        # Process the document
        chain_inputs = {
            "source_text": text,
            "query": query,
            "domain": domain,
            "schema": {}
        }
        
        result = chain(chain_inputs)
        
        return {
            "success": True,
            "filename": file.filename,
            "file_size": len(content),
            "extraction": result["extraction_result"],
            "triples_added": result["triples_added"],
            "response": result.get("response") if query else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File upload processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================================
# Schema Management Endpoints
# =====================================================================

@router.get("/schema")
async def get_schema(framework: ReflexiveComposition = Depends(get_framework)):
    """Get the current knowledge graph schema."""
    try:
        schema = framework.knowledge_graph.schema
        return JSONResponse(content=schema)
    except Exception as e:
        logger.error(f"Schema retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/schema/update")
async def update_schema(
    request: SchemaRequest,
    framework: ReflexiveComposition = Depends(get_framework)
):
    """Update the knowledge graph schema."""
    try:
        if request.validate and hasattr(framework, 'validator'):
            # Validate schema update through HITL if available
            validation_result = framework.validator.validate_schema_update(
                request.schema_update,
                framework.knowledge_graph.schema
            )
            
            if not validation_result.get('accepted', False):
                return {
                    "success": False,
                    "reason": validation_result.get('reason', 'Schema update rejected'),
                    "validation_result": validation_result
                }
        
        # Apply schema update
        success = framework.knowledge_graph.update_schema(request.schema_update)
        
        return {
            "success": success,
            "updated_schema": framework.knowledge_graph.schema,
            "version": framework.knowledge_graph.schema.get("version", 1)
        }
        
    except Exception as e:
        logger.error(f"Schema update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================================
# Metrics and Analytics Endpoints
# =====================================================================

@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(framework: ReflexiveComposition = Depends(get_framework)):
    """Get system metrics and performance data."""
    try:
        # Get framework stats
        framework_stats = framework.get_framework_stats() if hasattr(framework, 'get_framework_stats') else {}
        
        # Get component health
        component_health = {
            "knowledge_graph": framework.knowledge_graph is not None,
            "kb_llm": framework.kb_llm is not None,
            "target_llm": framework.target_llm is not None,
            "validator": framework.validator is not None
        }
        
        # Calculate performance metrics
        kg_stats = framework.knowledge_graph.get_stats()
        hitl_stats = framework.validator.get_stats() if framework.validator else {}
        
        performance_metrics = {
            "knowledge_graph": kg_stats,
            "validation": hitl_stats,
            "uptime": time.time() - getattr(framework, '_start_time', time.time())
        }
        
        return MetricsResponse(
            framework_stats=framework_stats,
            performance_metrics=performance_metrics,
            component_health=component_health,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Metrics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/knowledge-graph/stats")
async def get_kg_stats(framework: ReflexiveComposition = Depends(get_framework)):
    """Get detailed knowledge graph statistics."""
    try:
        stats = framework.knowledge_graph.get_stats()
        
        # Add additional analysis
        stats.update({
            "triples_by_confidence": _analyze_confidence_distribution(framework.knowledge_graph),
            "entity_degree_distribution": _analyze_entity_degrees(framework.knowledge_graph),
            "relationship_frequency": _analyze_relationship_types(framework.knowledge_graph)
        })
        
        return JSONResponse(content=stats)
        
    except Exception as e:
        logger.error(f"KG stats retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================================
# Utility Functions
# =====================================================================

def _analyze_confidence_distribution(kg) -> Dict[str, int]:
    """Analyze the distribution of confidence scores in the knowledge graph."""
    confidence_buckets = {"low": 0, "medium": 0, "high": 0}
    
    for triple in kg.triples:
        if triple.confidence < 0.5:
            confidence_buckets["low"] += 1
        elif triple.confidence < 0.8:
            confidence_buckets["medium"] += 1
        else:
            confidence_buckets["high"] += 1
    
    return confidence_buckets

def _analyze_entity_degrees(kg) -> Dict[str, int]:
    """Analyze the degree distribution of entities."""
    if hasattr(kg, 'graph'):
        degrees = dict(kg.graph.degree())
        return {
            "avg_degree": sum(degrees.values()) / len(degrees) if degrees else 0,
            "max_degree": max(degrees.values()) if degrees else 0,
            "min_degree": min(degrees.values()) if degrees else 0
        }
    return {}

def _analyze_relationship_types(kg) -> Dict[str, int]:
    """Analyze the frequency of different relationship types."""
    rel_freq = {}
    for triple in kg.triples:
        rel_freq[triple.predicate] = rel_freq.get(triple.predicate, 0) + 1
    return dict(sorted(rel_freq.items(), key=lambda x: x[1], reverse=True)[:10])

# =====================================================================
# Export
# =====================================================================

__all__ = [
    "router",
    "framework_dependency",
    "get_framework",
    "get_chain"
]
