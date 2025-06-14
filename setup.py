"""
Reflexive Composition Framework - Setup and Documentation
Complete setup guide with real API integrations and deployment examples.
"""

import os
import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import yaml

# Setup requirements
REQUIRED_PACKAGES = [
    "langchain>=0.1.0",
    "langgraph>=0.1.0",
    "langchain-openai>=0.1.0",
    "langchain-anthropic>=0.1.0",
    "networkx>=3.0",
    "pydantic>=2.0",
    "fastapi>=0.100.0",
    "uvicorn>=0.20.0",
    "aiosqlite>=0.19.0",
    "python-multipart>=0.0.6",
    "python-dotenv>=1.0.0"
]

# =====================================================================
# Environment Configuration
# =====================================================================

class EnvironmentConfig:
    """Manage environment configuration for Reflexive Composition."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or ".env"
        self.load_environment()
    
    def load_environment(self):
        """Load environment variables."""
        try:
            from dotenv import load_dotenv
            load_dotenv(self.config_path)
        except ImportError:
            logging.warning("python-dotenv not installed. Using system environment variables.")
    
    def get_llm_config(self, provider: str = "openai") -> Dict[str, Any]:
        """Get LLM configuration from environment."""
        if provider == "openai":
            return {
                "provider": "openai",
                "model": os.getenv("OPENAI_MODEL", "gpt-4"),
                "api_key": os.getenv("OPENAI_API_KEY"),
                "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.1")),
                "max_tokens": int(os.getenv("OPENAI_MAX_TOKENS", "2000"))
            }
        elif provider == "anthropic":
            return {
                "provider": "anthropic",
                "model": os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229"),
                "api_key": os.getenv("ANTHROPIC_API_KEY"),
                "temperature": float(os.getenv("ANTHROPIC_TEMPERATURE", "0.1")),
                "max_tokens": int(os.getenv("ANTHROPIC_MAX_TOKENS", "2000"))
            }
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration."""
        return {
            "type": os.getenv("DB_TYPE", "sqlite"),
            "host": os.getenv("DB_HOST", "localhost"),
            "port": int(os.getenv("DB_PORT", "5432")),
            "name": os.getenv("DB_NAME", "reflexive_composition"),
            "user": os.getenv("DB_USER", ""),
            "password": os.getenv("DB_PASSWORD", ""),
            "path": os.getenv("DB_PATH", "./reflexive_composition.db")
        }
    
    def get_hitl_config(self) -> Dict[str, Any]:
        """Get HITL configuration."""
        return {
            "auto_accept_threshold": float(os.getenv("HITL_AUTO_ACCEPT", "0.9")),
            "auto_reject_threshold": float(os.getenv("HITL_AUTO_REJECT", "0.3")),
            "interactive": os.getenv("HITL_INTERACTIVE", "false").lower() == "true",
            "validation_timeout": int(os.getenv("HITL_TIMEOUT", "300"))
        }

# =====================================================================
# Installation and Setup
# =====================================================================

def install_requirements():
    """Install required packages."""
    import subprocess
    import sys
    
    print("Installing required packages...")
    for package in REQUIRED_PACKAGES:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ Installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {package}: {e}")
            return False
    
    print("✓ All packages installed successfully")
    return True

def create_env_file():
    """Create a template .env file."""
    env_content = """
# Reflexive Composition Environment Configuration

# LLM Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4
OPENAI_TEMPERATURE=0.1
OPENAI_MAX_TOKENS=2000

ANTHROPIC_API_KEY=your_anthropic_api_key_here
ANTHROPIC_MODEL=claude-3-sonnet-20240229
ANTHROPIC_TEMPERATURE=0.1
ANTHROPIC_MAX_TOKENS=2000

# Human-in-the-Loop Configuration
HITL_AUTO_ACCEPT=0.9
HITL_AUTO_REJECT=0.3
HITL_INTERACTIVE=false
HITL_TIMEOUT=300

# Database Configuration
DB_TYPE=sqlite
DB_PATH=./reflexive_composition.db

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=false

# Monitoring Configuration
MONITORING_ENABLED=true
LOG_LEVEL=INFO
"""
    
    with open(".env", "w") as f:
        f.write(env_content.strip())
    
    print("✓ Created .env template file")
    print("  Please update it with your API keys and configuration")

def create_config_yaml():
    """Create a configuration YAML file."""
    config = {
        "reflexive_composition": {
            "schema": {
                "entity_types": [
                    "Person", "Organization", "Location", "Event",
                    "Product", "Technology", "Date", "Currency",
                    "Document", "Concept"
                ],
                "relationship_types": [
                    "WorksAt", "LocatedIn", "OccurredAt", "InvolvedIn",
                    "ManufacturedBy", "AcquiredBy", "PartOf", "RelatedTo",
                    "PublishedBy", "AuthoredBy", "Contains", "Mentions"
                ],
                "version": 1
            },
            "extraction": {
                "max_triples_per_document": 50,
                "confidence_threshold": 0.7,
                "domain_detection": True,
                "temporal_extraction": True
            },
            "validation": {
                "batch_size": 10,
                "parallel_validation": True,
                "escalation_enabled": True,
                "audit_trail": True
            },
            "knowledge_graph": {
                "max_entities": 100000,
                "max_relationships": 500000,
                "indexing_enabled": True,
                "backup_enabled": True,
                "backup_interval": "24h"
            }
        }
    }
    
    with open("config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print("✓ Created config.yaml file")

# =====================================================================
# FastAPI Integration
# =====================================================================

def create_fastapi_server():
    """Create a FastAPI server for the Reflexive Composition framework."""
    
    fastapi_code = '''
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
'''
    
    with open("api_server.py", "w") as f:
        f.write(fastapi_code)
    
    print("✓ Created api_server.py")

# =====================================================================
# Example Usage Scripts
# =====================================================================

def create_example_scripts():
    """Create example usage scripts."""
    
    # Basic usage example
    basic_example = '''
"""
Basic Reflexive Composition Usage Example
"""

import asyncio
import os
from reflexive_composition_agentic import (
    ProductionReflexiveComposition,
    EnvironmentConfig
)

async def main():
    """Run basic example."""
    # Load configuration
    env_config = EnvironmentConfig()
    
    config = {
        "llm": env_config.get_llm_config("openai"),
        "schema": {
            "entity_types": ["Person", "Event", "Location", "Organization"],
            "relationship_types": ["OccurredAt", "InvolvedIn", "LocatedIn"],
            "version": 1
        },
        "hitl": env_config.get_hitl_config()
    }
    
    # Initialize system
    system = ProductionReflexiveComposition(config)
    await system.initialize()
    
    # Example document
    document_text = """
    The OpenAI ChatGPT model was released in November 2022, revolutionizing 
    conversational AI. It was developed by OpenAI, a company based in San Francisco.
    The model gained over 100 million users within two months of its launch.
    """
    
    # Process document
    result = await system.agent.process_document(
        document_text=document_text,
        query="When was ChatGPT released?",
        domain="technology"
    )
    
    print("Processing Result:", result)
    
    # Query knowledge
    query_result = system.framework.generate_grounded_response(
        "What do you know about OpenAI?"
    )
    
    print("Query Result:", query_result)

if __name__ == "__main__":
    asyncio.run(main())
'''
    
    # API client example
    api_client_example = '''
"""
API Client Example for Reflexive Composition
"""

import requests
import json

class ReflexiveCompositionClient:
    """Client for the Reflexive Composition API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        
    def extract_knowledge(self, text: str, query: str = None, domain: str = "general"):
        """Extract knowledge from text."""
        url = f"{self.base_url}/extract"
        data = {
            "text": text,
            "query": query,
            "domain": domain
        }
        
        response = requests.post(url, json=data)
        return response.json()
    
    def query_knowledge(self, query: str):
        """Query the knowledge graph."""
        url = f"{self.base_url}/query"
        data = {"query": query}
        
        response = requests.post(url, json=data)
        return response.json()
    
    def get_metrics(self):
        """Get system metrics."""
        url = f"{self.base_url}/metrics"
        response = requests.get(url)
        return response.json()
    
    def get_schema(self):
        """Get knowledge graph schema."""
        url = f"{self.base_url}/schema"
        response = requests.get(url)
        return response.json()

def main():
    """Example usage of the API client."""
    client = ReflexiveCompositionClient()
    
    # Extract knowledge
    text = """
    Apple Inc. announced the iPhone 15 in September 2023. The device features 
    a USB-C port replacing the Lightning connector. The announcement was made 
    at Apple Park in Cupertino, California.
    """
    
    extraction_result = client.extract_knowledge(
        text=text,
        query="What did Apple announce in 2023?",
        domain="technology"
    )
    
    print("Extraction Result:")
    print(json.dumps(extraction_result, indent=2))
    
    # Query knowledge
    query_result = client.query_knowledge("Tell me about the iPhone 15")
    
    print("\\nQuery Result:")
    print(json.dumps(query_result, indent=2))
    
    # Get metrics
    metrics = client.get_metrics()
    print("\\nSystem Metrics:")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
'''
    
    # Create files
    with open("example_basic.py", "w") as f:
        f.write(basic_example)
    
    with open("example_api_client.py", "w") as f:
        f.write(api_client_example)
    
    print("✓ Created example scripts:")
    print("  - example_basic.py")
    print("  - example_api_client.py")

# =====================================================================
# Testing Framework
# =====================================================================

def create_test_suite():
    """Create a comprehensive test suite."""
    
    test_code = '''
"""
Test Suite for Reflexive Composition Framework
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from reflexive_composition_agentic import (
    ReflexiveCompositionAgent,
    ProductionReflexiveComposition,
    KnowledgeGraph,
    Triple
)

class TestKnowledgeGraph:
    """Test the KnowledgeGraph component."""
    
    def test_add_triple(self):
        """Test adding a single triple."""
        kg = KnowledgeGraph()
        triple = Triple(
            subject="Apple",
            predicate="headquartered_in",
            object="Cupertino",
            confidence=0.9
        )
        
        result = kg.add_triple(triple)
        assert result == True
        assert len(kg.triples) == 1
        assert kg.graph.has_node("Apple")
        assert kg.graph.has_node("Cupertino")
    
    def test_query_related(self):
        """Test querying related entities."""
        kg = KnowledgeGraph()
        
        # Add test triples
        triples = [
            Triple("Apple", "headquartered_in", "Cupertino", 0.9),
            Triple("Apple", "founded_by", "Steve Jobs", 0.9),
            Triple("Steve Jobs", "born_in", "San Francisco", 0.8)
        ]
        
        for triple in triples:
            kg.add_triple(triple)
        
        related = kg.query_related("Apple", max_hops=2)
        assert len(related) >= 2
    
    def test_get_subgraph(self):
        """Test subgraph retrieval."""
        kg = KnowledgeGraph()
        
        triple = Triple("OpenAI", "created", "ChatGPT", 0.9)
        kg.add_triple(triple)
        
        subgraph = kg.get_subgraph("What did OpenAI create?")
        assert len(subgraph) >= 1

class TestTriple:
    """Test the Triple data structure."""
    
    def test_triple_creation(self):
        """Test creating a triple."""
        triple = Triple(
            subject="Microsoft",
            predicate="acquired",
            object="Activision Blizzard",
            confidence=0.95
        )
        
        assert triple.subject == "Microsoft"
        assert triple.predicate == "acquired"
        assert triple.object == "Activision Blizzard"
        assert triple.confidence == 0.95
        assert triple.timestamp is not None
    
    def test_triple_to_dict(self):
        """Test converting triple to dictionary."""
        triple = Triple("A", "B", "C", 0.8)
        triple_dict = triple.to_dict()
        
        assert isinstance(triple_dict, dict)
        assert triple_dict["subject"] == "A"
        assert triple_dict["predicate"] == "B"
        assert triple_dict["object"] == "C"
        assert triple_dict["confidence"] == 0.8

@pytest.mark.asyncio
class TestReflexiveCompositionAgent:
    """Test the agentic workflow."""
    
    async def test_agent_initialization(self):
        """Test agent initialization."""
        # Mock LLM
        mock_llm = Mock()
        mock_llm._call = Mock(return_value="test response")
        mock_llm._llm_type = "mock"
        
        # Mock framework
        mock_framework = Mock()
        mock_framework.knowledge_graph = KnowledgeGraph()
        
        agent = ReflexiveCompositionAgent(mock_framework, mock_llm)
        assert agent is not None
        assert agent.tools is not None
        assert len(agent.tools) == 5  # Should have 5 tools

class TestProductionSystem:
    """Test the production system."""
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = {
            "llm": {
                "provider": "openai",
                "model": "gpt-4",
                "api_key": "test-key"
            }
        }
        
        system = ProductionReflexiveComposition(config)
        assert system.config == config

# =====================================================================
# Integration Tests
# =====================================================================

@pytest.mark.integration
class TestIntegration:
    """Integration tests (require API keys)."""
    
    @pytest.mark.skip(reason="Requires API keys")
    async def test_full_pipeline(self):
        """Test the complete pipeline with real LLMs."""
        config = {
            "llm": {
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "api_key": "your-api-key"
            }
        }
        
        system = ProductionReflexiveComposition(config)
        await system.initialize()
        
        document_text = "Test document for processing."
        result = await system.agent.process_document(document_text)
        
        assert result["success"] == True

# =====================================================================
# Performance Tests
# =====================================================================

@pytest.mark.performance
class TestPerformance:
    """Performance tests."""
    
    def test_knowledge_graph_scalability(self):
        """Test knowledge graph performance with many triples."""
        kg = KnowledgeGraph()
        
        # Add many triples
        import time
        start_time = time.time()
        
        for i in range(1000):
            triple = Triple(f"Entity{i}", "related_to", f"Entity{i+1}", 0.8)
            kg.add_triple(triple)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        assert processing_time < 5.0  # Should complete in under 5 seconds
        assert len(kg.triples) == 1000

if __name__ == "__main__":
    pytest.main([__file__])
'''
    
    # pytest.ini configuration
    pytest_config = '''
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --verbose
    --tb=short
    --strict-markers
markers =
    integration: marks tests as integration tests (may require API keys)
    performance: marks tests as performance tests (may be slow)
    unit: marks tests as unit tests (fast, no external dependencies)
'''
    
    with open("test_reflexive_composition.py", "w") as f:
        f.write(test_code)
    
    with open("pytest.ini", "w") as f:
        f.write(pytest_config)
    
    print("✓ Created test suite:")
    print("  - test_reflexive_composition.py")
    print("  - pytest.ini")

# =====================================================================
# Documentation Generation
# =====================================================================

def create_documentation():
    """Create comprehensive documentation."""
    
    readme_content = '''
# Reflexive Composition Framework

A sophisticated framework for bidirectional enhancement between Large Language Models (LLMs) and Knowledge Graphs with strategic human-in-the-loop validation.

## Features

- **LLM2KG**: Automated knowledge extraction from text using LLMs
- **Knowledge Graph**: Scalable graph storage with NetworkX backend
- **Human-in-the-Loop**: Strategic validation with configurable thresholds
- **KG2LLM**: Knowledge-grounded response generation
- **Agentic Workflows**: LangGraph-based autonomous processing
- **REST API**: Production-ready API with FastAPI
- **Monitoring**: Comprehensive metrics and health monitoring

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file:

```bash
# API Keys
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key

# Configuration
HITL_INTERACTIVE=false
HITL_AUTO_ACCEPT=0.9
HITL_AUTO_REJECT=0.3
```

### 3. Basic Usage

```python
import asyncio
from reflexive_composition_agentic import ProductionReflexiveComposition

async def main():
    config = {
        "llm": {
            "provider": "openai",
            "model": "gpt-4",
            "api_key": "your-api-key"
        }
    }
    
    system = ProductionReflexiveComposition(config)
    await system.initialize()
    
    result = await system.agent.process_document(
        document_text="Your text here",
        query="Your question here",
        domain="general"
    )
    
    print(result)

asyncio.run(main())
```

### 4. API Server

```bash
python api_server.py
```

Then access the API at `http://localhost:8000/docs`

## Architecture

The framework consists of three main components:

1. **LLM2KG**: Extracts structured knowledge from unstructured text
2. **HITL**: Validates extracted knowledge through human oversight
3. **KG2LLM**: Generates responses grounded in validated knowledge

## API Endpoints

- `POST /extract` - Extract knowledge from text
- `POST /query` - Query the knowledge graph
- `GET /metrics` - Get system metrics
- `GET /schema` - Get knowledge graph schema
- `POST /batch` - Process multiple documents
- `POST /upload` - Upload and process files

## Testing

```bash
pytest test_reflexive_composition.py
```

## Examples

See the `examples/` directory for comprehensive usage examples:

- `example_basic.py` - Basic framework usage
- `example_api_client.py` - API client usage

## Configuration

The system can be configured through environment variables or YAML files. See `config.yaml` for all available options.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the MIT License.
'''
    
    # API documentation
    api_docs = '''
# API Documentation

## Authentication

Currently, no authentication is required for the API endpoints. In production, consider implementing API key authentication.

## Endpoints

### POST /extract

Extract knowledge from text documents.

**Request Body:**
```json
{
    "text": "Your document text here",
    "query": "Optional question about the text",
    "domain": "general"
}
```

**Response:**
```json
{
    "success": true,
    "triples_added": 5,
    "kg_updated": true,
    "schema_evolved": false,
    "response": "Generated response to query",
    "knowledge_used": [...],
    "processing_time": 2.5,
    "errors": [],
    "warnings": []
}
```

### POST /query

Query the knowledge graph.

**Request Body:**
```json
{
    "query": "Your question here",
    "max_context_triples": 10
}
```

**Response:**
```json
{
    "response": "Answer based on knowledge graph",
    "knowledge_used": 3,
    "context_triples": [...],
    "confidence": 0.85
}
```

### GET /metrics

Get system performance metrics.

**Response:**
```json
{
    "performance": {
        "extraction": {
            "average_accuracy": 0.87,
            "total_extractions": 150
        },
        "validation": {
            "average_efficiency": 0.92,
            "total_validations": 200
        }
    },
    "framework_stats": {...}
}
```

## Error Handling

All endpoints return appropriate HTTP status codes:

- 200: Success
- 400: Bad Request (invalid input)
- 500: Internal Server Error
- 503: Service Unavailable (system not initialized)

Error responses include details:

```json
{
    "detail": "Error description"
}
```
'''
    
    # Create documentation files
    with open("README.md", "w") as f:
        f.write(readme_content)
    
    with open("API_DOCS.md", "w") as f:
        f.write(api_docs)
    
    print("✓ Created documentation:")
    print("  - README.md")
    print("  - API_DOCS.md")

# =====================================================================
# Main Setup Function
# =====================================================================

def setup_reflexive_composition():
    """Complete setup of the Reflexive Composition framework."""
    print("Setting up Reflexive Composition Framework...")
    print("=" * 50)
    
    # Create directory structure
    import os
    directories = [
        "examples",
        "tests", 
        "docs",
        "config",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Install requirements
    if os.environ.get("INSTALL_DEPS", "y").lower() == 'y':
        install_requirements()
    
    # Create configuration files
    create_env_file()
    create_config_yaml()
    
    # Create API server
    create_fastapi_server()
    
    # Create examples
    create_example_scripts()
    
    # Create test suite
    create_test_suite()
    
    # Create documentation
    create_documentation()
    
    # Create requirements.txt
    with open("requirements.txt", "w") as f:
        f.write("\n".join(REQUIRED_PACKAGES))
    print("✓ Created requirements.txt")
    
    print("\n" + "=" * 50)
    print("Setup completed successfully!")
    print("\nNext steps:")
    print("1. Update .env file with your API keys")
    print("2. Review and modify config.yaml as needed")
    print("3. Run: python example_basic.py")
    print("4. Start API server: python api_server.py")
    print("5. Run tests: pytest")

if __name__ == "__main__":
    setup_reflexive_composition()
