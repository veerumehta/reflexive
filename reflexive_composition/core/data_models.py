"""
Core data models and enums for Reflexive Composition framework.
Centralizes all Pydantic models, dataclasses, and type definitions.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator
from typing_extensions import Literal, TypedDict

# =====================================================================
# Enums
# =====================================================================

class TriggerType(Enum):
    """Types of knowledge graph update triggers."""
    QUERY_FAILURE = "query_failure"
    CONTRADICTION = "contradiction"
    SCHEMA_EVOLUTION = "schema_evolution"
    VALIDATION_FEEDBACK = "validation_feedback"
    SOURCE_UPDATE = "source_update"

class ValidationDecision(Enum):
    """Possible validation decisions."""
    ACCEPT = "accept"
    REJECT = "reject"
    MODIFY = "modify"
    ESCALATE = "escalate"

class ExtractionType(Enum):
    """Types of knowledge extraction."""
    GENERAL = "general"
    TEMPORAL = "temporal"
    DOMAIN_SPECIFIC = "domain_specific"
    SCHEMA_GUIDED = "schema_guided"

# =====================================================================
# Core Data Classes
# =====================================================================

@dataclass
class Triple:
    """Represents a knowledge graph triple."""
    subject: str
    predicate: str
    object: str
    confidence: float = 0.8
    source: Optional[str] = None
    timestamp: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Triple':
        return cls(**data)

@dataclass
class ValidationContext:
    """Context information for validation decisions."""
    triple: Triple
    source_text: str
    existing_knowledge: List[Triple]
    confidence_threshold: float
    domain: Optional[str] = None
    extraction_method: Optional[str] = None
    validator_id: Optional[str] = None

@dataclass
class ValidationResult:
    """Result of a validation decision."""
    decision: ValidationDecision
    original_triple: Triple
    modified_triple: Optional[Triple] = None
    reason: Optional[str] = None
    validator_id: Optional[str] = None
    timestamp: Optional[str] = None
    confidence: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()

@dataclass
class UpdateTrigger:
    """Represents a knowledge graph update trigger."""
    type: TriggerType
    source: str
    confidence: float
    details: Dict[str, Any]
    timestamp: str
    priority: int = 1  # 1 = low, 5 = high
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()

# =====================================================================
# Pydantic Models for API and LangChain
# =====================================================================

class ExtractedTriple(BaseModel):
    """Pydantic model for extracted knowledge triples."""
    subject: str = Field(description="The subject entity of the triple")
    predicate: str = Field(description="The relationship or property")
    object: str = Field(description="The object entity or value")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0", ge=0.0, le=1.0)
    source: Optional[str] = Field(description="Source of this information", default=None)
    metadata: Optional[Dict[str, Any]] = Field(description="Additional metadata", default_factory=dict)

class KnowledgeExtraction(BaseModel):
    """Pydantic model for knowledge extraction results."""
    triples: List[ExtractedTriple] = Field(description="List of extracted triples")
    domain: Optional[str] = Field(description="Detected domain", default=None)
    extraction_method: ExtractionType = Field(description="Method used for extraction", default=ExtractionType.GENERAL)
    total_confidence: float = Field(description="Overall confidence in extraction", default=0.0, ge=0.0, le=1.0)
    processing_time: Optional[float] = Field(description="Time taken for extraction", default=None)
    errors: List[str] = Field(description="Any errors encountered", default_factory=list)

class SchemaUpdate(BaseModel):
    """Pydantic model for schema update suggestions."""
    entity_types_to_add: List[str] = Field(description="New entity types to add", default_factory=list)
    relationship_types_to_add: List[str] = Field(description="New relationship types to add", default_factory=list)
    attributes_to_add: List[Dict[str, Any]] = Field(description="New attributes to add", default_factory=list)
    reasoning: str = Field(description="Explanation for the suggested updates")
    confidence: float = Field(description="Confidence in the schema update", ge=0.0, le=1.0)
    affected_domains: List[str] = Field(description="Domains affected by update", default_factory=list)

class QueryRequest(BaseModel):
    """Request model for knowledge queries."""
    query: str = Field(description="The user's query")
    domain: Optional[str] = Field(description="Domain context", default=None)
    max_context_triples: int = Field(description="Maximum context triples to use", default=10, ge=1, le=100)
    include_metadata: bool = Field(description="Include metadata in response", default=False)

class ExtractionRequest(BaseModel):
    """Request model for knowledge extraction."""
    text: str = Field(description="Text to extract knowledge from")
    domain: Optional[str] = Field(description="Domain context", default=None)
    extraction_type: ExtractionType = Field(description="Type of extraction", default=ExtractionType.GENERAL)
    schema: Optional[Dict[str, Any]] = Field(description="Schema to guide extraction", default=None)
    confidence_threshold: float = Field(description="Minimum confidence for auto-acceptance", default=0.7, ge=0.0, le=1.0)

# =====================================================================
# State Models for LangGraph
# =====================================================================

class AgentState(TypedDict):
    """State for the reflexive composition agent workflow."""
    # Input data
    source_text: str
    query: Optional[str]
    domain: str
    extraction_type: ExtractionType
    
    # Processing state
    extracted_knowledge: List[Dict[str, Any]]
    validation_results: List[Dict[str, Any]]
    validated_triples: List[Dict[str, Any]]
    
    # Knowledge graph state
    kg_updated: bool
    schema_evolved: bool
    schema_version: int
    
    # Response generation
    response: Optional[str]
    knowledge_used: List[Dict[str, Any]]
    context_metadata: Dict[str, Any]
    
    # Workflow control
    next_action: str
    iteration_count: int
    max_iterations: int
    workflow_state: str
    
    # Error handling and monitoring
    errors: List[str]
    warnings: List[str]
    processing_times: Dict[str, float]
    confidence_scores: List[float]

class WorkflowState(BaseModel):
    """Workflow state for monitoring and debugging."""
    workflow_id: str
    current_step: str
    steps_completed: List[str]
    start_time: datetime
    last_update: datetime
    status: Literal["running", "completed", "failed", "paused"]
    progress_percentage: float = Field(ge=0.0, le=100.0)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

# =====================================================================
# Configuration Models
# =====================================================================

class LLMConfig(BaseModel):
    """Configuration for LLM providers."""
    provider: Literal["openai", "anthropic", "google", "huggingface"] = "openai"
    model: str = "gpt-4"
    api_key: Optional[str] = None
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2000, ge=100, le=100000)
    timeout: float = Field(default=30.0, ge=1.0, le=300.0)
    max_retries: int = Field(default=3, ge=0, le=10)

class HITLConfig(BaseModel):
    """Configuration for Human-in-the-Loop validation."""
    auto_accept_threshold: float = Field(default=0.9, ge=0.0, le=1.0)
    auto_reject_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    interactive: bool = True
    validation_timeout: int = Field(default=300, ge=30, le=3600)  # seconds
    escalation_enabled: bool = True
    batch_validation: bool = False
    validator_assignment: Literal["round_robin", "expertise_based", "random"] = "round_robin"

class KnowledgeGraphConfig(BaseModel):
    """Configuration for Knowledge Graph."""
    storage_type: Literal["in_memory", "sqlite", "neo4j", "rdf"] = "in_memory"
    connection_string: Optional[str] = None
    max_entities: Optional[int] = Field(default=None, ge=1)
    max_relationships: Optional[int] = Field(default=None, ge=1)
    enable_indexing: bool = True
    backup_enabled: bool = False
    backup_interval: str = "24h"

class FrameworkConfig(BaseModel):
    """Main framework configuration."""
    kb_llm: LLMConfig
    target_llm: LLMConfig
    hitl: HITLConfig
    knowledge_graph: KnowledgeGraphConfig
    
    # Schema configuration
    initial_schema: Optional[Dict[str, Any]] = None
    enable_schema_evolution: bool = True
    schema_update_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    
    # Performance settings
    max_concurrent_extractions: int = Field(default=5, ge=1, le=50)
    cache_enabled: bool = True
    monitoring_enabled: bool = True
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, ge=1024, le=65535)
    api_debug: bool = False

# =====================================================================
# Response Models
# =====================================================================

class ExtractionResponse(BaseModel):
    """Response model for knowledge extraction."""
    success: bool
    extraction: Optional[KnowledgeExtraction] = None
    triples_extracted: int = 0
    processing_time: float
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

class ValidationResponse(BaseModel):
    """Response model for validation operations."""
    success: bool
    validation_results: List[Dict[str, Any]] = Field(default_factory=list)
    triples_validated: int = 0
    auto_decisions: int = 0
    human_decisions: int = 0
    processing_time: float
    errors: List[str] = Field(default_factory=list)

class QueryResponse(BaseModel):
    """Response model for knowledge queries."""
    response: str
    knowledge_used: int
    context_triples: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: float
    processing_time: float
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SystemHealth(BaseModel):
    """System health status model."""
    status: Literal["healthy", "degraded", "unhealthy"]
    components: Dict[str, bool]
    metrics: Dict[str, Any] = Field(default_factory=dict)
    last_check: datetime
    uptime: float  # seconds
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

# =====================================================================
# Utility Functions
# =====================================================================

def create_default_schema() -> Dict[str, Any]:
    """Create a default knowledge graph schema."""
    return {
        "entity_types": [
            "Person", "Organization", "Location", "Event", 
            "Product", "Technology", "Date", "Currency",
            "Document", "Concept"
        ],
        "relationship_types": [
            "WorksAt", "LocatedIn", "OccurredAt", "InvolvedIn",
            "ManufacturedBy", "AcquiredBy", "PartOf", "RelatedTo",
            "PublishedBy", "AuthoredBy", "Contains", "Mentions",
            "FoundedBy", "HeadquarteredIn", "OwnedBy"
        ],
        "attributes": {
            "Person": ["name", "age", "occupation", "nationality"],
            "Organization": ["name", "type", "founded", "headquarters"],
            "Location": ["name", "type", "coordinates", "population"],
            "Event": ["name", "date", "location", "participants"]
        },
        "version": 1,
        "created_at": datetime.utcnow().isoformat()
    }

def validate_config(config: Dict[str, Any]) -> FrameworkConfig:
    """Validate and parse framework configuration."""
    try:
        return FrameworkConfig(**config)
    except Exception as e:
        raise ValueError(f"Invalid configuration: {e}")

# =====================================================================
# Export commonly used items
# =====================================================================

__all__ = [
    # Enums
    "TriggerType", "ValidationDecision", "ExtractionType",
    
    # Data classes
    "Triple", "ValidationContext", "ValidationResult", "UpdateTrigger",
    
    # Pydantic models
    "ExtractedTriple", "KnowledgeExtraction", "SchemaUpdate",
    "QueryRequest", "ExtractionRequest",
    
    # State models
    "AgentState", "WorkflowState",
    
    # Configuration models
    "LLMConfig", "HITLConfig", "KnowledgeGraphConfig", "FrameworkConfig",
    
    # Response models
    "ExtractionResponse", "ValidationResponse", "QueryResponse", "SystemHealth",
    
    # Utility functions
    "create_default_schema", "validate_config"
]