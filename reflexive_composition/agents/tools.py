"""
LangGraph tools for Reflexive Composition framework.
Provides individual tools that can be used in agent workflows.
"""

import logging
from typing import Dict, List, Any, Optional, Type
from datetime import datetime

# LangChain imports
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

# Import your existing components
from ..core.data_models import Triple, ValidationContext, ValidationDecision, ExtractionType
from ..core import ReflexiveComposition

logger = logging.getLogger(__name__)

# =====================================================================
# Tool Input Models
# =====================================================================

class ExtractionInput(BaseModel):
    """Input model for knowledge extraction tool."""
    text: str = Field(description="Text to extract knowledge from")
    domain: str = Field(default="general", description="Domain context")
    extraction_type: str = Field(default="general", description="Type of extraction")

class ValidationInput(BaseModel):
    """Input model for validation tool."""
    triples: List[Dict[str, Any]] = Field(description="Triples to validate")
    source_text: str = Field(description="Source text")
    domain: str = Field(default="general", description="Domain context")

class KGUpdateInput(BaseModel):
    """Input model for knowledge graph update tool."""
    validated_triples: List[Dict[str, Any]] = Field(description="Validated triples to add")

class ResponseInput(BaseModel):
    """Input model for response generation tool."""
    query: str = Field(description="Query to answer")
    max_context_triples: int = Field(default=10, description="Maximum context triples")

class SchemaEvolutionInput(BaseModel):
    """Input model for schema evolution tool."""
    recent_extractions: List[Dict[str, Any]] = Field(description="Recent extraction results")

# =====================================================================
# Knowledge Extraction Tool
# =====================================================================

class KnowledgeExtractionTool(BaseTool):
    """Tool for extracting knowledge from text."""
    
    name: str = "extract_knowledge"
    description: str = """
    Extract structured knowledge from text in the form of triples (subject, predicate, object).
    Use this tool when you need to convert unstructured text into structured knowledge.
    """
    args_schema: Type[BaseModel] = ExtractionInput
    
    def __init__(self, framework: ReflexiveComposition):
        super().__init__()
        self.framework = framework
    
    def _run(self, text: str, domain: str = "general", extraction_type: str = "general") -> Dict[str, Any]:
        """Extract knowledge from text."""
        try:
            # Map extraction type string to enum
            extraction_type_enum = ExtractionType.GENERAL
            if extraction_type == "temporal":
                extraction_type_enum = ExtractionType.TEMPORAL
            elif extraction_type == "domain_specific":
                extraction_type_enum = ExtractionType.DOMAIN_SPECIFIC
            elif extraction_type == "schema_guided":
                extraction_type_enum = ExtractionType.SCHEMA_GUIDED
            
            # Use the framework's extraction method
            extraction = self.framework.llm2kg.extract_knowledge(
                source_text=text,
                schema=self.framework.knowledge_graph.schema,
                domain_context=domain
            )
            
            return {
                "success": True,
                "triples": [triple.dict() for triple in extraction.triples],
                "domain": extraction.domain,
                "method": extraction.extraction_method.value if hasattr(extraction.extraction_method, 'value') else str(extraction.extraction_method),
                "total_confidence": extraction.total_confidence,
                "processing_time": extraction.processing_time or 0.0,
                "errors": extraction.errors
            }
            
        except Exception as e:
            logger.error(f"Knowledge extraction tool error: {e}")
            return {
                "success": False,
                "error": str(e),
                "triples": [],
                "domain": domain,
                "method": extraction_type,
                "total_confidence": 0.0,
                "processing_time": 0.0,
                "errors": [str(e)]
            }
    
    async def _arun(self, text: str, domain: str = "general", extraction_type: str = "general") -> Dict[str, Any]:
        """Async version of extraction."""
        return self._run(text, domain, extraction_type)

# =====================================================================
# Validation Tool
# =====================================================================

class ValidationTool(BaseTool):
    """Tool for validating extracted knowledge."""
    
    name: str = "validate_knowledge"
    description: str = """
    Validate extracted knowledge triples through human-in-the-loop or automated validation.
    Use this tool to ensure quality before adding knowledge to the graph.
    """
    args_schema: Type[BaseModel] = ValidationInput
    
    def __init__(self, framework: ReflexiveComposition):
        super().__init__()
        self.framework = framework
    
    def _run(self, triples: List[Dict[str, Any]], source_text: str, domain: str = "general") -> Dict[str, Any]:
        """Validate knowledge triples."""
        try:
            validation_results = []
            validated_triples = []
            auto_decisions = 0
            human_decisions = 0
            
            for triple_data in triples:
                # Convert to Triple object
                try:
                    triple = Triple(
                        subject=triple_data["subject"],
                        predicate=triple_data["predicate"],
                        object=triple_data["object"],
                        confidence=triple_data.get("confidence", 0.8),
                        source=triple_data.get("source"),
                        metadata=triple_data.get("metadata", {})
                    )
                except Exception as convert_error:
                    logger.warning(f"Failed to convert triple data: {convert_error}")
                    continue
                
                # Get related knowledge for context
                related_knowledge = self.framework.knowledge_graph.query_related(
                    triple.subject, max_hops=1
                )
                
                # Create validation context
                context = ValidationContext(
                    triple=triple,
                    source_text=source_text,
                    existing_knowledge=related_knowledge,
                    confidence_threshold=self.framework.validator.auto_accept_threshold if self.framework.validator else 0.7,
                    domain=domain
                )
                
                # Validate using the framework's validator
                if self.framework.validator:
                    result = self.framework.validator.validate_triple(context)
                    validation_results.append({
                        "decision": result.decision.value,
                        "reason": result.reason,
                        "timestamp": result.timestamp,
                        "confidence": result.confidence
                    })
                    
                    # Track decision types
                    if result.decision in [ValidationDecision.ACCEPT]:
                        if "auto" in str(result.reason).lower():
                            auto_decisions += 1
                        else:
                            human_decisions += 1
                        
                        # Add to validated triples
                        if result.decision == ValidationDecision.ACCEPT:
                            validated_triples.append(triple.to_dict())
                        elif result.decision == ValidationDecision.MODIFY and result.modified_triple:
                            validated_triples.append(result.modified_triple.to_dict())
                    else:
                        human_decisions += 1
                else:
                    # No validator available, auto-accept based on confidence
                    if triple.confidence >= 0.7:
                        validated_triples.append(triple.to_dict())
                        auto_decisions += 1
                        validation_results.append({
                            "decision": "accept",
                            "reason": f"Auto-accepted (confidence: {triple.confidence})",
                            "timestamp": datetime.utcnow().isoformat(),
                            "confidence": triple.confidence
                        })
                    else:
                        validation_results.append({
                            "decision": "reject",
                            "reason": f"Auto-rejected (confidence: {triple.confidence})",
                            "timestamp": datetime.utcnow().isoformat(),
                            "confidence": triple.confidence
                        })
            
            return {
                "success": True,
                "validation_results": validation_results,
                "validated_triples": validated_triples,
                "total_validated": len(validated_triples),
                "auto_decisions": auto_decisions,
                "human_decisions": human_decisions
            }
            
        except Exception as e:
            logger.error(f"Validation tool error: {e}")
            return {
                "success": False,
                "error": str(e),
                "validation_results": [],
                "validated_triples": [],
                "total_validated": 0,
                "auto_decisions": 0,
                "human_decisions": 0
            }
    
    async def _arun(self, triples: List[Dict[str, Any]], source_text: str, domain: str = "general") -> Dict[str, Any]:
        """Async version of validation."""
        return self._run(triples, source_text, domain)

# =====================================================================
# Knowledge Graph Update Tool
# =====================================================================

class KnowledgeGraphUpdateTool(BaseTool):
    """Tool for updating the knowledge graph."""
    
    name: str = "update_knowledge_graph"
    description: str = """
    Update the knowledge graph with validated triples.
    Use this tool after knowledge has been extracted and validated.
    """
    args_schema: Type[BaseModel] = KGUpdateInput
    
    def __init__(self, framework: ReflexiveComposition):
        super().__init__()
        self.framework = framework
    
    def _run(self, validated_triples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Update knowledge graph with validated triples."""
        try:
            # Convert to Triple objects
            triples = []
            for triple_data in validated_triples:
                try:
                    triple = Triple(
                        subject=triple_data["subject"],
                        predicate=triple_data["predicate"],
                        object=triple_data["object"],
                        confidence=triple_data.get("confidence", 0.8),
                        source=triple_data.get("source"),
                        timestamp=triple_data.get("timestamp"),
                        metadata=triple_data.get("metadata", {})
                    )
                    triples.append(triple)
                except Exception as convert_error:
                    logger.warning(f"Failed to convert triple data for update: {convert_error}")
                    continue
            
            # Update knowledge graph
            added_count = self.framework.knowledge_graph.add_triples(triples)
            
            # Get updated statistics
            kg_stats = self.framework.knowledge_graph.get_stats()
            
            return {
                "success": True,
                "triples_added": added_count,
                "total_triples": kg_stats.get("num_triples", 0),
                "total_entities": kg_stats.get("num_entities", 0),
                "kg_stats": kg_stats
            }
            
        except Exception as e:
            logger.error(f"Knowledge graph update tool error: {e}")
            return {
                "success": False,
                "error": str(e),
                "triples_added": 0,
                "total_triples": 0,
                "total_entities": 0
            }
    
    async def _arun(self, validated_triples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Async version of update."""
        return self._run(validated_triples)

# =====================================================================
# Response Generation Tool
# =====================================================================

class ResponseGenerationTool(BaseTool):
    """Tool for generating knowledge-grounded responses."""
    
    name: str = "generate_response"
    description: str = """
    Generate a response to a query using knowledge from the knowledge graph.
    Use this tool when you need to answer questions based on stored knowledge.
    """
    args_schema: Type[BaseModel] = ResponseInput
    
    def __init__(self, framework: ReflexiveComposition):
        super().__init__()
        self.framework = framework
    
    def _run(self, query: str, max_context_triples: int = 10) -> Dict[str, Any]:
        """Generate a knowledge-grounded response."""
        try:
            # Use the framework's response generation method
            response_data = self.framework.kg2llm.generate_response(
                query, 
                self.framework.knowledge_graph,
                max_context_triples
            )
            
            return {
                "success": True,
                "response": response_data["response"],
                "knowledge_used": response_data["knowledge_used"],
                "context_triples": response_data["context_triples"],
                "query": response_data["query"]
            }
            
        except Exception as e:
            logger.error(f"Response generation tool error: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": f"I'm sorry, I encountered an error while generating a response: {str(e)}",
                "knowledge_used": 0,
                "context_triples": [],
                "query": query
            }
    
    async def _arun(self, query: str, max_context_triples: int = 10) -> Dict[str, Any]:
        """Async version of response generation."""
        return self._run(query, max_context_triples)

# =====================================================================
# Schema Evolution Tool
# =====================================================================

class SchemaEvolutionTool(BaseTool):
    """Tool for evolving the knowledge graph schema."""
    
    name: str = "evolve_schema"
    description: str = """
    Evolve the knowledge graph schema based on recent extractions.
    Use this tool when new types of entities or relationships are discovered.
    """
    args_schema: Type[BaseModel] = SchemaEvolutionInput
    
    def __init__(self, framework: ReflexiveComposition):
        super().__init__()
        self.framework = framework
    
    def _run(self, recent_extractions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evolve schema based on recent extractions."""
        try:
            # Convert extraction data to the format expected by the framework
            from ..core.data_models import KnowledgeExtraction, ExtractedTriple
            
            extractions = []
            for extraction_data in recent_extractions:
                triples = []
                for triple_data in extraction_data.get("triples", []):
                    try:
                        triple = ExtractedTriple(
                            subject=triple_data["subject"],
                            predicate=triple_data["predicate"],
                            object=triple_data["object"],
                            confidence=triple_data.get("confidence", 0.8),
                            source=triple_data.get("source"),
                            metadata=triple_data.get("metadata", {})
                        )
                        triples.append(triple)
                    except Exception as triple_error:
                        logger.warning(f"Failed to convert triple for schema evolution: {triple_error}")
                        continue
                
                if triples:  # Only add extraction if it has valid triples
                    extraction = KnowledgeExtraction(
                        triples=triples,
                        domain=extraction_data.get("domain"),
                        extraction_method=ExtractionType.GENERAL,  # Default
                        total_confidence=extraction_data.get("total_confidence", 0.0)
                    )
                    extractions.append(extraction)
            
            if not extractions:
                return {
                    "success": True,
                    "schema_updated": False,
                    "message": "No valid extractions provided for schema evolution"
                }
            
            # Use the framework's schema evolution method
            updated_schema = self.framework.evolve_schema(extractions)
            
            if updated_schema:
                return {
                    "success": True,
                    "schema_updated": True,
                    "new_schema": updated_schema,
                    "version": updated_schema.get("version", 1)
                }
            else:
                return {
                    "success": True,
                    "schema_updated": False,
                    "message": "No schema updates needed based on recent extractions"
                }
                
        except Exception as e:
            logger.error(f"Schema evolution tool error: {e}")
            return {
                "success": False,
                "error": str(e),
                "schema_updated": False
            }
    
    async def _arun(self, recent_extractions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Async version of schema evolution."""
        return self._run(recent_extractions)

# =====================================================================
# Additional Utility Tools
# =====================================================================

class KnowledgeSearchTool(BaseTool):
    """Tool for searching the knowledge graph."""
    
    name: str = "search_knowledge"
    description: str = """
    Search the knowledge graph for specific entities or relationships.
    Use this tool to find existing knowledge before adding new information.
    """
    
    def __init__(self, framework: ReflexiveComposition):
        super().__init__()
        self.framework = framework
    
    def _run(self, search_query: str, max_results: int = 10) -> Dict[str, Any]:
        """Search the knowledge graph."""
        try:
            # Use the knowledge graph's search functionality
            results = self.framework.knowledge_graph.get_subgraph(search_query, max_results)
            
            return {
                "success": True,
                "results": [triple.to_dict() for triple in results],
                "result_count": len(results),
                "search_query": search_query
            }
            
        except Exception as e:
            logger.error(f"Knowledge search tool error: {e}")
            return {
                "success": False,
                "error": str(e),
                "results": [],
                "result_count": 0,
                "search_query": search_query
            }

class KnowledgeGraphStatsTool(BaseTool):
    """Tool for getting knowledge graph statistics."""
    
    name: str = "get_kg_stats"
    description: str = """
    Get statistics about the current knowledge graph.
    Use this tool to understand the size and composition of the knowledge base.
    """
    
    def __init__(self, framework: ReflexiveComposition):
        super().__init__()
        self.framework = framework
    
    def _run(self) -> Dict[str, Any]:
        """Get knowledge graph statistics."""
        try:
            stats = self.framework.knowledge_graph.get_stats()
            
            return {
                "success": True,
                "stats": stats
            }
            
        except Exception as e:
            logger.error(f"Knowledge graph stats tool error: {e}")
            return {
                "success": False,
                "error": str(e),
                "stats": {}
            }

# =====================================================================
# Tool Factory
# =====================================================================

class ToolFactory:
    """Factory for creating tools for agent workflows."""
    
    @staticmethod
    def create_all_tools(framework: ReflexiveComposition) -> List[BaseTool]:
        """Create all available tools for the framework."""
        return [
            KnowledgeExtractionTool(framework),
            ValidationTool(framework),
            KnowledgeGraphUpdateTool(framework),
            ResponseGenerationTool(framework),
            SchemaEvolutionTool(framework),
            KnowledgeSearchTool(framework),
            KnowledgeGraphStatsTool(framework)
        ]
    
    @staticmethod
    def create_core_tools(framework: ReflexiveComposition) -> List[BaseTool]:
        """Create core tools for basic functionality."""
        return [
            KnowledgeExtractionTool(framework),
            ValidationTool(framework),
            KnowledgeGraphUpdateTool(framework),
            ResponseGenerationTool(framework)
        ]
    
    @staticmethod
    def create_tool_by_name(tool_name: str, framework: ReflexiveComposition) -> Optional[BaseTool]:
        """Create a specific tool by name."""
        tool_map = {
            "extract_knowledge": KnowledgeExtractionTool,
            "validate_knowledge": ValidationTool,
            "update_knowledge_graph": KnowledgeGraphUpdateTool,
            "generate_response": ResponseGenerationTool,
            "evolve_schema": SchemaEvolutionTool,
            "search_knowledge": KnowledgeSearchTool,
            "get_kg_stats": KnowledgeGraphStatsTool
        }
        
        tool_class = tool_map.get(tool_name)
        if tool_class:
            return tool_class(framework)
        
        return None

# =====================================================================
# Export
# =====================================================================

__all__ = [
    "KnowledgeExtractionTool",
    "ValidationTool",
    "KnowledgeGraphUpdateTool", 
    "ResponseGenerationTool",
    "SchemaEvolutionTool",
    "KnowledgeSearchTool",
    "KnowledgeGraphStatsTool",
    "ToolFactory"
]