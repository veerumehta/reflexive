"""
Reflexive Composition Framework with LangGraph Agentic Capabilities
Demonstrates how to extend the basic framework with multi-agent workflows
for autonomous knowledge graph construction and maintenance.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple, Annotated
from datetime import datetime
from enum import Enum

# LangChain and LangGraph imports
from langchain.llms.base import LLM
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import BaseTool, tool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.tools import BaseTool

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langgraph.checkpoint.sqlite import SqliteSaver
from typing_extensions import TypedDict

# Import our base framework
from reflexive_composition_langchain import (
    ReflexiveComposition, KnowledgeGraph, Triple, ValidationContext, 
    ValidationResult, ValidationDecision, TriggerType
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================================================
# State Management for LangGraph
# =====================================================================

class AgentState(TypedDict):
    """State for the reflexive composition agent workflow."""
    # Input data
    source_text: str
    query: Optional[str]
    domain: str
    
    # Processing state
    extracted_knowledge: List[Dict[str, Any]]
    validation_results: List[Dict[str, Any]]
    validated_triples: List[Dict[str, Any]]
    
    # Knowledge graph state
    kg_updated: bool
    schema_evolved: bool
    
    # Response generation
    response: Optional[str]
    knowledge_used: List[Dict[str, Any]]
    
    # Workflow control
    next_action: str
    iteration_count: int
    max_iterations: int
    
    # Error handling
    errors: List[str]
    warnings: List[str]

# =====================================================================
# Tools for Agentic Workflow
# =====================================================================

class KnowledgeExtractionTool(BaseTool):
    """Tool for extracting knowledge from text."""
    
    name: str = "extract_knowledge"
    description: str = """
    Extract structured knowledge from text in the form of triples (subject, predicate, object).
    Use this tool when you need to convert unstructured text into structured knowledge.
    """
    
    def __init__(self, reflexive_framework: ReflexiveComposition):
        super().__init__()
        self.framework = reflexive_framework
    
    def _run(self, text: str, domain: str = "general") -> Dict[str, Any]:
        """Extract knowledge from text."""
        try:
            extraction = self.framework.llm2kg.extract_knowledge(
                source_text=text,
                schema=self.framework.knowledge_graph.schema,
                domain_context=domain
            )
            
            return {
                "success": True,
                "triples": [triple.dict() for triple in extraction.triples],
                "domain": extraction.domain,
                "method": extraction.extraction_method,
                "total_confidence": extraction.total_confidence
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "triples": []
            }

class KnowledgeValidationTool(BaseTool):
    """Tool for validating extracted knowledge."""
    
    name: str = "validate_knowledge"
    description: str = """
    Validate extracted knowledge triples through human-in-the-loop or automated validation.
    Use this tool to ensure quality before adding knowledge to the graph.
    """
    
    def __init__(self, reflexive_framework: ReflexiveComposition):
        super().__init__()
        self.framework = reflexive_framework
    
    def _run(self, triples_data: List[Dict[str, Any]], source_text: str) -> Dict[str, Any]:
        """Validate knowledge triples."""
        try:
            validation_results = []
            validated_triples = []
            
            for triple_data in triples_data:
                # Convert to Triple object
                triple = Triple(
                    subject=triple_data["subject"],
                    predicate=triple_data["predicate"],
                    object=triple_data["object"],
                    confidence=triple_data["confidence"],
                    source=triple_data.get("source")
                )
                
                # Get validation context
                related_knowledge = self.framework.knowledge_graph.query_related(
                    triple.subject, max_hops=1
                )
                
                context = ValidationContext(
                    triple=triple,
                    source_text=source_text,
                    existing_knowledge=related_knowledge,
                    confidence_threshold=self.framework.hitl.auto_accept_threshold
                )
                
                # Validate
                result = self.framework.hitl.validate_triple(context)
                validation_results.append({
                    "decision": result.decision.value,
                    "reason": result.reason,
                    "timestamp": result.timestamp
                })
                
                # Add to validated if accepted or modified
                if result.decision == ValidationDecision.ACCEPT:
                    validated_triples.append(triple.to_dict())
                elif result.decision == ValidationDecision.MODIFY and result.modified_triple:
                    validated_triples.append(result.modified_triple.to_dict())
            
            return {
                "success": True,
                "validation_results": validation_results,
                "validated_triples": validated_triples,
                "total_validated": len(validated_triples)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "validation_results": [],
                "validated_triples": []
            }

class KnowledgeGraphUpdateTool(BaseTool):
    """Tool for updating the knowledge graph."""
    
    name: str = "update_knowledge_graph"
    description: str = """
    Update the knowledge graph with validated triples.
    Use this tool after knowledge has been extracted and validated.
    """
    
    def __init__(self, reflexive_framework: ReflexiveComposition):
        super().__init__()
        self.framework = reflexive_framework
    
    def _run(self, validated_triples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Update knowledge graph with validated triples."""
        try:
            # Convert to Triple objects
            triples = []
            for triple_data in validated_triples:
                triple = Triple(
                    subject=triple_data["subject"],
                    predicate=triple_data["predicate"],
                    object=triple_data["object"],
                    confidence=triple_data["confidence"],
                    source=triple_data.get("source"),
                    timestamp=triple_data.get("timestamp")
                )
                triples.append(triple)
            
            # Update knowledge graph
            added_count = self.framework.update_knowledge_graph(triples)
            
            return {
                "success": True,
                "triples_added": added_count,
                "total_triples": len(self.framework.knowledge_graph.triples),
                "kg_stats": self.framework.knowledge_graph.get_stats()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "triples_added": 0
            }

class ResponseGenerationTool(BaseTool):
    """Tool for generating knowledge-grounded responses."""
    
    name: str = "generate_response"
    description: str = """
    Generate a response to a query using knowledge from the knowledge graph.
    Use this tool when you need to answer questions based on stored knowledge.
    """
    
    def __init__(self, reflexive_framework: ReflexiveComposition):
        super().__init__()
        self.framework = reflexive_framework
    
    def _run(self, query: str) -> Dict[str, Any]:
        """Generate a knowledge-grounded response."""
        try:
            response_data = self.framework.generate_grounded_response(query)
            return {
                "success": True,
                "response": response_data["response"],
                "knowledge_used": response_data["knowledge_used"],
                "context_triples": response_data["context_triples"]
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response": f"Error generating response: {str(e)}"
            }

class SchemaEvolutionTool(BaseTool):
    """Tool for evolving the knowledge graph schema."""
    
    name: str = "evolve_schema"
    description: str = """
    Evolve the knowledge graph schema based on recent extractions.
    Use this tool when new types of entities or relationships are discovered.
    """
    
    def __init__(self, reflexive_framework: ReflexiveComposition):
        super().__init__()
        self.framework = reflexive_framework
    
    def _run(self, recent_extractions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evolve schema based on recent extractions."""
        try:
            # Convert to KnowledgeExtraction objects (simplified)
            from reflexive_composition_langchain import KnowledgeExtraction, ExtractedTriple
            
            extractions = []
            for extraction_data in recent_extractions:
                triples = [
                    ExtractedTriple(**triple_data) 
                    for triple_data in extraction_data.get("triples", [])
                ]
                extraction = KnowledgeExtraction(
                    triples=triples,
                    domain=extraction_data.get("domain"),
                    extraction_method=extraction_data.get("method", "general")
                )
                extractions.append(extraction)
            
            # Evolve schema
            updated_schema = self.framework.evolve_schema(extractions)
            
            if updated_schema:
                return {
                    "success": True,
                    "schema_updated": True,
                    "new_schema": updated_schema,
                    "version": updated_schema["version"]
                }
            else:
                return {
                    "success": True,
                    "schema_updated": False,
                    "message": "No schema updates needed"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "schema_updated": False
            }

# =====================================================================
# Agent Workflow Nodes
# =====================================================================

class ReflexiveCompositionAgent:
    """Agentic workflow for reflexive composition using LangGraph."""
    
    def __init__(self, reflexive_framework: ReflexiveComposition, llm: LLM):
        self.framework = reflexive_framework
        self.llm = llm
        
        # Initialize tools
        self.tools = [
            KnowledgeExtractionTool(reflexive_framework),
            KnowledgeValidationTool(reflexive_framework),
            KnowledgeGraphUpdateTool(reflexive_framework),
            ResponseGenerationTool(reflexive_framework),
            SchemaEvolutionTool(reflexive_framework)
        ]
        
        # Create tool executor
        self.tool_executor = ToolExecutor(self.tools)
        
        # Initialize workflow graph
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("extract_knowledge", self.extract_knowledge_node)
        workflow.add_node("validate_knowledge", self.validate_knowledge_node)
        workflow.add_node("update_graph", self.update_graph_node)
        workflow.add_node("evolve_schema", self.evolve_schema_node)
        workflow.add_node("generate_response", self.generate_response_node)
        workflow.add_node("error_handler", self.error_handler_node)
        
        # Define the flow
        workflow.set_entry_point("extract_knowledge")
        
        # Conditional edges
        workflow.add_conditional_edges(
            "extract_knowledge",
            self.should_continue_extraction,
            {
                "validate": "validate_knowledge",
                "error": "error_handler",
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "validate_knowledge",
            self.should_continue_validation,
            {
                "update": "update_graph",
                "error": "error_handler",
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "update_graph",
            self.should_continue_update,
            {
                "evolve": "evolve_schema",
                "respond": "generate_response",
                "error": "error_handler",
                "end": END
            }
        )
        
        workflow.add_edge("evolve_schema", "generate_response")
        workflow.add_edge("generate_response", END)
        workflow.add_edge("error_handler", END)
        
        return workflow
    
    def extract_knowledge_node(self, state: AgentState) -> AgentState:
        """Extract knowledge from source text."""
        logger.info("Executing knowledge extraction node")
        
        try:
            # Use the extraction tool
            tool_input = ToolInvocation(
                tool="extract_knowledge",
                tool_input={"text": state["source_text"], "domain": state["domain"]}
            )
            result = self.tool_executor.invoke(tool_input)
            
            if result["success"]:
                state["extracted_knowledge"] = [result]
                state["next_action"] = "validate"
            else:
                state["errors"].append(f"Knowledge extraction failed: {result.get('error')}")
                state["next_action"] = "error"
            
        except Exception as e:
            state["errors"].append(f"Error in knowledge extraction: {str(e)}")
            state["next_action"] = "error"
        
        state["iteration_count"] += 1
        return state
    
    def validate_knowledge_node(self, state: AgentState) -> AgentState:
        """Validate extracted knowledge."""
        logger.info("Executing knowledge validation node")
        
        try:
            if not state["extracted_knowledge"]:
                state["warnings"].append("No knowledge to validate")
                state["next_action"] = "end"
                return state
            
            # Get triples from extraction
            triples_data = state["extracted_knowledge"][0]["triples"]
            
            # Use the validation tool
            tool_input = ToolInvocation(
                tool="validate_knowledge",
                tool_input={
                    "triples_data": triples_data,
                    "source_text": state["source_text"]
                }
            )
            result = self.tool_executor.invoke(tool_input)
            
            if result["success"]:
                state["validation_results"] = result["validation_results"]
                state["validated_triples"] = result["validated_triples"]
                state["next_action"] = "update" if result["validated_triples"] else "end"
            else:
                state["errors"].append(f"Knowledge validation failed: {result.get('error')}")
                state["next_action"] = "error"
                
        except Exception as e:
            state["errors"].append(f"Error in knowledge validation: {str(e)}")
            state["next_action"] = "error"
        
        return state
    
    def update_graph_node(self, state: AgentState) -> AgentState:
        """Update the knowledge graph."""
        logger.info("Executing knowledge graph update node")
        
        try:
            if not state["validated_triples"]:
                state["warnings"].append("No validated triples to add")
                state["next_action"] = "respond"
                return state
            
            # Use the update tool
            tool_input = ToolInvocation(
                tool="update_knowledge_graph",
                tool_input={"validated_triples": state["validated_triples"]}
            )
            result = self.tool_executor.invoke(tool_input)
            
            if result["success"]:
                state["kg_updated"] = True
                # Decide whether to evolve schema
                if result["triples_added"] > 0:
                    state["next_action"] = "evolve"
                else:
                    state["next_action"] = "respond"
            else:
                state["errors"].append(f"Knowledge graph update failed: {result.get('error')}")
                state["next_action"] = "error"
                
        except Exception as e:
            state["errors"].append(f"Error in knowledge graph update: {str(e)}")
            state["next_action"] = "error"
        
        return state
    
    def evolve_schema_node(self, state: AgentState) -> AgentState:
        """Evolve the knowledge graph schema."""
        logger.info("Executing schema evolution node")
        
        try:
            # Use the schema evolution tool
            tool_input = ToolInvocation(
                tool="evolve_schema",
                tool_input={"recent_extractions": state["extracted_knowledge"]}
            )
            result = self.tool_executor.invoke(tool_input)
            
            if result["success"]:
                state["schema_evolved"] = result["schema_updated"]
                if result["schema_updated"]:
                    logger.info(f"Schema evolved to version {result['version']}")
            else:
                state["warnings"].append(f"Schema evolution failed: {result.get('error')}")
                
        except Exception as e:
            state["warnings"].append(f"Error in schema evolution: {str(e)}")
        
        return state
    
    def generate_response_node(self, state: AgentState) -> AgentState:
        """Generate a knowledge-grounded response."""
        logger.info("Executing response generation node")
        
        try:
            if not state.get("query"):
                state["response"] = "Knowledge processing completed successfully."
                state["knowledge_used"] = []
                return state
            
            # Use the response generation tool
            tool_input = ToolInvocation(
                tool="generate_response",
                tool_input={"query": state["query"]}
            )
            result = self.tool_executor.invoke(tool_input)
            
            if result["success"]:
                state["response"] = result["response"]
                state["knowledge_used"] = result["context_triples"]
            else:
                state["errors"].append(f"Response generation failed: {result.get('error')}")
                state["response"] = result["response"]  # Error response
                state["knowledge_used"] = []
                
        except Exception as e:
            state["errors"].append(f"Error in response generation: {str(e)}")
            state["response"] = f"Error generating response: {str(e)}"
            state["knowledge_used"] = []
        
        return state
    
    def error_handler_node(self, state: AgentState) -> AgentState:
        """Handle errors in the workflow."""
        logger.error(f"Error handler activated. Errors: {state['errors']}")
        
        state["response"] = f"Workflow encountered errors: {'; '.join(state['errors'])}"
        state["knowledge_used"] = []
        
        return state
    
    # Conditional edge functions
    def should_continue_extraction(self, state: AgentState) -> str:
        """Determine next step after knowledge extraction."""
        if state["errors"]:
            return "error"
        elif state["extracted_knowledge"]:
            return "validate"
        else:
            return "end"
    
    def should_continue_validation(self, state: AgentState) -> str:
        """Determine next step after validation."""
        if state["errors"]:
            return "error"
        elif state["validated_triples"]:
            return "update"
        else:
            return "end"
    
    def should_continue_update(self, state: AgentState) -> str:
        """Determine next step after knowledge graph update."""
        if state["errors"]:
            return "error"
        elif state["kg_updated"]:
            return "evolve"
        else:
            return "respond"
    
    async def process_document(self, 
                              document_text: str, 
                              query: Optional[str] = None,
                              domain: str = "general",
                              max_iterations: int = 10) -> Dict[str, Any]:
        """Process a document through the complete agentic workflow."""
        
        # Initialize state
        initial_state = AgentState(
            source_text=document_text,
            query=query,
            domain=domain,
            extracted_knowledge=[],
            validation_results=[],
            validated_triples=[],
            kg_updated=False,
            schema_evolved=False,
            response=None,
            knowledge_used=[],
            next_action="extract",
            iteration_count=0,
            max_iterations=max_iterations,
            errors=[],
            warnings=[]
        )
        
        # Compile and run the workflow
        app = self.workflow.compile()
        
        try:
            # Execute the workflow
            final_state = await app.ainvoke(initial_state)
            
            return {
                "success": True,
                "response": final_state.get("response"),
                "knowledge_used": final_state.get("knowledge_used", []),
                "triples_added": len(final_state.get("validated_triples", [])),
                "kg_updated": final_state.get("kg_updated", False),
                "schema_evolved": final_state.get("schema_evolved", False),
                "iterations": final_state.get("iteration_count", 0),
                "errors": final_state.get("errors", []),
                "warnings": final_state.get("warnings", [])
            }
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": None,
                "knowledge_used": [],
                "triples_added": 0
            }

# =====================================================================
# Integration and Usage Example
# =====================================================================

async def create_production_framework():
    """Create a production-ready Reflexive Composition framework."""
    
    # Initialize with real LLMs (you'll need to provide API keys)
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0.1,
        max_tokens=2000
    )
    
    # Alternative: Use Anthropic
    # llm = ChatAnthropic(
    #     model="claude-3-sonnet-20240229",
    #     temperature=0.1,
    #     max_tokens=2000
    # )
    
    # Initial schema for knowledge graph
    initial_schema = {
        "entity_types": ["Person", "Event", "Location", "Organization", "Date"],
        "relationship_types": ["OccurredAt", "InvolvedIn", "LocatedIn", "WorksAt", "HappenedOn"],
        "version": 1
    }
    
    # HITL configuration
    hitl_config = {
        "auto_accept_threshold": 0.9,
        "auto_reject_threshold": 0.3,
        "interactive": True  # Set to False for fully automated processing
    }
    
    # Create base framework
    reflexive_framework = ReflexiveComposition(
        llm=llm,
        initial_schema=initial_schema,
        hitl_config=hitl_config
    )
    
    # Create agentic layer
    agent = ReflexiveCompositionAgent(reflexive_framework, llm)
    
    return agent, reflexive_framework

async def demonstrate_agentic_workflow():
    """Demonstrate the agentic workflow capabilities."""
    
    print("Initializing Reflexive Composition Agentic Framework...")
    
    # Create framework (this would use real LLMs in production)
    try:
        agent, framework = await create_production_framework()
        print("✓ Framework initialized successfully")
    except Exception as e:
        print(f"✗ Framework initialization failed: {e}")
        return
    
    # Example documents for processing
    documents = [
        {
            "text": """
            Donald Trump survived an assassination attempt during a rally in Butler, Pennsylvania on July 13, 2024.
            He was grazed on the right ear by a bullet fired by a 20-year-old shooter, who was subsequently killed 
            by Secret Service agents. The incident occurred during a campaign event attended by thousands of supporters.
            Trump was immediately evacuated from the stage and later continued his campaign activities.
            """,
            "query": "What happened to Donald Trump at the rally in July 2024?",
            "domain": "news"
        },
        {
            "text": """
            Microsoft announced the acquisition of Activision Blizzard for $68.7 billion in January 2022.
            The deal faced regulatory scrutiny from multiple countries including the United States, United Kingdom,
            and European Union. After extensive review and negotiation, the acquisition was finally completed
            in October 2023, making it one of the largest gaming industry acquisitions in history.
            """,
            "query": "When did Microsoft complete the Activision Blizzard acquisition?",
            "domain": "business"
        }
    ]
    
    print(f"\nProcessing {len(documents)} documents through agentic workflow...")
    
    results = []
    for i, doc in enumerate(documents, 1):
        print(f"\n{'='*60}")
        print(f"Processing Document {i}: {doc['domain'].upper()} DOMAIN")
        print(f"{'='*60}")
        print(f"Text: {doc['text'][:100]}...")
        print(f"Query: {doc['query']}")
        
        try:
            # Process document through agentic workflow
            result = await agent.process_document(
                document_text=doc["text"],
                query=doc["query"],
                domain=doc["domain"]
            )
            
            # Display results
            if result["success"]:
                print(f"\n✓ Processing completed successfully")
                print(f"  - Triples added: {result['triples_added']}")
                print(f"  - Knowledge graph updated: {result['kg_updated']}")
                print(f"  - Schema evolved: {result['schema_evolved']}")
                print(f"  - Workflow iterations: {result['iterations']}")
                
                if result["warnings"]:
                    print(f"  - Warnings: {len(result['warnings'])}")
                    for warning in result["warnings"]:
                        print(f"    • {warning}")
                
                print(f"\nGenerated Response:")
                print(f"  {result['response']}")
                
                if result["knowledge_used"]:
                    print(f"\nKnowledge Used ({len(result['knowledge_used'])} triples):")
                    for j, triple in enumerate(result["knowledge_used"][:3], 1):  # Show first 3
                        print(f"  {j}. {triple['subject']} → {triple['predicate']} → {triple['object']}")
                    if len(result["knowledge_used"]) > 3:
                        print(f"  ... and {len(result['knowledge_used']) - 3} more triples")
                
            else:
                print(f"\n✗ Processing failed: {result.get('error')}")
            
            results.append(result)
            
        except Exception as e:
            print(f"\n✗ Error processing document: {e}")
            results.append({"success": False, "error": str(e)})
    
    # Display overall statistics
    print(f"\n{'='*60}")
    print("OVERALL STATISTICS")
    print(f"{'='*60}")
    
    framework_stats = framework.get_framework_stats()
    print(f"Framework Statistics:")
    print(f"  - Total extractions: {framework_stats['framework_stats']['extractions_performed']}")
    print(f"  - Total triples added: {framework_stats['framework_stats']['triples_added']}")
    print(f"  - Total responses generated: {framework_stats['framework_stats']['responses_generated']}")
    print(f"  - Schema updates: {framework_stats['framework_stats']['schema_updates']}")
    
    print(f"\nKnowledge Graph Statistics:")
    kg_stats = framework_stats['knowledge_graph_stats']
    print(f"  - Total entities: {kg_stats['num_entities']}")
    print(f"  - Total relationships: {kg_stats['num_relationships']}")
    print(f"  - Total triples: {kg_stats['num_triples']}")
    print(f"  - Schema version: {kg_stats['schema_version']}")
    
    print(f"\nHITL Statistics:")
    hitl_stats = framework_stats['hitl_stats']
    print(f"  - Total validations: {hitl_stats['total_validations']}")
    print(f"  - Auto-accepted: {hitl_stats['auto_accepted']}")
    print(f"  - Auto-rejected: {hitl_stats['auto_rejected']}")
    print(f"  - Human reviewed: {hitl_stats['human_reviewed']}")
    print(f"  - Modified: {hitl_stats['modified']}")
    
    return results, framework_stats

# =====================================================================
# Advanced Features and Extensions
# =====================================================================

class ReflexiveCompositionMonitor:
    """Monitoring and analytics for the Reflexive Composition framework."""
    
    def __init__(self):
        self.metrics = {
            "extraction_accuracy": [],
            "validation_efficiency": [],
            "response_quality": [],
            "schema_evolution_events": [],
            "error_rates": [],
            "processing_times": []
        }
        self.alerts = []
    
    def record_extraction_metrics(self, 
                                 extracted_count: int, 
                                 validated_count: int, 
                                 processing_time: float):
        """Record extraction performance metrics."""
        accuracy = validated_count / extracted_count if extracted_count > 0 else 0
        self.metrics["extraction_accuracy"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "accuracy": accuracy,
            "extracted": extracted_count,
            "validated": validated_count,
            "processing_time": processing_time
        })
        
        # Alert on low accuracy
        if accuracy < 0.5 and extracted_count > 5:
            self.alerts.append({
                "type": "low_extraction_accuracy",
                "timestamp": datetime.utcnow().isoformat(),
                "accuracy": accuracy,
                "message": f"Extraction accuracy dropped to {accuracy:.2%}"
            })
    
    def record_validation_metrics(self, 
                                auto_decisions: int, 
                                human_decisions: int,
                                validation_time: float):
        """Record validation efficiency metrics."""
        total_decisions = auto_decisions + human_decisions
        efficiency = auto_decisions / total_decisions if total_decisions > 0 else 0
        
        self.metrics["validation_efficiency"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "efficiency": efficiency,
            "auto_decisions": auto_decisions,
            "human_decisions": human_decisions,
            "validation_time": validation_time
        })
    
    def record_schema_evolution(self, 
                               old_version: int, 
                               new_version: int,
                               changes: Dict[str, Any]):
        """Record schema evolution events."""
        self.metrics["schema_evolution_events"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "old_version": old_version,
            "new_version": new_version,
            "changes": changes
        })
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of performance metrics."""
        summary = {}
        
        # Extraction accuracy
        if self.metrics["extraction_accuracy"]:
            accuracies = [m["accuracy"] for m in self.metrics["extraction_accuracy"]]
            summary["extraction"] = {
                "average_accuracy": sum(accuracies) / len(accuracies),
                "latest_accuracy": accuracies[-1] if accuracies else 0,
                "total_extractions": len(accuracies)
            }
        
        # Validation efficiency
        if self.metrics["validation_efficiency"]:
            efficiencies = [m["efficiency"] for m in self.metrics["validation_efficiency"]]
            summary["validation"] = {
                "average_efficiency": sum(efficiencies) / len(efficiencies),
                "latest_efficiency": efficiencies[-1] if efficiencies else 0,
                "total_validations": len(efficiencies)
            }
        
        # Schema evolution
        summary["schema"] = {
            "evolution_events": len(self.metrics["schema_evolution_events"]),
            "latest_version": self.metrics["schema_evolution_events"][-1]["new_version"] 
                            if self.metrics["schema_evolution_events"] else 1
        }
        
        # Alerts
        summary["alerts"] = {
            "total_alerts": len(self.alerts),
            "recent_alerts": [a for a in self.alerts 
                            if (datetime.utcnow() - datetime.fromisoformat(a["timestamp"])).days < 1]
        }
        
        return summary

class ReflexiveCompositionAPI:
    """REST API interface for the Reflexive Composition framework."""
    
    def __init__(self, agent: ReflexiveCompositionAgent):
        self.agent = agent
        self.monitor = ReflexiveCompositionMonitor()
    
    async def extract_knowledge_endpoint(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """API endpoint for knowledge extraction."""
        try:
            text = request_data.get("text", "")
            domain = request_data.get("domain", "general")
            
            if not text:
                return {"error": "Text is required", "status": 400}
            
            start_time = datetime.utcnow()
            
            # Process through agent
            result = await self.agent.process_document(
                document_text=text,
                domain=domain
            )
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Record metrics
            if result["success"]:
                self.monitor.record_extraction_metrics(
                    extracted_count=result.get("triples_added", 0),
                    validated_count=result.get("triples_added", 0),
                    processing_time=processing_time
                )
            
            return {
                "result": result,
                "processing_time": processing_time,
                "status": 200 if result["success"] else 500
            }
            
        except Exception as e:
            return {"error": str(e), "status": 500}
    
    async def query_knowledge_endpoint(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """API endpoint for querying knowledge."""
        try:
            query = request_data.get("query", "")
            
            if not query:
                return {"error": "Query is required", "status": 400}
            
            # Generate response using the framework
            response_data = self.agent.framework.generate_grounded_response(query)
            
            return {
                "response": response_data["response"],
                "knowledge_used": response_data["knowledge_used"],
                "context_triples": response_data["context_triples"],
                "status": 200
            }
            
        except Exception as e:
            return {"error": str(e), "status": 500}
    
    async def get_metrics_endpoint(self) -> Dict[str, Any]:
        """API endpoint for getting performance metrics."""
        try:
            performance_summary = self.monitor.get_performance_summary()
            framework_stats = self.agent.framework.get_framework_stats()
            
            return {
                "performance": performance_summary,
                "framework_stats": framework_stats,
                "status": 200
            }
            
        except Exception as e:
            return {"error": str(e), "status": 500}

# =====================================================================
# Production Deployment Example
# =====================================================================

class ProductionReflexiveComposition:
    """Production-ready deployment of Reflexive Composition."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agent = None
        self.framework = None
        self.api = None
        self.monitor = None
    
    async def initialize(self):
        """Initialize the production system."""
        try:
            # Initialize LLM based on config
            if self.config["llm"]["provider"] == "openai":
                llm = ChatOpenAI(
                    model=self.config["llm"]["model"],
                    api_key=self.config["llm"]["api_key"],
                    temperature=self.config["llm"].get("temperature", 0.1),
                    max_tokens=self.config["llm"].get("max_tokens", 2000)
                )
            elif self.config["llm"]["provider"] == "anthropic":
                llm = ChatAnthropic(
                    model=self.config["llm"]["model"],
                    api_key=self.config["llm"]["api_key"],
                    temperature=self.config["llm"].get("temperature", 0.1),
                    max_tokens=self.config["llm"].get("max_tokens", 2000)
                )
            else:
                raise ValueError(f"Unsupported LLM provider: {self.config['llm']['provider']}")
            
            # Initialize framework
            self.framework = ReflexiveComposition(
                llm=llm,
                initial_schema=self.config.get("schema", {}),
                hitl_config=self.config.get("hitl", {})
            )
            
            # Initialize agent
            self.agent = ReflexiveCompositionAgent(self.framework, llm)
            
            # Initialize API and monitoring
            self.api = ReflexiveCompositionAPI(self.agent)
            self.monitor = self.api.monitor
            
            logger.info("Production Reflexive Composition system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize production system: {e}")
            raise
    
    async def process_batch(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of documents."""
        results = []
        
        for doc in documents:
            try:
                result = await self.agent.process_document(
                    document_text=doc["text"],
                    query=doc.get("query"),
                    domain=doc.get("domain", "general")
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing document: {e}")
                results.append({"success": False, "error": str(e)})
        
        return results
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status."""
        return {
            "status": "healthy" if self.agent and self.framework else "unhealthy",
            "components": {
                "agent": self.agent is not None,
                "framework": self.framework is not None,
                "api": self.api is not None,
                "monitor": self.monitor is not None
            },
            "metrics": self.monitor.get_performance_summary() if self.monitor else {},
            "timestamp": datetime.utcnow().isoformat()
        }

# =====================================================================
# Configuration and Deployment
# =====================================================================

def create_production_config() -> Dict[str, Any]:
    """Create a production configuration."""
    return {
        "llm": {
            "provider": "openai",  # or "anthropic"
            "model": "gpt-4",
            "api_key": "your-api-key-here",
            "temperature": 0.1,
            "max_tokens": 2000
        },
        "schema": {
            "entity_types": [
                "Person", "Organization", "Location", "Event", 
                "Product", "Technology", "Date", "Currency"
            ],
            "relationship_types": [
                "WorksAt", "LocatedIn", "OccurredAt", "InvolvedIn",
                "ManufacturedBy", "AcquiredBy", "PartOf", "RelatedTo"
            ],
            "version": 1
        },
        "hitl": {
            "auto_accept_threshold": 0.9,
            "auto_reject_threshold": 0.3,
            "interactive": False  # Set to True for manual validation
        },
        "monitoring": {
            "enabled": True,
            "alert_thresholds": {
                "extraction_accuracy": 0.5,
                "validation_efficiency": 0.8
            }
        }
    }

if __name__ == "__main__":
    # Example usage
    print("Reflexive Composition Agentic Framework")
    print("=" * 50)
    
    # Run the demonstration
    asyncio.run(demonstrate_agentic_workflow())