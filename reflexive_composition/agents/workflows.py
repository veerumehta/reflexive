"""
LangGraph workflows for Reflexive Composition framework.
Provides autonomous agent workflows for knowledge processing.
"""

import logging
from typing import Dict, List, Any, Optional, Literal
from datetime import datetime

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langgraph.checkpoint.sqlite import SqliteSaver

# Import your existing components
from ..core.data_models import AgentState, ValidationDecision, Triple
from ..core.framework import ReflexiveComposition
from .tools import (
    KnowledgeExtractionTool, ValidationTool, KnowledgeGraphUpdateTool,
    ResponseGenerationTool, SchemaEvolutionTool
)

logger = logging.getLogger(__name__)

# =====================================================================
# Base Workflow Class
# =====================================================================

class ReflexiveWorkflow:
    """Base class for Reflexive Composition workflows."""
    
    def __init__(self, framework: ReflexiveComposition, checkpointer=None):
        self.framework = framework
        self.checkpointer = checkpointer or SqliteSaver.from_conn_string(":memory:")
        
        # Initialize tools
        self.tools = [
            KnowledgeExtractionTool(framework),
            ValidationTool(framework),
            KnowledgeGraphUpdateTool(framework),
            ResponseGenerationTool(framework),
            SchemaEvolutionTool(framework)
        ]
        
        # Create tool executor
        self.tool_executor = ToolExecutor(self.tools)
        
        # Build workflow graph
        self.graph = self._build_workflow()
        self.app = self.graph.compile(checkpointer=self.checkpointer)
    
    def _build_workflow(self) -> StateGraph:
        """Build the workflow graph. Override in subclasses."""
        raise NotImplementedError
    
    async def run(self, initial_state: AgentState) -> AgentState:
        """Run the workflow with the given initial state."""
        try:
            result = await self.app.ainvoke(initial_state)
            return result
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            raise

# =====================================================================
# Document Processing Workflow
# =====================================================================

class DocumentProcessingWorkflow(ReflexiveWorkflow):
    """Complete document processing workflow."""
    
    def _build_workflow(self) -> StateGraph:
        """Build the document processing workflow."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("extract_knowledge", self._extract_knowledge_node)
        workflow.add_node("validate_knowledge", self._validate_knowledge_node)
        workflow.add_node("update_graph", self._update_graph_node)
        workflow.add_node("evolve_schema", self._evolve_schema_node)
        workflow.add_node("generate_response", self._generate_response_node)
        workflow.add_node("error_handler", self._error_handler_node)
        
        # Set entry point
        workflow.set_entry_point("extract_knowledge")
        
        # Define conditional edges
        workflow.add_conditional_edges(
            "extract_knowledge",
            self._should_continue_after_extraction,
            {
                "validate": "validate_knowledge",
                "error": "error_handler",
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "validate_knowledge",
            self._should_continue_after_validation,
            {
                "update": "update_graph",
                "error": "error_handler",
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "update_graph",
            self._should_continue_after_update,
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
    
    def _extract_knowledge_node(self, state: AgentState) -> AgentState:
        """Extract knowledge from the source text."""
        logger.info("Executing knowledge extraction node")
        
        try:
            # Create tool invocation
            tool_input = ToolInvocation(
                tool="extract_knowledge",
                tool_input={
                    "text": state["source_text"],
                    "domain": state["domain"],
                    "extraction_type": state["extraction_type"].value
                }
            )
            
            # Execute tool
            start_time = datetime.utcnow()
            result = self.tool_executor.invoke(tool_input)
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Update state
            if result.get("success", False):
                state["extracted_knowledge"] = [result]
                state["next_action"] = "validate"
                state["processing_times"]["extraction"] = processing_time
                
                # Extract confidence scores
                triples = result.get("triples", [])
                state["confidence_scores"] = [t.get("confidence", 0.0) for t in triples]
            else:
                state["errors"].append(f"Knowledge extraction failed: {result.get('error', 'Unknown error')}")
                state["next_action"] = "error"
            
        except Exception as e:
            logger.error(f"Error in knowledge extraction node: {e}")
            state["errors"].append(f"Knowledge extraction error: {str(e)}")
            state["next_action"] = "error"
        
        state["iteration_count"] += 1
        return state
    
    def _validate_knowledge_node(self, state: AgentState) -> AgentState:
        """Validate extracted knowledge."""
        logger.info("Executing knowledge validation node")
        
        try:
            if not state["extracted_knowledge"]:
                state["warnings"].append("No knowledge to validate")
                state["next_action"] = "end"
                return state
            
            # Get extracted triples
            extraction_result = state["extracted_knowledge"][0]
            triples_data = extraction_result.get("triples", [])
            
            if not triples_data:
                state["warnings"].append("No triples extracted")
                state["next_action"] = "end"
                return state
            
            # Create tool invocation
            tool_input = ToolInvocation(
                tool="validate_knowledge",
                tool_input={
                    "triples": triples_data,
                    "source_text": state["source_text"],
                    "domain": state["domain"]
                }
            )
            
            # Execute validation
            start_time = datetime.utcnow()
            result = self.tool_executor.invoke(tool_input)
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Update state
            if result.get("success", False):
                state["validation_results"] = result["validation_results"]
                state["validated_triples"] = result["validated_triples"]
                state["processing_times"]["validation"] = processing_time
                
                if result["validated_triples"]:
                    state["next_action"] = "update"
                else:
                    state["warnings"].append("No triples passed validation")
                    state["next_action"] = "end"
            else:
                state["errors"].append(f"Knowledge validation failed: {result.get('error', 'Unknown error')}")
                state["next_action"] = "error"
                
        except Exception as e:
            logger.error(f"Error in knowledge validation node: {e}")
            state["errors"].append(f"Knowledge validation error: {str(e)}")
            state["next_action"] = "error"
        
        return state
    
    def _update_graph_node(self, state: AgentState) -> AgentState:
        """Update the knowledge graph."""
        logger.info("Executing knowledge graph update node")
        
        try:
            if not state["validated_triples"]:
                state["warnings"].append("No validated triples to add")
                state["next_action"] = "respond"
                return state
            
            # Create tool invocation
            tool_input = ToolInvocation(
                tool="update_knowledge_graph",
                tool_input={"validated_triples": state["validated_triples"]}
            )
            
            # Execute update
            start_time = datetime.utcnow()
            result = self.tool_executor.invoke(tool_input)
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Update state
            if result.get("success", False):
                state["kg_updated"] = True
                state["processing_times"]["kg_update"] = processing_time
                
                # Track current schema version
                current_version = self.framework.knowledge_graph.schema.get("version", 1)
                state["schema_version"] = current_version
                
                # Decide next action based on update results
                triples_added = result.get("triples_added", 0)
                if triples_added > 0:
                    # Check if schema evolution might be needed
                    if self._should_check_schema_evolution(state):
                        state["next_action"] = "evolve"
                    else:
                        state["next_action"] = "respond"
                else:
                    state["next_action"] = "respond"
            else:
                state["errors"].append(f"Knowledge graph update failed: {result.get('error', 'Unknown error')}")
                state["next_action"] = "error"
                
        except Exception as e:
            logger.error(f"Error in knowledge graph update node: {e}")
            state["errors"].append(f"Knowledge graph update error: {str(e)}")
            state["next_action"] = "error"
        
        return state
    
    def _evolve_schema_node(self, state: AgentState) -> AgentState:
        """Evolve the knowledge graph schema."""
        logger.info("Executing schema evolution node")
        
        try:
            # Create tool invocation
            tool_input = ToolInvocation(
                tool="evolve_schema",
                tool_input={"recent_extractions": state["extracted_knowledge"]}
            )
            
            # Execute schema evolution
            start_time = datetime.utcnow()
            result = self.tool_executor.invoke(tool_input)
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Update state
            if result.get("success", False):
                state["schema_evolved"] = result.get("schema_updated", False)
                state["processing_times"]["schema_evolution"] = processing_time
                
                if result.get("schema_updated", False):
                    new_version = result.get("version", state["schema_version"])
                    state["schema_version"] = new_version
                    logger.info(f"Schema evolved to version {new_version}")
            else:
                state["warnings"].append(f"Schema evolution failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.warning(f"Error in schema evolution node: {e}")
            state["warnings"].append(f"Schema evolution error: {str(e)}")
        
        return state
    
    def _generate_response_node(self, state: AgentState) -> AgentState:
        """Generate a knowledge-grounded response."""
        logger.info("Executing response generation node")
        
        try:
            query = state.get("query")
            
            if not query:
                # No query provided, generate summary response
                triples_added = len(state.get("validated_triples", []))
                schema_evolved = state.get("schema_evolved", False)
                
                response_parts = [f"Successfully processed document. Added {triples_added} triples to knowledge graph."]
                
                if schema_evolved:
                    response_parts.append(f"Schema evolved to version {state.get('schema_version', 'unknown')}.")
                
                state["response"] = " ".join(response_parts)
                state["knowledge_used"] = []
                return state
            
            # Generate response to query
            tool_input = ToolInvocation(
                tool="generate_response",
                tool_input={"query": query}
            )
            
            start_time = datetime.utcnow()
            result = self.tool_executor.invoke(tool_input)
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            if result.get("success", False):
                state["response"] = result["response"]
                state["knowledge_used"] = result["context_triples"]
                state["processing_times"]["response_generation"] = processing_time
            else:
                state["errors"].append(f"Response generation failed: {result.get('error', 'Unknown error')}")
                state["response"] = result.get("response", "Error generating response")
                state["knowledge_used"] = []
                
        except Exception as e:
            logger.error(f"Error in response generation node: {e}")
            state["errors"].append(f"Response generation error: {str(e)}")
            state["response"] = f"Error generating response: {str(e)}"
            state["knowledge_used"] = []
        
        return state
    
    def _error_handler_node(self, state: AgentState) -> AgentState:
        """Handle errors in the workflow."""
        logger.error(f"Error handler activated. Errors: {state['errors']}")
        
        # Generate error response
        error_summary = "; ".join(state["errors"])
        state["response"] = f"Workflow encountered errors: {error_summary}"
        state["knowledge_used"] = []
        state["workflow_state"] = "failed"
        
        return state
    
    # Conditional edge functions
    def _should_continue_after_extraction(self, state: AgentState) -> str:
        """Determine next step after knowledge extraction."""
        if state["errors"]:
            return "error"
        elif state["extracted_knowledge"]:
            return "validate"
        else:
            return "end"
    
    def _should_continue_after_validation(self, state: AgentState) -> str:
        """Determine next step after validation."""
        if state["errors"]:
            return "error"
        elif state["validated_triples"]:
            return "update"
        else:
            return "end"
    
    def _should_continue_after_update(self, state: AgentState) -> str:
        """Determine next step after knowledge graph update."""
        if state["errors"]:
            return "error"
        elif state["kg_updated"] and self._should_check_schema_evolution(state):
            return "evolve"
        else:
            return "respond"
    
    def _should_check_schema_evolution(self, state: AgentState) -> bool:
        """Determine if schema evolution should be checked."""
        # Check if we have new entity or relationship types
        if not state["validated_triples"]:
            return False
        
        # Simple heuristic: check if we added a significant number of triples
        # In practice, this would be more sophisticated
        triples_added = len(state["validated_triples"])
        return triples_added >= 5  # Threshold for considering schema evolution

# =====================================================================
# Query Processing Workflow
# =====================================================================

class QueryProcessingWorkflow(ReflexiveWorkflow):
    """Workflow focused on query processing and response generation."""
    
    def _build_workflow(self) -> StateGraph:
        """Build the query processing workflow."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("retrieve_context", self._retrieve_context_node)
        workflow.add_node("generate_response", self._generate_response_node)
        workflow.add_node("validate_response", self._validate_response_node)
        workflow.add_node("refine_response", self._refine_response_node)
        
        # Set entry point
        workflow.set_entry_point("retrieve_context")
        
        # Define edges
        workflow.add_edge("retrieve_context", "generate_response")
        
        workflow.add_conditional_edges(
            "generate_response",
            self._should_validate_response,
            {
                "validate": "validate_response",
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "validate_response",
            self._should_refine_response,
            {
                "refine": "refine_response",
                "end": END
            }
        )
        
        workflow.add_edge("refine_response", END)
        
        return workflow
    
    def _retrieve_context_node(self, state: AgentState) -> AgentState:
        """Retrieve relevant context from the knowledge graph."""
        logger.info("Retrieving context for query")
        
        try:
            query = state["query"]
            max_triples = 15  # Could be made configurable
            
            # Get relevant triples from knowledge graph
            relevant_triples = self.framework.knowledge_graph.get_subgraph(query, max_triples)
            
            # Store context
            state["knowledge_used"] = [triple.to_dict() for triple in relevant_triples]
            state["context_metadata"] = {
                "triples_retrieved": len(relevant_triples),
                "retrieval_method": "subgraph_matching"
            }
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            state["errors"].append(f"Context retrieval error: {str(e)}")
            state["knowledge_used"] = []
        
        return state
    
    def _generate_response_node(self, state: AgentState) -> AgentState:
        """Generate response using retrieved context."""
        return self._generate_response_node_impl(state)
    
    def _generate_response_node_impl(self, state: AgentState) -> AgentState:
        """Implementation of response generation."""
        logger.info("Generating response to query")
        
        try:
            tool_input = ToolInvocation(
                tool="generate_response",
                tool_input={"query": state["query"]}
            )
            
            start_time = datetime.utcnow()
            result = self.tool_executor.invoke(tool_input)
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            if result.get("success", False):
                state["response"] = result["response"]
                state["processing_times"]["response_generation"] = processing_time
            else:
                state["errors"].append(f"Response generation failed: {result.get('error', 'Unknown error')}")
                state["response"] = "I'm sorry, I couldn't generate a response to your query."
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            state["errors"].append(f"Response generation error: {str(e)}")
            state["response"] = "I'm sorry, an error occurred while generating the response."
        
        return state
    
    def _validate_response_node(self, state: AgentState) -> AgentState:
        """Validate the generated response against knowledge."""
        logger.info("Validating generated response")
        
        try:
            # Simple validation: check if response contradicts known facts
            response = state["response"]
            knowledge_used = state["knowledge_used"]
            
            # This is a simplified validation - in practice, this would be more sophisticated
            contradictions = []
            
            # Check for obvious contradictions (this is a placeholder)
            if "not found" in response.lower() and knowledge_used:
                contradictions.append("Response claims no information found but knowledge is available")
            
            state["context_metadata"]["contradictions"] = contradictions
            state["context_metadata"]["validation_passed"] = len(contradictions) == 0
            
        except Exception as e:
            logger.warning(f"Error validating response: {e}")
            state["warnings"].append(f"Response validation error: {str(e)}")
        
        return state
    
    def _refine_response_node(self, state: AgentState) -> AgentState:
        """Refine the response based on validation results."""
        logger.info("Refining response based on validation")
        
        try:
            # This is a placeholder for response refinement logic
            # In practice, this would involve more sophisticated refinement
            contradictions = state["context_metadata"].get("contradictions", [])
            
            if contradictions:
                # Add a disclaimer or regenerate
                original_response = state["response"]
                refined_response = f"{original_response}\n\nNote: This response has been flagged for potential inconsistencies and may require further verification."
                state["response"] = refined_response
                state["context_metadata"]["was_refined"] = True
            
        except Exception as e:
            logger.warning(f"Error refining response: {e}")
            state["warnings"].append(f"Response refinement error: {str(e)}")
        
        return state
    
    # Conditional edge functions
    def _should_validate_response(self, state: AgentState) -> str:
        """Determine if response should be validated."""
        # Always validate for now - could be made conditional based on confidence
        if state["errors"]:
            return "end"
        return "validate"
    
    def _should_refine_response(self, state: AgentState) -> str:
        """Determine if response should be refined."""
        validation_passed = state["context_metadata"].get("validation_passed", True)
        if not validation_passed:
            return "refine"
        return "end"

# =====================================================================
# Workflow Factory
# =====================================================================

class WorkflowFactory:
    """Factory for creating different types of workflows."""
    
    @staticmethod
    def create_document_processing_workflow(framework: ReflexiveComposition) -> DocumentProcessingWorkflow:
        """Create a document processing workflow."""
        return DocumentProcessingWorkflow(framework)
    
    @staticmethod
    def create_query_processing_workflow(framework: ReflexiveComposition) -> QueryProcessingWorkflow:
        """Create a query processing workflow."""
        return QueryProcessingWorkflow(framework)
    
    @staticmethod
    def create_workflow(workflow_type: Literal["document", "query"], 
                       framework: ReflexiveComposition) -> ReflexiveWorkflow:
        """Create a workflow of the specified type."""
        if workflow_type == "document":
            return WorkflowFactory.create_document_processing_workflow(framework)
        elif workflow_type == "query":
            return WorkflowFactory.create_query_processing_workflow(framework)
        else:
            raise ValueError(f"Unknown workflow type: {workflow_type}")

# =====================================================================
# Export
# =====================================================================

__all__ = [
    "ReflexiveWorkflow",
    "DocumentProcessingWorkflow", 
    "QueryProcessingWorkflow",
    "WorkflowFactory"
]
