# reflexive_composition/core.py
"""
Core components for the Reflexive Composition framework that enables 
bidirectional enhancement between Language Models and Knowledge Graphs.

The framework consists of three main components:
1. LLM2KG: LLM-based knowledge extraction and graph construction
2. HITL: Human-in-the-loop validation framework
3. KG2LLM: Knowledge graph enhanced LLM inference
"""
import logging
from typing import Dict, List, Optional, Any, Union, Tuple

logger = logging.getLogger(__name__)

class ReflexiveComposition:
    """
    Main framework class that orchestrates the reflexive interaction between
    LLMs, knowledge graphs, and human validators.
    """
    
    def __init__(self, 
                 kb_llm_config: Dict[str, Any],
                 target_llm_config: Dict[str, Any],
                 kg_config: Dict[str, Any],
                 hitl_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Reflexive Composition framework.
        
        Args:
            kb_llm_config: Configuration for the Knowledge Builder LLM
            target_llm_config: Configuration for the Target LLM
            kg_config: Configuration for the Knowledge Graph
            hitl_config: Configuration for the Human-in-the-loop component
        """
        self.kb_llm = None  # Will be initialized with LLM for extraction
        self.target_llm = None  # Will be initialized with LLM for generation
        self.knowledge_graph = None  # Will hold the knowledge graph
        self.validator = None  # Will hold the HITL validation component
        
        # Initialize components
        self._init_llm2kg(kb_llm_config)
        self._init_kg2llm(target_llm_config)
        self._init_knowledge_graph(kg_config)
        
        if hitl_config:
            self._init_hitl(hitl_config)
        else:
            logger.warning("No HITL configuration provided. Running in automated mode.")
    
    def _init_llm2kg(self, config: Dict[str, Any]) -> None:
        """Initialize the LLM2KG component."""
        from reflexive_composition.llm2kg import KnowledgeBuilderLLM
        self.kb_llm = KnowledgeBuilderLLM(**config)
    
    def _init_kg2llm(self, config: Dict[str, Any]) -> None:
        """Initialize the KG2LLM component."""
        from reflexive_composition.kg2llm import TargetLLM
        self.target_llm = TargetLLM(**config)
    
    def _init_knowledge_graph(self, config: Dict[str, Any]) -> None:
        """Initialize the Knowledge Graph component."""
        from reflexive_composition.knowledge_graph import KnowledgeGraph
        self.knowledge_graph = KnowledgeGraph(**config)
    
    def _init_hitl(self, config: Dict[str, Any]) -> None:
        """Initialize the HITL component."""
        from reflexive_composition.hitl import Validator
        self.validator = Validator(**config)
    
    def extract_knowledge(self, 
                         source_text: str, 
                         schema: Optional[Dict[str, Any]] = None,
                         confidence_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Extract knowledge from source text using the KB-LLM.
        
        Args:
            source_text: The text to extract knowledge from
            schema: Optional schema to guide extraction
            confidence_threshold: Threshold for automatic acceptance
            
        Returns:
            Extracted knowledge in structured format
        """
        # Extract candidate triples
        candidate_triples = self.kb_llm.extract(source_text, schema)
        
        # If validator is available, route low-confidence triples
        if self.validator:
            validated_triples = []
            for triple in candidate_triples:
                if triple.get('confidence', 0) >= confidence_threshold:
                    validated_triples.append(triple)
                else:
                    # Route to validator
                    validation_result = self.validator.validate_triple(
                        triple, source_text, self.knowledge_graph
                    )
                    if validation_result.get('accepted', False):
                        validated_triples.append(validation_result.get('triple', triple))
            
            return {'triples': validated_triples}
        
        # Without validator, return all triples above threshold
        return {
            'triples': [t for t in candidate_triples 
                       if t.get('confidence', 0) >= confidence_threshold]
        }
    
    def update_knowledge_graph(self, triples: List[Dict[str, Any]]) -> bool:
        """
        Update the knowledge graph with validated triples.
        
        Args:
            triples: List of validated triples to add to the KG
            
        Returns:
            Success status
        """
        return self.knowledge_graph.add_triples(triples)
    
    def generate_response(self, 
                         query: str, 
                         retrieve_context: bool = True,
                         max_context_items: int = 10) -> Dict[str, Any]:
        """
        Generate a response using the Target LLM with KG enhancement.
        
        Args:
            query: The user query
            retrieve_context: Whether to retrieve context from KG
            max_context_items: Maximum number of KG items to include
            
        Returns:
            Generated response with metadata
        """
        context = {}
        
        if retrieve_context:
            # Retrieve relevant subgraph
            context = self.knowledge_graph.retrieve_context(
                query, max_items=max_context_items
            )
        
        # Generate response with context
        response = self.target_llm.generate(query, context)
        
        # Check for contradictions
        contradictions = self._detect_contradictions(response, context)
        
        if contradictions and self.validator:
            # Route contradictions to validator
            validated_response = self.validator.validate_response(
                response, contradictions, context
            )
            return validated_response
        
        return response
    
    def _detect_contradictions(self, 
                              response: Dict[str, Any], 
                              context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect contradictions between response and KG context.
        
        Args:
            response: The generated response
            context: The KG context used for generation
            
        Returns:
            List of detected contradictions
        """
        # This would implement contradiction detection logic
        # For now, returning empty list as placeholder
        return []
    
    def reflexive_update(self, 
                         query: str, 
                         response: Dict[str, Any],
                         feedback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform a reflexive update based on feedback.
        
        This closes the loop by extracting knowledge from responses and feedback,
        validating it, and updating the knowledge graph.
        
        Args:
            query: The original query
            response: The generated response
            feedback: Optional feedback (e.g., corrections, ratings)
            
        Returns:
            Update statistics
        """
        update_stats = {
            'extracted': 0,
            'validated': 0,
            'added_to_kg': 0,
        }
        
        # Extract knowledge from response
        extracted = self.kb_llm.extract(response.get('text', ''))
        update_stats['extracted'] = len(extracted.get('triples', []))
        
        # If feedback is provided, use it for validation
        if feedback and self.validator:
            validated = self.validator.validate_with_feedback(
                extracted.get('triples', []), feedback
            )
            update_stats['validated'] = len(validated.get('triples', []))
            
            # Update KG with validated triples
            if validated.get('triples'):
                success = self.knowledge_graph.add_triples(validated['triples'])
                if success:
                    update_stats['added_to_kg'] = len(validated['triples'])
        
        return update_stats