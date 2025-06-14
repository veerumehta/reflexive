"""
LangChain adapter for Reflexive Composition framework.
Provides integration layer between existing RC components and LangChain ecosystem.
"""

import logging
from typing import Dict, List, Any, Optional, Type
from abc import ABC, abstractmethod

# LangChain imports
from langchain.chains.base import Chain
from langchain.prompts import PromptTemplate
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.schema.runnable import Runnable
from langchain.callbacks.base import BaseCallbackHandler
from langchain.llms.base import LLM
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser

# Import existing RC components
from ..core.data_models import (
    Triple, KnowledgeExtraction, SchemaUpdate, ExtractionRequest,
    ValidationContext, ValidationResult, LLMConfig
)
from ..core.framework import ReflexiveComposition
from ..llm2kg import KnowledgeBuilderLLM
from ..kg2llm import TargetLLM
from ..hitl import Validator
from ..knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)

# =====================================================================
# Base Adapter Classes
# =====================================================================

class ReflexiveCompositionAdapter(ABC):
    """Base adapter for integrating RC components with LangChain."""
    
    def __init__(self, rc_component, llm: Optional[LLM] = None):
        self.rc_component = rc_component
        self.llm = llm
        self._initialized = False
    
    @abstractmethod
    def adapt(self) -> Runnable:
        """Convert RC component to LangChain Runnable."""
        pass
    
    def initialize(self):
        """Initialize the adapter."""
        if not self._initialized:
            self._setup()
            self._initialized = True
    
    def _setup(self):
        """Setup method for subclasses to override."""
        pass

# =====================================================================
# Knowledge Extraction Chain
# =====================================================================

class KnowledgeExtractionChain(Chain):
    """LangChain wrapper for RC knowledge extraction."""
    
    def __init__(self, 
                 kb_llm: KnowledgeBuilderLLM,
                 llm: LLM,
                 callbacks: Optional[List[BaseCallbackHandler]] = None):
        super().__init__(callbacks=callbacks)
        self.kb_llm = kb_llm
        self.llm = llm
        
        # Initialize output parser
        self.output_parser = PydanticOutputParser(pydantic_object=KnowledgeExtraction)
        self.fixing_parser = OutputFixingParser.from_llm(
            parser=self.output_parser, 
            llm=llm
        )
        
        # Setup prompt
        self._setup_prompt()
    
    def _setup_prompt(self):
        """Setup the extraction prompt template."""
        template = """
        You are an expert knowledge extraction system. Extract structured knowledge 
        from the following text according to the provided schema.
        
        Text to analyze:
        {source_text}
        
        Domain context: {domain}
        Extraction type: {extraction_type}
        
        Schema guidelines:
        Entity types: {entity_types}
        Relationship types: {relationship_types}
        
        Instructions:
        1. Extract entities and relationships as triples (subject, predicate, object)
        2. Only extract information explicitly stated in the text
        3. Assign confidence scores between 0.0 and 1.0
        4. Use specific entity names, not general concepts
        5. Follow the provided schema when possible
        
        {format_instructions}
        """
        
        self.prompt = PromptTemplate(
            template=template,
            input_variables=[
                "source_text", "domain", "extraction_type", 
                "entity_types", "relationship_types"
            ],
            partial_variables={
                "format_instructions": self.output_parser.get_format_instructions()
            }
        )
    
    @property
    def input_keys(self) -> List[str]:
        return ["source_text", "domain", "extraction_type", "schema"]
    
    @property
    def output_keys(self) -> List[str]:
        return ["extraction_result"]
    
    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the knowledge extraction chain."""
        try:
            # Prepare schema information
            schema = inputs.get("schema", {})
            entity_types = ", ".join(schema.get("entity_types", []))
            relationship_types = ", ".join(schema.get("relationship_types", []))
            
            # Format prompt
            prompt_input = {
                "source_text": inputs["source_text"],
                "domain": inputs.get("domain", "general"),
                "extraction_type": inputs.get("extraction_type", "general"),
                "entity_types": entity_types,
                "relationship_types": relationship_types
            }
            
            formatted_prompt = self.prompt.format(**prompt_input)
            
            # Generate with LLM
            response = self.llm(formatted_prompt)
            
            # Parse response
            try:
                extraction = self.output_parser.parse(response)
            except Exception as parse_error:
                logger.warning(f"Primary parsing failed, using fixing parser: {parse_error}")
                extraction = self.fixing_parser.parse(response)
            
            return {"extraction_result": extraction}
            
        except Exception as e:
            logger.error(f"Error in knowledge extraction chain: {e}")
            return {
                "extraction_result": KnowledgeExtraction(
                    triples=[],
                    errors=[str(e)]
                )
            }

# =====================================================================
# Validation Chain
# =====================================================================

class ValidationChain(Chain):
    """LangChain wrapper for RC validation."""
    
    def __init__(self, 
                 validator: Validator,
                 knowledge_graph: KnowledgeGraph,
                 callbacks: Optional[List[BaseCallbackHandler]] = None):
        super().__init__(callbacks=callbacks)
        self.validator = validator
        self.knowledge_graph = knowledge_graph
    
    @property
    def input_keys(self) -> List[str]:
        return ["triples", "source_text", "domain"]
    
    @property
    def output_keys(self) -> List[str]:
        return ["validation_results", "validated_triples"]
    
    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the validation chain."""
        try:
            triples_data = inputs["triples"]
            source_text = inputs["source_text"]
            domain = inputs.get("domain", "general")
            
            validation_results = []
            validated_triples = []
            
            for triple_data in triples_data:
                # Convert to RC Triple format
                if isinstance(triple_data, dict):
                    triple = Triple(
                        subject=triple_data["subject"],
                        predicate=triple_data["predicate"],
                        object=triple_data["object"],
                        confidence=triple_data.get("confidence", 0.8),
                        source=triple_data.get("source")
                    )
                else:
                    triple = triple_data
                
                # Get related knowledge for context
                related_knowledge = self.knowledge_graph.query_related(
                    triple.subject, max_hops=1
                )
                
                # Create validation context
                context = ValidationContext(
                    triple=triple,
                    source_text=source_text,
                    existing_knowledge=related_knowledge,
                    confidence_threshold=self.validator.auto_accept_threshold,
                    domain=domain
                )
                
                # Validate
                result = self.validator.validate_triple(context)
                validation_results.append(result)
                
                # Add to validated if accepted or modified
                from ..core.data_models import ValidationDecision
                if result.decision == ValidationDecision.ACCEPT:
                    validated_triples.append(triple)
                elif result.decision == ValidationDecision.MODIFY and result.modified_triple:
                    validated_triples.append(result.modified_triple)
            
            return {
                "validation_results": validation_results,
                "validated_triples": validated_triples
            }
            
        except Exception as e:
            logger.error(f"Error in validation chain: {e}")
            return {
                "validation_results": [],
                "validated_triples": []
            }

# =====================================================================
# Response Generation Chain
# =====================================================================

class ResponseGenerationChain(Chain):
    """LangChain wrapper for RC response generation."""
    
    def __init__(self, 
                 target_llm: TargetLLM,
                 knowledge_graph: KnowledgeGraph,
                 callbacks: Optional[List[BaseCallbackHandler]] = None):
        super().__init__(callbacks=callbacks)
        self.target_llm = target_llm
        self.knowledge_graph = knowledge_graph
        
        # Setup prompt
        self._setup_prompt()
    
    def _setup_prompt(self):
        """Setup the response generation prompt."""
        template = """
        You are a helpful assistant that provides accurate information based on verified knowledge.
        
        User Query: {query}
        
        Relevant Knowledge Context:
        {knowledge_context}
        
        Instructions:
        1. Answer the user's query using the provided knowledge context when available
        2. If the knowledge context contains relevant information, prioritize it over your training data
        3. If the knowledge context is insufficient, clearly state what information is missing
        4. Do not make up facts not present in the knowledge context
        5. Cite specific pieces of knowledge when relevant
        6. Maintain a helpful and informative tone
        
        Response:
        """
        
        self.prompt = PromptTemplate(
            template=template,
            input_variables=["query", "knowledge_context"]
        )
    
    @property
    def input_keys(self) -> List[str]:
        return ["query", "max_context_triples"]
    
    @property
    def output_keys(self) -> List[str]:
        return ["response", "knowledge_used", "context_triples"]
    
    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the response generation chain."""
        try:
            query = inputs["query"]
            max_context_triples = inputs.get("max_context_triples", 10)
            
            # Retrieve relevant knowledge
            relevant_triples = self.knowledge_graph.get_subgraph(
                query, max_context_triples
            )
            
            # Format knowledge context
            if relevant_triples:
                context_lines = []
                for i, triple in enumerate(relevant_triples, 1):
                    context_lines.append(
                        f"{i}. {triple.subject} {triple.predicate} {triple.object} "
                        f"(confidence: {triple.confidence:.2f})"
                    )
                knowledge_context = "\n".join(context_lines)
            else:
                knowledge_context = "No relevant knowledge found in the knowledge graph."
            
            # Format prompt
            formatted_prompt = self.prompt.format(
                query=query,
                knowledge_context=knowledge_context
            )
            
            # Generate response using target LLM
            response = self.target_llm.generate(formatted_prompt)
            
            return {
                "response": response.get("text", ""),
                "knowledge_used": len(relevant_triples),
                "context_triples": [triple.to_dict() for triple in relevant_triples]
            }
            
        except Exception as e:
            logger.error(f"Error in response generation chain: {e}")
            return {
                "response": f"Error generating response: {str(e)}",
                "knowledge_used": 0,
                "context_triples": []
            }

# =====================================================================
# Complete Reflexive Composition Chain
# =====================================================================

class ReflexiveCompositionChain(Chain):
    """Complete LangChain wrapper for Reflexive Composition."""
    
    def __init__(self, 
                 framework: ReflexiveComposition,
                 llm: LLM,
                 callbacks: Optional[List[BaseCallbackHandler]] = None):
        super().__init__(callbacks=callbacks)
        self.framework = framework
        self.llm = llm
        
        # Initialize sub-chains
        self.extraction_chain = KnowledgeExtractionChain(
            kb_llm=framework.kb_llm,
            llm=llm,
            callbacks=callbacks
        )
        
        self.validation_chain = ValidationChain(
            validator=framework.validator,
            knowledge_graph=framework.knowledge_graph,
            callbacks=callbacks
        )
        
        self.generation_chain = ResponseGenerationChain(
            target_llm=framework.target_llm,
            knowledge_graph=framework.knowledge_graph,
            callbacks=callbacks
        )
    
    @property
    def input_keys(self) -> List[str]:
        return ["source_text", "query", "domain", "schema"]
    
    @property
    def output_keys(self) -> List[str]:
        return [
            "extraction_result", "validation_results", "response",
            "triples_added", "knowledge_used"
        ]
    
    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the complete reflexive composition pipeline."""
        try:
            results = {}
            
            # Step 1: Extract knowledge
            extraction_result = self.extraction_chain(inputs)
            results["extraction_result"] = extraction_result["extraction_result"]
            
            # Step 2: Validate extracted knowledge
            validation_inputs = {
                "triples": [t.dict() for t in extraction_result["extraction_result"].triples],
                "source_text": inputs["source_text"],
                "domain": inputs.get("domain", "general")
            }
            validation_result = self.validation_chain(validation_inputs)
            results["validation_results"] = validation_result["validation_results"]
            
            # Step 3: Update knowledge graph
            validated_triples = validation_result["validated_triples"]
            triples_added = self.framework.knowledge_graph.add_triples(validated_triples)
            results["triples_added"] = triples_added
            
            # Step 4: Generate response (if query provided)
            if inputs.get("query"):
                generation_inputs = {
                    "query": inputs["query"],
                    "max_context_triples": inputs.get("max_context_triples", 10)
                }
                generation_result = self.generation_chain(generation_inputs)
                results.update(generation_result)
            else:
                results.update({
                    "response": f"Successfully processed document. Added {triples_added} triples to knowledge graph.",
                    "knowledge_used": 0,
                    "context_triples": []
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in reflexive composition chain: {e}")
            return {
                "extraction_result": KnowledgeExtraction(triples=[], errors=[str(e)]),
                "validation_results": [],
                "response": f"Error processing request: {str(e)}",
                "triples_added": 0,
                "knowledge_used": 0
            }

# =====================================================================
# Adapter Factory
# =====================================================================

class AdapterFactory:
    """Factory for creating LangChain adapters for RC components."""
    
    @staticmethod
    def create_extraction_chain(framework: ReflexiveComposition, 
                              llm: LLM,
                              callbacks: Optional[List[BaseCallbackHandler]] = None) -> KnowledgeExtractionChain:
        """Create knowledge extraction chain."""
        return KnowledgeExtractionChain(
            kb_llm=framework.kb_llm,
            llm=llm,
            callbacks=callbacks
        )
    
    @staticmethod
    def create_validation_chain(framework: ReflexiveComposition,
                              callbacks: Optional[List[BaseCallbackHandler]] = None) -> ValidationChain:
        """Create validation chain."""
        return ValidationChain(
            validator=framework.validator,
            knowledge_graph=framework.knowledge_graph,
            callbacks=callbacks
        )
    
    @staticmethod
    def create_generation_chain(framework: ReflexiveComposition,
                              callbacks: Optional[List[BaseCallbackHandler]] = None) -> ResponseGenerationChain:
        """Create response generation chain."""
        return ResponseGenerationChain(
            target_llm=framework.target_llm,
            knowledge_graph=framework.knowledge_graph,
            callbacks=callbacks
        )
    
    @staticmethod
    def create_complete_chain(framework: ReflexiveComposition,
                            llm: LLM,
                            callbacks: Optional[List[BaseCallbackHandler]] = None) -> ReflexiveCompositionChain:
        """Create complete reflexive composition chain."""
        return ReflexiveCompositionChain(
            framework=framework,
            llm=llm,
            callbacks=callbacks
        )

# =====================================================================
# Utility Functions
# =====================================================================

def convert_rc_config_to_langchain(rc_config: LLMConfig) -> Dict[str, Any]:
    """Convert RC LLM config to LangChain LLM initialization parameters."""
    return {
        "model": rc_config.model,
        "temperature": rc_config.temperature,
        "max_tokens": rc_config.max_tokens,
        "timeout": rc_config.timeout,
        "max_retries": rc_config.max_retries
    }

def create_langchain_llm(rc_config: LLMConfig) -> LLM:
    """Create LangChain LLM from RC configuration."""
    config_dict = convert_rc_config_to_langchain(rc_config)
    
    if rc_config.provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            api_key=rc_config.api_key,
            **config_dict
        )
    elif rc_config.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            api_key=rc_config.api_key,
            **config_dict
        )
    elif rc_config.provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            google_api_key=rc_config.api_key,
            **config_dict
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {rc_config.provider}")

# =====================================================================
# Export
# =====================================================================

__all__ = [
    "ReflexiveCompositionAdapter",
    "KnowledgeExtractionChain",
    "ValidationChain", 
    "ResponseGenerationChain",
    "ReflexiveCompositionChain",
    "AdapterFactory",
    "convert_rc_config_to_langchain",
    "create_langchain_llm"
]
