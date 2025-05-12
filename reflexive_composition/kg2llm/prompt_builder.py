# reflexive_composition/kg2llm/prompt_builder.py
"""
Prompt construction for knowledge graph enhanced LLM inference.

This module handles the construction of prompts that incorporate knowledge
graph context for improved response generation.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple

logger = logging.getLogger(__name__)

class PromptBuilder:
    """
    Builds prompts for knowledge-enhanced LLM inference.
    
    This class handles the integration of knowledge graph context into
    prompts for LLM generation.
    """
    
    def __init__(self, 
                 default_template: Optional[str] = None,
                 max_context_length: int = 2000):
        """
        Initialize the prompt builder.
        
        Args:
            default_template: Default prompt template
            max_context_length: Maximum length for context in tokens
        """
        self.max_context_length = max_context_length
        
        # Set default template if none provided
        self.default_template = default_template or """
        Context information:
        {context}
        
        User query: {query}
        
        Answer the query based on the provided context information. If the context doesn't contain 
        sufficient information to answer the query fully, indicate this clearly. Avoid making up
        information that is not supported by the provided context.
        """
    
    def build_prompt(self, 
                    query: str, 
                    context: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Build a prompt with knowledge graph context.
        
        Args:
            query: User query
            context: Knowledge graph context
            
        Returns:
            Tuple of (prompt, metadata)
        """
        # Track metadata for the prompt construction process
        meta = {
            "context_size": 0,
            "triples_used": 0,
            "format": "default"
        }
        
        # If no context is provided, use a simple prompt
        if not context:
            return f"User query: {query}", meta
        
        # Format the context based on its structure
        formatted_context, context_meta = self._format_knowledge_context(context)
        meta.update(context_meta)
        
        # Apply the template
        prompt = self.default_template.format(
            context=formatted_context,
            query=query
        )
        
        return prompt, meta
    
    def _format_knowledge_context(self, 
                                 context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Format knowledge context for inclusion in the prompt.
        
        Args:
            context: Knowledge graph context
            
        Returns:
            Tuple of (formatted_context, metadata)
        """
        meta = {
            "context_size": 0,
            "triples_used": 0,
            "context_format": "unknown"
        }
        
        # Check if we have triples
        if "triples" in context:
            return self._format_triples_context(context["triples"], context.get("schema"))
        
        # Check if we have entities
        if "entities" in context:
            return self._format_entities_context(context["entities"])
        
        # Default fallback - just convert to string
        formatted_context = str(context)
        meta["context_size"] = len(formatted_context.split())
        meta["context_format"] = "raw"
        
        return formatted_context, meta
    
    def _format_triples_context(self, 
                               triples: List[Dict[str, Any]], 
                               schema: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Format triples context for inclusion in the prompt.
        
        Args:
            triples: List of triple dictionaries
            schema: Knowledge graph schema
            
        Returns:
            Tuple of (formatted_context, metadata)
        """
        meta = {
            "context_size": 0,
            "triples_used": 0,
            "context_format": "triples"
        }
        
        # Sort triples by priority if available
        sorted_triples = sorted(
            triples,
            key=lambda t: 0 if t.get("priority") == "high" else 1
        )
        
        # Format each triple
        context_items = []
        triple_count = 0
        token_count = 0
        
        for triple in sorted_triples:
            if triple_count >= 100 or token_count >= self.max_context_length:
                # Limit reached
                break
            
            # Extract components
            subject = triple.get("subject", "")
            predicate = triple.get("predicate", "")
            obj = triple.get("object", "")
            
            # Format as a natural language statement
            statement = f"- {subject} {predicate} {obj}"
            
            # Add to context
            context_items.append(statement)
            
            # Update counts
            triple_count += 1
            token_count += len(statement.split())
        
        # Combine into a single string
        formatted_context = "\n".join(context_items)
        
        # Update metadata
        meta["context_size"] = token_count
        meta["triples_used"] = triple_count
        
        return formatted_context, meta
    
    def _format_entities_context(self, 
                               entities: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        """
        Format entities context for inclusion in the prompt.
        
        Args:
            entities: List of entity dictionaries
            
        Returns:
            Tuple of (formatted_context, metadata)
        """
        meta = {
            "context_size": 0,
            "entities_used": 0,
            "context_format": "entities"
        }
        
        # Format each entity
        context_items = []
        entity_count = 0
        token_count = 0
        
        for entity in entities:
            if entity_count >= 20 or token_count >= self.max_context_length:
                # Limit reached
                break
            
            # Extract entity information
            entity_id = entity.get("id", "")
            entity_type = entity.get("type", "")
            properties = entity.get("properties", {})
            
            # Format as a structured description
            lines = [f"Entity: {entity_id} (Type: {entity_type})"]
            
            # Add properties
            for key, value in properties.items():
                lines.append(f"  {key}: {value}")
            
            entity_text = "\n".join(lines)
            
            # Add to context
            context_items.append(entity_text)
            
            # Update counts
            entity_count += 1
            token_count += len(entity_text.split())
        
        # Combine into a single string
        formatted_context = "\n\n".join(context_items)
        
        # Update metadata
        meta["context_size"] = token_count
        meta["entities_used"] = entity_count
        
        return formatted_context, meta
    
    def build_domain_specific_prompt(self, 
                                    query: str, 
                                    context: Dict[str, Any], 
                                    domain: str) -> Tuple[str, Dict[str, Any]]:
        """
        Build a domain-specific prompt with knowledge context.
        
        Args:
            query: User query
            context: Knowledge graph context
            domain: Domain identifier
            
        Returns:
            Tuple of (prompt, metadata)
        """
        # Get domain-specific template
        template = self._get_domain_template(domain)
        
        # Format the context
        formatted_context, context_meta = self._format_knowledge_context(context)
        
        # Apply the template
        prompt = template.format(
            context=formatted_context,
            query=query
        )
        
        # Metadata
        meta = {
            **context_meta,
            "domain": domain,
            "format": "domain_specific"
        }
        
        return prompt, meta
    
    def _get_domain_template(self, domain: str) -> str:
        """
        Get a domain-specific prompt template.
        
        Args:
            domain: Domain identifier
            
        Returns:
            Prompt template
        """
        if domain == "medical":
            return """
            Clinical Information:
            {context}
            
            Patient Query: {query}
            
            Provide a medical response based on the clinical information provided. 
            Only include information that is supported by the context. If the context 
            doesn't contain enough information, acknowledge this limitation. Avoid any 
            diagnostic or treatment recommendations not supported by the provided information.
            """
        
        elif domain == "temporal":
            return """
            Current Knowledge (as of the present moment):
            {context}
            
            User Query: {query}
            
            Answer the query using only the provided current knowledge. Be precise about 
            dates and times. If the provided knowledge doesn't contain specific temporal 
            information needed to answer the query, clearly indicate this limitation.
            """
        
        elif domain == "code":
            return """
            Code Context and Documentation:
            {context}
            
            Programming Query: {query}
            
            Provide code or technical advice based only on the provided context. Focus 
            on security, best practices, and currently supported APIs as specified in the 
            context. Avoid suggesting deprecated or insecure patterns. If the context lacks 
            necessary information, acknowledge this limitation.
            """
        
        else:
            logger.warning(f"Unknown domain: {domain}. Using default template.")
            return self.default_template