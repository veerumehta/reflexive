
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
