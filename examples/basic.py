
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
