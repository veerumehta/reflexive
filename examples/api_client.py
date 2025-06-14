
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
    
    print("\nQuery Result:")
    print(json.dumps(query_result, indent=2))
    
    # Get metrics
    metrics = client.get_metrics()
    print("\nSystem Metrics:")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
