
# Reflexive Composition Framework

A sophisticated framework for bidirectional enhancement between Large Language Models (LLMs) and Knowledge Graphs with strategic human-in-the-loop validation.

## Features

- **LLM2KG**: Automated knowledge extraction from text using LLMs
- **Knowledge Graph**: Scalable graph storage with NetworkX backend
- **Human-in-the-Loop**: Strategic validation with configurable thresholds
- **KG2LLM**: Knowledge-grounded response generation
- **Agentic Workflows**: LangGraph-based autonomous processing
- **REST API**: Production-ready API with FastAPI
- **Monitoring**: Comprehensive metrics and health monitoring

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file:

```bash
# API Keys
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key

# Configuration
HITL_INTERACTIVE=false
HITL_AUTO_ACCEPT=0.9
HITL_AUTO_REJECT=0.3
```

### 3. Basic Usage

```python
import asyncio
from reflexive_composition_agentic import ProductionReflexiveComposition

async def main():
    config = {
        "llm": {
            "provider": "openai",
            "model": "gpt-4",
            "api_key": "your-api-key"
        }
    }
    
    system = ProductionReflexiveComposition(config)
    await system.initialize()
    
    result = await system.agent.process_document(
        document_text="Your text here",
        query="Your question here",
        domain="general"
    )
    
    print(result)

asyncio.run(main())
```

### 4. API Server

```bash
python api_server.py
```

Then access the API at `http://localhost:8000/docs`

## Architecture

The framework consists of three main components:

1. **LLM2KG**: Extracts structured knowledge from unstructured text
2. **HITL**: Validates extracted knowledge through human oversight
3. **KG2LLM**: Generates responses grounded in validated knowledge

## API Endpoints

- `POST /extract` - Extract knowledge from text
- `POST /query` - Query the knowledge graph
- `GET /metrics` - Get system metrics
- `GET /schema` - Get knowledge graph schema
- `POST /batch` - Process multiple documents
- `POST /upload` - Upload and process files

## Testing

```bash
pytest test_reflexive_composition.py
```

## Examples

See the `examples/` directory for comprehensive usage examples:

- `example_basic.py` - Basic framework usage
- `example_api_client.py` - API client usage

## Configuration

The system can be configured through environment variables or YAML files. See `config.yaml` for all available options.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the MIT License.
