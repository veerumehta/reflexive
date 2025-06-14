
# API Documentation

## Authentication

Currently, no authentication is required for the API endpoints. In production, consider implementing API key authentication.

## Endpoints

### POST /extract

Extract knowledge from text documents.

**Request Body:**
```json
{
    "text": "Your document text here",
    "query": "Optional question about the text",
    "domain": "general"
}
```

**Response:**
```json
{
    "success": true,
    "triples_added": 5,
    "kg_updated": true,
    "schema_evolved": false,
    "response": "Generated response to query",
    "knowledge_used": [...],
    "processing_time": 2.5,
    "errors": [],
    "warnings": []
}
```

### POST /query

Query the knowledge graph.

**Request Body:**
```json
{
    "query": "Your question here",
    "max_context_triples": 10
}
```

**Response:**
```json
{
    "response": "Answer based on knowledge graph",
    "knowledge_used": 3,
    "context_triples": [...],
    "confidence": 0.85
}
```

### GET /metrics

Get system performance metrics.

**Response:**
```json
{
    "performance": {
        "extraction": {
            "average_accuracy": 0.87,
            "total_extractions": 150
        },
        "validation": {
            "average_efficiency": 0.92,
            "total_validations": 200
        }
    },
    "framework_stats": {...}
}
```

## Error Handling

All endpoints return appropriate HTTP status codes:

- 200: Success
- 400: Bad Request (invalid input)
- 500: Internal Server Error
- 503: Service Unavailable (system not initialized)

Error responses include details:

```json
{
    "detail": "Error description"
}
```
