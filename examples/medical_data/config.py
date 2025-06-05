import os
from typing import Optional, Dict


def get_kb_llm_config(env: Optional[Dict[str, str]] = None) -> Dict:
    env = env or os.environ
    return {
        "model_name": env.get("KB_LLM_MODEL", "gemini-2.0-flash"),
        "api_key": env.get("GEMINI_API_KEY"),
        "model_provider": "google",
    }


def get_target_llm_config(env: Optional[Dict[str, str]] = None) -> Dict:
    env = env or os.environ
    return {
        "model_name": env.get("TARGET_LLM_MODEL", "gemini-2.0-flash"),
        "api_key": env.get("GEMINI_API_KEY"),
        "model_provider": "google",
    }


def get_hitl_config(env: Optional[Dict[str, str]] = None) -> Dict:
    env = env or os.environ
    return {
        "high_confidence_threshold": float(env.get("HITL_HIGH", 0.85)),
        "low_confidence_threshold": float(env.get("HITL_LOW", 0.6)),
    }


def get_kg_config() -> Dict:
    return {
        "storage_type": "in_memory",
        "schema": {
            "entity_types": ["Patient", "Medication", "Condition", "Allergy", "Code"],
            "relationship_types": ["HasCondition", "Prescribed", "ContraindicatedWith", "MappedTo"],
            "version": 1
        }
    }