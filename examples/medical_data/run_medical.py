"""
Medical QA Case Study — Reflexive Composition Framework

This script demonstrates how structured and private clinical knowledge can be extracted,
validated, and grounded to support patient-specific question answering.
"""

import os
import logging
from reflexive_composition.core import ReflexiveComposition
from reflexive_composition.hitl.interface import ConsoleValidationInterface
from reflexive_composition.knowledge_graph.graph import KnowledgeGraph

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def main():
    print("\n=== Medical QA Case Study: FHIR + RxNorm Grounding ===\n")

    # --- Configuration ---
    kb_llm_config = {
        "model_name": os.environ.get("KB_LLM_MODEL", "medlm-1"),
        "api_key": os.environ.get("MEDLM_API_KEY"),
        "model_provider": "custom",
    }

    target_llm_config = {
        "model_name": os.environ.get("TARGET_LLM_MODEL", "gpt-4"),
        "api_key": os.environ.get("OPENAI_API_KEY"),
        "model_provider": "openai",
    }

    kg_config = {
        "storage_type": "in_memory",
        "schema": {
            "entity_types": ["Patient", "Medication", "Condition", "Allergy", "Code"],
            "relationship_types": ["HasCondition", "Prescribed", "ContraindicatedWith", "MappedTo"],
            "version": 1
        }
    }

    hitl_config = {
        "high_confidence_threshold": 0.85,
        "low_confidence_threshold": 0.6
    }

    # --- Initialize Reflexive Composition framework ---
    rc = ReflexiveComposition(
        kb_llm_config=kb_llm_config,
        target_llm_config=target_llm_config,
        kg_config=kg_config,
        hitl_config=hitl_config
    )

    # --- Source Text ---
    # Optional: Load from structured patient data
    json_path = os.path.join("data", "fhir_patient.json")
    if os.path.exists(json_path):
        import json
        with open(json_path) as f:
            structured_data = json.load(f)
        print("\nLoaded structured data:")
        for rel in structured_data["relations"]:
            print(" -", rel["subject"], rel["predicate"], rel["object"])
        print("\n(You can wire this into a KG directly instead of using free text.)\n")
    source_text = \"\"\"
    Patient John Doe, age 45, diagnosed with hypertension and Type 2 diabetes.
    Currently taking metformin and lisinopril. Reports mild allergy to sulfa drugs.
    \"\"\"

    print("Extracting structured knowledge from FHIR-style patient record...\n")
    extraction_result = rc.extract_knowledge(
        source_text=source_text,
        schema=kg_config["schema"],
        confidence_threshold=0.7
    )

    print("\nExtracted Triples:")
    for triple in extraction_result["triples"]:
        print(" -", triple)

    print("\nGenerating answer to clinical question...\n")
    prompt_result = rc.generate_output(
        query="Are there any known medication risks for this patient profile?",
        grounded=True
    )

    print("Generated Answer:")
    print(prompt_result["response"])


if __name__ == "__main__":
    main()
