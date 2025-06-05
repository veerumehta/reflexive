import os
from reflexive_composition.core import ReflexiveComposition
from reflexive_composition.utils.eval_utils import run_query_loop
from prompt_templates import WITH_CONTEXT_TEMPLATE, NO_CONTEXT_TEMPLATE
from medical_config import (get_kb_llm_config, get_target_llm_config, get_hitl_config, get_kg_config)

    # --- Configuration ---

kb_llm_config = get_kb_llm_config()
target_llm_config = get_target_llm_config()
hitl_config = get_hitl_config()
kg_config = get_kg_config()

# --- Initialize Reflexive Composition framework ---
rc = ReflexiveComposition(
    kb_llm_config=kb_llm_config,
    target_llm_config=target_llm_config,
    kg_config=kg_config,
    hitl_config=hitl_config
)

# Path to input queries and gold answers
# Get the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
queries_path = os.path.join(BASE_DIR, "data", "medical_data_queries_v1.jsonl")
gold_path = os.path.join(BASE_DIR, "data", "gold_medical_answers_v1.jsonl")

# Run evaluation
run_query_loop(
    rc=rc,
    queries_path=queries_path,
    template_with_kg=WITH_CONTEXT_TEMPLATE,
    template_without_kg=NO_CONTEXT_TEMPLATE,
    context_label="Context (extracted from clinical data)",
    save_results="results_medical.jsonl",
    gold_answers_path=gold_path,
    contradiction_check=True
)
