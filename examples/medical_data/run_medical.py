import os
import json
import argparse
from typing import Optional, Dict, Any

from reflexive_composition.core import ReflexiveComposition
from reflexive_composition.utils.llm_utils import log_result
from prompt_templates import WITH_CONTEXT_TEMPLATE, NO_CONTEXT_TEMPLATE
from config import (
    get_kb_llm_config,
    get_target_llm_config,
    get_hitl_config,
    get_kg_config
)


def create_reflexive_pipeline() -> ReflexiveComposition:
    return ReflexiveComposition(
        kb_llm_config=get_kb_llm_config(),
        target_llm_config=get_target_llm_config(),
        hitl_config=get_hitl_config(),
        kg_config=get_kg_config()
    )


def setup_kg(rc: ReflexiveComposition, interactive: bool = True):
    base_dir = os.path.dirname(__file__)
    source_path = os.path.join(base_dir, "data", "fhir_patient.json")
    with open(source_path, "r") as f:
        source_text = f.read()

    result = rc.extract_knowledge(source_text, interactive=interactive)
    rc.kg.add_triples(result["triples"])

    rx_path = os.path.join(base_dir, "data", "synthetic_rxnorm_triples.json")
    if os.path.exists(rx_path):
        with open(rx_path, "r") as f:
            rx_triples = json.load(f)
            rc.kg.add_triples(rx_triples)


def generate_response_for_query(
    rc: ReflexiveComposition,
    query: str,
    grounded: bool,
    template: Optional[str] = None,
    interactive: bool = True
) -> Dict[str, Any]:
    if not template:
        template = WITH_CONTEXT_TEMPLATE if grounded else NO_CONTEXT_TEMPLATE

    return rc.generate_response(
        query=query,
        grounded=grounded,
        template=template,
        context_label="Context (extracted from clinical data)"
    )


def run_query_file(
    rc: ReflexiveComposition,
    path: str,
    grounded: bool,
    interactive: bool = True
):
    with open(path) as f:
        for line in f:
            q = json.loads(line)
            response = generate_response_for_query(
                rc, q["query"], grounded, interactive=interactive
            )
            log_result({**response, "query": q["query"]})


def run_single_query(
    rc: ReflexiveComposition,
    query: str,
    grounded: bool,
    interactive: bool = True
):
    response = generate_response_for_query(
        rc, query, grounded, interactive=interactive
    )
    log_result({**response, "query": query})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, help="Single query string")
    parser.add_argument("--query-file", type=str, help="Path to .jsonl file of queries")
    parser.add_argument("--grounded", action="store_true", default=True, help="Use grounded context (default: True)")
    parser.add_argument("--no-grounded", dest="grounded", action="store_false", help="Disable grounding")
    parser.add_argument("--no-hitl", action="store_true", help="Disable manual HITL validation prompts")
    args = parser.parse_args()

    interactive = not args.no_hitl

    print("\n=== Medical QA Case Study: FHIR + RxNorm Grounding ===\n")

    rc = create_reflexive_pipeline()
    setup_kg(rc, interactive=interactive)

    if args.query_file:
        run_query_file(rc, args.query_file, args.grounded, interactive=interactive)
    else:
        query = args.query or "Is this patient currently on a blood pressure medication?"
        run_single_query(rc, query, args.grounded, interactive=interactive)