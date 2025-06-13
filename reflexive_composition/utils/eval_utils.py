import json
import os
from typing import List, Dict, Optional, Tuple
from reflexive_composition.kg2llm.contradiction_detector import ContradictionDetector

def compute_f1(pred: str, gold: str) -> Tuple[float, float, float]:
    """Compute token-level precision, recall, and F1."""
    pred_tokens = set(pred.lower().split())
    gold_tokens = set(gold.lower().split())
    common = pred_tokens & gold_tokens
    if not common:
        return 0.0, 0.0, 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1

def run_query_loop(
    rc,
    queries_path: str,
    template_with_kg: str,
    template_without_kg: str,
    context_label: str = "Facts",
    save_results: Optional[str] = None,
    gold_answers_path: Optional[str] = None,
    contradiction_check: bool = False
):
    """Evaluate queries with and without KG, compute F1 if gold answers are available, and detect contradictions."""
    
    # Load queries
    with open(queries_path) as f:
        query_entries = [json.loads(line) for line in f]

    # Load gold answers
    gold_lookup = {}
    if gold_answers_path:
        with open(gold_answers_path) as f:
            for line in f:
                item = json.loads(line)
                gold_lookup[item["id"]] = item["gold_answer"]

    results = []
    detector = ContradictionDetector() if contradiction_check else None

    for entry in query_entries:
        query = entry["query"]
        qid = entry.get("id", query[:30].replace(" ", "_"))

        for grounded, template, mode in [
            (False, template_without_kg, "no_kg"),
            (True, template_with_kg, "with_kg")
        ]:
            print(f"\n[ID: {qid}] [{mode}] {query}")
            response = rc.generate_response(
                query=query,
                grounded=grounded,
                template=template,
                context_label=context_label
            )
            output_text = response["text"]
            gold_answer = gold_lookup.get(qid)
            f1_score = None
            p = r = f1 = 0.0
            if gold_answer:
                p, r, f1 = compute_f1(output_text, gold_answer)
                f1_score = round(f1, 3)

            contradiction = None
            if contradiction_check and detector:
                # Get the knowledge graph context for contradiction detection
                if grounded:
                    # Retrieve the same context that was used for generation
                    kg_context = rc.kg.retrieve_context(query, max_items=10)
                    context_triples = kg_context.get("triples", [])
                else:
                    # No context available for non-grounded responses
                    context_triples = []
                
                # Detect contradictions with proper context
                contradictions_list = detector.detect_contradictions(output_text, context_triples)
                
                # Format the contradiction result to match expected structure
                contradiction = {
                    "has_contradiction": len(contradictions_list) > 0,
                    "details": contradictions_list
                }

            results.append({
                "id": qid,
                "mode": mode,
                "query": query,
                "response": output_text,
                "gold_answer": gold_answer,
                "precision": round(p, 3),
                "recall": round(r, 3),
                "f1": f1_score,
                "has_contradiction": contradiction.get("has_contradiction") if contradiction else None,
                "contradiction_details": contradiction.get("details") if contradiction else None
            })

    # Save results if needed
    if save_results:
        with open(save_results, "w") as f:
            for row in results:
                f.write(json.dumps(row) + "\n")

    return results
