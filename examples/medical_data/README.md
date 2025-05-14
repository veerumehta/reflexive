# Medical QA Case Study — FHIR and RxNorm Integration

This case study demonstrates how the **Reflexive Composition** framework can be used in a healthcare setting to extract, validate, and reason over structured patient data derived from unstructured clinical notes.

## 🧬 Scenario

Given a fictional patient's clinical summary (based on FHIR-style elements), the system:
1. Extracts entities and relationships (conditions, medications, allergies)
2. Applies human-in-the-loop validation
3. Answers a clinical query grounded in the validated knowledge graph

## 🔁 Reflexive Composition Flow

- **LLM2KG**: Extracts patient facts from unstructured notes (e.g., diagnosis, meds)
- **HITL Validation**: Applies thresholds for precision; optional human review
- **KG2LLM**: Uses validated KG to answer:
  _"Are there any known medication risks for this patient?"_

## 📂 Files

- `run_medical.py`: End-to-end case study runner
- `data/`: Folder for structured FHIR-style data (optional)

## 🚀 How to Run

```bash
python run_medical.py