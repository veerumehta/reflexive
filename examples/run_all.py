import os
import subprocess
from reflexive_composition.eval import metrics

# Define use case runners
USE_CASES = {
    "code_security": "examples/code_security/run_codegen.py",
    "temporal_qa": "examples/temporal_qa/run_temporal.py",
    "medical_data": "examples/medical_data/run_medical.py"
}

# Define output paths for post-analysis
OUTPUT_LOGS = {
    "code_security": "outputs/code_security_eval.jsonl",
    "temporal_qa": "outputs/temporal_eval.jsonl",
    "medical_data": "outputs/medical_eval.jsonl"
}

def run_use_case(label, script_path):
    print(f"Running: {label}")
    result = subprocess.run(["python", script_path], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error running {label}: {result.stderr}")

def evaluate_outputs():
    for label, output_path in OUTPUT_LOGS.items():
        if not os.path.exists(output_path):
            print(f"[{label}] No output log found at: {output_path}")
            continue
        print(f"[{label}] Evaluating output at: {output_path}")
        cases = metrics.load_jsonl(output_path)
        if "code" in label:
            results = metrics.evaluate_hitl_intervention(cases)
        elif "temporal" in label or "medical" in label:
            gold = [c["gold_answer"] for c in cases if "gold_answer" in c]
            pred = [c["generated_answer"] for c in cases if "generated_answer" in c]
            results = metrics.evaluate_generation_accuracy(gold, pred)
        else:
            results = {}
        print(f"[{label}] Metrics: {results}")

if __name__ == "__main__":
    for label, script in USE_CASES.items():
        run_use_case(label, script)
    evaluate_outputs()