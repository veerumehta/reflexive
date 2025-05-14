# case_studies/code_security/code_security_case_study.py
"""
Case Study: Historical Bias Mitigation in Code Generation using Reflexive Composition.

This script demonstrates the application of the Reflexive Composition framework to mitigate
historical bias in code generation, by constructing a security-focused knowledge graph
and using it to ground LLM code generation.
"""

import os
import logging
import json
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import Reflexive Composition
from reflexive_composition.core import ReflexiveComposition
from reflexive_composition.hitl.interface import ConsoleValidationInterface
from reflexive_composition.knowledge_graph.graph import KnowledgeGraph

def initialize_security_schema() -> Dict[str, Any]:
    """
    Initialize the security knowledge graph schema.
    
    Returns:
        Schema definition as dictionary
    """
    return {
        "entity_types": [
            "APIMethod", 
            "PythonVersion", 
            "ReplacementMethod", 
            "SecurityRiskLevel",
            "VulnerabilityType",
            "SecureCodingPattern"
        ],
        "relationship_types": [
            "deprecatedInVersion", 
            "recommendedReplacement", 
            "securityRiskLevel",
            "hasVulnerability",
            "hasSecureAlternative",
            "mitigates",
            "affects"
        ],
        "version": 1
    }

def extract_security_knowledge(data_path: str) -> List[Dict[str, Any]]:
    """
    Extract structured knowledge from security data sources.
    
    This is a simplified implementation for the case study; in a real deployment,
    this would use more sophisticated extraction methods.
    
    Args:
        data_path: Path to security data
        
    Returns:
        List of extracted triples
    """
    triples = []
    
    # Load sample data from a JSON file
    try:
        with open(data_path, 'r') as f:
            security_data = json.load(f)
        
        # Process deprecation information
        for item in security_data.get('deprecations', []):
            api_name = item.get('api')
            deprecated_version = item.get('deprecated_in')
            replacement = item.get('replacement')
            security_risk = item.get('security_risk', 'Low')
            
            if api_name and deprecated_version:
                # Create deprecation triple
                triples.append({
                    "subject": api_name,
                    "predicate": "deprecatedInVersion",
                    "object": deprecated_version,
                    "confidence": 0.95,
                    "source": data_path
                })
                
                # Create replacement triple if available
                if replacement:
                    triples.append({
                        "subject": api_name,
                        "predicate": "recommendedReplacement",
                        "object": replacement,
                        "confidence": 0.95,
                        "source": data_path
                    })
                
                # Create security risk triple
                triples.append({
                    "subject": api_name,
                    "predicate": "securityRiskLevel",
                    "object": security_risk,
                    "confidence": 0.9,
                    "source": data_path
                })
        
        # Process vulnerability information
        for item in security_data.get('vulnerabilities', []):
            api_name = item.get('api')
            vuln_type = item.get('type')
            secure_alt = item.get('secure_alternative')
            
            if api_name and vuln_type:
                # Create vulnerability triple
                triples.append({
                    "subject": api_name,
                    "predicate": "hasVulnerability",
                    "object": vuln_type,
                    "confidence": 0.9,
                    "source": data_path
                })
                
                # Create secure alternative triple if available
                if secure_alt:
                    triples.append({
                        "subject": api_name,
                        "predicate": "hasSecureAlternative",
                        "object": secure_alt,
                        "confidence": 0.9,
                        "source": data_path
                    })
        
        logger.info(f"Extracted {len(triples)} security triples from {data_path}")
        return triples
    
    except Exception as e:
        logger.error(f"Error extracting security knowledge: {e}")
        return []

def create_example_data():
    """
    Create example security data for the case study.
    
    This function generates sample security knowledge that would typically
    come from authoritative sources like API docs and vulnerability databases.
    """
    example_data = {
        "deprecations": [
            {
                "api": "datetime.utcnow",
                "deprecated_in": "Python 3.12",
                "replacement": "datetime.now(tz=datetime.UTC)",
                "security_risk": "Medium",
                "description": "Prone to timezone handling inconsistencies affecting timestamps"
            },
            {
                "api": "urllib.urlopen",
                "deprecated_in": "Python 3.0",
                "replacement": "requests.get",
                "security_risk": "High",
                "description": "Lacks SSL verification and proper error handling"
            },
            {
                "api": "ast.Num",
                "deprecated_in": "Python 3.8",
                "replacement": "ast.Constant",
                "security_risk": "Low",
                "description": "Parser compatibility issues"
            },
            {
                "api": "threading.activeCount",
                "deprecated_in": "Python 3.10",
                "replacement": "threading.active_count",
                "security_risk": "Low",
                "description": "Thread synchronization issues"
            }
        ],
        "vulnerabilities": [
            {
                "api": "hashlib.md5",
                "type": "InsecureHashFunction",
                "secure_alternative": "hashlib.sha256",
                "description": "MD5 is cryptographically broken and unsuitable for security applications"
            },
            {
                "api": "random.random",
                "type": "InsecurePseudorandomGenerator",
                "secure_alternative": "secrets.token_bytes",
                "description": "Not cryptographically secure for sensitive operations"
            },
            {
                "api": "subprocess.call",
                "type": "CommandInjectionRisk",
                "secure_alternative": "subprocess.run with shell=False",
                "description": "Vulnerable to shell injection attacks if used with shell=True"
            },
            {
                "api": "pickle.loads",
                "type": "DeserializationVulnerability",
                "secure_alternative": "json.loads",
                "description": "Arbitrary code execution risk when deserializing untrusted data"
            }
        ],
        "secure_patterns": [
            {
                "id": "SecureTokenGeneration",
                "description": "Use cryptographically secure random token generation",
                "applies_to": ["random.random", "random.randint"],
                "recommendation": "secrets.token_bytes or secrets.token_hex"
            },
            {
                "id": "ParamQuery",
                "description": "Use parameterized queries for database operations",
                "applies_to": ["sqlite3.execute", "psycopg2.execute"],
                "recommendation": "Use query parameters instead of string formatting"
            }
        ]
    }
    
    # Save example data to file
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    data_path = os.path.join(data_dir, "security_data.json")
    
    with open(data_path, 'w') as f:
        json.dump(example_data, f, indent=2)
    
    logger.info(f"Created example security data at {data_path}")
    return data_path

def generate_test_cases() -> List[Dict[str, Any]]:
    """
    Generate test cases for secure code generation.
    
    Returns:
        List of test case dictionaries
    """
    return [
        {
            "prompt": "Create a Python function to get the current UTC time",
            "deprecated_api": "datetime.utcnow",
            "secure_alternative": "datetime.now(tz=datetime.UTC)",
            "risk": "Timezone handling inconsistencies"
        },
        {
            "prompt": "Write a function to check if a thread is active",
            "deprecated_api": "threading.activeCount",
            "secure_alternative": "threading.active_count",
            "risk": "Thread synchronization issues"
        },
        {
            "prompt": "Write code to fetch data from a URL",
            "deprecated_api": "urllib.urlopen",
            "secure_alternative": "requests.get with verify=True",
            "risk": "Missing SSL verification"
        },
        {
            "prompt": "Create a function to generate a random token",
            "deprecated_api": "random.random",
            "secure_alternative": "secrets.token_hex",
            "risk": "Insecure randomness"
        }
    ]

def construct_secure_prompt(query: str, security_context: Dict[str, Any]) -> str:
    """
    Construct a secure coding prompt with knowledge graph context.
    
    Args:
        query: Original coding query
        security_context: Security knowledge from the graph
    
    Returns:
        Enhanced prompt with security guidance
    """
    # Extract relevant security information
    apis = security_context.get("apis", [])
    patterns = security_context.get("patterns", [])
    
    # Construct security guidance section
    security_guidance = ""
    if apis:
        security_guidance += "SECURITY CONSIDERATIONS:\n"
        for api in apis:
            if api.get("deprecated"):
                security_guidance += f"- {api['name']} is deprecated since {api['deprecated_in']}. "
                if api.get("replacement"):
                    security_guidance += f"Use {api['replacement']} instead.\n"
                else:
                    security_guidance += "\n"
            
            if api.get("vulnerability"):
                security_guidance += f"- {api['name']} has security risk: {api['vulnerability']}. "
                if api.get("secure_alternative"):
                    security_guidance += f"Use {api['secure_alternative']} instead.\n"
                else:
                    security_guidance += "\n"
    
    if patterns:
        security_guidance += "\nRECOMMENDED PATTERNS:\n"
        for pattern in patterns:
            security_guidance += f"- {pattern['description']}\n"
    
    # Construct the final prompt
    prompt = f"""
You are a secure code assistant tasked with writing high-quality, secure code.

TASK: {query}

{security_guidance}

Generate secure Python code that follows these security recommendations.
"""
    return prompt

def evaluate_code_security(generated_code: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate the security of generated code.
    
    Args:
        generated_code: Generated code snippet
        test_case: Test case with security expectations
    
    Returns:
        Evaluation results
    """
    deprecated_api = test_case["deprecated_api"]
    
    # Check for deprecated API usage
    uses_deprecated = deprecated_api in generated_code
    
    # Check for secure alternative
    secure_alternative = test_case["secure_alternative"]
    uses_secure = secure_alternative in generated_code
    
    return {
        "uses_deprecated_api": uses_deprecated,
        "uses_secure_alternative": uses_secure,
        "security_score": 0 if uses_deprecated else (1 if uses_secure else 0.5),
        "test_case": test_case
    }

def main():
    """Run the code security case study."""
    print("=== Code Security Case Study: Historical Bias Mitigation ===\n")
    
    # Create example data
    data_path = create_example_data()
    
    # Set up configuration
    kb_llm_config = {
        "model_name": os.environ.get("KB_LLM_MODEL", "gpt-3.5-turbo"),
        "api_key": os.environ.get("OPENAI_API_KEY"),
        "model_provider": "openai",
    }
    
    target_llm_config = {
        "model_name": os.environ.get("TARGET_LLM_MODEL", "gpt-3.5-turbo"),
        "api_key": os.environ.get("OPENAI_API_KEY"),
        "model_provider": "openai"
    }
    
    kg_config = {
        "storage_type": "in_memory",
        "schema": initialize_security_schema()
    }
    
    hitl_config = {
        "high_confidence_threshold": 0.9,
        "low_confidence_threshold": 0.7
    }
    
    # Initialize Reflexive Composition framework
    print("Initializing Reflexive Composition framework...")
    rc = ReflexiveComposition(
        kb_llm_config=kb_llm_config,
        target_llm_config=target_llm_config,
        kg_config=kg_config,
        hitl_config=hitl_config
    )
    print("Framework initialized!\n")
    
    # Extract knowledge from security data
    print("=== Step 1: Knowledge Extraction ===")
    print("Extracting security knowledge from data sources...")
    
    extracted_triples = extract_security_knowledge(data_path)
    
    extraction_result = {
        "triples": extracted_triples
    }
    
    print(f"Extracted {len(extraction_result['triples'])} security triples")
    for i, triple in enumerate(extraction_result['triples'][:5], 1):  # Show first 5 as example
        print(f"Triple {i}: {triple.get('subject')} - {triple.get('predicate')} - {triple.get('object')}")
    if len(extraction_result['triples']) > 5:
        print(f"... (and {len(extraction_result['triples']) - 5} more)")
    print()
    
    # Update knowledge graph with extracted triples
    print("=== Step 2: Knowledge Graph Update ===")
    print("Updating knowledge graph with security knowledge...")
    
    update_success = rc.update_knowledge_graph(extraction_result['triples'])
    print(f"Update success: {update_success}")
    
    kg_stats = rc.knowledge_graph.get_stats()
    print(f"Knowledge graph stats: {kg_stats}\n")
    
    # Generate code with and without the knowledge graph
    print("=== Step 3: Code Generation Evaluation ===")
    
    test_cases = generate_test_cases()
    results_baseline = []
    results_kg_enhanced = []
    
    print("Evaluating code generation with and without security knowledge...\n")
    
    for test_case in test_cases:
        query = test_case["prompt"]
        print(f"Test case: {query}")
        
        # Generate code without KG enhancement (baseline)
        print("Generating code WITHOUT security knowledge...")
        baseline_response = rc.target_llm.generate(query)
        baseline_code = baseline_response["text"]
        baseline_eval = evaluate_code_security(baseline_code, test_case)
        results_baseline.append(baseline_eval)
        
        # Generate code with KG enhancement
        print("Generating code WITH security knowledge...")
        context = rc.knowledge_graph.retrieve_context(query, max_items=10)
        enhanced_response = rc.generate_response(query, retrieve_context=True)
        enhanced_code = enhanced_response["text"]
        enhanced_eval = evaluate_code_security(enhanced_code, test_case)
        results_kg_enhanced.append(enhanced_eval)
        
        # Show brief results for this test case
        print(f"Baseline security score: {baseline_eval['security_score']}")
        print(f"KG-enhanced security score: {enhanced_eval['security_score']}")
        print()
    
    # Calculate overall results
    baseline_deprecated_count = sum(r["uses_deprecated_api"] for r in results_baseline)
    baseline_secure_count = sum(r["uses_secure_alternative"] for r in results_baseline)
    baseline_avg_score = sum(r["security_score"] for r in results_baseline) / len(results_baseline)
    
    enhanced_deprecated_count = sum(r["uses_deprecated_api"] for r in results_kg_enhanced)
    enhanced_secure_count = sum(r["uses_secure_alternative"] for r in results_kg_enhanced)
    enhanced_avg_score = sum(r["security_score"] for r in results_kg_enhanced) / len(results_kg_enhanced)
    
    # Report summary
    print("=== Results Summary ===")
    print(f"Total test cases: {len(test_cases)}")
    print("\nBaseline LLM (No Security Knowledge):")
    print(f"- Uses deprecated APIs: {baseline_deprecated_count}/{len(test_cases)} ({baseline_deprecated_count/len(test_cases)*100:.1f}%)")
    print(f"- Uses secure alternatives: {baseline_secure_count}/{len(test_cases)} ({baseline_secure_count/len(test_cases)*100:.1f}%)")
    print(f"- Average security score: {baseline_avg_score:.2f}/1.0")
    
    print("\nKG-Enhanced LLM (With Security Knowledge):")
    print(f"- Uses deprecated APIs: {enhanced_deprecated_count}/{len(test_cases)} ({enhanced_deprecated_count/len(test_cases)*100:.1f}%)")
    print(f"- Uses secure alternatives: {enhanced_secure_count}/{len(test_cases)} ({enhanced_secure_count/len(test_cases)*100:.1f}%)")
    print(f"- Average security score: {enhanced_avg_score:.2f}/1.0")
    
    improvement = (enhanced_avg_score - baseline_avg_score) / baseline_avg_score * 100
    print(f"\nSecurity improvement: {improvement:.1f}%")
    
    print("\nCase study complete!")

if __name__ == "__main__":
    main()
