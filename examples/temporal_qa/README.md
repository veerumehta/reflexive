# US Presidents Current Affairs Benchmark

A framework for evaluating Large Language Models' (LLMs) performance on current affairs knowledge, focusing on US Presidents. This benchmark specifically tests LLMs' ability to handle time-sensitive information and measures their knowledge cutoff limitations.

## Overview

The benchmark framework evaluates LLMs on their ability to:
- Provide accurate current information about US Presidents
- Handle temporal queries appropriately
- Maintain consistency across different query phrasings
- Recognize and acknowledge knowledge boundaries

### Tested Models

1. Early/Base Models (2019-2021)
- GPT-2 XL
- GPT-3 (davinci-002)
- T5 (3B)
- GPT-J (6B)

2. Mid-Generation (2022)
- Bloom (7B)
- Flan-T5 XL
- ChatGPT (GPT-3.5)

3. Recent Models (2023)
- GPT-4
- Llama-2 (7B) & Llama-2 Chat (7B)
- Falcon (7B) & Falcon Instruct (7B)
- Vicuna v1.5 (7B)
- Mistral v0.1 (7B) & Mistral Instruct v0.1 (7B)
- Mixtral 8x7B v0.1 & Mixtral 8x7B Instruct v0.1

4. Latest Models (2024)
- OLMo (1B & 7B)
- Llama-3 (8B) and Llama-3 Instruct (8B)
- OpenELM (270M & 1.1B & 3B)

## Installation

```bash
git clone https://github.com/yourusername/presidents-benchmark.git
cd presidents-benchmark
pip install -r requirements.txt
```

## Usage

1. Configure model access:

```python
# config.py
API_KEYS = {
    'openai': 'your-openai-key',
    'anthropic': 'your-anthropic-key',
    # Add other API keys as needed
}
```

2. Run the benchmark:

```bash
python benchmark_evaluator.py
```

3. View results:

```python
import pandas as pd
results = pd.read_csv('benchmark_results.csv')
print(results.groupby('year').mean())
```

## Benchmark Structure

### Categories

1. Basic Facts
- Current role/status
- Current location
- Recent appearances
- Public activities

2. Security Incidents
- Life attempts
- Security breaches
- Threats
- Investigations

3. Political Activities
- Recent speeches
- Campaign events
- Policy statements
- Endorsements

4. Legal & Administrative
- Current investigations
- Court appearances
- Executive orders
- Appointments

### Validation Sources

- Official White House Records
- Court Documents
- Major News Organizations
- Government Agency Announcements

### Evaluation Metrics

- Accuracy: Factual correctness
- Currency: Up-to-date information
- Consistency: Uniform answers across rewordings
- Attribution: Source awareness and citation

## Example Query Format

```python
# Example query variations
trump_shooting_queries = [
    "When was the last attempt on Donald Trump's life?",
    "Has there been any recent security incident involving Donald Trump?",
    "What security incidents involving Donald Trump occurred in 2024?",
    "Was there any shooting incident involving Donald Trump at a courthouse?",
    "Has Donald Trump faced any security threats at recent court appearances?"
]
```

## Adding New Facts

Add new facts to `benchmark_facts.json`:

```json
{
    "id": "event_id",
    "subject": "President Name",
    "event": "event_type",
    "date": "YYYY-MM-DD",
    "category": "category_name",
    "current_answer": "correct current answer",
    "queries": [
        "question 1",
        "question 2",
        "question 3"
    ],
    "sources": [
        {
            "url": "source_url",
            "type": "source_type",
            "date": "YYYY-MM-DD"
        }
    ]
}
```

## Output Format

Results are saved in CSV format with the following columns:
- Model name
- Release year
- Correct responses (%)
- Outdated responses (%)
- Irrelevant responses (%)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@misc{presidents-benchmark-2024,
    title={US Presidents Current Affairs Benchmark},
    author={Your Name},
    year={2024},
    publisher={GitHub},
    journal={GitHub repository},
    howpublished={\url{https://github.com/yourusername/presidents-benchmark}}
}
```
