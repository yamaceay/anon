# K-Anonymizer

*by Yamaç Eren Ay*

This is a Proof of Concept for K-Anonymization designed to protect sensitive textual data by systematically anonymizing content while maintaining its utility. This tool could be particularly useful for handling medical records, patient notes, and other confidential documents.

## Introduction

This K-Anonymizer ensures that sensitive documents remain private yet useful by employing advanced text re-identification (TRI) techniques. The framework guarantees that entities within documents cannot be easily re-identified by requiring at least k attempts for successful identification, where k is a parameter you can configure.

### Use Case Example

Consider a cloud provider serving multiple hospitals as customers, which wants to effectively protect patient privacy while maintaining document usefulness. Each hospital maintains isolated subaccounts containing sensitive documents such as medical records. The challenge is to ensure that even if someone (with background knowledge of all people) attempts to re-identify a person within these records, they would need at least k guesses.

We outline a system that respects the following principles:
- *Isolated Subaccount Security*: Process each customer's records independently to prevent data leakage and cross-customer contamination. Background knowledge of one customer is available to the attacker, but the attacker cannot find out which customer a document belongs to.
- *Configurable Anonymization*: Set your preferred minimum re-identification rank (k) to control privacy levels. Achieve this by utilizing some explainability methods to identify Most Disclosive Tokens (MDTs) for targeted protection.

### Re-Identification Protection

Two main approaches are tested for protection against text re-identification:
1. Baseline BERT TRI: Uses hard-coded labels with pre-trained entity embeddings. This was used in the original PETRE framework.
2. New RAG-based SBERT TRI: Implements a flexible, scalable approach using efficient vector database storage.

### Training Methodology

The system employs two primary training approaches:
1. *Self-Supervised Contrastive Learning* to develop optimal embedding spaces which learn to distinguish between different entities and documents.
2. *Optional SFT & RLHF* for fine-tuning and continuous improvement, allowing the model to adapt further to specific domains or datasets.

### System Operation

1. Document Processing:
    - Accept documents from customer subaccounts
    - Apply re-identification attempts using BERT or FAISS-based systems
    - Calculate entity probability rankings

2. Anonymization Process:
    - Check if ranking meets required k-anonymity level
    - Identify and mask / modify most disclosive elements if needed
    - Repeat until adequate protection is achieved

3. Quality Assurance:
    - Evaluate results using standard metrics (Recall@k, Precision@k, F1@k, MRR@k)
    - Verify anonymization effectiveness

### Advantages Over Traditional Methods

Traditional BERT-based NER models have some limitations:
- Require complete retraining for new entities
- Inflexible due to hard-coded label positions
- Need extensive domain-specific preprocessing

This RAG-based approach offers:
- Combined document and entity embeddings
- Efficient semantic space mapping
- Dynamic updates without complete retraining

### Performance Considerations

| Aspect | Details |
|--------|----------|
| Data Handling | Careful management required for large datasets due to quadratic scaling, e.g. 1k entities requiring 1M pairwise relationships to be learnt |
| Privacy Boundaries | Separate models recommended unless data overlap exists, e.g. two hospitals with 95% patients in common |

## Installation

Requires Python 3.8 or later:

Either using standard Python tools:

```bash
pip3 install -r requirements.txt
```

Or using [uv](https://github.com/astral-sh/uv):

```bash
# uv init anonymizer # Initialize the project with uv
uv venv .venv # Create a virtual environment
source .venv/bin/activate # Activate the virtual environment
uv pip install -r requirements.txt # Install dependencies
```

## Usage

Data Preprocessing:
```sh
[uv run] python -m data [-i <path_to_input_data>] [-o <path_to_processed_data>]
```

Model Training:
```sh
[uv run] python -m tri [-i <path_to_processed_data>] [-o <path_to_sbert_model>]
```

Framework Execution:
```sh
[uv run] python -m petre [-i <path_to_processed_data>] [-s <path_to_sbert_model>] [-o <path_to_output_data>]
```

## Credits

Based on work by [@BenetManzanaresSalor](https://github.com/BenetManzanaresSalor)

PETRE Implementation:
- Paper: Manzanares-Salor, B., & Sánchez, D. (2025). Enhancing text anonymization via re-identification risk-based explainability. Knowledge-Based Systems, 310, 112945.
- Repository: https://github.com/BenetManzanaresSalor/PETRE

Text Re-Identification Research:
- Paper: Manzanares-Salor, B., Sánchez, D., & Lison, P. (2024). Evaluating the disclosure risk of anonymized documents via a machine learning-based re-identification attack. Data Mining and Knowledge Discovery, 38(6), 4040-4075.
- Repository: https://github.com/BenetManzanaresSalor/TextRe-Identification