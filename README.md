# K-Anonymizer

*by Yamaç Eren Ay*

Welcome to K-Anonymizer, a framework designed to protect sensitive textual data by systematically anonymizing content while maintaining its utility. This tool is particularly useful for handling medical records, patient notes, and other confidential documents.

## Introduction

K-Anonymizer ensures that sensitive documents remain private yet useful by employing advanced text re-identification (TRI) techniques. The framework guarantees that entities within documents cannot be easily re-identified by requiring at least k attempts for successful identification, where k is a parameter you can configure.

### Use Case Example

Consider a cloud provider serving multiple hospitals as customers. Each hospital maintains isolated subaccounts containing sensitive documents such as medical records. K-Anonymizer ensures that even if someone attempts to re-identify an entity within these documents, they would need at least k attempts to succeed, effectively protecting patient privacy while maintaining document usefulness.

### Key Features

- Isolated Subaccount Security: Process each customer's documents independently to prevent data leakage and cross-customer contamination
- Configurable Anonymization: Set your preferred minimum re-identification rank (k) to control privacy levels
- Explainable Results: Utilize SHAP and other methods to identify Most Disclosive Tokens (MDTs) for targeted protection

### Re-Identification Protection

We implement two main approaches for protection against text re-identification:
1. Baseline BERT TRI: Uses hard-coded labels with pre-trained entity embeddings
2. FAISS + Sentence Transformers: Implements a flexible, scalable approach using efficient vector database storage

### Training Methodology

Our system employs two primary training approaches:
1. Self-Supervised Contrastive Learning to develop optimal embedding spaces
2. Optional SFT & RLHF for fine-tuning and continuous improvement

### System Operation

1. Document Processing:
    - Accept documents from customer subaccounts
    - Apply re-identification attempts using BERT or FAISS-based systems
    - Calculate entity probability rankings

2. Anonymization Process:
    - Check if ranking meets required k-anonymity level
    - Identify and modify most disclosive elements if needed
    - Repeat until adequate protection is achieved

3. Quality Assurance:
    - Evaluate results using standard metrics (Recall@k, Precision@k, F1@k, MRR@k)
    - Verify anonymization effectiveness

### Advantages Over Traditional Methods

Traditional BERT-based NER models have limitations:
- Require complete retraining for new entities
- Inflexible due to hard-coded label positions
- Need extensive domain-specific preprocessing

Our FAISS-based approach offers:
- Combined document and entity embeddings
- Efficient semantic space mapping
- Dynamic updates without complete retraining

### Performance Considerations

| Aspect | Details |
|--------|----------|
| Data Handling | Careful management required for large datasets due to quadratic scaling |
| Privacy Boundaries | Separate models recommended unless data overlap exists |
| Model Updates | Optional continuous learning available but resource-intensive |

## Installation

Requires Python 3.8 or later:

```sh
pip install -r requirements.txt
```

## Usage

Data Preprocessing:
```sh
python -m data [-i <path_to_input_data>] [-o <path_to_processed_data>]
```

Model Training:
```sh
python -m tri [-i <path_to_processed_data>] [-o <path_to_sbert_model>]
```

Framework Execution:
```sh
python -m petre [-i <path_to_processed_data>] [-s <path_to_sbert_model>] [-o <path_to_output_data>]
```

## Credits

Based on work by [@BenetManzanaresSalor](https://github.com/BenetManzanaresSalor)

PETRE Implementation:
- Paper: Manzanares-Salor, B., & Sánchez, D. (2025). Enhancing text anonymization via re-identification risk-based explainability. Knowledge-Based Systems, 310, 112945.
- Repository: https://github.com/BenetManzanaresSalor/PETRE

Text Re-Identification Research:
- Paper: Manzanares-Salor, B., Sánchez, D., & Lison, P. (2024). Evaluating the disclosure risk of anonymized documents via a machine learning-based re-identification attack. Data Mining and Knowledge Discovery, 38(6), 4040-4075.
- Repository: https://github.com/BenetManzanaresSalor/TextRe-Identification