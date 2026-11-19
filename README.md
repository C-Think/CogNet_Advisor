# CogNet Advisor: Integrating Cognitive Knowledge Graphs and Causal Machine Learning for Personalized and Explainable Course Recommendations

This repository contains the implementation, dataset, and evaluation reports for **CogNet Advisor**, an explainable course recommendation framework. The system integrates the Felder-Silverman Learning Style Model (FSLSM) as a cognitive graph, Double Machine Learning (DML) for causal inference, and Retrieval-Augmented Generation (RAG) using Large Language Models (LLMs) for extracting pedagogical features from course syllabi.

*Note: This repository is anonymized for double-blind peer review.*

## Repository Structure
* `app.py`: The main Streamlit application code.
* `CogNet_Advisory_Data.csv`: The evaluation dataset containing 5,000 student records (features normalized to 0-1).
* `requirements.txt`: The Python dependencies required to run the system.
* `.env.example`: A template for setting up your local environment variables (see below).
* `cached_results/`: (If applicable) Contains pre-generated JSON extractions and PDF reports so reviewers can verify outputs without running the LLM.

## How to Reproduce and Run the System

### 1. Environment Setup
We recommend using Python 3.9+ and a virtual environment.
```bash
# Clone the repository
git clone https://github.com/C-Think/CogNet_Advisor.git
cd CogNet_Advisor

# Install dependencies
pip install -r requirements.txtCogNet1234@
