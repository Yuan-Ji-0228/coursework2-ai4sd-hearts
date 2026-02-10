# coursework2-ai4sd-hearts

Coursework project for AI for Sustainable Development: reproduction and contextual adaptation of the HEARTS framework for explainable and sustainable text stereotype detection.

## What is reproduced (Part A.1)
Baseline reproduction: Logistic Regression (Embeddings) on EMGSD (binary stereotype vs non-stereotype).

## Environment
- OS: Windows 10
- Python: 3.10 (conda)
- GPU: RTX 3050

## Result
- Macro-F1 (my run): 0.638
- Paper (Table 1): 0.634
- Difference: 0.4% (within ±5%)

## How to run
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
python run_baseline.py