# HEARTS Replication: WinoBias Adaptation for Gender Stereotype Detection


[![License: MIT](https://opensource.org/licenses/MIT)
[![HEARTS Paper](https://arxiv.org/abs/2409.11579)

## 📋 Overview

This project replicates the [HEARTS (Holistic Framework for Explainable, Sustainable and Robust Text Stereotype Detection)](https://arxiv.org/abs/2409.11579) methodology and adapts it to detect gender-based occupational stereotypes using the WinoBias dataset.

**Original Paper:** HEARTS - Holistic Framework for Explainable, Sustainable Text Stereotype Detection  
**Original Dataset:** EMGSD (57,201 samples, 6 dimensions)  
**Adapted Dataset:** WinoBias (3,168 samples, gender focus)  
**Model:** ALBERT-V2 (11M parameters, low-carbon)  
**SDG Alignment:** SDG 5 (Gender Equality), SDG 8 (Decent Work)

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_lg

# Run baseline (Part A.1)
cd Part_A1_Baseline && python run_baseline.py

# Preprocess data (Part A.3)
cd Part_A3_Preprocessing && python preprocess_winobias.py

# Train & evaluate (Part A.4-A.5)
cd Part_A4_A5_Training_Evaluation && python train_winobias_hearts.py

# Optional: Explainability analysis
python explainability_analysis.py

# Generate poster visualizations
python generate_poster_visualizations.py
```

---

## 📁 Project Structure

coursework2-ai4sd-hearts/
├── dataset_winobias/                    
│   ├── train.xlsx
│   ├── valid.xlsx
│   └── test.xlsx
│
├── models/                               
│   ├── albert_winobias_hearts/
│   └── hearts_emgsd_baseline/
│
├── evaluation_results/                   
│   └── albert_winobias_hearts/
│       ├── classification_report.xlsx
│       └── full_results.xlsx
│
├── explainability_results/               
│   ├── lime_results.xlsx
│   ├── shap_results.xlsx
│   ├── sampled_data.xlsx
│   └── sentence_similarity_metrics.xlsx
│
├── outputs/                              
│   └── results.json
│
├── poster_visualizations/                
│   └── (generated plots)
│
├── baseline_model.py                                          
├── preprocessing.py                      
├── train.py                             
├── explainability_analysis.py           
├── generate_poster_visualizations.py     
│
├── requirements.txt
└── README.md
```
▶️ How to Run

All commands should be executed from the project root directory.

1️⃣ Install Dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_lg

2️⃣ Run Baseline 
python baseline_model.py

Outputs:

outputs/results.json

May update files in evaluation_results/

3️⃣ Preprocess WinoBias Dataset
python preprocessing.py

Outputs:

Processed train/valid/test splits saved in dataset_winobias/

4️⃣ Train and Evaluate Model
python train.py

Outputs:

Trained model saved in models/

Evaluation reports saved in evaluation_results/albert_winobias_hearts/

5️⃣ Run Explainability Analysis (Optional)
python explainability_analysis.py

Outputs:

SHAP and LIME results saved in explainability_results/

6️⃣ Generate Poster Visualizations
python generate_poster_visualizations.py

Outputs:

Figures saved in poster_visualizations/


## 🎨 Poster Visualizations

Generate all required plots:

```bash
python generate_poster_visualizations.py
```

Outputs:
- `confusion_matrix.png` → Model Performance section
- `performance_comparison.png` → Evaluation section  
- `shap_example.png` → Discussion section (if explainability run)
- `carbon_footprint.png` → Sustainability metrics

---

## 🔧 Troubleshooting

**CUDA Out of Memory:** Reduce batch size to 32  
**Accelerate Error:** `pip install --upgrade accelerate`  
**spaCy Model:** `python -m spacy download en_core_web_lg`

---

## 📝 Citation

```bibtex
@article{hearts2024,
  title={HEARTS: Explainable, Sustainable Text Stereotype Detection},
  journal={arXiv preprint arXiv:2409.11579},
  year={2024}
}
```


---

## 👥 Contact

**[Yuan Ji]**  
GitHub: https://github.com/Yuan-Ji-0228/coursework2-ai4sd-hearts
Email: ucabjid@ucl.ac.uk

---

**Last Updated:** February 2026
