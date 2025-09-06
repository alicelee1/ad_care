# CARE-AD: Multi-Agent System for Alzheimer's Disease Risk Prediction

This repository implements **CARE-AD**, a multi-agent pipeline designed to assess the risk of Alzheimer's disease by analyzing patient clinical notes. It uses **fine-tuned LLaMA-3.1 8B models** for binary and multi-class classification, alongside a large language model for specialist reasoning.

This project is associated with the publication:  
🔗 [Nature Digital Medicine Article](https://www.nature.com/articles/s41746-025-01940-4)

---

## 📂 Project Structure

```
ad_care-main/
├── care_ad_system.py       # Core assessment pipeline and multi-agent specialization
├── host_services.py        # REST API server hosting classifiers & LLM
├── run_evaluation.py       # Evaluation pipeline with metrics reporting
├── train_classifer.py      # Fine-tuning script for LLaMA-3 classifiers
├── README.md               # Project documentation
```

---

## ⚙️ Installation

1. **Clone the repository**:

```bash
git clone https://github.com/<your-username>/ad_care-main.git
cd ad_care-main
```

2. **Install dependencies**:

You will also need:
- **PyTorch** with CUDA enabled
- **Transformers** (latest version)
- **FastAPI & Uvicorn**
- **scikit-learn**, **pandas**, **numpy**
- **GPUtil** & **psutil** for system monitoring

3. **Download Base Models**  
Ensure you have access to **Meta-LLaMA-3.1 8B and LlaMa-3-70B** weights from HuggingFace and accept their license.

---

## 🚀 Workflow Overview

The CARE-AD system consists of **4 main stages**:

### **1️⃣ Train the Classifiers**

You need to fine-tune **two models**:
- **Binary** classifier: Determines whether a sentence is AD-related or not.
- **Multi-class** classifier: Categorizes AD-related symptoms into categories (Cognitive Impairment, Neuropsychiatric Symptoms, etc.)

**Example commands:**

```bash
# Train binary classifier
python train_classifer.py \
    --model_name "meta-llama/Meta-Llama-3.1-8B" \
    --train_data "sentiment_train.csv" \
    --val_data "sentiment_val.csv" \
    --task_type "binary" \
    --output_dir "./llama3-binary-classifier" \
    --batch_size 4 \
    --num_epochs 3

# Train multi-class classifier
python train_classifer.py \
    --model_name "meta-llama/Meta-Llama-3.1-8B" \
    --train_data "topic_train.csv" \
    --val_data "topic_val.csv" \
    --task_type "multiclass" \
    --output_dir "./llama3-multi-classifier" \
    --batch_size 4 \
    --num_epochs 3
```

---

### **2️⃣ Host the Services**

Launch the FastAPI server that hosts the **two classifiers** and a **LLaMA-3-70B Instruct** model for generation.

```bash
python host_services.py
```

This starts endpoints:
- `POST /classify/binary`
- `POST /classify/multi`
- `POST /generate`
- `GET /health` – system health check

---

### **3️⃣ Prepare Patient Data**

Your test data must be in the following JSON format:

```json
[
  {
    "patient_info": {
      "patient_id": "PT001",
      "sex": "Female",
      "birth_year": 1955,
      "race": "Asian",
      "ethnicity": "Not Hispanic"
    },
    "clinical_notes": [
      {
        "age": 67,
        "text": "Patient reports memory lapses, forgetting names and appointments..."
      },
      {
        "age": 68,
        "text": "Increased confusion with medication management. Difficulty following conversations..."
      }
    ],
    "ground_truth": "Yes"
  },
  {
    "patient_info": {
      "patient_id": "PT002",
      "sex": "Male",
      "birth_year": 1960,
      "race": "Caucasian",
      "ethnicity": "Not Hispanic"
    },
    "clinical_notes": [
      {
        "age": 60,
        "text": "Patient reports normal memory function..."
      }
    ],
    "ground_truth": "No"
  }
]
```

---

### **4️⃣ Run the Evaluation**

```bash
python run_evaluation.py test_data.json
```

This will:
1. Load test data
2. Pass it through the CARE-AD system (`care_ad_system.py`)
3. Collect predictions
4. Output metrics (accuracy, precision, recall, F1)
5. Save reports if configured

---

## 🧠 How the System Works

1. **Data Extraction**:  
   - Splits clinical notes into sentences.
   - Filters AD-related sentences via the binary classifier.

2. **Symptom Categorization**:  
   - Uses the multi-class classifier to assign each AD-related sentence into predefined categories.

3. **Specialist Agent Analysis**:  
   - Five simulated virtual specialists provide textual analyses from their expertise.

4. **Final Decision**:  
   - An AD specialist model integrates all specialist opinions into a **binary risk assessment**.

---

## 📜 Key Files

- **`train_classifer.py`** — Fine-tunes LLaMA models for binary/multi-class classification.
- **`host_services.py`** — Runs a FastAPI server to provide model inference endpoints.
- **`care_ad_system.py`** — Defines the CARE-AD multi-agent pipeline logic.
- **`run_evaluation.py`** — Evaluates performance against test sets and reports results.

---

## 📊 Example Output

After running evaluation:

```
PERFORMANCE REPORT
```
---

## ⚠️ Notes

- You must **host the models** before running an evaluation.
- Large models (LLaMA-3-70B) require substantial VRAM (≥ 80GB for full precision inference; FP16 recommended).
---

## 📄 License

Please see the license terms of the Meta-LLaMA weights and the Nature publication accompanying this code before use.

---
