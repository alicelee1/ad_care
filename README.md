care-ad, a multi agent system for ad prediction. v1

code for https://www.nature.com/articles/s41746-025-01940-4

1) train the classifiers

python train_classifer.py \
    --model_name "meta-llama/Meta-Llama-3.1-8B" \
    --train_data "sentiment_train.csv" \
    --val_data "sentiment_val.csv" \
    --task_type "binary" \
    --output_dir "./llama3-binary-classifier" \
    --batch_size 4 \
    --num_epochs 3

python train_classifer.py \
    --model_name "meta-llama/Meta-Llama-3.1-8B" \
    --train_data "topic_train.csv" \
    --val_data "topic_val.csv" \
    --task_type "multiclass" \
    --output_dir "./llama3-multi-classifier" \
    --batch_size 4 \
    --num_epochs 3

2) host the services, including 2 classifiers and 1 LLM
   
python host_services.py

3) prepare the data

Data format:

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
        "text": "Patient reports memory lapses, forgetting names and appointments. Family notes repetitive questioning and getting lost in familiar places."
      },
      {
        "age": 68,
        "text": "Increased confusion with medication management. Difficulty following conversations and trouble finding words."
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
        "text": "Patient reports normal memory function. Active lifestyle, manages own finances and medications without issues."
      }
    ],
    "ground_truth": "No"
  }
]

   
4) run the evaluation
   
python care_ad_evaluator.py test_data.json

