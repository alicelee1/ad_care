v1. multi agent care-ad system

1) train the classifiers

2) host the services, including 2 classifiers and 1 LLM

4) prepare the data
   
5) run the evaluation


Data format:

cases = [
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
                    "birth_year": 1948,
                    "race": "Caucasian",
                    "ethnicity": "Not Hispanic"
                },
                "clinical_notes": [
                    {
                        "age": 72,
                        "text": "Progressive memory decline over past year. Disorientation in time and place. Difficulty with complex tasks and planning."
                    }
                ],
                "ground_truth": "Yes"
            }
        ]
