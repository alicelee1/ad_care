import requests
import json
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class PatientProfile:
    """Patient profile structure"""
    patient_id: str
    sex: str
    birth_year: int
    race: str
    ethnicity: str
    ad_symptoms_by_age: Dict[int, List[Dict[str, str]]]
    
    def to_string(self) -> str:
        """Convert patient profile to formatted string"""
        profile_str = f"Patient Profile for Pt ID {self.patient_id}:\n\n"
        profile_str += "Demographic Information\n"
        profile_str += f"• Sex: {self.sex}\n"
        profile_str += f"• Birth Year: {self.birth_year}\n"
        profile_str += f"• Race: {self.race}\n"
        profile_str += f"• Ethnicity: {self.ethnicity}\n\n"
        profile_str += "AD-related signs and symptoms by Age\n"
        
        for age in sorted(self.ad_symptoms_by_age.keys()):
            profile_str += f"\nAge {age}:\n"
            for symptom in self.ad_symptoms_by_age[age]:
                profile_str += f"• [{symptom['category']}]: {symptom['sentence']}\n"
        
        return profile_str


class CAREADSystem:
    """CARE-AD System for Alzheimer's Disease Risk Assessment"""
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url
        
        # Category mapping for multi-class classification
        self.category_mapping = {
            -1: "Others",
            0: "Cognitive Impairment",
            1: "Neuropsychiatric Symptoms", 
            2: "Notice by Others",
            3: "Physiological Changes",
            4: "Requiring Assistance"
        }
        
        # Specialist prompts
        self.specialist_prompts = {
            "primary_care": {
                "role": "a primary care physician with comprehensive knowledge of adult health and preventive care",
                "instruction": "Please assess the patient's overall physical health, medical history, comorbid conditions, and daily functioning. Identify any concerns that may be related to Alzheimer's disease, and briefly explain why these findings are concerning.",
                "additional": "• Highlight any red flags that warrant further neurological or cognitive evaluation.\n• Summarize key clinical observations that could guide next steps in the diagnostic process."
            },
            "neurologist": {
                "role": "a neurologist specializing in disorders of the brain and nervous system",
                "instruction": "Please evaluate the patient's neurological status, including memory, executive function, and any motor or sensory findings. Identify concerns that may be related to Alzheimer's disease and explain the reasoning behind each concern.",
                "additional": "• Focus on symptoms such as memory loss, language difficulties, or other cognitive deficits.\n• Provide a concise rationale for suspected AD-related signs."
            },
            "geriatrician": {
                "role": "a geriatrician with expertise in the care of older adults",
                "instruction": "Please assess the patient's overall functional status, comorbidities, and geriatric syndromes (e.g., fall risk, frailty, polypharmacy). Identify any concerns that may be related to Alzheimer's disease and provide a brief explanation for each.",
                "additional": "• Consider the patient's aging trajectory, functional independence, and support system.\n• Highlight interactions between comorbidities and cognitive decline.\n• Integrate information about daily living activities and any noticeable changes in routine or self-care."
            },
            "psychiatrist": {
                "role": "a psychiatrist specializing in mental and emotional health",
                "instruction": "Please evaluate the patient's psychiatric presentation, including mood, affect, and behavioral changes. Identify any concerns potentially linked to Alzheimer's disease and articulate the basis for each concern.",
                "additional": "• Assess depression, anxiety, or other psychiatric comorbidities that may mask or exacerbate cognitive decline.\n• Note any behavioral disturbances, personality changes, or psychotic features that could be indicative of AD.\n• Consider how cognitive deficits might interact with mental health conditions in formulating your overall clinical impression."
            },
            "clinical_psychologist": {
                "role": "a clinical psychologist focusing on cognitive assessment",
                "instruction": "Please evaluate the patient's cognitive domains (memory, attention, language, executive function, and visuospatial skills). Identify concerns that may indicate Alzheimer's disease and explain the underlying reasons for these concerns.",
                "additional": "• Highlight patterns of cognitive performance characteristic of Alzheimer's disease.\n• Include relevant psychological theories or models that support your assessment."
            }
        }
        
        self.ad_specialist_prompt = """You are a highly qualified specialist in Alzheimer's disease, with extensive clinical and research experience. You are presented with a comprehensive profile of a patient's AD-related signs and symptoms, alongside evaluations from multiple specialty clinicians. Please integrate this information in light of recognized diagnostic criteria and determine whether the patient is likely to develop Alzheimer's disease in the future. Provide your assessment as either "Yes" or "No", followed by one or two concise sentences explaining the basis for your conclusion."""

    def classify_sentence_binary(self, sentence: str) -> Dict:
        """Step 1: Binary classification to check if sentence is AD-related"""
        try:
            response = requests.post(
                f"{self.server_url}/classify/binary",
                json={
                    "text": sentence,
                    "return_probabilities": True
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error in binary classification: {e}")
            return {"predicted_class": 0, "confidence": 0.0}

    def classify_sentence_multi(self, sentence: str) -> Dict:
        """Step 1: Multi-class classification to categorize AD-related symptoms"""
        try:
            response = requests.post(
                f"{self.server_url}/classify/multi",
                json={
                    "text": sentence,
                    "return_probabilities": True
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error in multi-class classification: {e}")
            return {"predicted_class": -1, "confidence": 0.0}  # Default to -1#need to handle separately

    def extract_patient_data(self, clinical_notes: List[Dict], patient_info: Dict) -> PatientProfile:
        """Step 1: Data extraction from clinical notes"""
        print("=" * 80)
        print("STEP 1: DATA EXTRACTION")
        print("=" * 80)
        
        ad_symptoms_by_age = defaultdict(list)
        
        for note in clinical_notes:
            patient_age = note.get("age", 0)
            sentences = self._split_into_sentences(note.get("text", ""))
            
            print(f"\nProcessing clinical note for age {patient_age}:")
            print(f"Found {len(sentences)} sentences to analyze")
            
            for i, sentence in enumerate(sentences):
                if len(sentence.strip()) < 10:  # Skip very short sentences
                    continue
                    
                print(f"\nSentence {i+1}: {sentence[:100]}...")
                
                # Binary classification first
                binary_result = self.classify_sentence_binary(sentence)
                print(f"Binary classification: {binary_result['predicted_class']} (confidence: {binary_result['confidence']:.3f})")
                
                # If classified as AD-related (assuming 1 = AD-related, 0 = not AD-related)
                if binary_result['predicted_class'] == 1 and binary_result['confidence'] > 0.5:
                    # Multi-class classification for categorization
                    multi_result = self.classify_sentence_multi(sentence)
                    category_id = multi_result['predicted_class']
                    category_name = self.category_mapping.get(category_id, "Other")
                    
                    print(f"Multi-class classification: {category_name} (confidence: {multi_result['confidence']:.3f})")
                    
                    ad_symptoms_by_age[patient_age].append({
                        "sentence": sentence,
                        "category": category_name,
                        "binary_confidence": binary_result['confidence'],
                        "multi_confidence": multi_result['confidence']
                    })
                    #print("✓ Added to AD symptoms profile")
                else:
                    pass
                    #print("✗ Not classified as AD-related")
        
        # Create patient profile
        profile = PatientProfile(
            patient_id=patient_info.get("patient_id", "unknown"),
            sex=patient_info.get("sex", "Unknown"),
            birth_year=patient_info.get("birth_year", 1950),
            race=patient_info.get("race", "Unknown"),
            ethnicity=patient_info.get("ethnicity", "Unknown"),
            ad_symptoms_by_age=dict(ad_symptoms_by_age)
        )
        
        print("\n" + "=" * 80)
        print("EXTRACTED PATIENT PROFILE:")
        print("=" * 80)
        print(profile.to_string())
        
        return profile

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting (can be improved with more sophisticated methods)
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def generate_specialist_analysis(self, specialist_name: str, patient_profile: PatientProfile) -> str:
        """Step 2: Generate specialist analysis using Llama 70B"""
        prompt_config = self.specialist_prompts[specialist_name]
        
        prompt = f"""You are {prompt_config['role']}. A new patient has arrived for evaluation. {prompt_config['instruction']}

Additional Instructions:
{prompt_config['additional']}

Patient information: {patient_profile.to_string()}"""
        
        try:
            response = requests.post(
                f"{self.server_url}/generate",
                json={
                    "prompt": prompt,
                    "max_new_tokens": 500,
                    "temperature": 0.3,
                    "top_p": 0.9
                },
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            return result["generated_text"].strip()
        except requests.exceptions.RequestException as e:
            print(f"Error generating {specialist_name} analysis: {e}")
            return f"Error: Could not generate analysis for {specialist_name}"

    def run_specialist_agents(self, patient_profile: PatientProfile) -> Dict[str, str]:
        """Step 2: Run all 5 specialist agents"""
        print("\n" + "=" * 80)
        print("STEP 2: SPECIALIST AGENTS ANALYSIS")
        print("=" * 80)
        
        specialist_analyses = {}
        
        for specialist_name in self.specialist_prompts.keys():
            print(f"\n{'-' * 60}")
            print(f"RUNNING {specialist_name.upper().replace('_', ' ')} ANALYSIS")
            print(f"{'-' * 60}")
            
            analysis = self.generate_specialist_analysis(specialist_name, patient_profile)
            specialist_analyses[specialist_name] = analysis
            
            print(f"\n{specialist_name.replace('_', ' ').title()} Analysis:")
            print(analysis)
        
        return specialist_analyses

    def generate_ad_specialist_decision(self, patient_profile: PatientProfile, specialist_analyses: Dict[str, str]) -> str:
        """Step 3: AD Specialist final decision"""
        # Format specialist analyses for the prompt
        specialist_text = ""
        for specialist, analysis in specialist_analyses.items():
            specialist_text += f"\n{specialist.replace('_', ' ').title()}: {analysis}\n"
        
        prompt = f"""{self.ad_specialist_prompt}

Patient information: {patient_profile.to_string()}

Specialist analysis: {specialist_text}"""
        
        try:
            response = requests.post(
                f"{self.server_url}/generate",
                json={
                    "prompt": prompt,
                    "max_new_tokens": 300,
                    "temperature": 0.1,  # Lower temperature for more consistent decisions
                    "top_p": 0.8
                },
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            return result["generated_text"].strip()
        except requests.exceptions.RequestException as e:
            print(f"Error generating AD specialist decision: {e}")
            return "Error: Could not generate final decision"

    def parse_ad_decision(self, decision_text: str) -> str:
        """Parse AD specialist decision according to the rules"""
        decision_lower = decision_text.lower()
        
        # Check for explicit "No" or statements about no AD risk
        if ("no" in decision_lower and ("ad risk" in decision_lower or "alzheimer" in decision_lower)) or \
           ("not related to ad" in decision_lower) or \
           ("no risk" in decision_lower):
            return "No"
        
        # Check for explicit "Yes"
        if decision_text.strip().startswith("Yes") or "yes" in decision_lower[:20]:
            return "Yes"
        
        return "Yes"

    def assess_patient(self, clinical_notes: List[Dict], patient_info: Dict) -> Dict:
        """Complete CARE-AD assessment workflow"""
        print("STARTING CARE-AD SYSTEM ASSESSMENT")
        print("=" * 80)
        
        # Step 1: Data Extraction
        patient_profile = self.extract_patient_data(clinical_notes, patient_info)
        
        # Step 2: Specialist Agents
        specialist_analyses = self.run_specialist_agents(patient_profile)
        
        # Step 3: AD Specialist Decision
        print("\n" + "=" * 80)
        print("STEP 3: AD SPECIALIST FINAL DECISION")
        print("=" * 80)
        
        final_decision_text = self.generate_ad_specialist_decision(patient_profile, specialist_analyses)
        print(f"\nAD Specialist Raw Decision:")
        print(final_decision_text)
        
        final_decision = self.parse_ad_decision(final_decision_text)
        print(f"\nParsed Final Decision: {final_decision}")
        
        return {
            "patient_profile": patient_profile,
            "specialist_analyses": specialist_analyses,
            "final_decision_text": final_decision_text,
            "final_decision": final_decision,
            "assessment_timestamp": datetime.now().isoformat()
        }


# Example usage and test case
def main():
    """Example usage of the CARE-AD system"""
    
    # Initialize the system
    care_ad = CAREADSystem()
    
    # Example patient data
    patient_info = {
        "patient_id": "PT001234",
        "sex": "Female",
        "birth_year": 1955,
        "race": "Asian",
        "ethnicity": "Not Hispanic"
    }
    
    # Example clinical notes with different ages
    clinical_notes = [
        {
            "age": 67,
            "text": """Patient reports occasionally forgetting grocery lists and spouse notes she asked about weekend plans twice in one day. Mild hearing loss in left ear per audiology consult; denies impact on daily life. Son states patient misplaced passport before trip and anxious about airport directions. Patient describes trouble recalling new coworker names saying they tell me, but it slips away."""
        },
        {
            "age": 68, 
            "text": """Patient shows increased irritability when routines are disrupted and snaps if dinner is late. Patient mentions difficulty focusing during budget meetings saying numbers get jumbled in my head. Patient appears generally well oriented and cooperative during visit."""
        },
        {
            "age": 70,
            "text": """Daughter notes patient repeatedly checked stove knobs before leaving home last month. Patient complains of muffled taste for coffee. Spouse describes apathy toward book club saying used to love it, now says it's too much work. Patient states I keep losing track of time and missed two haircuts this year."""
        }
    ]
    
    # Run the assessment
    try:
        results = care_ad.assess_patient(clinical_notes, patient_info)
        
        print("\n" + "=" * 80)
        print("CARE-AD ASSESSMENT COMPLETE")
        print("=" * 80)
        print(f"Final Decision: {results['final_decision']}")
        print(f"Assessment completed at: {results['assessment_timestamp']}")
        
        return results
        
    except Exception as e:
        print(f"Error during assessment: {e}")
        return None


if __name__ == "__main__":
    # Test the system
    results = main()
    
    # Additional test with different patient
    print("\n\n" + "=" * 100)
    print("TESTING WITH DIFFERENT PATIENT")
    print("=" * 100)
    
    care_ad = CAREADSystem()
    
    test_patient = {
        "patient_id": "PT005678",
        "sex": "Male", 
        "birth_year": 1948,
        "race": "Caucasian",
        "ethnicity": "Not Hispanic"
    }
    
    test_notes = [
        {
            "age": 72,
            "text": """Patient presents for routine follow-up. Blood pressure well controlled on current medications. Patient reports feeling well and maintaining active lifestyle with daily walks. No cognitive complaints. Family reports no concerns about memory or function."""
        },
        {
            "age": 73,
            "text": """Patient continues to do well. Manages own medications and finances independently. Still driving without issues. Plays bridge weekly and volunteers at local library. No memory concerns reported by patient or family."""
        }
    ]
    
    try:
        test_results = care_ad.assess_patient(test_notes, test_patient)
        print(f"\nTest Patient Final Decision: {test_results['final_decision']}")
    except Exception as e:
        print(f"Error in test assessment: {e}")


# Client code for easy testing
"""
# To test the system, you can also run individual components:

care_ad = CAREADSystem()

# Test binary classification
sentence = "Patient reports forgetting names frequently"
binary_result = care_ad.classify_sentence_binary(sentence)
print(f"Binary: {binary_result}")

# Test multi-class classification  
multi_result = care_ad.classify_sentence_multi(sentence)
print(f"Multi-class: {multi_result}")

# Test specialist analysis
patient_profile = PatientProfile(...)
analysis = care_ad.generate_specialist_analysis("neurologist", patient_profile)
print(f"Neurologist analysis: {analysis}")
"""
