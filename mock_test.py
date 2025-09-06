import json
import random
import re
from datetime import datetime
from typing import Dict, List, Any
from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn

# Mock service models
class ClassificationRequest(BaseModel):
    text: str
    return_probabilities: bool = False

class ClassificationResponse(BaseModel):
    predicted_class: int
    confidence: float
    probabilities: Dict[str, float] = None
    processing_time: float

class GenerationRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    stream: bool = False

class GenerationResponse(BaseModel):
    generated_text: str
    prompt_tokens: int
    generated_tokens: int
    processing_time: float

# Mock FastAPI service
app = FastAPI(title="Mock CARE-AD Model Service", version="1.0.0")

class MockModelService:
    """Mock service that simulates the classification and generation models"""
    
    def __init__(self):
        # Keywords for binary classification (AD-related vs not)
        self.ad_keywords = [
            'memory', 'forget', 'confusion', 'lost', 'disoriented', 'cognitive',
            'dementia', 'alzheimer', 'recall', 'remember', 'names', 'faces',
            'irritability', 'mood', 'personality', 'behavior', 'agitation',
            'anxiety', 'depression', 'apathy', 'wandering', 'repeating',
            'difficulty', 'trouble', 'problems', 'decline', 'deterioration',
            'impairment', 'disability', 'functional', 'independence'
        ]
        
        # Multi-class category keywords (removed category 4 "Other")
        self.category_keywords = {
            0: ['memory', 'forget', 'recall', 'cognitive', 'confusion', 'names', 'faces', 'remember', 'difficulty', 'trouble', 'problems', 'focusing', 'concentration'],  # Cognitive Impairment
            1: ['irritability', 'mood', 'personality', 'behavior', 'agitation', 'anxiety', 'depression', 'apathy', 'snaps', 'disrupted', 'overwhelmed'],  # Neuropsychiatric
            2: ['family', 'spouse', 'daughter', 'son', 'noticed', 'reports', 'states', 'observed', 'notes', 'describes'],  # Notice by Others
            3: ['hearing', 'vision', 'taste', 'smell', 'physical', 'motor', 'balance', 'coordination', 'muffled', 'loss']  # Physiological
        }
        
        # Specialist response templates
        self.specialist_templates = {
            'primary_care': self._primary_care_template,
            'neurologist': self._neurologist_template,
            'geriatrician': self._geriatrician_template,
            'psychiatrist': self._psychiatrist_template,
            'clinical_psychologist': self._clinical_psychologist_template
        }
    
    def binary_classify(self, text: str, return_probabilities: bool = False) -> Dict:
        """Mock binary classification"""
        text_lower = text.lower()
        
        # Count AD-related keywords
        ad_score = sum(1 for keyword in self.ad_keywords if keyword in text_lower)
        
        # Determine if AD-related based on keyword presence
        is_ad_related = ad_score > 0
        confidence = min(0.6 + (ad_score * 0.1), 0.95) if is_ad_related else max(0.8 - (ad_score * 0.1), 0.3)
        
        # Add some randomness
        if random.random() < 0.1:  # 10% chance of random classification
            is_ad_related = not is_ad_related
            confidence = random.uniform(0.5, 0.7)
        
        predicted_class = 1 if is_ad_related else 0
        
        result = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'processing_time': random.uniform(0.1, 0.3)
        }
        
        if return_probabilities:
            prob_ad = confidence if is_ad_related else (1 - confidence)
            result['probabilities'] = {
                '0': 1 - prob_ad,
                '1': prob_ad
            }
        
        return result
    
    def multi_classify(self, text: str, return_probabilities: bool = False) -> Dict:
        """Mock multi-class classification"""
        text_lower = text.lower()
        
        # Score each category (excluding "Other" category 4)
        category_scores = {}
        for category_id, keywords in self.category_keywords.items():
            if category_id == 4:  # Skip "Other" category
                continue
            score = sum(1 for keyword in keywords if keyword in text_lower)
            category_scores[category_id] = score
        
        # Find best category from meaningful categories (0-3)
        if category_scores and max(category_scores.values()) > 0:
            best_category = max(category_scores.keys(), key=lambda k: category_scores[k])
            confidence = min(0.6 + (category_scores[best_category] * 0.15), 0.9)
        else:
            # If no specific keywords found, default to Cognitive Impairment (most common)
            best_category = 0
            confidence = 0.55
        
        result = {
            'predicted_class': best_category,
            'confidence': confidence,
            'processing_time': random.uniform(0.1, 0.3)
        }
        
        if return_probabilities:
            total_score = sum(category_scores.values()) or 1
            probabilities = {}
            for cat_id in range(4):  # Only categories 0-3
                prob = category_scores.get(cat_id, 0) / total_score if total_score > 0 else 0.25
                probabilities[str(cat_id)] = prob
            result['probabilities'] = probabilities
        
        return result
    
    def generate_text(self, prompt: str) -> Dict:
        """Mock text generation - simplified to avoid parameter conflicts"""
        # Detect which specialist is being called
        specialist_type = self._detect_specialist_type(prompt)
        
        if specialist_type:
            generated_text = self.specialist_templates[specialist_type](prompt)
        else:
            # AD Specialist decision
            generated_text = self._ad_specialist_template(prompt)
        
        # Simulate token counts
        prompt_tokens = len(prompt.split()) * 1.3  # Rough estimate
        generated_tokens = len(generated_text.split())
        
        return {
            'generated_text': generated_text,
            'prompt_tokens': int(prompt_tokens),
            'generated_tokens': generated_tokens,
            'processing_time': random.uniform(1.0, 3.0)
        }
    
    def _detect_specialist_type(self, prompt: str) -> str:
        """Detect which specialist is being called based on prompt"""
        prompt_lower = prompt.lower()
        if 'primary care physician' in prompt_lower:
            return 'primary_care'
        elif 'neurologist' in prompt_lower:
            return 'neurologist'
        elif 'geriatrician' in prompt_lower:
            return 'geriatrician'
        elif 'psychiatrist' in prompt_lower:
            return 'psychiatrist'
        elif 'clinical psychologist' in prompt_lower:
            return 'clinical_psychologist'
        return None
    
    def _extract_patient_age_and_symptoms(self, prompt: str) -> tuple:
        """Extract patient information from prompt for more realistic responses"""
        ages = re.findall(r'Age (\d+):', prompt)
        symptoms = []
        
        # Extract symptom categories mentioned
        if 'Cognitive Impairment' in prompt:
            symptoms.append('cognitive')
        if 'Neuropsychiatric Symptoms' in prompt:
            symptoms.append('behavioral')
        if 'Notice by Others' in prompt:
            symptoms.append('family_observed')
        if 'Physiological Changes' in prompt:
            symptoms.append('physical')
            
        return ages, symptoms
    
    def _primary_care_template(self, prompt: str) -> str:
        ages, symptoms = self._extract_patient_age_and_symptoms(prompt)
        
        responses = [
            "Based on my assessment, I have several concerns regarding this patient's presentation.",
            "The reported memory difficulties and functional changes are particularly noteworthy.",
            "The pattern of cognitive decline observed across multiple domains suggests the need for further evaluation.",
            "Family observations of personality changes and increased confusion are significant red flags.",
            "I recommend immediate referral to neurology for comprehensive cognitive assessment and possible neuroimaging."
        ]
        
        if 'cognitive' in symptoms:
            responses.append("The memory complaints, particularly difficulty with names and forgetting recent conversations, are concerning for early cognitive impairment.")
        if 'behavioral' in symptoms:
            responses.append("The personality changes and irritability represent significant behavioral shifts that warrant investigation.")
        
        return " ".join(random.sample(responses, min(3, len(responses))))
    
    def _neurologist_template(self, prompt: str) -> str:
        ages, symptoms = self._extract_patient_age_and_symptoms(prompt)
        
        responses = [
            "Neurological examination reveals concerning patterns consistent with early-stage cognitive decline.",
            "The memory deficits, particularly episodic memory impairment, are suggestive of medial temporal lobe dysfunction.",
            "Executive function difficulties noted in budget management and task completion indicate frontal lobe involvement.",
            "The progression of symptoms across multiple cognitive domains is highly concerning for neurodegenerative process.",
            "Recommend formal neuropsychological testing and brain MRI to evaluate for structural changes consistent with Alzheimer's disease."
        ]
        
        if ages and int(ages[0]) > 65:
            responses.append(f"Given the patient's age of {ages[0]}, the constellation of symptoms is particularly concerning for AD pathology.")
        
        return " ".join(random.sample(responses, min(3, len(responses))))
    
    def _geriatrician_template(self, prompt: str) -> str:
        ages, symptoms = self._extract_patient_age_and_symptoms(prompt)
        
        responses = [
            "From a geriatric perspective, this patient shows significant functional decline affecting activities of daily living.",
            "The combination of cognitive symptoms with physical manifestations suggests a systemic neurodegenerative process.",
            "Safety concerns are paramount given the reported incidents of stove-checking and getting lost.",
            "The patient's support system appears engaged, which is positive for ongoing care planning.",
            "Comprehensive geriatric assessment reveals multiple domains of impairment consistent with dementia syndrome."
        ]
        
        if 'family_observed' in symptoms:
            responses.append("Family reports of functional changes are often the most reliable indicators of cognitive decline in older adults.")
        
        return " ".join(random.sample(responses, min(3, len(responses))))
    
    def _psychiatrist_template(self, prompt: str) -> str:
        ages, symptoms = self._extract_patient_age_and_symptoms(prompt)
        
        responses = [
            "Mental status examination reveals mood changes and behavioral symptoms that may be prodromal to dementia.",
            "The apathy and loss of interest in previously enjoyed activities are concerning neuropsychiatric features.",
            "Irritability and personality changes often precede more obvious cognitive symptoms in neurodegenerative disorders.",
            "No evidence of primary psychiatric disorder; symptoms appear secondary to underlying cognitive pathology.",
            "The behavioral profile is consistent with early-stage Alzheimer's disease rather than primary mood disorder."
        ]
        
        if 'behavioral' in symptoms:
            responses.append("The neuropsychiatric symptoms described are highly characteristic of early AD pathology.")
        
        return " ".join(random.sample(responses, min(3, len(responses))))
    
    def _clinical_psychologist_template(self, prompt: str) -> str:
        ages, symptoms = self._extract_patient_age_and_symptoms(prompt)
        
        responses = [
            "Cognitive assessment reveals impairments across multiple domains including memory, executive function, and attention.",
            "The pattern of deficits is consistent with cortical dementia, specifically Alzheimer's type pathology.",
            "Episodic memory impairment with relative preservation of procedural memory supports AD diagnosis.",
            "Executive dysfunction evidenced by planning difficulties and task management problems.",
            "Recommend comprehensive neuropsychological battery to fully characterize the cognitive profile."
        ]
        
        if 'cognitive' in symptoms:
            responses.append("The memory encoding and retrieval difficulties follow the typical pattern seen in early Alzheimer's disease.")
        
        return " ".join(random.sample(responses, min(3, len(responses))))
    
    def _ad_specialist_template(self, prompt: str) -> str:
        """Generate AD specialist decision"""
        ages, symptoms = self._extract_patient_age_and_symptoms(prompt)
        
        # Analyze the severity based on symptoms present
        symptom_count = len(symptoms)
        has_multiple_domains = symptom_count >= 2
        has_family_concerns = 'family_observed' in symptoms
        has_cognitive_symptoms = 'cognitive' in symptoms
        
        # Decision logic
        if has_cognitive_symptoms and (has_multiple_domains or has_family_concerns):
            decision = "Yes"
            reasoning = "The patient demonstrates cognitive decline across multiple domains with functional impairment and family-observed changes, consistent with early Alzheimer's disease pathology."
        elif symptom_count >= 2:
            decision = "Yes" 
            reasoning = "Multiple concerning symptoms and specialist evaluations support high risk for Alzheimer's disease development."
        elif symptom_count == 1 and has_cognitive_symptoms:
            decision = "Further evaluation needed"
            reasoning = "While some cognitive concerns are present, additional testing is required to establish definitive risk."
        else:
            decision = "No"
            reasoning = "Current symptoms do not demonstrate sufficient evidence for significant Alzheimer's disease risk."
        
        return f"{decision}. {reasoning}"

# Initialize mock service
mock_service = MockModelService()

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": ["binary_classification", "multi_classification", "llama_70b_generation"],
        "gpu_memory": {"mock": "No GPU needed for mock service"},
        "system_memory": {"mock": "Mock service uses minimal memory"},
        "timestamp": datetime.now().isoformat()
    }

@app.post("/classify/binary", response_model=ClassificationResponse)
async def binary_classification(request: ClassificationRequest):
    try:
        result = mock_service.binary_classify(request.text, request.return_probabilities)
        return ClassificationResponse(**result)
    except Exception as e:
        print(f"Error in binary classification: {e}")
        return ClassificationResponse(
            predicted_class=0,
            confidence=0.5,
            processing_time=0.1
        )

@app.post("/classify/multi", response_model=ClassificationResponse)
async def multi_classification(request: ClassificationRequest):
    try:
        result = mock_service.multi_classify(request.text, request.return_probabilities)
        return ClassificationResponse(**result)
    except Exception as e:
        print(f"Error in multi classification: {e}")
        return ClassificationResponse(
            predicted_class=0,
            confidence=0.5,
            processing_time=0.1
        )

@app.post("/generate", response_model=GenerationResponse)
async def text_generation(request: GenerationRequest):
    try:
        # Only pass the prompt to avoid parameter conflicts
        result = mock_service.generate_text(request.prompt)
        return GenerationResponse(**result)
    except Exception as e:
        print(f"Error in text generation: {e}")
        # Return a fallback response
        return GenerationResponse(
            generated_text="I apologize, but I'm unable to provide a detailed analysis at this time. Please consult with a healthcare professional for proper evaluation.",
            prompt_tokens=len(request.prompt.split()),
            generated_tokens=20,
            processing_time=0.1
        )

@app.get("/models")
async def list_models():
    return {
        "models": {
            "binary_classification": {"loaded": True, "type": "mock"},
            "multi_classification": {"loaded": True, "type": "mock", "num_labels": 4},
            "llama_70b_generation": {"loaded": True, "type": "mock"}
        }
    }

# Test data for CARE-AD system
class TestDataGenerator:
    """Generate realistic test data for CARE-AD system"""
    
    def __init__(self):
        self.patient_templates = [
            # High-risk AD patient
            {
                "patient_info": {
                    "patient_id": "PT_AD_001",
                    "sex": "Female",
                    "birth_year": 1955,
                    "race": "Caucasian",
                    "ethnicity": "Not Hispanic"
                },
                "clinical_notes": [
                    {
                        "age": 67,
                        "text": "Patient reports occasionally forgetting grocery lists and spouse notes she asked about weekend plans twice in one day. Mild hearing loss in left ear per audiology consult; denies impact on daily life. Son states patient misplaced passport before trip and anxious about airport directions. Patient describes trouble recalling new coworker names saying they tell me, but it slips away."
                    },
                    {
                        "age": 68,
                        "text": "Patient shows increased irritability when routines are disrupted and snaps if dinner is late. Patient mentions difficulty focusing during budget meetings saying numbers get jumbled in my head. Patient reports feeling overwhelmed by simple tasks that were previously manageable."
                    },
                    {
                        "age": 70,
                        "text": "Daughter notes patient repeatedly checked stove knobs before leaving home last month. Patient complains of muffled taste for coffee. Spouse describes apathy toward book club saying used to love it, now says it's too much work. Patient states I keep losing track of time and missed two haircuts this year. Family reports patient got lost driving to familiar grocery store."
                    }
                ]
            }
        ]
    
    def get_test_patients(self) -> List[Dict]:
        """Return all test patients"""
        return self.patient_templates
    
    def get_patient_by_risk(self, risk_level: str = "high") -> Dict:
        """Get patient by risk level"""
        return self.patient_templates[0]  # For now, just return the high-risk patient

def run_mock_service():
    """Run the mock service"""
    print("Starting Fixed Mock CARE-AD Service...")
    print("Service will be available at: http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "service":
        # Run mock service
        run_mock_service()
    else:
        # Show test data
        test_gen = TestDataGenerator()
        print("Fixed Mock Service Ready!")
        print("Run with: python fixed_mock_service.py service")