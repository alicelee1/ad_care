from care_ad_system import CAREADSystem
from mock_test import TestDataGenerator

# Initialize
care_ad = CAREADSystem(server_url="http://localhost:8000")
test_gen = TestDataGenerator()

# Test with different risk levels
high_risk = test_gen.get_patient_by_risk("high")
results = care_ad.assess_patient(
    high_risk["clinical_notes"],
    high_risk["patient_info"]
)

print(f"Result: {results['final_decision']}")
