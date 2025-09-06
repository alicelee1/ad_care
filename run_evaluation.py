import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import sys

class CAREADEvaluator:
    """Evaluator for CARE-AD system performance"""
    
    def __init__(self, system):
        self.system = system
        self.results = []
    
    def load_test_data(self, file_path: str) -> List[Dict]:
        """Load test data with ground truth labels"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Validate test data structure
            if not self._validate_test_data(data):
                raise ValueError("Invalid test data format")
                
            print(f"Successfully loaded {len(data)} test cases from {file_path}")
            return data
            
        except FileNotFoundError:
            print(f"Error: Test data file {file_path} not found")
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in {file_path}")
            sys.exit(1)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    def _validate_test_data(self, data: List[Dict]) -> bool:
        """Validate test data structure"""
        if not isinstance(data, list):
            return False
        
        for i, test_case in enumerate(data):
            if not isinstance(test_case, dict):
                print(f"Error: Test case {i} is not a dictionary")
                return False
            
            # Check required fields
            required_fields = ['patient_info', 'clinical_notes', 'ground_truth']
            for field in required_fields:
                if field not in test_case:
                    print(f"Error: Missing '{field}' in test case {i}")
                    return False
            
            # Validate patient_info
            patient_info = test_case['patient_info']
            required_patient_fields = ['patient_id', 'sex', 'birth_year', 'race', 'ethnicity']
            for field in required_patient_fields:
                if field not in patient_info:
                    print(f"Error: Missing '{field}' in patient_info for test case {i}")
                    return False
            
            # Validate clinical_notes
            clinical_notes = test_case['clinical_notes']
            if not isinstance(clinical_notes, list):
                print(f"Error: clinical_notes must be a list in test case {i}")
                return False
            
            for j, note in enumerate(clinical_notes):
                if not isinstance(note, dict):
                    print(f"Error: Clinical note {j} in test case {i} is not a dictionary")
                    return False
                if 'age' not in note or 'text' not in note:
                    print(f"Error: Clinical note {j} in test case {i} missing 'age' or 'text'")
                    return False
            
            # Validate ground_truth
            ground_truth = test_case['ground_truth']
            if ground_truth not in ['Yes', 'No']:
                print(f"Error: ground_truth must be 'Yes' or 'No' in test case {i}")
                return False
        
        return True
    
    def run_evaluation(self, test_data: List[Dict], max_patients: int = None) -> Dict:
        """Run evaluation on test data"""
        if not test_data:
            print("Error: No test data available")
            return {}
        
        if max_patients:
            test_data = test_data[:max_patients]
        
        print(f"Starting evaluation on {len(test_data)} patients...")
        print("=" * 80)
        
        results = []
        start_time = time.time()
        
        for i, test_case in enumerate(test_data, 1):
            print(f"\nProcessing patient {i}/{len(test_data)}: {test_case['patient_info']['patient_id']}")
            print("-" * 60)
            
            try:
                # Run the CARE-AD system
                result = self.system.assess_patient(
                    test_case["clinical_notes"],
                    test_case["patient_info"]
                )
                
                # Add ground truth for evaluation
                result["ground_truth"] = test_case["ground_truth"]
                result["patient_id"] = test_case["patient_info"]["patient_id"]
                
                results.append(result)
                
                print(f"Prediction: {result['final_decision']} | Ground Truth: {test_case['ground_truth']}")
                
            except Exception as e:
                print(f"Error processing patient {test_case['patient_info']['patient_id']}: {e}")
                # Add error result
                results.append({
                    "patient_id": test_case["patient_info"]["patient_id"],
                    "final_decision": "Error",
                    "ground_truth": test_case["ground_truth"],
                    "error": str(e)
                })
        
        total_time = time.time() - start_time
        avg_time_per_patient = total_time / len(test_data) if test_data else 0
        
        self.results = results
        
        return {
            "results": results,
            "total_time": total_time,
            "avg_time_per_patient": avg_time_per_patient,
            "timestamp": datetime.now().isoformat()
        }
    
    def calculate_metrics(self) -> Dict:
        """Calculate performance metrics"""
        if not self.results:
            return {}
        
        # Extract predictions and ground truth
        y_true = []
        y_pred = []
        valid_results = []
        
        for result in self.results:
            if result['final_decision'] in ['Yes', 'No'] and 'ground_truth' in result:
                y_true.append(result['ground_truth'])
                y_pred.append(result['final_decision'])
                valid_results.append(result)
        
        if not y_true:
            return {"error": "No valid results for metric calculation"}
        
        # Convert to binary (Yes=1, No=0)
        y_true_binary = [1 if gt == "Yes" else 0 for gt in y_true]
        y_pred_binary = [1 if pred == "Yes" else 0 for pred in y_pred]
        
        # Calculate metrics
        accuracy = accuracy_score(y_true_binary, y_pred_binary)
        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true_binary, y_pred_binary)
        
        # Additional detailed metrics
        classification_rep = classification_report(y_true_binary, y_pred_binary, 
                                                 target_names=['No AD risk', 'AD risk'],
                                                 output_dict=True)
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm.tolist(),
            "classification_report": classification_rep,
            "total_patients": len(self.results),
            "valid_predictions": len(valid_results),
            "error_count": len([r for r in self.results if r.get('error')])
        }
    
    def generate_report(self, metrics: Dict, evaluation_stats: Dict) -> str:
        """Generate comprehensive performance report"""
        report = []
        report.append("CARE-AD SYSTEM PERFORMANCE REPORT")
        report.append("=" * 50)
        report.append(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Patients Processed: {metrics.get('total_patients', 0)}")
        report.append(f"Valid Predictions: {metrics.get('valid_predictions', 0)}")
        report.append(f"Errors: {metrics.get('error_count', 0)}")
        report.append("")
        
        report.append("PERFORMANCE METRICS")
        report.append("-" * 30)
        report.append(f"Accuracy: {metrics.get('accuracy', 0):.3f}")
        report.append(f"Precision: {metrics.get('precision', 0):.3f}")
        report.append(f"Recall: {metrics.get('recall', 0):.3f}")
        report.append(f"F1 Score: {metrics.get('f1_score', 0):.3f}")
        report.append("")
        
        report.append("CONFUSION MATRIX")
        report.append("-" * 30)
        cm = metrics.get('confusion_matrix', [[0, 0], [0, 0]])
        report.append(f"True Negatives: {cm[0][0]} | False Positives: {cm[0][1]}")
        report.append(f"False Negatives: {cm[1][0]} | True Positives: {cm[1][1]}")
        report.append("")
        
        report.append("PROCESSING STATISTICS")
        report.append("-" * 30)
        report.append(f"Total Processing Time: {evaluation_stats.get('total_time', 0):.2f} seconds")
        report.append(f"Average Time per Patient: {evaluation_stats.get('avg_time_per_patient', 0):.2f} seconds")
        report.append("")
        
        # Detailed classification report
        class_report = metrics.get('classification_report', {})
        if class_report:
            report.append("DETAILED CLASSIFICATION REPORT")
            report.append("-" * 30)
            for class_name, metrics_dict in class_report.items():
                if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                    report.append(f"{class_name}:")
                    report.append(f"  Precision: {metrics_dict.get('precision', 0):.3f}")
                    report.append(f"  Recall: {metrics_dict.get('recall', 0):.3f}")
                    report.append(f"  F1-score: {metrics_dict.get('f1-score', 0):.3f}")
                    report.append(f"  Support: {metrics_dict.get('support', 0)}")
        
        return "\n".join(report)
    

    
    def save_results(self, results: List[Dict], file_path: str):
        """Save results to JSON file"""
        try:
            with open(file_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {file_path}")
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def export_detailed_report(self, metrics: Dict, evaluation_stats: Dict, file_path: str):
        """Export detailed report to text file"""
        report = self.generate_report(metrics, evaluation_stats)
        try:
            with open(file_path, 'w') as f:
                f.write(report)
            print(f"Detailed report saved to {file_path}")
        except Exception as e:
            print(f"Error saving report: {e}")

def main():
    """Main function to run the evaluation"""
    
    # Check command line arguments for test data file
    if len(sys.argv) != 2:
        print("Usage: python care_ad_evaluator.py <test_data_file.json>")
        sys.exit(1)
    
    test_data_file = sys.argv[1]
    
    # Initialize the CARE-AD system
    care_ad_system = CAREADSystem(server_url="http://localhost:8000")
    
    # Initialize evaluator
    evaluator = CAREADEvaluator(care_ad_system)
    
    # Load test data
    test_data = evaluator.load_test_data(test_data_file)
    
    # Run evaluation
    evaluation_stats = evaluator.run_evaluation(test_data)
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics()
    
    # Generate and print report
    report = evaluator.generate_report(metrics, evaluation_stats)
    print("\n" + "=" * 80)
    print("PERFORMANCE REPORT")
    print("=" * 80)
    print(report)

if __name__ == "__main__":
    main()
