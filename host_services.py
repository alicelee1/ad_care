import torch
import uvicorn
import json
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    pipeline,
    TextStreamer
)
from typing import Dict, List, Optional, Union, Any
import logging
import os
import gc
from datetime import datetime
import psutil
import GPUtil


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model storage
models = {}
tokenizers = {}
pipelines = {}


class ModelConfig:
    """Configuration for each model"""
    def __init__(self):
        self.binary_classifier = {
            "path": "./llama3-8b-binary-classifier",  # Path to your fine-tuned binary model
            "type": "classification",
            "num_labels": 2,
            "max_length": 512
        }
        
        self.multi_classifier = {
            "path": "./llama3-8b-multiclass-classifier",  # Path to your fine-tuned multiclass model
            "type": "classification", 
            "num_labels": None,  # Will be loaded from training_info.json
            "max_length": 512
        }
        
        self.llama_70b = {
            "path": "meta-llama/Meta-Llama-3-70B-Instruct",  # Hugging Face model
            "type": "generation",
            "max_length": 4096
        }


# Request/Response models
class ClassificationRequest(BaseModel):
    text: str = Field(..., description="Text to classify")
    return_probabilities: bool = Field(False, description="Return class probabilities")


class ClassificationResponse(BaseModel):
    predicted_class: Union[str, int]
    confidence: float
    probabilities: Optional[Dict[str, float]] = None
    processing_time: float


class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt for text generation")
    max_new_tokens: int = Field(512, description="Maximum number of tokens to generate")
    temperature: float = Field(0.7, description="Sampling temperature")
    top_p: float = Field(0.9, description="Top-p sampling parameter")
    top_k: int = Field(50, description="Top-k sampling parameter")
    do_sample: bool = Field(True, description="Whether to use sampling")
    stream: bool = Field(False, description="Whether to stream the response")


class GenerationResponse(BaseModel):
    generated_text: str
    prompt_tokens: int
    generated_tokens: int
    processing_time: float


class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]
    gpu_memory: Dict[str, Any]
    system_memory: Dict[str, Any]
    timestamp: str


# Model loading functions
async def load_classification_model(model_name: str, model_path: str, num_labels: int = None):
    """Load a classification model"""
    try:
        logger.info(f"Loading classification model: {model_name}")
        
        # Load training info if available
        training_info_path = os.path.join(model_path, "training_info.json")
        if os.path.exists(training_info_path):
            with open(training_info_path, 'r') as f:
                training_info = json.load(f)
                if num_labels is None:
                    num_labels = training_info.get("num_labels", 2)
        
        # Load label mapping if available
        label_mapping = None
        label_mapping_path = os.path.join(model_path, "label_mapping.json")
        if not os.path.exists(label_mapping_path):
            # Try alternative naming
            task_type = "binary" if num_labels == 2 else "multiclass"
            label_mapping_path = f"label_mapping_{task_type}.json"
        
        if os.path.exists(label_mapping_path):
            with open(label_mapping_path, 'r') as f:
                label_mapping = json.load(f)
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        models[model_name] = {
            "model": model,
            "tokenizer": tokenizer,
            "label_mapping": label_mapping,
            "num_labels": num_labels,
            "reverse_mapping": {v: k for k, v in label_mapping.items()} if label_mapping else None
        }
        
        logger.info(f"Successfully loaded {model_name}")
        return True
        
    except Exception as e:
        logger.error(f"Error loading {model_name}: {str(e)}")
        return False


async def load_generation_model(model_name: str, model_path: str):
    """Load a text generation model"""
    try:
        logger.info(f"Loading generation model: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            use_cache=True
        )
        
        # Create text generation pipeline
        text_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        models[model_name] = {
            "model": model,
            "tokenizer": tokenizer,
            "pipeline": text_pipeline
        }
        
        logger.info(f"Successfully loaded {model_name}")
        return True
        
    except Exception as e:
        logger.error(f"Error loading {model_name}: {str(e)}")
        return False


async def load_all_models():
    """Load all models on startup"""
    config = ModelConfig()
    
    # Load binary classifier
    await load_classification_model(
        "binary_classification",
        config.binary_classifier["path"],
        config.binary_classifier["num_labels"]
    )
    
    # Load multiclass classifier
    await load_classification_model(
        "multi_classification", 
        config.multi_classifier["path"],
        config.multi_classifier["num_labels"]
    )
    
    # Load Llama 70B for generation
    await load_generation_model(
        "llama_70b_generation",
        config.llama_70b["path"]
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting model server...")
    await load_all_models()
    logger.info("All models loaded successfully!")
    yield
    # Shutdown
    logger.info("Shutting down model server...")
    # Clear GPU memory
    for model_name in models:
        if "model" in models[model_name]:
            del models[model_name]["model"]
    torch.cuda.empty_cache()
    gc.collect()


# Initialize FastAPI app
app = FastAPI(
    title="Llama Model Server",
    description="FastAPI server hosting Llama-3 models for classification and text generation",
    version="1.0.0",
    lifespan=lifespan
)


# Helper functions
def get_system_stats():
    """Get system resource usage"""
    # GPU stats
    gpu_stats = {}
    try:
        gpus = GPUtil.getGPUs()
        for i, gpu in enumerate(gpus):
            gpu_stats[f"GPU_{i}"] = {
                "name": gpu.name,
                "memory_used": f"{gpu.memoryUsed}MB",
                "memory_total": f"{gpu.memoryTotal}MB",
                "memory_percent": f"{gpu.memoryUtil * 100:.1f}%",
                "temperature": f"{gpu.temperature}°C"
            }
    except:
        gpu_stats["error"] = "Could not fetch GPU stats"
    
    # System memory
    memory = psutil.virtual_memory()
    memory_stats = {
        "total": f"{memory.total / (1024**3):.1f}GB",
        "used": f"{memory.used / (1024**3):.1f}GB",
        "percent": f"{memory.percent:.1f}%"
    }
    
    return gpu_stats, memory_stats


def classify_text(model_name: str, text: str, return_probabilities: bool = False):
    """Classify text using specified model"""
    if model_name not in models:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    model_data = models[model_name]
    model = model_data["model"]
    tokenizer = model_data["tokenizer"]
    
    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    
    # Move to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class = torch.argmax(logits, dim=-1).item()
        confidence = probabilities[0][predicted_class].item()
    
    # Convert to label if mapping exists
    if model_data["reverse_mapping"]:
        predicted_label = model_data["reverse_mapping"][predicted_class]
    else:
        predicted_label = predicted_class
    
    result = {
        "predicted_class": predicted_label,
        "confidence": confidence
    }
    
    if return_probabilities:
        prob_dict = {}
        for i, prob in enumerate(probabilities[0]):
            label = model_data["reverse_mapping"][i] if model_data["reverse_mapping"] else i
            prob_dict[str(label)] = prob.item()
        result["probabilities"] = prob_dict
    
    return result


# API Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    gpu_stats, memory_stats = get_system_stats()
    
    return HealthResponse(
        status="healthy",
        models_loaded=list(models.keys()),
        gpu_memory=gpu_stats,
        system_memory=memory_stats,
        timestamp=datetime.now().isoformat()
    )


@app.post("/classify/binary", response_model=ClassificationResponse)
async def binary_classification(request: ClassificationRequest):
    """Binary classification endpoint"""
    start_time = datetime.now()
    
    try:
        result = classify_text("binary_classification", request.text, request.return_probabilities)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ClassificationResponse(
            predicted_class=result["predicted_class"],
            confidence=result["confidence"],
            probabilities=result.get("probabilities"),
            processing_time=processing_time
        )
    
    except Exception as e:
        logger.error(f"Binary classification error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify/multi", response_model=ClassificationResponse)
async def multi_classification(request: ClassificationRequest):
    """Multi-class classification endpoint"""
    start_time = datetime.now()
    
    try:
        result = classify_text("multi_classification", request.text, request.return_probabilities)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ClassificationResponse(
            predicted_class=result["predicted_class"],
            confidence=result["confidence"],
            probabilities=result.get("probabilities"),
            processing_time=processing_time
        )
    
    except Exception as e:
        logger.error(f"Multi-class classification error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate", response_model=GenerationResponse)
async def text_generation(request: GenerationRequest):
    """Text generation endpoint using Llama 70B"""
    start_time = datetime.now()
    
    if "llama_70b_generation" not in models:
        raise HTTPException(status_code=404, detail="Generation model not loaded")
    
    try:
        model_data = models["llama_70b_generation"]
        pipeline_obj = model_data["pipeline"]
        tokenizer = model_data["tokenizer"]
        
        # Count input tokens
        input_tokens = len(tokenizer.encode(request.prompt))
        
        # Generate text
        result = pipeline_obj(
            request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            do_sample=request.do_sample,
            pad_token_id=tokenizer.eos_token_id,
            return_full_text=False
        )
        
        generated_text = result[0]["generated_text"]
        generated_tokens = len(tokenizer.encode(generated_text))
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return GenerationResponse(
            generated_text=generated_text,
            prompt_tokens=input_tokens,
            generated_tokens=generated_tokens,
            processing_time=processing_time
        )
    
    except Exception as e:
        logger.error(f"Text generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def list_models():
    """List all loaded models"""
    model_info = {}
    for model_name, model_data in models.items():
        info = {"loaded": True}
        if "num_labels" in model_data:
            info["num_labels"] = model_data["num_labels"]
        if "label_mapping" in model_data and model_data["label_mapping"]:
            info["labels"] = list(model_data["label_mapping"].keys())
        model_info[model_name] = info
    
    return {"models": model_info}


@app.post("/reload/{model_name}")
async def reload_model(model_name: str, background_tasks: BackgroundTasks):
    """Reload a specific model"""
    if model_name not in ["binary_classification", "multi_classification", "llama_70b_generation"]:
        raise HTTPException(status_code=400, detail="Invalid model name")
    
    def reload_task():
        config = ModelConfig()
        if model_name == "binary_classification":
            asyncio.run(load_classification_model(
                model_name, config.binary_classifier["path"], config.binary_classifier["num_labels"]
            ))
        elif model_name == "multi_classification":
            asyncio.run(load_classification_model(
                model_name, config.multi_classifier["path"], config.multi_classifier["num_labels"]
            ))
        elif model_name == "llama_70b_generation":
            asyncio.run(load_generation_model(model_name, config.llama_70b["path"]))
    
    background_tasks.add_task(reload_task)
    return {"message": f"Reloading {model_name} in background"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,  # Important: Use only 1 worker to avoid model loading issues
        log_level="info"
    )


# Example client usage:
"""
import requests
import json

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Binary classification
binary_request = {
    "text": "I love this product! It's amazing!",
    "return_probabilities": True
}
response = requests.post("http://localhost:8000/classify/binary", json=binary_request)
print("Binary classification:", response.json())

# Multi-class classification
multi_request = {
    "text": "The latest iPhone has incredible camera quality",
    "return_probabilities": True
}
response = requests.post("http://localhost:8000/classify/multi", json=multi_request)
print("Multi classification:", response.json())

# Text generation
generation_request = {
    "prompt": "Explain quantum computing in simple terms:",
    "max_new_tokens": 200,
    "temperature": 0.7
}
response = requests.post("http://localhost:8000/generate", json=generation_request)
print("Generated text:", response.json())

# List models
response = requests.get("http://localhost:8000/models")
print("Available models:", response.json())
"""