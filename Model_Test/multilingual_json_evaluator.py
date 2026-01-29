#!/usr/bin/env python3
"""
Multilingual JSON Response Evaluator for LLMs
Tests models on Hindi, English, and Bengali JSON generation accuracy.

Author: Koushik (PhD Research - Bias in Multilingual LLMs)
"""

import os
import json
import re
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch
from dotenv import load_dotenv
from huggingface_hub import login, snapshot_download
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig
)
from tqdm import tqdm
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


@dataclass
class ModelConfig:
    """Configuration for each model."""
    name: str
    hf_id: str
    family: str
    params: str
    trust_remote_code: bool = False
    use_flash_attention: bool = False
    max_memory_gb: float = 24.0


@dataclass
class EvaluationResult:
    """Results from a single evaluation."""
    model_name: str
    language: str
    prompt_id: str
    prompt_type: str
    is_valid_json: bool
    json_structure_score: float
    key_accuracy: float
    value_accuracy: float
    language_consistency: float
    response_time: float
    raw_response: str
    parsed_json: Optional[Dict] = None
    expected_keys: List[str] = field(default_factory=list)
    found_keys: List[str] = field(default_factory=list)
    error_message: str = ""


# Model configurations
MODELS = [
    ModelConfig(
        name="Qwen2.5-1.5B-Instruct",
        hf_id="Qwen/Qwen2.5-1.5B-Instruct",
        family="Qwen",
        params="1.5B",
        trust_remote_code=True
    ),
    ModelConfig(
        name="Llama-3.2-1B-Instruct",
        hf_id="meta-llama/Llama-3.2-1B-Instruct",
        family="Llama",
        params="1B"
    ),
    ModelConfig(
        name="Gemma-2-2B-IT",
        hf_id="google/gemma-2-2b-it",
        family="Gemma",
        params="2B"
    ),
    ModelConfig(
        name="mGPT-1.3B",
        hf_id="ai-forever/mGPT",
        family="mGPT",
        params="1.3B"
    ),
    ModelConfig(
        name="BLOOMZ-7B1",
        hf_id="bigscience/bloomz-7b1",
        family="BLOOM",
        params="7B"
    ),
]


# Multilingual test prompts
TEST_PROMPTS = {
    "english": [
        {
            "id": "en_001",
            "type": "entity_extraction",
            "prompt": """Extract the following information from the text and return as JSON:
Text: "Dr. Amit Kumar works at Apollo Hospital in Mumbai. He specializes in cardiology and has 15 years of experience."

Return JSON with keys: name, workplace, city, specialization, experience_years""",
            "expected_keys": ["name", "workplace", "city", "specialization", "experience_years"],
            "expected_values": {
                "name": "Dr. Amit Kumar",
                "workplace": "Apollo Hospital",
                "city": "Mumbai",
                "specialization": "cardiology",
                "experience_years": 15
            }
        },
        {
            "id": "en_002",
            "type": "classification",
            "prompt": """Classify the sentiment of these reviews and return as JSON array:
1. "The food was amazing and service was excellent!"
2. "Terrible experience, never coming back."
3. "It was okay, nothing special."

Return JSON with format: {"reviews": [{"id": 1, "sentiment": "positive/negative/neutral", "confidence": 0.0-1.0}]}""",
            "expected_keys": ["reviews"],
            "expected_structure": {"reviews": [{"id": int, "sentiment": str, "confidence": float}]}
        },
        {
            "id": "en_003",
            "type": "data_transformation",
            "prompt": """Convert this information to JSON format:
Student Name: Priya Sharma
Age: 22
Subjects: Mathematics, Physics, Computer Science
GPA: 3.8
University: IIT Delhi

Return as a properly formatted JSON object.""",
            "expected_keys": ["name", "age", "subjects", "gpa", "university"],
        },
        {
            "id": "en_004",
            "type": "nested_json",
            "prompt": """Create a JSON representation of this company structure:
Company: TechCorp India
CEO: Rajesh Gupta
Departments:
- Engineering (Head: Anita Desai, Employees: 50)
- Marketing (Head: Vikram Singh, Employees: 20)
- Finance (Head: Meera Patel, Employees: 15)

Return nested JSON with company details and department array.""",
            "expected_keys": ["company", "ceo", "departments"],
        },
        {
            "id": "en_005",
            "type": "qa_extraction",
            "prompt": """Answer the following questions about India and return as JSON:
1. What is the capital?
2. What is the currency?
3. What is the population (approximate)?
4. Name 3 major languages.

Return JSON: {"capital": "", "currency": "", "population": "", "languages": []}""",
            "expected_keys": ["capital", "currency", "population", "languages"],
        }
    ],
    "hindi": [
        {
            "id": "hi_001",
            "type": "entity_extraction",
            "prompt": """निम्नलिखित पाठ से जानकारी निकालें और JSON के रूप में लौटाएं:
पाठ: "डॉ. अमित कुमार मुंबई के अपोलो अस्पताल में काम करते हैं। वे हृदय रोग विशेषज्ञ हैं और उनके पास 15 वर्षों का अनुभव है।"

JSON में ये keys होने चाहिए: नाम, कार्यस्थल, शहर, विशेषज्ञता, अनुभव_वर्ष""",
            "expected_keys": ["नाम", "कार्यस्थल", "शहर", "विशेषज्ञता", "अनुभव_वर्ष"],
            "expected_values": {
                "नाम": "डॉ. अमित कुमार",
                "कार्यस्थल": "अपोलो अस्पताल",
                "शहर": "मुंबई",
                "विशेषज्ञता": "हृदय रोग",
                "अनुभव_वर्ष": 15
            }
        },
        {
            "id": "hi_002",
            "type": "classification",
            "prompt": """इन समीक्षाओं की भावना वर्गीकृत करें और JSON array के रूप में लौटाएं:
1. "खाना बहुत स्वादिष्ट था और सेवा उत्कृष्ट थी!"
2. "भयानक अनुभव, फिर कभी नहीं आऊंगा।"
3. "ठीक-ठाक था, कुछ खास नहीं।"

JSON format: {"समीक्षाएं": [{"क्रमांक": 1, "भावना": "सकारात्मक/नकारात्मक/तटस्थ", "विश्वास": 0.0-1.0}]}""",
            "expected_keys": ["समीक्षाएं"],
        },
        {
            "id": "hi_003",
            "type": "data_transformation",
            "prompt": """इस जानकारी को JSON format में बदलें:
छात्र का नाम: प्रिया शर्मा
आयु: 22 वर्ष
विषय: गणित, भौतिकी, कंप्यूटर विज्ञान
GPA: 3.8
विश्वविद्यालय: आईआईटी दिल्ली

हिंदी में keys के साथ JSON object लौटाएं।""",
            "expected_keys": ["नाम", "आयु", "विषय", "gpa", "विश्वविद्यालय"],
        },
        {
            "id": "hi_004",
            "type": "nested_json",
            "prompt": """इस कंपनी संरचना का JSON प्रतिनिधित्व बनाएं:
कंपनी: टेककॉर्प इंडिया
सीईओ: राजेश गुप्ता
विभाग:
- इंजीनियरिंग (प्रमुख: अनीता देसाई, कर्मचारी: 50)
- मार्केटिंग (प्रमुख: विक्रम सिंह, कर्मचारी: 20)
- वित्त (प्रमुख: मीरा पटेल, कर्मचारी: 15)

कंपनी विवरण और विभाग array के साथ nested JSON लौटाएं।""",
            "expected_keys": ["कंपनी", "सीईओ", "विभाग"],
        },
        {
            "id": "hi_005",
            "type": "qa_extraction",
            "prompt": """भारत के बारे में निम्नलिखित प्रश्नों के उत्तर दें और JSON के रूप में लौटाएं:
1. राजधानी क्या है?
2. मुद्रा क्या है?
3. जनसंख्या कितनी है (अनुमानित)?
4. 3 प्रमुख भाषाओं के नाम बताएं।

JSON: {"राजधानी": "", "मुद्रा": "", "जनसंख्या": "", "भाषाएं": []}""",
            "expected_keys": ["राजधानी", "मुद्रा", "जनसंख्या", "भाषाएं"],
        }
    ],
    "bengali": [
        {
            "id": "bn_001",
            "type": "entity_extraction",
            "prompt": """নিম্নলিখিত পাঠ থেকে তথ্য বের করুন এবং JSON হিসাবে ফেরত দিন:
পাঠ: "ডাঃ অমিত কুমার মুম্বাইয়ের অ্যাপোলো হাসপাতালে কাজ করেন। তিনি হৃদরোগ বিশেষজ্ঞ এবং তাঁর ১৫ বছরের অভিজ্ঞতা রয়েছে।"

JSON-এ এই keys থাকা উচিত: নাম, কর্মস্থল, শহর, বিশেষজ্ঞতা, অভিজ্ঞতা_বছর""",
            "expected_keys": ["নাম", "কর্মস্থল", "শহর", "বিশেষজ্ঞতা", "অভিজ্ঞতা_বছর"],
            "expected_values": {
                "নাম": "ডাঃ অমিত কুমার",
                "কর্মস্থল": "অ্যাপোলো হাসপাতাল",
                "শহর": "মুম্বাই",
                "বিশেষজ্ঞতা": "হৃদরোগ",
                "অভিজ্ঞতা_বছর": 15
            }
        },
        {
            "id": "bn_002",
            "type": "classification",
            "prompt": """এই রিভিউগুলির sentiment শ্রেণীবদ্ধ করুন এবং JSON array হিসাবে ফেরত দিন:
1. "খাবার অসাধারণ ছিল এবং পরিষেবা চমৎকার ছিল!"
2. "ভয়ানক অভিজ্ঞতা, আর কখনো আসব না।"
3. "ঠিকঠাক ছিল, বিশেষ কিছু না।"

JSON format: {"রিভিউ": [{"ক্রমিক": 1, "sentiment": "ইতিবাচক/নেতিবাচক/নিরপেক্ষ", "আত্মবিশ্বাস": 0.0-1.0}]}""",
            "expected_keys": ["রিভিউ"],
        },
        {
            "id": "bn_003",
            "type": "data_transformation",
            "prompt": """এই তথ্যকে JSON format-এ রূপান্তর করুন:
ছাত্রের নাম: প্রিয়া শর্মা
বয়স: ২২ বছর
বিষয়: গণিত, পদার্থবিদ্যা, কম্পিউটার বিজ্ঞান
GPA: ৩.৮
বিশ্ববিদ্যালয়: আইআইটি দিল্লি

বাংলায় keys সহ JSON object ফেরত দিন।""",
            "expected_keys": ["নাম", "বয়স", "বিষয়", "gpa", "বিশ্ববিদ্যালয়"],
        },
        {
            "id": "bn_004",
            "type": "nested_json",
            "prompt": """এই কোম্পানি কাঠামোর JSON উপস্থাপনা তৈরি করুন:
কোম্পানি: টেককর্প ইন্ডিয়া
সিইও: রাজেশ গুপ্তা
বিভাগসমূহ:
- ইঞ্জিনিয়ারিং (প্রধান: অনিতা দেশাই, কর্মচারী: ৫০)
- মার্কেটিং (প্রধান: বিক্রম সিং, কর্মচারী: ২০)
- অর্থ (প্রধান: মীরা প্যাটেল, কর্মচারী: ১৫)

কোম্পানির বিবরণ এবং বিভাগ array সহ nested JSON ফেরত দিন।""",
            "expected_keys": ["কোম্পানি", "সিইও", "বিভাগসমূহ"],
        },
        {
            "id": "bn_005",
            "type": "qa_extraction",
            "prompt": """ভারত সম্পর্কে নিম্নলিখিত প্রশ্নের উত্তর দিন এবং JSON হিসাবে ফেরত দিন:
1. রাজধানী কী?
2. মুদ্রা কী?
3. জনসংখ্যা কত (আনুমানিক)?
4. ৩টি প্রধান ভাষার নাম বলুন।

JSON: {"রাজধানী": "", "মুদ্রা": "", "জনসংখ্যা": "", "ভাষাসমূহ": []}""",
            "expected_keys": ["রাজধানী", "মুদ্রা", "জনসংখ্যা", "ভাষাসমূহ"],
        }
    ]
}


class JSONEvaluator:
    """Evaluates JSON response quality."""
    
    @staticmethod
    def extract_json_from_response(response: str) -> Tuple[Optional[Dict], str]:
        """Extract JSON from model response."""
        # Try direct parsing
        try:
            return json.loads(response.strip()), ""
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON in code blocks
        patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
            r'\{[\s\S]*\}',
            r'\[[\s\S]*\]'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.MULTILINE)
            for match in matches:
                try:
                    # Clean the match
                    cleaned = match.strip()
                    if cleaned.startswith('{') or cleaned.startswith('['):
                        return json.loads(cleaned), ""
                except json.JSONDecodeError:
                    continue
        
        return None, "Could not extract valid JSON from response"
    
    @staticmethod
    def calculate_key_accuracy(expected_keys: List[str], found_keys: List[str]) -> float:
        """Calculate accuracy of keys in JSON."""
        if not expected_keys:
            return 1.0 if not found_keys else 0.5
        
        expected_set = set(expected_keys)
        found_set = set(found_keys)
        
        # Check for exact matches and normalized matches
        matches = len(expected_set & found_set)
        
        # Also check lowercase normalized
        expected_lower = {k.lower() for k in expected_keys}
        found_lower = {k.lower() for k in found_keys}
        matches_lower = len(expected_lower & found_lower)
        
        best_matches = max(matches, matches_lower)
        return best_matches / len(expected_keys)
    
    @staticmethod
    def calculate_structure_score(json_obj: Dict, expected_keys: List[str]) -> float:
        """Calculate JSON structure quality score."""
        if json_obj is None:
            return 0.0
        
        score = 0.0
        
        # Valid JSON: 0.4 points
        score += 0.4
        
        # Has expected keys: up to 0.3 points
        if expected_keys:
            found_keys = list(json_obj.keys()) if isinstance(json_obj, dict) else []
            key_ratio = len(set(found_keys) & set(expected_keys)) / len(expected_keys)
            score += 0.3 * key_ratio
        else:
            score += 0.3
        
        # Proper nesting/structure: up to 0.3 points
        if isinstance(json_obj, dict):
            # Check for non-empty values
            non_empty = sum(1 for v in json_obj.values() if v is not None and v != "")
            if json_obj:
                score += 0.3 * (non_empty / len(json_obj))
        elif isinstance(json_obj, list) and json_obj:
            score += 0.3
        
        return min(score, 1.0)
    
    @staticmethod
    def detect_language(text: str) -> str:
        """Detect predominant language in text."""
        # Simple heuristic based on character ranges
        hindi_chars = len(re.findall(r'[\u0900-\u097F]', text))
        bengali_chars = len(re.findall(r'[\u0980-\u09FF]', text))
        ascii_chars = len(re.findall(r'[a-zA-Z]', text))
        
        total = hindi_chars + bengali_chars + ascii_chars
        if total == 0:
            return "unknown"
        
        if hindi_chars / total > 0.3:
            return "hindi"
        elif bengali_chars / total > 0.3:
            return "bengali"
        else:
            return "english"
    
    @staticmethod
    def calculate_language_consistency(
        response: str, 
        expected_language: str,
        json_obj: Optional[Dict]
    ) -> float:
        """Calculate how consistent the response language is."""
        if json_obj is None:
            return 0.0
        
        # Extract text content from JSON
        def extract_text(obj):
            texts = []
            if isinstance(obj, dict):
                for k, v in obj.items():
                    texts.append(str(k))
                    texts.extend(extract_text(v))
            elif isinstance(obj, list):
                for item in obj:
                    texts.extend(extract_text(item))
            elif isinstance(obj, str):
                texts.append(obj)
            return texts
        
        all_text = " ".join(extract_text(json_obj))
        detected = JSONEvaluator.detect_language(all_text)
        
        if expected_language == detected:
            return 1.0
        elif detected == "unknown":
            return 0.5
        else:
            return 0.3


class MultilingualJSONTester:
    """Main tester class for multilingual JSON evaluation."""
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.evaluator = JSONEvaluator()
        self.results: List[EvaluationResult] = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Login to HuggingFace
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            login(token=hf_token)
            logger.info("Successfully logged in to HuggingFace")
        else:
            logger.warning("HF_TOKEN not found in .env file")
    
    def load_model(self, config: ModelConfig) -> Tuple[Any, Any]:
        """Load model and tokenizer."""
        logger.info(f"Loading model: {config.name}")
        
        # Configure quantization for larger models
        quantization_config = None
        if config.params == "7B":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.hf_id,
            trust_remote_code=config.trust_remote_code,
            padding_side="left"
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model_kwargs = {
            "trust_remote_code": config.trust_remote_code,
            "torch_dtype": torch.float16,
            "device_map": "auto",
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        
        model = AutoModelForCausalLM.from_pretrained(
            config.hf_id,
            **model_kwargs
        )
        
        logger.info(f"Model {config.name} loaded successfully")
        return model, tokenizer
    
    def generate_response(
        self, 
        model: Any, 
        tokenizer: Any, 
        prompt: str,
        max_new_tokens: int = 512
    ) -> Tuple[str, float]:
        """Generate response from model."""
        # Format prompt based on model type
        messages = [{"role": "user", "content": prompt}]
        
        try:
            # Try chat template first
            formatted_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except Exception:
            # Fallback for models without chat template
            formatted_prompt = f"User: {prompt}\n\nAssistant:"
        
        inputs = tokenizer(
            formatted_prompt, 
            return_tensors="pt", 
            truncation=True,
            max_length=2048
        ).to(model.device)
        
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        response_time = time.time() - start_time
        
        # Decode response
        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        return response.strip(), response_time
    
    def evaluate_response(
        self,
        model_name: str,
        language: str,
        prompt_data: Dict,
        response: str,
        response_time: float
    ) -> EvaluationResult:
        """Evaluate a single response."""
        # Extract JSON
        parsed_json, error_msg = self.evaluator.extract_json_from_response(response)
        
        is_valid = parsed_json is not None
        expected_keys = prompt_data.get("expected_keys", [])
        found_keys = list(parsed_json.keys()) if isinstance(parsed_json, dict) else []
        
        # Calculate metrics
        key_accuracy = self.evaluator.calculate_key_accuracy(expected_keys, found_keys) if is_valid else 0.0
        structure_score = self.evaluator.calculate_structure_score(parsed_json, expected_keys)
        lang_consistency = self.evaluator.calculate_language_consistency(response, language, parsed_json)
        
        # Value accuracy (if expected values provided)
        value_accuracy = 0.0
        if is_valid and "expected_values" in prompt_data:
            expected_vals = prompt_data["expected_values"]
            matches = 0
            for key, expected_val in expected_vals.items():
                if key in parsed_json:
                    actual_val = parsed_json[key]
                    if isinstance(expected_val, (int, float)):
                        matches += 1 if actual_val == expected_val else 0
                    else:
                        # String comparison (case-insensitive, normalized)
                        if str(actual_val).lower().strip() == str(expected_val).lower().strip():
                            matches += 1
                        elif str(expected_val).lower() in str(actual_val).lower():
                            matches += 0.5
            value_accuracy = matches / len(expected_vals) if expected_vals else 0.0
        
        return EvaluationResult(
            model_name=model_name,
            language=language,
            prompt_id=prompt_data["id"],
            prompt_type=prompt_data["type"],
            is_valid_json=is_valid,
            json_structure_score=structure_score,
            key_accuracy=key_accuracy,
            value_accuracy=value_accuracy,
            language_consistency=lang_consistency,
            response_time=response_time,
            raw_response=response,
            parsed_json=parsed_json,
            expected_keys=expected_keys,
            found_keys=found_keys,
            error_message=error_msg
        )
    
    def test_model(self, config: ModelConfig) -> List[EvaluationResult]:
        """Test a single model on all prompts."""
        results = []
        
        try:
            model, tokenizer = self.load_model(config)
        except Exception as e:
            logger.error(f"Failed to load model {config.name}: {e}")
            return results
        
        for language, prompts in TEST_PROMPTS.items():
            logger.info(f"Testing {config.name} on {language}...")
            
            for prompt_data in tqdm(prompts, desc=f"{config.name} - {language}"):
                try:
                    response, response_time = self.generate_response(
                        model, tokenizer, prompt_data["prompt"]
                    )
                    
                    result = self.evaluate_response(
                        model_name=config.name,
                        language=language,
                        prompt_data=prompt_data,
                        response=response,
                        response_time=response_time
                    )
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error evaluating {prompt_data['id']}: {e}")
                    results.append(EvaluationResult(
                        model_name=config.name,
                        language=language,
                        prompt_id=prompt_data["id"],
                        prompt_type=prompt_data["type"],
                        is_valid_json=False,
                        json_structure_score=0.0,
                        key_accuracy=0.0,
                        value_accuracy=0.0,
                        language_consistency=0.0,
                        response_time=0.0,
                        raw_response="",
                        error_message=str(e)
                    ))
        
        # Clear GPU memory
        del model
        del tokenizer
        torch.cuda.empty_cache()
        
        return results
    
    def run_all_tests(self, models: List[ModelConfig] = None):
        """Run tests on all models."""
        if models is None:
            models = MODELS
        
        logger.info(f"Starting evaluation of {len(models)} models")
        
        for config in models:
            logger.info(f"\n{'='*50}")
            logger.info(f"Testing: {config.name}")
            logger.info(f"{'='*50}")
            
            model_results = self.test_model(config)
            self.results.extend(model_results)
            
            # Save intermediate results
            self.save_results()
        
        logger.info("\nAll tests completed!")
        return self.results
    
    def save_results(self):
        """Save results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert results to DataFrame
        results_data = []
        for r in self.results:
            results_data.append({
                "model": r.model_name,
                "language": r.language,
                "prompt_id": r.prompt_id,
                "prompt_type": r.prompt_type,
                "is_valid_json": r.is_valid_json,
                "structure_score": r.json_structure_score,
                "key_accuracy": r.key_accuracy,
                "value_accuracy": r.value_accuracy,
                "language_consistency": r.language_consistency,
                "response_time": r.response_time,
                "error": r.error_message
            })
        
        df = pd.DataFrame(results_data)
        
        # Save detailed results
        df.to_csv(self.output_dir / f"detailed_results_{timestamp}.csv", index=False)
        
        # Save summary statistics
        summary = df.groupby(["model", "language"]).agg({
            "is_valid_json": "mean",
            "structure_score": "mean",
            "key_accuracy": "mean",
            "value_accuracy": "mean",
            "language_consistency": "mean",
            "response_time": "mean"
        }).round(4)
        
        summary.to_csv(self.output_dir / f"summary_{timestamp}.csv")
        
        # Save raw responses
        raw_data = []
        for r in self.results:
            raw_data.append({
                "model": r.model_name,
                "language": r.language,
                "prompt_id": r.prompt_id,
                "raw_response": r.raw_response,
                "parsed_json": json.dumps(r.parsed_json, ensure_ascii=False) if r.parsed_json else None
            })
        
        with open(self.output_dir / f"raw_responses_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(raw_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Results saved to {self.output_dir}")
        
        return summary
    
    def generate_report(self) -> str:
        """Generate a markdown report of results."""
        if not self.results:
            return "No results to report."
        
        report = ["# Multilingual JSON Response Evaluation Report\n"]
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"**Total Tests:** {len(self.results)}\n\n")
        
        # Overall summary
        report.append("## Overall Summary\n")
        
        df = pd.DataFrame([{
            "model": r.model_name,
            "language": r.language,
            "is_valid_json": r.is_valid_json,
            "structure_score": r.json_structure_score,
            "key_accuracy": r.key_accuracy,
            "language_consistency": r.language_consistency
        } for r in self.results])
        
        # Model-wise summary
        report.append("### Model Performance\n")
        model_summary = df.groupby("model").agg({
            "is_valid_json": "mean",
            "structure_score": "mean",
            "key_accuracy": "mean",
            "language_consistency": "mean"
        }).round(4)
        
        report.append("| Model | Valid JSON % | Structure | Key Accuracy | Lang Consistency |")
        report.append("|-------|-------------|-----------|--------------|------------------|")
        for model, row in model_summary.iterrows():
            report.append(f"| {model} | {row['is_valid_json']*100:.1f}% | {row['structure_score']:.3f} | {row['key_accuracy']:.3f} | {row['language_consistency']:.3f} |")
        report.append("\n")
        
        # Language-wise summary
        report.append("### Language Performance\n")
        lang_summary = df.groupby("language").agg({
            "is_valid_json": "mean",
            "structure_score": "mean",
            "key_accuracy": "mean"
        }).round(4)
        
        report.append("| Language | Valid JSON % | Structure | Key Accuracy |")
        report.append("|----------|-------------|-----------|--------------|")
        for lang, row in lang_summary.iterrows():
            report.append(f"| {lang.capitalize()} | {row['is_valid_json']*100:.1f}% | {row['structure_score']:.3f} | {row['key_accuracy']:.3f} |")
        report.append("\n")
        
        # Cross-tabulation
        report.append("### Model × Language Matrix (Valid JSON %)\n")
        cross_tab = df.pivot_table(
            index="model",
            columns="language",
            values="is_valid_json",
            aggfunc="mean"
        ).round(3) * 100
        
        report.append("| Model | English | Hindi | Bengali |")
        report.append("|-------|---------|-------|---------|")
        for model in cross_tab.index:
            en = cross_tab.loc[model, "english"] if "english" in cross_tab.columns else 0
            hi = cross_tab.loc[model, "hindi"] if "hindi" in cross_tab.columns else 0
            bn = cross_tab.loc[model, "bengali"] if "bengali" in cross_tab.columns else 0
            report.append(f"| {model} | {en:.1f}% | {hi:.1f}% | {bn:.1f}% |")
        
        return "\n".join(report)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multilingual JSON Evaluation")
    parser.add_argument("--models", nargs="+", help="Specific models to test")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--skip-download", action="store_true", help="Skip model download check")
    args = parser.parse_args()
    
    tester = MultilingualJSONTester(output_dir=args.output)
    
    # Filter models if specified
    models_to_test = MODELS
    if args.models:
        models_to_test = [m for m in MODELS if m.name in args.models]
    
    # Run tests
    tester.run_all_tests(models_to_test)
    
    # Generate and save report
    report = tester.generate_report()
    report_path = Path(args.output) / "evaluation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"\n{report}")
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
