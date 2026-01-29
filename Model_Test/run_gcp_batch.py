#!/usr/bin/env python3
"""
GCP Batch Runner - Run evaluation with checkpointing and resume capability.
Designed for long-running experiments on cloud instances.

Usage:
    python run_gcp_batch.py                    # Run all models
    python run_gcp_batch.py --resume           # Resume from checkpoint
    python run_gcp_batch.py --models Qwen Llama  # Run specific models
"""

import os
import sys
import json
import time
import pickle
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch
from dotenv import load_dotenv
from huggingface_hub import login

# Import main evaluator
from multilingual_json_evaluator import (
    MODELS, TEST_PROMPTS, ModelConfig, EvaluationResult,
    MultilingualJSONTester, JSONEvaluator
)

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('gcp_batch.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BatchRunner:
    """Batch runner with checkpointing for GCP execution."""
    
    def __init__(
        self,
        output_dir: str = "results",
        checkpoint_dir: str = "checkpoints"
    ):
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_file = self.checkpoint_dir / "batch_checkpoint.pkl"
        self.results_file = self.checkpoint_dir / "partial_results.json"
        
        self.tester = MultilingualJSONTester(output_dir=output_dir)
        
        # Track state
        self.completed_models: List[str] = []
        self.all_results: List[dict] = []
        
    def save_checkpoint(self, current_model: str, model_results: List[EvaluationResult]):
        """Save checkpoint after each model."""
        checkpoint = {
            "completed_models": self.completed_models,
            "current_model": current_model,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(self.checkpoint_file, "wb") as f:
            pickle.dump(checkpoint, f)
        
        # Also save results as JSON
        results_data = []
        for r in self.all_results:
            if isinstance(r, EvaluationResult):
                results_data.append({
                    "model": r.model_name,
                    "language": r.language,
                    "prompt_id": r.prompt_id,
                    "is_valid_json": r.is_valid_json,
                    "structure_score": r.json_structure_score,
                    "key_accuracy": r.key_accuracy,
                    "response_time": r.response_time
                })
            else:
                results_data.append(r)
        
        with open(self.results_file, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Checkpoint saved: {len(self.completed_models)} models completed")
    
    def load_checkpoint(self) -> Optional[str]:
        """Load checkpoint if exists."""
        if not self.checkpoint_file.exists():
            return None
        
        try:
            with open(self.checkpoint_file, "rb") as f:
                checkpoint = pickle.load(f)
            
            self.completed_models = checkpoint.get("completed_models", [])
            
            # Load partial results
            if self.results_file.exists():
                with open(self.results_file, "r", encoding="utf-8") as f:
                    self.all_results = json.load(f)
            
            logger.info(f"Loaded checkpoint: {len(self.completed_models)} models already completed")
            return checkpoint.get("current_model")
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def get_system_info(self) -> dict:
        """Get system information for logging."""
        info = {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "timestamp": datetime.now().isoformat()
        }
        
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
            info["cuda_version"] = torch.version.cuda
        
        return info
    
    def run(
        self,
        models: List[str] = None,
        resume: bool = False
    ):
        """Run batch evaluation."""
        logger.info("=" * 60)
        logger.info("GCP Batch Runner - Multilingual JSON Evaluation")
        logger.info("=" * 60)
        
        # Log system info
        sys_info = self.get_system_info()
        logger.info(f"System Info: {json.dumps(sys_info, indent=2)}")
        
        # HF login
        token = os.getenv("HF_TOKEN")
        if token:
            login(token=token)
            logger.info("Logged in to HuggingFace")
        else:
            logger.warning("HF_TOKEN not found - gated models will fail")
        
        # Filter models
        models_to_run = MODELS
        if models:
            models_to_run = [m for m in MODELS if any(
                s.lower() in m.name.lower() for s in models
            )]
        
        logger.info(f"Models to evaluate: {[m.name for m in models_to_run]}")
        
        # Check for resume
        if resume:
            last_model = self.load_checkpoint()
            if last_model:
                # Skip completed models
                models_to_run = [
                    m for m in models_to_run 
                    if m.name not in self.completed_models
                ]
                logger.info(f"Resuming... {len(models_to_run)} models remaining")
        
        if not models_to_run:
            logger.info("No models to run!")
            return
        
        # Run evaluation
        start_time = time.time()
        
        for i, config in enumerate(models_to_run):
            logger.info(f"\n{'='*60}")
            logger.info(f"[{i+1}/{len(models_to_run)}] Testing: {config.name}")
            logger.info(f"{'='*60}")
            
            model_start = time.time()
            
            try:
                # Run tests for this model
                results = self.tester.test_model(config)
                
                # Store results
                for r in results:
                    self.all_results.append(r)
                    self.tester.results.append(r)
                
                self.completed_models.append(config.name)
                
                model_time = time.time() - model_start
                logger.info(f"✓ {config.name} completed in {model_time:.1f}s")
                
                # Calculate quick stats
                valid_count = sum(1 for r in results if r.is_valid_json)
                logger.info(f"  Valid JSON: {valid_count}/{len(results)} ({100*valid_count/len(results):.1f}%)")
                
            except Exception as e:
                logger.error(f"✗ {config.name} failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
            
            # Save checkpoint after each model
            self.save_checkpoint(config.name, results if 'results' in dir() else [])
            
            # Clear GPU memory
            torch.cuda.empty_cache()
            
            # Small delay between models
            time.sleep(5)
        
        total_time = time.time() - start_time
        logger.info(f"\n{'='*60}")
        logger.info(f"Batch completed in {total_time/60:.1f} minutes")
        logger.info(f"{'='*60}")
        
        # Generate final report
        self.generate_final_report()
    
    def generate_final_report(self):
        """Generate final evaluation report."""
        logger.info("Generating final report...")
        
        # Save all results
        self.tester.save_results()
        
        # Generate markdown report
        report = self.tester.generate_report()
        
        report_path = self.output_dir / "final_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        
        logger.info(f"Report saved to: {report_path}")
        
        # Print summary to console
        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE")
        print("=" * 60)
        print(report)


def main():
    parser = argparse.ArgumentParser(
        description="GCP Batch Runner for Multilingual JSON Evaluation"
    )
    parser.add_argument(
        "--models", "-m",
        nargs="+",
        help="Model names to run (partial match supported)"
    )
    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        help="Resume from last checkpoint"
    )
    parser.add_argument(
        "--output", "-o",
        default="results",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    runner = BatchRunner(output_dir=args.output)
    runner.run(models=args.models, resume=args.resume)


if __name__ == "__main__":
    main()
