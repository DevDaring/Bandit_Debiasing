"""
Interactive inference CLI for testing trained system.
"""

import argparse
import logging
from pathlib import Path

from config.model_config import ModelConfig
from config.bandit_config import BanditConfig
from src.pipeline.inference_pipeline import MABDebiasInferencePipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def interactive_mode(pipeline: MABDebiasInferencePipeline, show_details: bool = True):
    """Run interactive inference loop."""
    print("\n" + "="*60)
    print("MAB DEBIASING - INTERACTIVE MODE")
    print("="*60)
    print("Type your input text and press Enter.")
    print("Type 'quit' or 'exit' to stop.")
    print("="*60 + "\n")

    while True:
        try:
            # Get user input
            user_input = input("Input: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Exiting...")
                break

            if not user_input:
                continue

            # Process input
            result = pipeline.process(user_input, return_details=show_details)

            # Display result
            if show_details:
                print("\n" + "-"*60)
                print(f"Response: {result['response']}")
                print("-"*60)
                print(f"Selected Arm: {result['selected_arm']}")
                print(f"Confidence: {result['arm_confidence']:.3f}")
                print(f"Reward: {result['reward']:.3f}")
                print(f"Bias Score: {result['bias_score']:.3f} (lower is better)")
                print(f"Quality Score: {result['quality_score']:.3f}")
                print(f"Generation Time: {result['generation_time']:.2f}s")
                print("-"*60 + "\n")
            else:
                print(f"\nResponse: {result}\n")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Exiting...")
            break
        except Exception as e:
            logger.error(f"Error processing input: {e}", exc_info=True)
            print(f"Error: {e}\n")


def batch_mode(pipeline: MABDebiasInferencePipeline, input_file: str, output_file: str, show_details: bool = True):
    """Run inference on batch of inputs from file."""
    print("\n" + "="*60)
    print("MAB DEBIASING - BATCH MODE")
    print("="*60)
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print("="*60 + "\n")

    # Read inputs
    with open(input_file, 'r', encoding='utf-8') as f:
        inputs = [line.strip() for line in f if line.strip()]

    logger.info(f"Processing {len(inputs)} inputs...")

    # Process batch
    results = pipeline.process_batch(inputs, return_details=show_details)

    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, (input_text, result) in enumerate(zip(inputs, results)):
            f.write(f"Input {i+1}: {input_text}\n")

            if show_details:
                f.write(f"Response: {result['response']}\n")
                f.write(f"Selected Arm: {result['selected_arm']}\n")
                f.write(f"Confidence: {result['arm_confidence']:.3f}\n")
                f.write(f"Reward: {result['reward']:.3f}\n")
                f.write(f"Bias Score: {result['bias_score']:.3f}\n")
                f.write(f"Quality Score: {result['quality_score']:.3f}\n")
            else:
                f.write(f"Response: {result}\n")

            f.write("\n" + "-"*60 + "\n\n")

    logger.info(f"Results saved to {output_file}")
    print(f"\nBatch processing complete! Results saved to {output_file}")


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Run inference with MAB Debiasing System')

    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to trained bandit checkpoint (optional)')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-7B-Instruct',
                        help='Model name or path')
    parser.add_argument('--bandit_type', type=str, default='linucb',
                        choices=['linucb', 'thompson', 'neural'],
                        help='Bandit algorithm type')

    parser.add_argument('--mode', type=str, default='interactive',
                        choices=['interactive', 'batch'],
                        help='Inference mode')
    parser.add_argument('--input_file', type=str, default=None,
                        help='Input file for batch mode (one input per line)')
    parser.add_argument('--output_file', type=str, default='inference_results.txt',
                        help='Output file for batch mode')

    parser.add_argument('--no_details', action='store_true',
                        help='Disable detailed output (only show response text)')
    parser.add_argument('--no_learning', action='store_true',
                        help='Disable bandit learning during inference')

    args = parser.parse_args()

    # Validate arguments
    if args.mode == 'batch' and not args.input_file:
        parser.error("--input_file is required for batch mode")

    logger.info("="*60)
    logger.info("MAB DEBIASING INFERENCE")
    logger.info("="*60)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Bandit: {args.bandit_type}")
    logger.info(f"Checkpoint: {args.checkpoint or 'None (untrained)'}")
    logger.info(f"Mode: {args.mode}")
    logger.info("="*60)

    # Create pipeline
    logger.info("Initializing pipeline...")
    pipeline = MABDebiasInferencePipeline(
        model_name=args.model_name,
        bandit_type=args.bandit_type,
        enable_learning=not args.no_learning
    )

    # Load components
    logger.info("Loading components...")
    pipeline.load_components()

    # Load checkpoint if provided
    if args.checkpoint:
        logger.info(f"Loading checkpoint from {args.checkpoint}...")
        pipeline.load_state(args.checkpoint)
    else:
        logger.warning("No checkpoint provided - using untrained bandit")

    # Run inference
    try:
        if args.mode == 'interactive':
            interactive_mode(pipeline, show_details=not args.no_details)
        else:  # batch mode
            batch_mode(pipeline, args.input_file, args.output_file, show_details=not args.no_details)

    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        raise

    finally:
        # Cleanup
        pipeline.unload()

    logger.info("Inference complete!")


if __name__ == "__main__":
    main()
