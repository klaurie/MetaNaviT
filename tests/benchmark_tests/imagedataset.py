"""
LLaVA Model Evaluation Script with DeepEval

This script:
1. Loads the DiagramQG dataset (or checks if it's already downloaded)
2. Allows users to select a topic/category
3. Randomly selects images from different sections
4. Evaluates LLaVA's responses using DeepEval
5. Generates visual benchmarks and analysis reports
"""

import os
import sys
import random
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import ollama
from datasets import load_dataset
from PIL import Image
from io import BytesIO
import argparse
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import torch

# DeepEval imports
from deepeval import evaluate
from deepeval.metrics import GEval, HallucinationMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase, ImageInput
from deepeval.models import DeepEvalBaseLLM
from deepeval.utils import plot_single_metric, plot_test_case_results

# Define paths
DATASET_CACHE_PATH = os.path.join(os.path.dirname(__file__), "cache")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
REPORT_PATH = os.path.join(RESULTS_DIR, "llava_evaluation_report.txt")
BENCHMARK_IMG_PATH = os.path.join(RESULTS_DIR, "benchmark_results.png")

# Ensure directories exist
os.makedirs(DATASET_CACHE_PATH, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

class MistralLLM(DeepEvalBaseLLM):
    """DeepEval compatible Mistral-7B LLM for evaluation"""
    
    def __init__(self):
        # Use Mistral 7B model for evaluation
        # You can replace this with a different model if needed
        if torch.cuda.is_available():
            print("Using GPU for evaluation model")
            self.device = "cuda"
        else:
            print("Using CPU for evaluation model (this may be slow)")
            self.device = "cpu"
            
        # Load model from Ollama for evaluation
        self.model_name = "mistral"

    def generate(self, prompt: str, **kwargs):
        """Generate text using Ollama's Mistral model"""
        try:
            response = ollama.generate(model=self.model_name, prompt=prompt)
            return response["response"]
        except Exception as e:
            print(f"Error generating with Mistral: {e}")
            return "Error generating evaluation response"

    async def a_generate(self, prompt: str, **kwargs):
        """Async generate implementation"""
        return self.generate(prompt, **kwargs)
    
    def get_model_name(self):
        return "Mistral-7B"


class LLaVaEvaluator:
    """Handles dataset loading, model inference and evaluation"""
    
    def __init__(self, dataset_name: str = "zhibei1204/DiagramQG"):
        self.dataset_name = dataset_name
        self.dataset = None
        self.categories = None
        self.eval_llm = MistralLLM()
        
    def load_dataset(self) -> None:
        """Load the dataset if not already loaded"""
        # Check if we have a cached version
        dataset_cache_file = os.path.join(DATASET_CACHE_PATH, f"{self.dataset_name.replace('/', '_')}_info.json")
        
        if os.path.exists(dataset_cache_file):
            print(f"Loading dataset info from cache: {dataset_cache_file}")
            with open(dataset_cache_file, 'r') as f:
                cache_info = json.load(f)
                self.categories = cache_info.get('categories', [])
                print(f"Found {len(self.categories)} categories in cached dataset info")
        
        # Download dataset if not already present
        print(f"Loading dataset: {self.dataset_name}")
        self.dataset = load_dataset(self.dataset_name)
        print(f"Dataset loaded: {self.dataset}")
        
        # Extract categories if not already cached
        if not self.categories:
            # Extract categories based on dataset structure
            # This is dataset-specific and might need adjustment
            if 'train' in self.dataset:
                # Example: Extract categories based on fields in the dataset
                if 'category' in self.dataset['train'].features:
                    self.categories = self.dataset['train']['category']
                    # Get unique categories
                    self.categories = list(set(self.categories))
                else:
                    # If no explicit category, try to identify document types or domains
                    self.categories = ["diagrams", "charts", "graphs", "tables", 
                                      "scientific_figures", "infographics", 
                                      "architectural", "engineering"]
            
            # Cache the categories
            os.makedirs(os.path.dirname(dataset_cache_file), exist_ok=True)
            with open(dataset_cache_file, 'w') as f:
                json.dump({'categories': self.categories}, f)
        
        print(f"Available categories: {self.categories}")
    
    def get_topic_options(self) -> List[str]:
        """Return available topics/categories for selection"""
        if not self.dataset:
            self.load_dataset()
        return self.categories or ["general"]
    
    def sample_images_by_topic(self, topic: str, samples_per_section: int = 2, num_sections: int = 8) -> List[Dict]:
        """
        Sample images from the dataset based on the selected topic
        
        Args:
            topic: The selected topic/category
            samples_per_section: Number of samples to select from each section
            num_sections: Number of sections to sample from
            
        Returns:
            List of dictionaries containing image data and metadata
        """
        if not self.dataset:
            self.load_dataset()
        
        # Select the split (use train if available, otherwise the first available split)
        split = 'train' if 'train' in self.dataset else list(self.dataset.keys())[0]
        
        # Filter by topic if needed and if the dataset has category information
        filtered_dataset = self.dataset[split]
        if topic != "all" and 'category' in filtered_dataset.features:
            filtered_indices = [i for i, cat in enumerate(filtered_dataset['category']) if topic.lower() in cat.lower()]
            if not filtered_indices:
                print(f"No images found for topic '{topic}'. Using all images.")
                filtered_indices = list(range(len(filtered_dataset)))
        else:
            filtered_indices = list(range(len(filtered_dataset)))
            
        # Determine how to divide dataset into sections
        section_size = max(1, len(filtered_indices) // num_sections)
        sections = []
        
        for i in range(0, min(num_sections, len(filtered_indices)), section_size):
            section_indices = filtered_indices[i:i+section_size]
            # Randomly sample from each section
            if len(section_indices) <= samples_per_section:
                sampled_indices = section_indices
            else:
                sampled_indices = random.sample(section_indices, samples_per_section)
            sections.extend(sampled_indices)
            
        # Get the actual samples
        samples = []
        for idx in sections:
            sample = filtered_dataset[idx]
            # Extract image and relevant metadata
            image = sample.get('image', sample.get('img', None))
            if image is not None:
                samples.append({
                    'id': idx,
                    'image': image,
                    'question': sample.get('question', ''),
                    'answer': sample.get('answer', ''),
                    'category': sample.get('category', topic)
                })
                
        print(f"Selected {len(samples)} images for evaluation from topic '{topic}'")
        return samples
    
    def run_llava_inference(self, image, prompt: str = "Describe this image in detail. Identify any text, diagrams, or notable elements.") -> str:
        """
        Run inference using LLaVA model through Ollama
        
        Args:
            image: PIL Image object
            prompt: Text prompt to send with the image
            
        Returns:
            LLaVA's response text
        """
        try:
            # Convert PIL image to bytes
            with BytesIO() as buffer:
                image.save(buffer, format='PNG')
                image_bytes = buffer.getvalue()
            
            # Run inference using Ollama
            full_response = ''
            for response in ollama.generate(
                model='llava:13b-v1.6',
                prompt=prompt,
                images=[image_bytes],
                stream=True
            ):
                full_response += response['response']
                
            return full_response
        except Exception as e:
            print(f"Error during LLaVA inference: {e}")
            return f"Error: {str(e)}"
    
    def create_test_cases(self, samples: List[Dict]) -> List[LLMTestCase]:
        """
        Create DeepEval test cases from image samples
        
        Args:
            samples: List of image samples with metadata
            
        Returns:
            List of DeepEval LLMTestCase objects
        """
        test_cases = []
        
        for sample in tqdm(samples, desc="Creating test cases"):
            # Process image and get LLaVa response
            prompt = "Describe this image in detail. Identify any text, diagrams, or notable elements."
            
            # Get LLaVa's response for this image
            image = sample['image']
            if isinstance(image, dict) and 'bytes' in image:
                # Handle dataset-specific image format
                img = Image.open(BytesIO(image['bytes']))
            elif hasattr(image, 'convert'):
                # Image is already a PIL Image
                img = image
            else:
                # Try to interpret as PIL Image
                try:
                    img = Image.open(BytesIO(image))
                except Exception as e:
                    print(f"Error loading image: {e}")
                    continue
            
            llava_response = self.run_llava_inference(img, prompt)
            
            # Create test case
            test_case = LLMTestCase(
                input=prompt,
                actual_output=llava_response,
                expected_output=sample.get('answer', ''),
                input_image=ImageInput(image=img),
                context=sample.get('question', '')
            )
            
            test_cases.append(test_case)
            
        return test_cases
    
    def evaluate_responses(self, test_cases: List[LLMTestCase]) -> Dict[str, Dict]:
        """
        Evaluate LLaVA responses using DeepEval metrics
        
        Args:
            test_cases: List of DeepEval test cases
            
        Returns:
            Dictionary of evaluation results
        """
        results = {}
        
        # Define metrics
        metrics = [
            GEval(
                name="Image Understanding",
                evaluation_steps=[
                    "Does the response accurately describe the key elements in the image?",
                    "Is the text in the image correctly identified and transcribed?",
                    "Are the relationships between different visual elements explained correctly?",
                    "Is the overall meaning or purpose of the diagram/chart understood?",
                ],
                model=self.eval_llm
            ),
            HallucinationMetric(threshold=0.7, model=self.eval_llm),
            ContextualRelevancyMetric(threshold=0.7, model=self.eval_llm)
        ]
        
        # Evaluate each test case
        for i, test_case in enumerate(tqdm(test_cases, desc="Evaluating responses")):
            case_results = {}
            
            # Apply each metric
            for metric in metrics:
                try:
                    metric.measure(test_case)
                    case_results[metric.__class__.__name__] = {
                        'score': metric.score,
                        'reason': metric.reason
                    }
                except Exception as e:
                    print(f"Error applying metric {metric.__class__.__name__}: {e}")
                    case_results[metric.__class__.__name__] = {
                        'score': 0.0,
                        'reason': f"Error: {str(e)}"
                    }
            
            results[f"case_{i}"] = case_results
            
        return results
    
    def generate_report(self, samples: List[Dict], test_cases: List[LLMTestCase], results: Dict) -> None:
        """
        Generate evaluation report and visualizations
        
        Args:
            samples: Original image samples with metadata
            test_cases: Evaluated test cases
            results: Evaluation results
        """
        # Create report text file
        with open(REPORT_PATH, 'w') as f:
            f.write("# LLaVA-13B Evaluation Report\n\n")
            
            # Overall statistics
            f.write("## Overall Performance\n\n")
            
            # Calculate average scores by metric
            metric_scores = {}
            for case_id, case_results in results.items():
                for metric_name, metric_data in case_results.items():
                    if metric_name not in metric_scores:
                        metric_scores[metric_name] = []
                    metric_scores[metric_name].append(metric_data['score'])
            
            # Write average scores
            f.write("### Average Scores\n\n")
            for metric_name, scores in metric_scores.items():
                avg_score = sum(scores) / len(scores) if scores else 0
                f.write(f"- {metric_name}: {avg_score:.3f}\n")
            
            f.write("\n## Detailed Results\n\n")
            
            # Individual case details
            for i, (test_case, sample) in enumerate(zip(test_cases, samples)):
                case_id = f"case_{i}"
                if case_id in results:
                    f.write(f"### Image {i+1}\n\n")
                    f.write(f"Category: {sample.get('category', 'Unknown')}\n\n")
                    f.write(f"Prompt: {test_case.input}\n\n")
                    f.write(f"LLaVA Response:\n{test_case.actual_output}\n\n")
                    
                    if test_case.expected_output:
                        f.write(f"Expected Answer:\n{test_case.expected_output}\n\n")
                    
                    f.write("Evaluation Results:\n\n")
                    for metric_name, metric_data in results[case_id].items():
                        f.write(f"- {metric_name}: {metric_data['score']:.3f}\n")
                        f.write(f"  Reason: {metric_data['reason']}\n\n")
                    
                    f.write("-" * 80 + "\n\n")
        
        print(f"Evaluation report saved to: {REPORT_PATH}")
        
        # Generate visualizations
        self.generate_visualizations(results)
    
    def generate_visualizations(self, results: Dict) -> None:
        """Generate benchmark visualizations from evaluation results"""
        # Extract scores by metric
        metrics = {}
        for case_results in results.values():
            for metric_name, metric_data in case_results.items():
                if metric_name not in metrics:
                    metrics[metric_name] = []
                metrics[metric_name].append(metric_data['score'])
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Plot boxplots for each metric
        plt.subplot(2, 1, 1)
        df = pd.DataFrame(metrics)
        sns.boxplot(data=df)
        plt.title("LLaVA-13B Performance by Metric")
        plt.ylim(0, 1)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot average scores
        plt.subplot(2, 1, 2)
        avg_scores = {metric: sum(scores)/len(scores) for metric, scores in metrics.items()}
        sns.barplot(x=list(avg_scores.keys()), y=list(avg_scores.values()))
        plt.title("Average Score by Metric")
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(BENCHMARK_IMG_PATH)
        print(f"Benchmark visualization saved to: {BENCHMARK_IMG_PATH}")


def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(description="Evaluate LLaVA-13B on image datasets using DeepEval")
    parser.add_argument("--topic", type=str, default="all", help="Topic/category to focus on")
    parser.add_argument("--samples", type=int, default=2, help="Number of samples per section")
    parser.add_argument("--sections", type=int, default=8, help="Number of sections to sample from")
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = LLaVaEvaluator()
    
    # Load dataset
    evaluator.load_dataset()
    
    # Get available topics and validate selected topic
    topics = evaluator.get_topic_options()
    if args.topic != "all" and args.topic not in topics:
        print(f"Topic '{args.topic}' not found. Available topics: {topics}")
        print(f"Defaulting to 'all'")
        topic = "all"
    else:
        topic = args.topic
    
    # Sample images based on topic
    samples = evaluator.sample_images_by_topic(
        topic, 
        samples_per_section=args.samples,
        num_sections=args.sections
    )
    
    if not samples:
        print("No samples found. Please try a different topic or dataset.")
        return
    
    # Create test cases
    test_cases = evaluator.create_test_cases(samples)
    
    # Evaluate responses
    results = evaluator.evaluate_responses(test_cases)
    
    # Generate report and visualizations
    evaluator.generate_report(samples, test_cases, results)
    
    print("Evaluation complete!")


if __name__ == "__main__":
    main() 