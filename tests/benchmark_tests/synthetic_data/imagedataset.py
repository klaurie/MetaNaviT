"""
LLaVA Model Evaluation Script with DeepEval

This script:
1. Uses the local DiagramQG dataset in the synthetic_data directory
2. Allows users to select a topic/category 
3. Randomly selects images from different sections
4. Uses questions from questions.json to test LLaVA's understanding
5. Evaluates LLaVA's responses using DeepEval
6. Generates visual benchmarks and analysis reports
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
from PIL import Image
from io import BytesIO
import argparse
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import torch
import glob

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

# DeepEval imports
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, HallucinationMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import DeepEvalBaseLLM

# Import the evaluation LLM from common directory
from benchmark_tests.common.eval_llm import EvalLLM_4Bit

# Define paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(CURRENT_DIR, "DiagramQG")
RESULTS_DIR = os.path.join(CURRENT_DIR, "results")
REPORT_PATH = os.path.join(RESULTS_DIR, "llava_evaluation_report.txt")
BENCHMARK_IMG_PATH = os.path.join(RESULTS_DIR, "benchmark_results.png")

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)

class LLaVaEvaluator:
    """Handles dataset loading, model inference and evaluation"""
    
    def __init__(self):
        self.dataset_path = DATASET_PATH
        self.categories = None
        try:
            # Try to use the established EvalLLM from common directory
            self.eval_llm = EvalLLM_4Bit()
            print("Using EvalLLM_4Bit for evaluation")
        except Exception as e:
            print(f"Error loading EvalLLM_4Bit: {e}")
            print("Metrics will run without evaluation model")
            self.eval_llm = None
        
    def load_dataset(self) -> None:
        """Load the local DiagramQG dataset"""
        # Check if dataset exists
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")
            
        # Get all topic directories (categories)
        self.categories = [d for d in os.listdir(self.dataset_path) 
                          if os.path.isdir(os.path.join(self.dataset_path, d))]
        
        print(f"Found {len(self.categories)} categories in the dataset")
        print(f"Available categories: {self.categories}")
    
    def get_topic_options(self) -> List[str]:
        """Return available topics/categories for selection"""
        if not self.categories:
            self.load_dataset()
        return self.categories
    
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
        # If "all" is selected, choose randomly from available categories
        if topic == "all":
            if not self.categories:
                self.load_dataset()
            topic = random.choice(self.categories)
            print(f"Randomly selected topic: {topic}")
        
        # Get the path for the topic
        topic_path = os.path.join(self.dataset_path, topic)
        if not os.path.exists(topic_path):
            print(f"Topic '{topic}' not found in dataset at {topic_path}")
            return []
            
        # Find all images in the topic directory
        image_files = glob.glob(os.path.join(topic_path, "*.png"))
        if not image_files:
            print(f"No images found for topic '{topic}'")
            return []
            
        # Load questions JSON if available
        questions_file = os.path.join(topic_path, "questions.json")
        questions_data = {}
        if os.path.exists(questions_file):
            try:
                with open(questions_file, 'r') as f:
                    questions_list = json.load(f)
                    # Create a dictionary mapping image_name to question data
                    for q in questions_list:
                        image_name = q.get('image_name')
                        if image_name:
                            questions_data[image_name] = q
                    print(f"Loaded {len(questions_data)} questions for topic '{topic}'")
            except Exception as e:
                print(f"Error loading questions file: {e}")
        
        # Divide images into sections
        num_images = len(image_files)
        sections = min(num_sections, num_images)
        section_size = max(1, num_images // sections)
        
        samples = []
        for i in range(0, sections):
            start_idx = i * section_size
            end_idx = min((i + 1) * section_size, num_images)
            section_images = image_files[start_idx:end_idx]
            
            # Sample from each section
            selected_images = random.sample(section_images, 
                                         min(samples_per_section, len(section_images)))
            
            for img_path in selected_images:
                img_name = os.path.basename(img_path)
                question_data = questions_data.get(img_name, {})
                
                # Add image and its metadata to samples
                samples.append({
                    'id': len(samples),
                    'image_path': img_path,
                    'image_name': img_name,
                    'question': question_data.get('question', ''),
                    'answer': question_data.get('answer', ''),
                    'context': question_data.get('context', ''),
                    'choices': question_data.get('choices', {}),
                    'category': topic
                })
                
        print(f"Selected {len(samples)} images for evaluation from topic '{topic}'")
        return samples
    
    def run_llava_inference(self, image, prompt: str) -> str:
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
            
            # Run inference using Ollama - try different LLaVA model versions
            model_versions = ['llava:13b-v1.6', 'llava:latest', 'llava:13b']
            model_found = False
            
            for model_version in model_versions:
                try:
                    print(f"Trying LLaVA model: {model_version}")
                    full_response = ''
                    for response in ollama.generate(
                        model=model_version,
                        prompt=prompt,
                        images=[image_bytes],
                        stream=True
                    ):
                        full_response += response['response']
                    model_found = True
                    break
                except Exception as e:
                    if "not found" in str(e).lower():
                        continue
                    else:
                        raise e
            
            if not model_found:
                print("No LLaVA model found. Please run: ollama pull llava:13b to download it.")
                return "Error: LLaVA model not available. Please install it using 'ollama pull llava:13b'"
                
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
        general_prompt = "Describe this image in detail. Identify and explain any diagrams, charts, or textual elements present."
        
        for sample in tqdm(samples, desc="Creating test cases"):
            try:
                # Load the image
                img = Image.open(sample['image_path'])
                # Save a copy to results directory
                img_filename = os.path.basename(sample['image_path'])
                result_img_path = os.path.join(RESULTS_DIR, f"image_{len(test_cases)}_{img_filename}")
                img.save(result_img_path)
                
                # First get a general description
                general_response = self.run_llava_inference(img, general_prompt)
                general_response = f"General Description:\n {general_response}"
                
                # If there's a specific question, ask it
                specific_question = False
                if sample.get('question'):
                    specific_question = True
                    question_text = sample.get('question')
                    
                    # Format the question based on whether it has choices
                    if sample.get('choices'):
                        choices_text = "\n".join([f"{choice_id}) {choice_text}" for choice_id, choice_text in sample.get('choices', {}).items()])
                        question_prompt = f"Based on this image, answer the following question: {question_text}\nChoose from these options:\n{choices_text}\n"
                    else:
                        question_prompt = f"Based on this image, answer the following question: {question_text}"
                    
                    # Get response to the specific question
                    specific_response = self.run_llava_inference(img, question_prompt)
                    specific_response = f"\nQuestion Response:\n {specific_response}"
                    
                    # Combine responses
                    combined_response = general_response + specific_response
                else:
                    # Just use the general description if no specific question
                    question_prompt = general_prompt
                    combined_response = general_response
                
                # Create context for the test case - extract relevant information
                context_strings = []
                if sample.get('context'):
                    context_strings.append(sample.get('context'))
                
                # Create test case
                test_case = LLMTestCase(
                    input=question_prompt,
                    actual_output=combined_response,
                    expected_output=sample.get('answer', ''),
                    # For ContextualRelevancyMetric
                    retrieval_context=context_strings, 
                    # For HallucinationMetric
                    context=context_strings
                )
                
                test_cases.append(test_case)
                
            except Exception as e:
                print(f"Error processing image {sample['image_path']}: {e}")
                continue
            
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
        
        # Process each test case individually
        for i, test_case in enumerate(tqdm(test_cases, desc="Evaluating responses")):
            case_results = {}
            
            # Apply metrics one at a time with proper error handling
            
            # 1. AnswerRelevancyMetric - only needs input and actual_output
            try:
                answer_metric = AnswerRelevancyMetric(
                    threshold=0.5, 
                    model=self.eval_llm
                )
                answer_metric.measure(test_case)
                case_results["AnswerRelevancyMetric"] = {
                    'score': answer_metric.score,
                    'reason': answer_metric.reason
                }
            except Exception as e:
                print(f"Error with AnswerRelevancyMetric: {e}")
                case_results["AnswerRelevancyMetric"] = {
                    'score': 0.0, 
                    'reason': f"Error: {str(e)}"
                }
            
            # 2. HallucinationMetric - needs context
            try:
                hallucination_metric = HallucinationMetric(
                    threshold=0.5,
                    model=self.eval_llm
                )
                hallucination_metric.measure(test_case)
                case_results["HallucinationMetric"] = {
                    'score': hallucination_metric.score, 
                    'reason': hallucination_metric.reason
                }
            except Exception as e:
                print(f"Error with HallucinationMetric: {e}")
                case_results["HallucinationMetric"] = {
                    'score': 0.0, 
                    'reason': f"Error: {str(e)}"
                }
                
            # 3. ContextualRelevancyMetric - needs retrieval_context
            try:
                relevancy_metric = ContextualRelevancyMetric(
                    threshold=0.5,
                    model=self.eval_llm
                )
                relevancy_metric.measure(test_case)
                case_results["ContextualRelevancyMetric"] = {
                    'score': relevancy_metric.score, 
                    'reason': relevancy_metric.reason
                }
            except Exception as e:
                print(f"Error with ContextualRelevancyMetric: {e}")
                case_results["ContextualRelevancyMetric"] = {
                    'score': 0.0, 
                    'reason': f"Error: {str(e)}"
                }
            
            # Add custom metrics based on simple rules
            
            # Check if response contains the expected answer
            expected = test_case.expected_output.lower()
            if expected and expected in test_case.actual_output.lower():
                answer_score = 1.0
                answer_reason = f"Response contains the expected answer: {expected}"
            else:
                answer_score = 0.0
                answer_reason = "Response does not contain the expected answer"
                
            case_results["DirectAnswerMetric"] = {
                'score': answer_score,
                'reason': answer_reason
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
                    f.write(f"### Image {i+1}: {os.path.basename(sample.get('image_path', ''))}\n\n")
                    f.write(f"Category: {sample.get('category', 'Unknown')}\n\n")
                    
                    if sample.get('question'):
                        f.write(f"Question: {sample.get('question')}\n\n")
                        
                        if sample.get('choices'):
                            f.write("Choices:\n")
                            for choice_id, choice_text in sample.get('choices', {}).items():
                                f.write(f"- {choice_id}) {choice_text}\n")
                            f.write("\n")
                    
                    if sample.get('answer'):
                        f.write(f"Correct Answer: {sample.get('answer')}\n\n")
                    
                    if sample.get('context'):
                        f.write(f"Context: {sample.get('context')}\n\n")
                    
                    f.write(f"Prompt: {test_case.input}\n\n")
                    f.write(f"LLaVA Response:\n{test_case.actual_output}\n\n")
                    
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