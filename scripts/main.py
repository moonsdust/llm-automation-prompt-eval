from datasets import load_dataset
from benchmark_update import benchmark_update
from huggingface_hub import login
import os
from dotenv import load_dotenv

def main(benchmark_dataset, gpqa_dataset):
    """This function begins process of replacing values in <benchmark dataset>,
    automating prompting of different GPT models to answer the prompts 
    in the benchmark dataset and <gpqa_dataset> dataset, and having the same GPT model 
    evaluate its own response. Using all this information, output a CSV file. 
    """
    # Replace values in benchmark dataset 
    benchmark_update(benchmark_dataset)
    
if __name__ == "__main__":
    # Number of examples we want from each data set
    num_of_examples = 10
    # Load in .env file 
    load_dotenv()
    # Log into hugging face and pass it YOUR access token from .env file 
    login(token = os.getenv('HUGGING_FACE_ACCESS_TOKEN'))
    # Load in datasets (Only load in the first 10 rows)
    benchmark_dataset = load_dataset("openai/gsm8k", "main", split=f'train[:{num_of_examples}]')
    gpqa_dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split=f'train[:{num_of_examples}]')
    # Pass in dataset to main function
    main(benchmark_dataset, gpqa_dataset)