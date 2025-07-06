from datasets import load_dataset
from benchmark_update import benchmark_update
from prompting_eval import prompting_eval_qualitative
from huggingface_hub import login
import os
from dotenv import load_dotenv

def main(benchmark_dataset, gpqa_dataset, qualitative_dataset):
    """This function begins process of replacing values in <benchmark dataset>,
    automating prompting of different GPT models to answer the prompts 
    in the benchmark dataset, <gpqa_dataset> dataset, <qualitative_dataset> dataset, and having the same GPT model 
    evaluate its own response. Using all this information, output a CSV file. 
    """
    # Replace values in benchmark dataset 
    # benchmark_update(benchmark_dataset)
    # Prompt GPT on the custom_dataset
    prompting_eval_qualitative(qualitative_dataset)
    
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
    # Pass in qualitative-type prompts dataset
    qualitative_dataset = ["Write me a paragraph about Charles Darwin", # Vague
                      "Write me a paragraph about Charles Darwin's trip to the United States", # Hallucination  
                      "Write me a paragraph about Charles Darwin's trip to Galapagos islands and how it helped him create the theory of evolution", # More focused
                      "Write me a paragraph about how tokens works", # Ambiguity with which field's "tokens" they are referring to
                      "Write be a paragraph about how tokens works in large language models", # More specific in terms of the field 
                      "Come up with paper ideas related to exercise", # General 
                      "Come up with paper ideas related to exercise for a 4th year kinesiology class", # More specific towards 4th year
                      "Come up with paper ideas related to exercise for a 4th year kinesiology class if I'm interested in focusing on the heart in exercise and please reference https://www.functionalkinesiology.co.uk/heart-kinesiology/ to fact-check the paper idea",  # More tailored
                      "Write me a paragraph wishing a 6 year kid, happy birthday, where the 6 year old kid will be reading it", # Uses simpler words
                      "Write me a paragraph wishing a 60 year old person, happy birthday, where the 60 year old person will be reading it" # Uses complex words
                      ]
    # Pass in dataset to main function
    main(benchmark_dataset, gpqa_dataset, qualitative_dataset)