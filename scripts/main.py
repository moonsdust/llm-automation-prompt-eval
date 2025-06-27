from datasets import load_dataset
from benchmark_update import benchmark_update

def main(benchmark_dataset):
    """This function begins process of replacing values in benchmark dataset,
    automating prompting of different GPT models to answer the prompts 
    in the benchmark dataset, and having the same GPT model evaluate its own response. 
    Using all this information, output a CSV file. 
    """
    training_set = benchmark_dataset["train"]
    testing_set = benchmark_dataset["test"]
    # Replace values in benchmark dataset 
    benchmark_update(training_set, testing_set)
    
if __name__ == "__main__":
    # Load in dataset
    benchmark_dataset = load_dataset("openai/gsm8k", "main")
    # Pass in dataset to main function
    main(benchmark_dataset)