from datasets import load_dataset
from benchmark_update import benchmark_update

def main(benchmark_dataset, gpqa_dataset):
    """This function begins process of replacing values in <benchmark dataset>,
    automating prompting of different GPT models to answer the prompts 
    in the benchmark dataset and <gpqa_dataset> dataset, and having the same GPT model 
    evaluate its own response. Using all this information, output a CSV file. 
    """
    training_set_benchmark = benchmark_dataset["train"]
    testing_set_benchmark = benchmark_dataset["test"]
    training_set_gpqa_dataset = gpqa_dataset["train"]
    testing_set_gpqa_dataset = gpqa_dataset["test"]
    
    # Replace values in benchmark dataset 
    benchmark_update(training_set_benchmark, testing_set_benchmark)
    
if __name__ == "__main__":
    # Load in datasets
    benchmark_dataset = load_dataset("openai/gsm8k", "main")
    gpqa_dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond") #
    
    # Pass in dataset to main function
    main(benchmark_dataset, gpqa_dataset)