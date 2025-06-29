import re

def update_values(a_row, numbers_in_question, numbers_in_answer, division_in_question): 
    """Helper function for benchmark_update."""
    for number in range(len(numbers_in_question)):
        # Check if the number contains a decimal or not 
        if re.findall(r'\d+\.*\d+', numbers_in_question[number]) != []:
            new_num = str(float(numbers_in_question[number]) + 1.0)
        else: 
            new_num = str(int(numbers_in_question[number]) + 1)
        # Find the number in answer and replace with new_num 
        a_row["question"][0] = a_row["question"][0].replace(numbers_in_question[number], new_num)
        print(a_row["question"][0])
        # Check for any divisions 
        if division_in_question != []:
            for div in range(len(division_in_question)): 
                num_denom = division_in_question[div].split('/')
                print(int(num_denom[0])/int(num_denom[1]))

def benchmark_update(benchmark_dataset): 
    """In this function, update the benchmark dataset <benchmark_dataset> with 
    different values."""
    # Iterate through each row of the datasets 
    for a_row in benchmark_dataset.iter(batch_size=1):
        print(a_row)
        # Obtain lists of all the number that shows up in the question and answer. 
        numbers_in_question = re.findall(r'\d+\.*\d+', a_row["question"][0])
        numbers_in_answer = re.findall(r'\d+\.*\d+', a_row["answer"][0])
        division_in_question = re.findall(r'\d+\/+\d+', a_row["question"][0])
        update_values(a_row, numbers_in_question, numbers_in_answer, division_in_question)