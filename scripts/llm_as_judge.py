from openai import OpenAI
import os
import csv

def llm_as_a_judge(a_dataset, output_file_name, cleaned_data_group):
    """
    """
    client = OpenAI(api_key=os.getenv('OPENAI_API_TOKEN')) # Insert API token
    model = "gpt-4o" # Model used for "LLM as a judge"
    entries = []
    for i in range(len(a_dataset)): 
        # Prompts
        the_prompt = a_dataset[i]
        system_prompt_eval = "Only reply with a number. You do not have to say the dimension name."
        
        # Dimension - Relevance
        evaluation_prompt_relevance = f'Participants are asked to write a prompt to get an AI chatbot (assume ChatGPT, or GPT-4o) to correctly estimate the number of chickpeas in \
            pictures that shows the bottom and side of the jar. One of the participants wrote the following prompt: \
            {the_prompt}. Rate the prompt on a scale of 1 (Worse than expected) to 5 (Better than expected) under \
        the dimension relevance (Is the prompt relevant to the task and does the prompt reference the goal?)'
        # Construct conversation list
        eval_conversation_relevance = [{"role": "system", "content": system_prompt_eval}, {"role": "user", "content": evaluation_prompt_relevance}]
        # Get GPT's response
        eval_response_relevance = client.chat.completions.create(
                    model=f'{model}',
                    messages=eval_conversation_relevance,
                    max_tokens=150,
                )
        assistant_reply_eval_relevance = eval_response_relevance.choices[0].message.content.strip()
        
        # Dimension - Quality 
        evaluation_prompt_quality = f'Participants are asked to write a prompt to get an AI chatbot (assume ChatGPT, or GPT-4o) to correctly estimate the number of chickpeas in \
            pictures that shows the bottom and side of the jar. One of the participants wrote the following prompt: \
            {the_prompt}. Rate the prompt on a scale of 1 (Worse than expected) to 5 (Better than expected) under \
        the dimension quality (Is the prompt free of spelling and grammar mistakes? Does the prompt include in-context examples or using prompting techniques like chain-of-thought)'
        # Construct conversation list
        eval_conversation_quality = [{"role": "system", "content": system_prompt_eval}, {"role": "user", "content": evaluation_prompt_quality}]
        # Get GPT's response
        eval_response_quality = client.chat.completions.create(
                    model=f'{model}',
                    messages=eval_conversation_quality,
                    max_tokens=150,
                )
        assistant_reply_eval_quality = eval_response_quality.choices[0].message.content.strip()
        
        # Dimension - Coherence
        evaluation_prompt_coherence = f'Participants are asked to write a prompt to get an AI chatbot (assume ChatGPT, or GPT-4o) to correctly estimate the number of chickpeas in \
            pictures that shows the bottom and side of the jar. One of the participants wrote the following prompt: \
            {the_prompt}. Rate the prompt on a scale of 1 (Worse than expected) to 5 (Better than expected) under \
        the dimensions coherence (Are the ideas in the prompt connected?)'
        # Construct conversation list
        eval_conversation_coherence = [{"role": "system", "content": system_prompt_eval}, {"role": "user", "content": evaluation_prompt_coherence}]
        # Get GPT's response
        eval_response_coherence= client.chat.completions.create(
                    model=f'{model}',
                    messages=eval_conversation_coherence,
                    max_tokens=150,
                )
        assistant_reply_eval_coherence = eval_response_coherence.choices[0].message.content.strip()
        
        # Append results to entries list (so it can be outputted as a CSV later)
        entries.append([the_prompt, cleaned_data_group[i], assistant_reply_eval_relevance, assistant_reply_eval_quality, assistant_reply_eval_coherence])
        # Write a CSV
        fields = ["prompt", "group", "relevance", "quality", "coherence"]
        with open(output_file_name, 'w') as csv_file: 
            # creating a csv writer object
            the_csv_writer = csv.writer(csv_file)
            # writing the fields
            the_csv_writer.writerow(fields)
            # writing the data rows
            the_csv_writer.writerows(entries)