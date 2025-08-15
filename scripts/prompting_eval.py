from openai import OpenAI
import os
import csv

def prompting_eval(a_dataset, output_file_name):
    """In this function, iterate across different GPT models to prompt <a_dataset> a dataset and have
    the GPT model evaluate its response for the following dimensions on a scale of 0 (Worse than expected) to 100 (Better than expected): 
    quality, accuracy, length, and completeness. Then, output this into a CSV with the specified name <output_file_name>."""
    client = OpenAI(api_key=os.getenv('OPENAI_API_TOKEN')) # Insert API token
    gpt_models = ["gpt-3.5-turbo"] # All models to iterate through
    entries = []
    for prompt in a_dataset: 
        for model in gpt_models: 
            # Prompts
            the_prompt = prompt
            # system_prompt = "Keep your response under 200 words."
            system_prompt = ""
            # Construct conversation list
            conversation = [{"role": "system", "content": system_prompt}, {"role": "user", "content": the_prompt}]
            # Get GPT's response
            response = client.chat.completions.create(
                        model=f'{model}',
                        messages=conversation,
                        max_tokens=400,
                    )
            assistant_reply = response.choices[0].message.content.strip()
            # Append GPT's response to conversation list
            conversation.append({"role": "assistant", "content": assistant_reply})
            print(conversation)
            print("\n")
            # Have model evaluate model's response under the following dimensions on a scale of 0 
            # (Worse than expected) to 100 (Better than expected): quality, accuracy, length, and completeness
            system_prompt_eval = "Only return a number for each dimension, which are quality, accuracy, length, and completeness."
            evaluation_prompt = f'I gave a prompt of {prompt}. Rate the following response on a scale of 0 (Worse than expected) to 100 (Better than expected) under \
            the dimensions quality, accuracy, length, and completeness: {assistant_reply}'
            # Construct conversation list
            eval_conversation = [{"role": "system", "content": system_prompt_eval}, {"role": "user", "content": evaluation_prompt}]
            # Get GPT's response
            eval_response = client.chat.completions.create(
                        model=f'{model}',
                        messages=eval_conversation,
                        max_tokens=150,
                    )
            assistant_reply_eval = eval_response.choices[0].message.content.strip()
            # Append GPT's response to conversation list
            eval_conversation.append({"role": "assistant", "content": assistant_reply_eval})
            print(eval_conversation)
            print("\n")
            # Append results to entries list (so it can be outputted as a CSV later)
            entries.append([the_prompt, assistant_reply, model, assistant_reply_eval])
            # entries.append([the_prompt, assistant_reply, model])
        # Write a CSV
        fields = ["Prompt", "Output", "Model", "GPT's Evaluation"]
        # fields = ["Prompt", "Output", "Model"]
        with open(output_file_name, 'w') as csv_file: 
            # creating a csv writer object
            the_csv_writer = csv.writer(csv_file)
            # writing the fields
            the_csv_writer.writerow(fields)
            # writing the data rows
            the_csv_writer.writerows(entries)