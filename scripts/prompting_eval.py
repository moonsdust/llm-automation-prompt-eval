from openai import OpenAI
import os
import csv
import base64

def encode_image(image_path):
    """This function encodes images"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
def extract_values_output(a_dataset, output_file_name, group = None, pid = None):
    """Extract value from <a_dataset>. This function is for the prompting challenge"""
    client = OpenAI(api_key=os.getenv('OPENAI_API_TOKEN')) # Insert API token
    the_model = "gpt-4o-mini" 
    entries = []
    for i in range(len(a_dataset)): 
        # Prompts
        the_prompt = a_dataset[i]
        # system_prompt = "You are given responses for the number of chickpeas. \
        #     Extract the final answer and only return a number for the final answer number of chickpeas.\
        #         if no final answer is given, return an empty string."
        system_prompt = "You are given responses commenting on a post someone made asking if they are the asshole or not. \
            Extract the final answer and only return a number where 1 means that the response thinks that person is the asshole and 0 if they don't think they are the asshole.\
                If no final answer is given, return an empty string."
        # For prompting challenge: Construct conversation list
        conversation = [{"role": "system", "content": system_prompt}, 
                        {"role": "user", "content": a_dataset[i]}]
        # For prompting challenge: get GPT's response
        response = client.chat.completions.create(
                    model=f'{the_model}',
                    messages=conversation
                )
        assistant_reply = response.choices[0].message.content.strip()
        # Append GPT's response to conversation list
        conversation.append({"role": "assistant", "content": assistant_reply})
        print(conversation)
        print("\n")
        # entries.append([pid[i], the_prompt, assistant_reply, group[i]])
        entries.append([pid[i], the_prompt, assistant_reply, group[i]])
        # Write a CSV
        fields = ["prompt", "gpt_answer_full", "is_asshole_gpt", "is_asshole_human"]
        with open(output_file_name, 'w') as csv_file: 
            # creating a csv writer object
            the_csv_writer = csv.writer(csv_file)
            # writing the fields
            the_csv_writer.writerow(fields)
            # writing the data rows
            the_csv_writer.writerows(entries)

def prompting_eval(a_dataset, output_file_name, group = None, pid = None):
    """In this function, iterate across different GPT models to prompt <a_dataset> a dataset and have
    the GPT model evaluate its response for the following dimensions on a scale of 0 (Worse than expected) to 100 (Better than expected): 
    quality, accuracy, length, and completeness. Then, output this into a CSV with the specified name <output_file_name>. 
    There are also optional parameters <group> and <pid>, which represents the group and participant id, respectively."""
    client = OpenAI(api_key=os.getenv('OPENAI_API_TOKEN')) # Insert API token
    gpt_models = ["gpt-4o-mini"] # All models to iterate through
    entries = []
    for i in range(len(a_dataset)): 
        for model in gpt_models: 
            # # Only for prompting challenge
            # image_1 = "images/chickpea_bottom.png"
            # image_2 = "images/chickpea_side.png"
            # # Encode image
            # base64_image_1 = encode_image(image_1)
            # base64_image_2 = encode_image(image_2)
            
            # Prompts
            the_prompt = a_dataset[i]
            # system_prompt = "Keep your response under 200 words."
            system_prompt = "Please judge the following prompt and indicate if am I the asshole or not at the beginning and explain why. AITA stands for am I the asshole."
            # Construct conversation list
            conversation = [{"role": "system", "content": system_prompt}, {"role": "user", "content": the_prompt}]
            # For prompting challenge: Construct conversation list
            # conversation = [{"role": "system", "content": system_prompt}, 
            #                 {"role": "user", 
            #                  "content": [
            #                     {"type": "text", "text": the_prompt},
            #                     {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image_1}"}},
            #                     {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image_2}"}}
            #                     ]
            #                  }]
            # Get GPT's response
            response = client.chat.completions.create(
                        model=f'{model}',
                        messages=conversation,
                        max_tokens=200,
                        temperature=0.7
                    )
            # # For prompting challenge: get GPT's response
            # response = client.chat.completions.create(
            #             model=f'{model}',
            #             messages=conversation,
            #         )
            assistant_reply = response.choices[0].message.content.strip()
            # Append GPT's response to conversation list
            conversation.append({"role": "assistant", "content": assistant_reply})
            print(conversation)
            print("\n")
            # # Have model evaluate model's response under the following dimensions on a scale of 0 
            # # (Worse than expected) to 100 (Better than expected): quality, accuracy, length, and completeness
            # system_prompt_eval = "Only return a number for each dimension, which are quality, accuracy, length, and completeness."
            # evaluation_prompt = f'I gave a prompt of {prompt}. Rate the following response on a scale of 0 (Worse than expected) to 100 (Better than expected) under \
            # the dimensions quality, accuracy, length, and completeness: {assistant_reply}'
            # # Construct conversation list
            # eval_conversation = [{"role": "system", "content": system_prompt_eval}, {"role": "user", "content": evaluation_prompt}]
            # # Get GPT's response
            # eval_response = client.chat.completions.create(
            #             model=f'{model}',
            #             messages=eval_conversation,
            #             max_tokens=150,
            #         )
            # assistant_reply_eval = eval_response.choices[0].message.content.strip()
            # # Append GPT's response to conversation list
            # eval_conversation.append({"role": "assistant", "content": assistant_reply_eval})
            # print(eval_conversation)
            # print("\n")
            # Append results to entries list (so it can be outputted as a CSV later)
            # entries.append([the_prompt, assistant_reply, model, assistant_reply_eval])
            # entries.append([pid[i], the_prompt, assistant_reply, group[i]])
            entries.append([the_prompt, assistant_reply, group])
        # Write a CSV
        # fields = ["Prompt", "Output", "Model", "GPT's Evaluation"]
        # fields = ["ID", "prompt", "output", "group"]
        fields = ["prompt", "gpt_eval", "human_eval"]
        with open(output_file_name, 'w') as csv_file: 
            # creating a csv writer object
            the_csv_writer = csv.writer(csv_file)
            # writing the fields
            the_csv_writer.writerow(fields)
            # writing the data rows
            the_csv_writer.writerows(entries)