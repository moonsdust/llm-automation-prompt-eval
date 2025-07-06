from openai import OpenAI
import os

def prompting_eval_qualitative(a_dataset):
    """In this function, iterate across different GPT models to prompt <a_dataset> a qualitative dataset and have
    the GPT model evaluate its response for the following dimensions on a scale of 0 (Worse than expected) to 100 (Better than expected): 
    quality, accuracy, length, and completeness. Then, output this into a CSV."""
    client = OpenAI(api_key=os.getenv('OPENAI_API_TOKEN')) # Insert API token
    gpt_models = ["gpt-3.5-turbo"] # All models to iterate through
    for prompt in a_dataset: 
        for model in gpt_models: 
            # Prompts
            the_prompt = prompt
            system_prompt = "Keep your response under 200 words."
            # Construct conversation list
            conversation = [{"role": "system", "content": system_prompt}, {"role": "user", "content": the_prompt}]
            # Get GPT's response
            response = client.chat.completions.create(
                        model=f'{model}',
                        messages=conversation,
                        max_tokens=150,
                    )
            assistant_reply = response.choices[0].message.content.strip()
            # Append GPT's response to conversation list
            conversation.append({"role": "assistant", "content": assistant_reply})
            print(conversation)
            print("\n")