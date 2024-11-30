# Call this script as 'python3.9 sort_prompts.py <input_file> <output_file> <num_prompts>
import sys
import json
import os
from colors import *

def json_loader(file_path):
    """
    Input: file_path: Path to the json file containing all the queries.

    File format looks like the following:
    {"prompt": "A seated man cleans a shoe in a classroom setting with other individuals. the man"}
    {"prompt": "Two girls are sailing on a lake. they"}

    Output: This function returns a list of prompts to be used by the draft LLM.
    """
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

def create_strided_prompts_from_config(config):
    input_file = config['full_prompt_input_file']
    output_file = config['strided_prompt_output_file']
    num_prompts = config['num_input_prompts']
    print(f"Creating strided input prompt file from {input_file} w/ {num_prompts}  prompts.")
    output_file_name = output_file.replace(".json", "") + "_" + str(num_prompts) + ".json"
    print(f"Writing to {output_file_name}")

    output_file_name = create_strided_prompts(input_file, output_file, num_prompts)
    cprint(GREEN, "Done making strided prompts!")
    return output_file_name

def create_strided_prompts(input_file, output_file, num_prompts):
    output_file_name = output_file.replace(".json", "") + "_" + str(num_prompts) + ".json"

    # If we already have the strided input prompts, use that instead of making again
    if os.path.exists(output_file_name):
        print(f"Strided prompts with size {num_prompts} already exists! Using cached prompt file.")
        return output_file_name

    # If strided input prompts doesn't exist, then make it
    data = json_loader(input_file)
    prompts = []
    for prompt in data:
        prompts.append(prompt['prompt'].replace('"', '\\"'))

    prompts = sorted(prompts, key=len)
    stride = len(prompts)//num_prompts
    strided_prompts = prompts[0:-1:stride]
    print(f"There are {len(strided_prompts)} prompts in the input file")

    with open(output_file_name, "w") as f:
        for sp in strided_prompts:
            string = '{"prompt": '+ '"'+ sp+ '"}\n'
            f.write(string) 
    return output_file_name

# Call this script as 'python3.9 sort_prompts.py <input_file> <output_file> <num_prompts>
if __name__ == "__main__":
    '''
    num_prompts = 1000
    output_file = "strided_prompts.json"
    input_file = "hellaswag.json"
    '''

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    num_prompts = sys.argv[3]
    create_strided_prompts(input_file, output_file, num_prompts)


