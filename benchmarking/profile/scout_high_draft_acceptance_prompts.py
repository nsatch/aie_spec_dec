import os

# Gather all of the prompts and their relevant data
#   (prompt, prompt length, # successes, runtime)
def parse_output_log(output_log_dir):
    prompts = []
    with open(output_log_dir, "r") as f:
        lines = f.read().splitlines()
        try:
            for i, line in enumerate(lines):
                final_line_number = i
                if "Current prompt:" in line:
                    curr_prompt = {}
                    curr_prompt["prompt"] = line.replace("Current prompt: ", "")
                    curr_prompt["prompt_length"] = len(curr_prompt["prompt"])
                    curr_prompt["draft_success"] = int(lines[i+1].replace("Draft successes: ", ""))
                if "Prompt runtime = " in line:
                    curr_prompt["runtime"] = float(line.replace("Prompt runtime = ", ""))
                    prompts.append(curr_prompt)

        except:
            print(f"Hit except. Final line number was {final_line_number}")
            return prompts

    return prompts


# Sort prompt list by the length of the prompt and the number draft acceptances
def sort_prompts(prompt_list, output_dir, output_name):
    # Sort by length of the prompt
    sort_length = sorted(prompt_list, key=lambda d: d['prompt_length'])
    write_prompts_to_file(sort_length, output_dir, output_name+"_sorted_length")

    # Sort by the number of draft acceptances
    sort_acceptance = reversed(sorted(prompt_list, key=lambda d: d['draft_success']))
    '''
    for sl in sort_acceptance:
        print(sl)
        input()
    '''
    write_prompts_to_file(sort_acceptance, output_dir, output_name+"_sorted_acceptance")
    return sort_length, sort_acceptance


def write_prompts_to_file(prompts, output_dir, output_name):
    output_file = os.path.join(output_dir, output_name+".json")
    with open(output_file, "w") as f:
        for p_dict in prompts:
            prompt = p_dict["prompt"].replace('"', '\\"') # Escape double quotes
            string = '{"prompt": ' + '"' + prompt + '"}\n'
            f.write(string)
    print(f"Wrote to {output_file}")
    return


output_log_dir_default = "../../output_logs/cheese2.txt"
output_name = "hellaswag"
sorted_output_dir = "../prompts"


if __name__ == "__main__":
    print("Parsing output log")
    prompts = parse_output_log(output_log_dir_default)

    print("Sorting prompts")
    sort_prompts(prompts, sorted_output_dir, output_name)
# This script will find prompts that yield the highest draft token acceptance rate
# Outputs two prompt files
#   - One sorted by the draft acceptance rate (highest at top of file)
#   - One sorted by the length of the input prompts (shorter prompts at top of file
