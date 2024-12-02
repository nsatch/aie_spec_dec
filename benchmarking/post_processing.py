import pandas as pd
import yaml
import os
import colors as c
import shutil

def construct_output_directory(config, output_directory):
    test_name = config['test_name']
    test_result_directory = os.path.join(output_directory, test_name)
    #print(f"Test result directory = {test_result_directory}")
    # If directory exists, verify with user that its okay to overwrite contents
    if os.path.exists(test_result_directory):
        print(f"{c.RED} WARNING: {c.ENDC} A directory already exists for {test_name}! Okay to overwrite? (y/n)")
        ans = input()
        if ans.lower() != 'y':
            print("Quitting")
            quit()
        print("Overwriting directory for this test")
        shutil.rmtree(test_result_directory)

    os.mkdir(test_result_directory)

    # Write the config file for this test into the directory
    config_archive_path = os.path.join(test_result_directory, "config.yaml")
    print(f"Writing this test suite's config file to {config_archive_path}")
    with open(config_archive_path, "w") as f:
        yaml.dump(config, f)

    return test_result_directory

# Read output log files and parse relevant data into csv
def write_runtime(output_log, runtime):
    print(f"Trying to write runtime to {output_log}")
    with open(output_log, "a") as f:
        f.write(f"\n\nThe runtime is {runtime}")

def write_test_info(output_log, test_info):
    print(f"Trying to write test info to {output_log}")
    test_info_file = output_log.replace(".log", "_info.log")
    with open(test_info_file, "a") as f:
        f.writelines(test_info)

def write_test_suite_runtime(test_suite_runtime, runtime_file):
    with open(runtime_file, "w") as f:
        f.write(f"{test_suite_runtime}s")

def parse_output_log(output_log_dir):
    prompts = []
    with open(output_log_dir, "r") as f:
        lines = f.read().splitlines()
        for i, line in enumerate(lines):
            final_line_number = i
            if "Current prompt:" in line:
                curr_prompt = {}
                curr_prompt["prompt"] = line.replace("Current prompt: ", "")
                curr_prompt["prompt_length"] = len(curr_prompt["prompt"])
            if "Draft successes: " in line:
                curr_prompt["draft_success"] = int(line.replace("Draft successes: ", ""))
            if "Prompt runtime = " in line:
                curr_prompt["runtime"] = float(line.replace("Prompt runtime = ", ""))
                # Tests without drafter models won't have draft successes
                if "draft_success" not in curr_prompt.keys():
                    curr_prompt["draft_success"] = None
                prompts.append(curr_prompt)
            if "The runtime is " in line:
                total_runtime = float(line.replace("The runtime is ", ""))
    return prompts, total_runtime


def extract_output_log_summary(output_log):
    prompts, total_runtime = parse_output_log(output_log)
    suc_cum = 0
    run_cum = 0
    for prompt in prompts:
        draft_success = prompt["draft_success"]
        prompt_runtime = prompt["runtime"]
        if draft_success != None:
            suc_cum += draft_success
            run_cum += prompt_runtime
    avg_accepted_tokens = suc_cum / len(prompts)
    avg_prompt_runtime = run_cum / len(prompts)

    return avg_accepted_tokens, avg_prompt_runtime, total_runtime


def add_testcase_to_summary(testcase, output_log, testcases_summary):
    testcase_success = verify_testcase_status(output_log)
    avg_accepted_tokens, avg_prompt_runtime, total_runtime = extract_output_log_summary(output_log)
    new_data = [
                testcase_success, 
                testcase.oracle,
                testcase.drafter1,
                testcase.drafter2,
                testcase.num_spec_tokens,
                testcase.switch_threshold,
                avg_accepted_tokens,
                avg_prompt_runtime,
                total_runtime
                ]
    testcases_summary.append(new_data)
    return testcases_summary

def testcases_summary_to_csv(testcases_summary, output_file_loc):
    cols = [
            "Testcase Succeeded",
            "Oracle Model",
            "Drafter 1 Model",
            "Drafter 2 Model",
            "Num Spec Tokens",
            "Switch Threshold",
            "Average Num Accepted Tokens",
            "Average Prompt Runtime",
            "Testcase Runtime",
            ]
    summary_df = pd.DataFrame(data=testcases_summary, columns = cols)
    summary_df.to_csv(output_file_loc, index=False)
    return

def verify_testcase_status(output_log):
    with open(output_log, "r") as f:
        lines = f.read().splitlines()
        for line in lines:
            if " exits successfully." in line:
                return True
    return False

# Parse csv into plots
