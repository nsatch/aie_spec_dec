import os
import subprocess

# Point to new culled input prompts
def update_input_file_in_param_code(strided_input_prompts, run_script):
    # TODO: Should make this more gneral 
    BENCHMARK_PATH = "benchmarking"
    strided_input_prompts = os.path.join(BENCHMARK_PATH, strided_input_prompts)
    replace_text = f"    test_json = json_loader(\"{strided_input_prompts}\")"
    #print(f"Replace text = '{replace_text}'")
    subprocess.run(f"cd .. && sed -i '/test_json = /c\\{replace_text}' {run_script}", shell=True)
    print("Pointed AI param code to use new strided input prompt file")
    return 

# Edit param python file to setup for test

# Run the test and record output
def run_test():
    return

# When test finishes write into benchmarking progress file
