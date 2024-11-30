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

    return test_result_directory

# Read output log files and parse relevant data into csv
def write_runtime(output_log, runtime):
    print(f"Trying to write to {output_log}")
    with open(output_log, "a") as f:
        f.write(f"\n\nThe runtime is {runtime}")

def verify_testcase_status(output_log):
    print("NOT YET IMPLEMENTED VERIFICATION OF TESTCASE STATUS!!!!")
    return
# Parse csv into plots
