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


def verify_testcase_status(output_log):
    print("NOT YET IMPLEMENTED VERIFICATION OF TESTCASE STATUS!!!!")
    return
# Parse csv into plots
