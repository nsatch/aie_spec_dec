from colors import *
import subprocess
import sys
import yaml

class test:
    def __init__():
        self.oracle = None
        self.USE_DRAFTER = None
        self.drafter1 = None
        self.drafter2 = None
        self.num_spec_tokens = None

    def __init__(self, oracle, USE_DRAFTER, drafter1, drafter2, num_spec_tokens):
        self.oracle = oracle
        self.USE_DRAFTER = USE_DRAFTER
        self.drafter1 = drafter1
        self.drafter2 = drafter2
        self.num_spec_tokens = num_spec_tokens

    def print(self):
        print(f"\t\t{PINK}Oracle{ENDC} = {self.oracle}")
        print(f"\t\t{PINK}Drafter 1 Model{ENDC} = {self.drafter1}")
        print(f"\t\t{PINK}Drafter 2 Model{ENDC} = {self.drafter2}")
        print(f"\t\t{PINK}Number of speculative tokens{ENDC} = {self.num_spec_tokens}")

def create_test_suite(config):
    tests = []
    # baseline_no_draft (bnd)
    #   - Turn off drafter (USE_DRAFTER = False)
    #   - Select oracle model
    bnd = config['baseline_no_draft']
    print(bnd)
    # TODO: Get rid of class and just use dictionary instead . . .
    # TODO: Add support for lists (so you can have a list of oracle models
    if bnd['active']:
        testcase = test(bnd['oracle_model'], False, None, None, 0)
        tests.append(testcase)
        cprint(CYAN, f"\tAdded baseline no draft test")
        testcase.print()

    num_spec_tokens_list = config['num_spec_tokens']
    #print(num_spec_tokens_list)
    # baseline_single_draft (bsd)
    for nst in num_spec_tokens_list:
        bsd = config['baseline_single_draft']
        # TODO: Add support for lists (so you can have a list of oracle models and drafter models
        # and get the cross product of them?
        if bsd['active']:
            oracle_model_list = bsd['oracle_model']
            drafter_model_list = bsd['draft1_model']
            for oracle_model in oracle_model_list:
                for drafter_model in drafter_model_list:
                    testcase = test(oracle_model, True, drafter_model, None, nst)
                    tests.append(testcase)
                    cprint(CYAN, f"\tAdded baseline single draft test")
                    testcase.print()

    # double draft (dd)
    for nst in num_spec_tokens_list:
        dd = config['double_draft']
        # TODO: Add support for lists (so you can have a list of oracle models and drafter models
        # and get the cross product of them?
        if dd['active']:
            oracle_model_list = dd['oracle_model']
            drafter1_model_list = dd['draft1_model']
            drafter2_model_list = dd['draft2_model']

            if len(drafter1_model_list) != len(drafter2_model_list):
                print(f"{RED} ERROR: {ENDC} The double draft test do not have the same number of draft 1 and draft 2 models")

            for oracle_model in oracle_model_list:
                for i, drafter1_model in enumerate(drafter1_model_list):
                    testcase = test(oracle_model, True, drafter1_model, drafter2_model_list[i], nst)
                    tests.append(testcase)
                    cprint(CYAN, f"\t Added double drafter model test")
                    testcase.print()
    return tests


def update_test_info(test_info, new_string):
    print(new_string)
    test_info += new_string + "\n"
    return test_info
                                       
def construct_test(testcase, run_script):
    test_info = ""

    # Update oracle
    search_text = "oracle_model = AutoModelForCausalLM.from_pretrained("
    oracle = testcase.oracle
    replace_text = f"    oracle_model = AutoModelForCausalLM.from_pretrained(\"{oracle}\", torch_dtype=torch.float16)"
    subprocess.run(f"cd .. && sed -i '/{search_text}/c\\{replace_text}' {run_script}", shell=True)
    test_info = update_test_info(test_info, f"Updating oracle model to {oracle}")

    # Update tokenizer to match oracle
    search_text = "AutoTokenizer.from_pretrained("
    replace_text = f"    tokenizer = AutoTokenizer.from_pretrained(\"{oracle}\", torch_dtype=torch.float16)"
    subprocess.run(f"cd .. && sed -i '/{search_text}/c\\{replace_text}' {run_script}", shell=True)
    test_info = update_test_info(test_info, f"Updating tokenizer to match oracle model ({oracle})")

    # Update USE_DRAFTER
    search_text = "USE_DRAFTER    = "
    replace_text = f"USE_DRAFTER    = {testcase.USE_DRAFTER}"
    subprocess.run(f"cd .. && sed -i '/{search_text}/c\\{replace_text}' {run_script}", shell=True)
    test_info = update_test_info(test_info, f"Updating USE_DRAFTER to {testcase.USE_DRAFTER}")

    # Update USE_DOUBLE_DRAFTER
    search_text = "USE_DOUBLE_DRAFTER    = "
    if testcase.drafter2 == None:
        use_double_drafter = "False"
    else:
        use_double_drafter = "True"
    replace_text = f"USE_DOUBLE_DRAFTER    = {use_double_drafter}"
    subprocess.run(f"cd .. && sed -i '/{search_text}/c\\{replace_text}' {run_script}", shell=True)
    test_info = update_test_info(test_info, f"Updating USE_DOUBLE_DRAFTER to {use_double_drafter}")

    # Update drafter 1
    if testcase.drafter1 != None:
        search_text = "draft_model = AutoModelForCausalLM.from_pretrained("
        replace_text = f"    draft_model = AutoModelForCausalLM.from_pretrained(\"{testcase.drafter1}\")"
        subprocess.run(f"cd .. && sed -i '/{search_text}/c\\{replace_text}' {run_script}", shell=True)
        test_info = update_test_info(test_info, f"Updating draft model one to {testcase.drafter1}")

    # Update drafter 2
    if testcase.drafter2 != None:
        search_text = "draft2_model = AutoModelForCausalLM.from_pretrained("
        replace_text = f"    draft2_model = AutoModelForCausalLM.from_pretrained(\"{testcase.drafter2}\")"
        subprocess.run(f"cd .. && sed -i '/{search_text}/c\\{replace_text}' {run_script}", shell=True)
        test_info = update_test_info(test_info, f"Updating draft model one to {testcase.drafter2}")

    # Update the number of speculative tokens generated
    num_spec_tokens = testcase.num_spec_tokens
    search_text = "max_tokens = "
    replace_text = f"    max_tokens = {num_spec_tokens}"
    subprocess.run(f"cd .. && sed -i '/{search_text}/c\\{replace_text}' {run_script}", shell=True)
    test_info = update_test_info(test_info, f"Updating max_tokens to {num_spec_tokens} (number of speculative tokens generated)")

    return test_info


def verify_venv():
    in_venv = sys.prefix != sys.base_prefix
    if not in_venv:
        print("Please run this code in virtual environment")
        print("Run 'activate' in terminal")
        quit()
    print("Running in virtual environment! Continuing to benchmark!")
    return 

def parse_config_file(config_loc):
    with open(config_loc) as stream:
        try:
            config = yaml.safe_load(stream) 
            print(f"Successfully loaded config file! \n Contents:\n {config}")
        except yaml.YAMLError as exc:
            print(exc)
    return config
