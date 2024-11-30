import subprocess
import sys
import yaml

class test:
    def __init__():
        self.oracle = None
        self.USE_DRAFTER = None
        self.drafter1 = None
        self.drafter2 = None

    def __init__(self, oracle, USE_DRAFTER, drafter1, drafter2):
        self.oracle = oracle
        self.USE_DRAFTER = USE_DRAFTER
        self.drafter1 = drafter1
        self.drafter2 = drafter2

def create_test_suite(config):
    tests = []
    # baseline_no_draft (bnd)
    #   - Turn off drafter (USE_DRAFTER = False)
    #   - Select oracle model
    bnd = config['baseline_no_draft']
    print(bnd)
    # TODO: Add support for lists (so you can have a list of oracle models
    if bnd['active']:
        tests.append(test(bnd['oracle_model'], False, None, None))
        print(f"\tAdded baseline no draft test")
    '''
    print(tests[0].oracle)
    print(tests[0].USE_DRAFTER)
    '''

    # baseline_single_draft (bsd)
    bsd = config['baseline_single_draft']
    # TODO: Add support for lists (so you can have a list of oracle models and drafter models
    # and get the cross product of them?
    if bsd['active']:
        tests.append(test(bsd['oracle_model'], True, bsd['draft1_model'], None))
        print(f"\tAdded baseline single draft test")

    # double draft (dd)


    return tests

def construct_test(testcase, run_script):
    # Update oracle
    search_text = "oracle_model = AutoModelForCausalLM.from_pretrained("
    oracle = testcase.oracle
    replace_text = f"    oracle_model = AutoModelForCausalLM.from_pretrained(\"{oracle}\", torch_dtype=torch.float16)"
    subprocess.run(f"cd .. && sed -i '/{search_text}/c\\{replace_text}' {run_script}", shell=True)
    print(f"Updating oracle model to {oracle}")

    # Update tokenizer to match oracle
    search_text = "AutoTokenizer.from_pretrained("
    replace_text = f"    tokenizer = AutoTokenizer.from_pretrained(\"{oracle}\", torch_dtype=torch.float16)"
    subprocess.run(f"cd .. && sed -i '/{search_text}/c\\{replace_text}' {run_script}", shell=True)
    print(f"Updating tokenizer to match oracle model ({oracle})")

    # Update USE_DRAFTER
    search_text = "USE_DRAFTER    = "
    replace_text = f"USE_DRAFTER    = {testcase.USE_DRAFTER}"
    subprocess.run(f"cd .. && sed -i '/{search_text}/c\\{replace_text}' {run_script}", shell=True)
    print(f"Updating USE_DRAFTER to {testcase.USE_DRAFTER}")

    # Update drafter 1
    print(f"TODO NOT YET IMPLEMENTED - Updating drafter 1 ")
    # Update drafter 2
    print(f"TODO NOT YET IMPLEMENTED - Updating drafter 2 ")

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
