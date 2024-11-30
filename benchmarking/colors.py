RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
CYAN = '\033[96m'
WHITE = '\033[97m'

ENDC = '\033[0m'

def cprint(color, string):
    print(f"{color} {string} {ENDC}")

