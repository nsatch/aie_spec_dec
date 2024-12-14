
# Running Code
Install dependencies as specified at the reference code repository below

Write a config file in `config.yaml`. Example configs in a more readable format can be found in config files containing `archive` in their name.
- demo_target: 0 = Run real deployed setup
- demo_target: 1 = Run simulated setup

Execute the testbench which runs multiple configurations via `cd benchmarking & python3.9 benchmark.py`

Results of each test can be found in the `output_logs` folder with the name specified in the config file.


# Reference Code
Check out the reference code at https://github.com/uw-mad-dash/decoding-speculative-decoding
