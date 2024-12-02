# To Code
- Config file
    - Baseline toggle
        - No drafter
        - One drafter
            - Change drafter
    - Two drafters
        - Choose 2 drafters
    - Number of prompts
    - Drafter switch thresholds

- Collect run times
    - Whole run time to finish all input prompts
    - Runtime of an individual propmpt
        - Make sure to exclude log dumping from run time

- Output log organization


# General Goals
- Run baseline 
    - No drafter model
    - One drafter model
        - Test all three

Tests
- Within the prompt itself 
    - Use smallest model for < second half and largest model for > first half
    - Move threshold around
- Between prompts (longer prompts might

# Sunday 
Test Suite Summary Sheet
    - Row = Summary of one testcase
        - Testcase params
        - Average number of accepted tokens for all prompts
        - Runtime of the testcase
        - Testcase verification result

Individual Testcase Sheet
    - Each prompt's 
        - Runtime
        - Number of accepted tokens
        - Prompt itself?
        - Outputprompt???

- Finish post processing (at least dumping to CSV)
    - Verification test case succeeded
    - Individual prompt run time
    - Total run time
    - NUmber of accepted draft tokens
        - Average number of accepted tokesn
    
Maybe look into profiling prompts that take the longest on the base model?
    - Theoretically since basemodel runs slowly on these, the draft model should help?
    It may be that draft model with high acceptance is high because the base model 
    is already so good at processing that they don't need draft model???

