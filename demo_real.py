from time import time
import torch
import torch.nn as nn
import re
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import deepspeed
import torch.distributed as dist
import json
import os

import random

seed = 42
random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

### Supppress pad and eos warnings
import warnings
warnings.filterwarnings('ignore')  # Suppress all other warnings
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  # Suppress transformer warnings

# Param variables

# DON"T DELETE EXTRA SPACES HERE!
# It will screw up parameterization from benchmarking script!!
USE_DRAFTER    = True
USE_DOUBLE_DRAFTER    = True
# DON"T DELETE EXTRA SPACES HERE!
# It will screw up parameterization from benchmarking script!!

def parse_tensors_from_file(file_path):
    """
    Input: file_path: Path to the file containing the ground truth tensors from large LLM execution.

    This demo is for those who doesn't have the resource to execute large LLM but wish to deploy and test speculative decoding.

    File format looks like the following:
    tensor([[  518, 799, 596, 18873,  1265,   322,  1510,   372,  1283,   304,   596,
          7875, 29991, 2]])
    tensor([[  518, 10568, 29962,  1522,  2924, 29889,   518, 10568, 29962,  1522,
          1176,   681, 29889,   518, 10568, 29962,  1522,   752,   465,   291]])

    Output: This function returns a list of tensor to be used by the draft LLM.
    """
    with open(file_path, 'r') as file:
        data = file.read()

    # Find all tensor strings using regular expression
    tensor_strings = re.findall(r"tensor\(\[\[([\s\S]+?)\]\]\)", data)

    tensor_list = []
    iter = 0
    prev_length = 0
    for tensor_str in tensor_strings:
        # Clean up the tensor string and split into rows
        tensor_rows = tensor_str.strip().split('\n')

        # Create a list of tensors, one for each row
        row_tensors = []
        for row in tensor_rows:
            # Here we strip to remove spaces and trailing commas
            row = row.strip().rstrip(',').lstrip(',')
            if row:  # Make sure row is not empty
                # Convert row string to a list of integers
                tensor_values = [int(value) for value in row.split(',')]
                # Convert the list to a tensor and append to row_tensors
                row_tensors.append(torch.tensor(tensor_values))
                # print(row_tensors)

        # Create a tensor from the list of lists and add to our tensor list
        final_tensor = torch.cat(row_tensors, dim=0)
        # if final_tensor.size(0) != prev_length:
        #     print(iter, final_tensor.size(0), prev_length)
        prev_length = final_tensor.size(0)
        tensor_list.append(torch.cat(row_tensors, dim=0))

        iter += 1
    return tensor_list


def json_loader(file_path):
    """
    Input: file_path: Path to the json file containing all the queries.

    File format looks like the following:
    {"prompt": "A seated man cleans a shoe in a classroom setting with other individuals. the man"}
    {"prompt": "Two girls are sailing on a lake. they"}

    Output: This function returns a list of prompts to be used by the draft LLM.
    """
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data


if __name__ == "__main__":
    # I have to use double space between USE_DRAFTER and = becuase then nit gets replaced on sed command lmao
    # Actually I will make the replace target use triple space ghetto aah solution
    print(f"Sanity Check: USE_DRAFTER  = {USE_DRAFTER}")

    # local_rank = int(os.getenv("LOCAL_RANK", "0"))
    # world_size = int(os.getenv("WORLD_SIZE", "1"))
    # dist.init_process_group(backend='nccl', rank=local_rank, world_size=world_size)
    dist.init_process_group(backend='nccl')
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    input_file_path = 'llama-65b-hellaswag.txt'  # Replace with your input file path
    ground_truth_tensor_list = parse_tensors_from_file(input_file_path)

    test_json = json_loader("benchmarking/prompts/hellaswag_shortened_topn_10.json")

    # The LLaMA tokenizer does not have a pad token.
    # Modify the tokenizer to add a pad token and change the model configs accordingly.
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-hf", torch_dtype=torch.float16)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = "[PAD]"
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    # Feel free to change it to the draft model of your choice
    draft_model = AutoModelForCausalLM.from_pretrained("minghaoyan/Wide-Sheared-LLaMA-290M")

    draft_model.resize_token_embeddings(len(tokenizer))
    draft_model.config.pad_token_id = tokenizer.pad_token_id
    draft_model.embed_tokens = nn.Embedding(draft_model.config.vocab_size, draft_model.config.hidden_size, padding_idx=draft_model.config.pad_token_id)


    # Launch the draft model with deepspeed on 1 node. Alternatively, you could use HF or load from a checkpoint.
    draft_model = deepspeed.init_inference(
                    draft_model,
                    replace_with_kernel_inject=False,
                    tp={"tp_size": 1,},
                    dtype=torch.float16,
                    #checkpoint=checkpoint_dict,
                   )

    # Ashley, please take a look at this and see if this is set up correctly for draft 2
    ### START OF CODE ADDED BY NICK FOR DRAFT 2 ###
    # Feel free to change it to the draft model of your choice
    draft2_model = AutoModelForCausalLM.from_pretrained("minghaoyan/Wide-Sheared-LLaMA-290M")
    draft2_model.resize_token_embeddings(len(tokenizer))
    draft2_model.config.pad_token_id = tokenizer.pad_token_id
    draft2_model.embed_tokens = nn.Embedding(draft2_model.config.vocab_size, draft2_model.config.hidden_size, padding_idx=draft2_model.config.pad_token_id)


    # Launch the draft model with deepspeed on 1 node. Alternatively, you could use HF or load from a checkpoint.
    draft2_model = deepspeed.init_inference(
        draft2_model,
        replace_with_kernel_inject=False,
        tp={"tp_size": 1, },
        dtype=torch.float16,
        # checkpoint=checkpoint_dict,
    )

    # This variable acts as the pointer for which draft model to use
    #   0 = Use draft 1
    #   1 = Use draft 2
    draft_model_ptr = 0

    # Percent threshold of length through the prompt where we switch between drafter models
    # Units: 50 = 50% 
    # NOT 0.5 = 50%
    ### DON"T DELETE THE EXTRA SPACES!!!!
    drafter_switch_threshold   = 75
    ### DON"T DELETE THE EXTRA SPACES!!!!
    # Disable drafter toggle in one draft case. Can never be 200% of the way through a prompt!
    if not USE_DOUBLE_DRAFTER:
        drafter_switch_threshold = 200
    #### END OF CODE ADDED BY NICK FOR DRAFT 2 ####

    current_prompts = []
    curr_count = 0

    # Set hyperparameters for speculative decoding
    batch_size = 1
    max_new_tokens = 4 # Draft model generates max_new_tokens per iteration
    output_file = "demo_real_output.txt" # Change this to your output name
    #output_file = "real_test_multi_draft_6.txt" # Change this to your output name

    processed_batches = 0

    for batch in test_json:
        draft_model_ptr = 0
        data_time_start = time()

        # Constructing the prompt for each question
        current_prompt = batch['prompt']
        print("\n\nCurrent prompt: ", current_prompt)

        current_prompts.append(batch['prompt'])
        processed_batches += 1
        curr_count += 1
        if curr_count == batch_size:

            draft_input_ids = tokenizer.batch_encode_plus(current_prompts, padding='longest')
            current_prompts = []
            curr_count = 0

            if batch_size == 1:
                ground_truth_slice = ground_truth_tensor_list[processed_batches - 1]
                ground_truth_tensor = ground_truth_slice.unsqueeze(0).cuda(local_rank)
            else:
                ground_truth_slice = ground_truth_tensor_list[
                    (processed_batches - 1) * batch_size:processed_batches * batch_size]
                ground_truth_tensor = torch.stack(ground_truth_slice, dim=0).cuda(local_rank)

            max_length = ground_truth_tensor.size(1) - max_new_tokens - 2
            current_length = 0
            iter_count = 0

            total_matched = torch.zeros(batch_size, dtype=torch.int32).cuda(local_rank)

            draft_successes = 0
            while current_length < max_length:

                # The first iteration uses in the input prompt
                # The following iterations use input constructed from the last iteration based on matched tokens
                if iter_count == 0:
                    iter_count += 1

                    output_len = len(draft_input_ids["input_ids"][0]) + max_new_tokens

                    input_tensors = torch.tensor(draft_input_ids["input_ids"]).cuda(local_rank)
                else:
                    output_len = len(new_inputs[0]) + max_new_tokens
                    input_tensors = new_inputs
                    if batch_size == 1:
                        input_tensors.unsqueeze(0)

                if current_length / max_length * 100 >= drafter_switch_threshold:
                    #print(f"Current length is at {current_length} out of {max_length}. Switching drafters!")
                    draft_model_ptr = 1

                if draft_model_ptr == 0:
                    cat_tensor = draft_model.generate(input_tensors, max_new_tokens=max_new_tokens,
                                                  pad_token_id=tokenizer.pad_token_id)
                    '''
                    print("Input tensor size: ", input_tensor.size(1))
                    print("Using 1")
                    '''
                else:
                    '''
                    print("Input tensor size: ", input_tensor.size(1))
                    print("Using 2")
                    '''
                    cat_tensor = draft2_model.generate(input_tensors, max_new_tokens=max_new_tokens,
                                                  pad_token_id=tokenizer.pad_token_id)
                '''
                print("Draft output: ", draft_output)
                print("Draft output size: ", draft_output.size())
                '''

                # cat_tensor = draft_model.generate(input_tensors, max_new_tokens=max_new_tokens,
                #                                   pad_token_id=tokenizer.pad_token_id)

                next_token_id = cat_tensor[:, -max_new_tokens:]

                # Create a range tensor from 0 to max_new_tokens, which will be used to get a slice of length max_new_tokens+1
                range_tensor = torch.arange(0, max_new_tokens).unsqueeze(0).expand(total_matched.size(0), -1).cuda(
                    local_rank)

                # Add the start positions to the range tensor to get the actual indices
                indices = total_matched.unsqueeze(1) + range_tensor

                # Now use torch.gather to get the slices from ground_truth_tensor
                slices = torch.gather(ground_truth_tensor, 1, indices)

                correct_predictions = (next_token_id == slices)

                # Step 1: Convert the boolean tensor to float tensor
                correct_predictions_float = correct_predictions.float()

                # Step 2: Compute the cumulative sum
                cumsum = correct_predictions_float.cumsum(dim=1)

                # Step 3: Find the position of the first False (0) in each row
                # If there is no False in the row, the position will be set to the length of the row
                first_false_positions = torch.full((correct_predictions_float.size(0),),
                                                   correct_predictions_float.size(1),
                                                   device=correct_predictions_float.device)

                # Find the positions of all False values.
                false_positions = (correct_predictions_float == 0).nonzero(as_tuple=True)

                if len(false_positions[0]) > 0:
                    if false_positions[0][0] == 0:
                        draft_successes += 0
                    else:
                        draft_successes += (first_incorrect + 1)
                else:
                    draft_successes += max_new_tokens

                # Update first_false_positions with the first False position for each row.
                for row, col in zip(*false_positions):
                    first_false_positions[row] = min(first_false_positions[row], col)

                # Compute the number of matched tokens in a batch
                matched_tokens = first_false_positions + torch.ones_like(first_false_positions)

                input_list = []

                # Construct the input for the next iteration based on matched tokens in the current batch
                for idx, matched in enumerate(matched_tokens):
                    input_list.append(torch.cat((torch.zeros(torch.max(matched_tokens) - matched_tokens[idx],
                                                             dtype=torch.int32).cuda(local_rank),
                                                 input_tensors[idx],
                                                 ground_truth_tensor[idx][
                                                 total_matched[idx]: total_matched[idx] + matched_tokens[idx]]),
                                                dim=0))

                new_inputs = torch.stack(input_list)
                total_matched += matched_tokens

                if local_rank == 0:
                    with open(output_file, "a") as f:  # Replace with your file path
                        f.write(str(total_matched.tolist()) + str("\n"))

                current_length = min(total_matched)

        else:
            continue
        data_time_end = time()
        print("Draft successes: ", draft_successes)
        print(f"Prompt runtime = {data_time_end - data_time_start}")
        print("Final output: ", tokenizer.decode(new_inputs[0], skip_special_tokens=True))

