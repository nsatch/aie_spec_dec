##### CODE ADAPTED FROM: https://github.com/uw-mad-dash/decoding-speculative-decoding/blob/main/spec_decoding_deployment.py #####

import torch
import torch.nn as nn
import re
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import deepspeed
import torch.distributed as dist
import json
import os

### Supppress pad and eos warnings
import warnings
warnings.filterwarnings('ignore')  # Suppress all other warnings
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  # Suppress transformer warnings


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

def oracle_verification(model, input_tensor, draft_output, max_new_tokens, local_rank, iter_count, past_key_values):
    """
    Verifies the predictions of an oracle model by comparing the generated tokens against actual tokens.

    Args:
        model (torch.nn.Module): The oracle model used for generating predictions.
        input_tensor (torch.Tensor): The input tensor containing tokens for prediction.
        max_new_tokens (int): The maximum number of new tokens to be generated by the model.
        local_rank (int): The local rank identifier for distributed training.
        iter_count (int): The current iteration count, used to determine the first call to the model.
        past_key_values (torch.Tensor): Cached past key values for accelerating generation in subsequent calls.

    Returns:
        tuple: A tuple containing:
               - The positions of the first incorrect predictions in each row.
               - The updated past_key_values tensor with dimensions adjusted based on the first incorrect prediction.
    """
    # print()
    # print("ORACLE VERIFICATION FUNCTION")
    # print("input tensor: ", input_tensor)
    # print(input_tensor.size())
    # print("draft guess: ", draft_output)
    # print(draft_output.size())

    for i in range(max_new_tokens):
        target_model_guess = model.generate(input_ids=input_tensor, max_new_tokens=1)
        # print("target model guess: ", target_model_guess)
        # print(target_model_guess.size())
        # print("target model guess: ", tokenizer.decode(target_model_guess[0], skip_special_tokens=True))

        curr_draft_guess = draft_output[0][-(max_new_tokens-i)].item()
        curr_target_guess = target_model_guess[0][-1].item()

        # print("draft token: ", curr_draft_guess)
        # print("target token: ", curr_target_guess)

        if curr_draft_guess != curr_target_guess:
            # print("first incorrect index: ", i)
            return i, target_model_guess

        input_tensor = target_model_guess
    
    return



if __name__ == "__main__":
    dist.init_process_group(backend='nccl')
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    print(world_size, rank, local_rank)
    os.environ['TRANSFORMERS_CACHE'] = "cache"  # Replace with your transformers cache directory

    test_json = json_loader("./hellaswag_shortened.json")  # Replace with your file path

    # Define the checkpoint dict. You may need to convert *.safetensors to
    # *.bin for this work. Make sure you get all the *.bin and *.pt files in
    # the checkpoint_files list.
    checkpoint_dir = "ckpt"

    # Change ckpt names if your .bin files are named differently
    checkpoint_files = [
        os.path.join(checkpoint_dir, f"pytorch_model-{i:05d}-of-00029.bin")
        for i in range(1, 30)  # Change number of bin files based on your model
    ]

    checkpoint_dict = {
        "type": "DS_MODEL",
        "checkpoints": checkpoint_files,
        "version": 1.0,
    }

    oracle_model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-3B', torch_dtype=torch.float16)

    oracle_model = deepspeed.init_inference(
        oracle_model,
        replace_with_kernel_inject=False,
        # tp={"tp_size": tensor_parallel_degrees, },
        tp={"tp_size": world_size, },
        dtype=torch.float16,
        checkpoint=checkpoint_dict,
    )

    # The LLaMA tokenizer does not have a pad token.
    # Modify the tokenizer to add a pad token and change the model configs accordingly.
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B', torch_dtype=torch.float16)

    # Feel free to change it to the draft model of your choice
    draft_model = AutoModelForCausalLM.from_pretrained("minghaoyan/Wide-Sheared-LLaMA-796M", torch_dtype=torch.float16)
    draft_model.resize_token_embeddings(len(tokenizer))


    # Launch the draft model with deepspeed on 1 node. Alternatively, you could use HF or load from a checkpoint.
    draft_model = deepspeed.init_inference(
        draft_model,
        replace_with_kernel_inject=False,
        tp={"tp_size": 1, },
        dtype=torch.float16,
        # checkpoint=checkpoint_dict,
    )

    # Set hyperparameters for speculative decoding
    max_tokens = 7
    output_file = "3b_796m_shortened.txt"
    past_key_values = None

    iteration = 0
    for data in test_json:
        # if iteration == 0:
        current_prompt = data["prompt"]

        print("Current prompt: ", current_prompt)

        draft_input_ids = tokenizer.encode(current_prompt, return_tensors="pt").cuda(local_rank)

        # print("Number of draft ids: ", draft_input_ids.size())
        # print("Draft input ids: ", draft_input_ids)

        # Calculating the maximum length for the generated sequence
        max_length = 200 - max_tokens - 2
        current_length = 0
        iter_count = 0

        # Intialize tensor to keep track of total matched tokens
        total_matched = torch.zeros(1, dtype=torch.int32).cuda(local_rank)

        draft_successes = 0

        while current_length < max_length:
            # print("Iteration: ", current_length, max_length)

            if iter_count == 0: 
                # For the first iteration, use the input prompt
                iter_count += 1
                input_tensor = draft_input_ids
            else: 
                # For subsequent iterations, use new inputs based on matched tokens
                input_tensor = new_inputs

            # print("Input tensor size: ", input_tensor.size(1))
            # print("Input tensor: ", input_tensor)

            output_len = input_tensor.size(1) + max_tokens

            # Generate predictions
            draft_output = draft_model.generate(input_tensor, max_new_tokens=max_tokens).to(dtype=torch.int32)
            # print("Draft output: ", draft_output)
            # print("Draft output size: ", draft_output.size())

            # print("Decode draft output: ", tokenizer.decode(draft_output[0], skip_special_tokens=True))


            # Verifying the generated sequence against the ground truth
            # first_false_position, past_key_values = oracle_verification(
            #     oracle_model, input_tensor, draft_output, max_tokens, local_rank, iter_count, past_key_values
            # )
            first_false_position, updated_input = oracle_verification(
                oracle_model, input_tensor, draft_output, max_tokens, local_rank, iter_count, past_key_values
            )

            if first_false_position > 0:
                draft_successes += 1

            matched_tokens = first_false_position + 1
            
            # matched_tokens = first_false_position
            matched_tokens = first_false_position + 1
            # print("Matched tokens: ", matched_tokens)

            new_inputs = updated_input

            # new_inputs = torch.cat([input_tensor[:, :matched_tokens], generated_tokens[:, :matched_tokens]], dim=-1)

            # print("New inputs: ", new_inputs)
            # print(new_inputs.size())

            total_matched += matched_tokens
            # print("Total matched: ", total_matched)

            # Save results
            if local_rank == 0:
                with open(output_file, "a") as f:
                    f.write(str(total_matched.tolist()) + "\n")

            current_length = total_matched.item()

        print("Draft successes: ", draft_successes)
        print("Final output: ", tokenizer.decode(new_inputs[0], skip_special_tokens=True))
        print(new_inputs.size())
        print()
