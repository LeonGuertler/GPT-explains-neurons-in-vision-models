import os
import json
import time
import random
import numpy as np
import scipy.stats as stats
import openai, tiktoken

import torch
import config 

from typing import List

def load_caption_activation_data(target_layer:str):
    # Loading JSON files for activation dictionary, caption dictionary, neuron_min_max, and few_shot_examples
    with open(os.path.join(config.DATA_DIR, "activation_dict.json"), "r") as f:
        activation_dict = json.load(f)
    
    with open(os.path.join(config.DATA_DIR, "caption_dict.json"), "r") as f:
        caption_dict = json.load(f)

    with open(os.path.join(config.DATA_DIR, "neuron_min_max.json"), "r") as f:
        neuron_min_max = json.load(f)

    with open(os.path.join(
        config.DATA_DIR, 
        f"few_shot_dict_{target_layer.replace('.', '_')}.json"
    ), "r") as f:
        few_shot_examples = json.load(f)

    return caption_dict, activation_dict, neuron_min_max, few_shot_examples



class ScaleActivations:
    def __init__(self, min:float, max:float):
        self.min = min
        self.max = max
    def __call__(self, activation:float) -> int:
        return int(np.round((activation - self.min) / (self.max - self.min) * 10, 0))
    

def load_scaling_functions(target_layer:str, target_neuron:int, neuron_min_max:dict) -> dict:
    # Creating a dictionary to scale activations
    scaling_dict = {key: ScaleActivations(
        min=neuron_min_max[target_layer][key]["min"],
        max=neuron_min_max[target_layer][key]["max"]
        ) for key in ['0', '1', '2', str(target_neuron)]
    }
    return scaling_dict

# Generate neuron explanation prompt
def get_neuron_explanation_prompt(neuron_idx: int, target_layer: str, activation_dict: dict, 
                                  scaling_dict: dict, caption_dict: dict) -> str:
    # Create a list of tuples each containing a key and corresponding activation value
    activation_list = [(key, activation_dict[key][target_layer][str(neuron_idx)]) for key in activation_dict.keys()]

    # Sort by activation
    activation_list = sorted(activation_list, key=lambda x: x[1], reverse=True)

    # Building a prompt string using the activations and their corresponding captions
    prompt = f"Neuron {neuron_idx}\nActivations:\n" + config.GPT_START_TOKEN + "\n"
    for key, _ in activation_list[:config.M_EXPLANATION][::-1]:
        caption = caption_dict[key]
        activation = activation_dict[key][target_layer][str(neuron_idx)]
        scaled_activation = scaling_dict[str(neuron_idx)](activation)
        prompt += f"{caption}{config.GPT_SEP_TOKEN}{scaled_activation}\n"
    prompt += config.GPT_END_TOKEN + "\n"
    prompt += f"Explanation of neuron {neuron_idx} behavior: the main thing this neuron does is find "
    return prompt


# Create LLM explanation prompt
def get_llm_explanation_prompt(target_layer: str, target_neuron: int, scaling_dict: dict, 
                               few_shot_examples: dict, caption_dict: dict, activation_dict: dict) -> List[dict]:
    # Initialize the system prompt 
    gpt_prompt = [{"role": "system", "content": config.EXP_BASE_PROMPT}]

    # Add few-shot examples to the prompt
    for few_shot_key in np.random.choice(list(few_shot_examples.keys()), config.NUM_FEW_SHOT_EXPLANATION, replace=False):
        few_shot_prompt = get_neuron_explanation_prompt(neuron_idx=few_shot_key, target_layer=target_layer, 
                                                        activation_dict=activation_dict, 
                                                        scaling_dict=scaling_dict, caption_dict=caption_dict)
        few_shot_reply = few_shot_examples[few_shot_key]["explanation"]
        gpt_prompt.extend([{"role": "user", "content": few_shot_prompt}, {"role": "assistant", "content": few_shot_reply}])

    # Add the target neuron prompt
    target_prompt = get_neuron_explanation_prompt(neuron_idx=target_neuron, target_layer=target_layer, 
                                                  activation_dict=activation_dict, 
                                                  scaling_dict=scaling_dict, caption_dict=caption_dict)
    gpt_prompt.append({"role": "user", "content": target_prompt})

    return gpt_prompt


# Generate LLM explanation 
def generate_llm_explanation(prompt: List[dict]) -> str:
    print("Generating LLM explanation...\n")
    while True:
        try:
            # Create a chat completion using OpenAI's GPT model
            response = openai.ChatCompletion.create(model=config.GPT_VERSION, messages=prompt, max_tokens=64, temperature=1.0, n=1)
            llm_explanation = response.choices[0].message.content
            break
        except openai.InvalidRequestError as e:
            if "This model's maximum input length" in str(e):
                raise ValueError(
                    "Input text is too long for Explanation Generation. Try reducing the number of few-shot examples or caption-activation pairs."
                )
            else:
                print(f"API call failed. Retrying... {e}")
                time.sleep(2)
        except Exception as e:
            print(f"API call failed. Retrying... {e}")
            time.sleep(2)

    print(f"LLM explanation: {llm_explanation}\n")
    return llm_explanation

# Check if the half width of the confidence interval is less than an acceptable margin
def check_CI(score_list, verbose=False):
    score_list = [s for s in score_list if not np.isnan(s)]
    sample_mean = np.mean(score_list)
    sample_std = np.std(score_list)
    sample_size = len(score_list)
    se = sample_std / np.sqrt(sample_size)
    ci = stats.t.interval(0.9, df=sample_size-1, loc=sample_mean, scale=se)
    half_width = stats.t.ppf(0.95, df=sample_size-1) * se  

    if verbose:
        print(f"Sample mean: {sample_mean}")
        print(f"Score list: {score_list}")
        print(f"95% confidence interval for the population mean: {ci}")
        print(f"Sample size: {sample_size}")
        print(f"Half width: {half_width}\n")

    return half_width >= config.ACCPETABLE_MARGIN


def get_neuron_assessment_prompt_top_50_50(
        explanation:str, 
        neuron_idx:int, 
        target_layer:str, 
        activation_dict:dict, 
        scaling_dict:dict, 
        caption_dict:dict
    ):
    """
    This 50% top positive activations and 50% zero activations set-up is only used
    for the few-shot examples.
    """
    prompt = f"Neuron {neuron_idx}\nExplanation of neuron {neuron_idx} "+\
        f"behavior: the main thing this neuron does is find {explanation}\nActivations:\n"
    prompt += config.GPT_START_TOKEN + "\n"
    activation_list, description_list = [], []
    activation_list_top = []
    # get all activation, key pairs
    for key in activation_dict.keys():
        activation_list_top.append(
            (key, activation_dict[key][target_layer][neuron_idx])
        )
    # sort the activation list
    activation_list_top = sorted(activation_list_top, key=lambda x: x[1], reverse=True)

    # get the zero list
    zero_list = []
    for key in activation_dict.keys():
        if activation_dict[key][target_layer][neuron_idx] == 0:
            zero_list.append(key)

    # join the two lists
    final_list = []
    for key, _ in activation_list_top[:int(config.M_ASSESSMENT*0.5)][::-1]:
        final_list.append(key)
    for key in np.random.choice(zero_list, int(config.M_ASSESSMENT*0.5), replace=False):
        final_list.append(key)

    # shuffle the joined list 
    random.shuffle(final_list)

    # get the actual prompt
    for key in final_list:
        # extract the caption and activation
        caption = caption_dict[key]
        activation = activation_dict[key][target_layer][neuron_idx]
        # scale activation
        scaled_activation = scaling_dict[neuron_idx](activation)
        
        prompt += f"{caption}{config.GPT_SEP_TOKEN}{config.GPT_UNK_TOKEN}\n"

        activation_list.append(scaled_activation)
        description_list.append(caption)

    prompt += config.GPT_END_TOKEN + "\n"
    return prompt , activation_list, description_list

def get_neuron_assessment_prompt(
        explanation:str, 
        neuron_idx:int, 
        target_layer:str, 
        activation_dict:dict, 
        scaling_dict:dict, 
        caption_dict:dict
    ):
    prompt = f"Neuron {neuron_idx}\nExplanation of neuron {neuron_idx} "+\
        f"behavior: the main thing this neuron does is find {explanation}\nActivations:\n"
    prompt += config.GPT_START_TOKEN + "\n"
    activation_list, description_list = [], []
    for key in np.random.choice(list(activation_dict.keys()), config.M_ASSESSMENT, replace=False):
        caption = caption_dict[key]
        activation = activation_dict[key][target_layer][str(neuron_idx)]
        # scale activation
        scaled_activation = scaling_dict[str(neuron_idx)](activation)
        
        prompt += f"{caption}{config.GPT_SEP_TOKEN}{config.GPT_UNK_TOKEN}\n"

        activation_list.append(scaled_activation)
        description_list.append(caption)

    prompt += config.GPT_END_TOKEN + "\n"
    return prompt , activation_list, description_list


def get_few_shot_assessment_reply(
        few_shot_activation:List, 
        few_shot_description:List
    )->str:
    reply = config.GPT_START_TOKEN + "\n"
    for weight, description in zip(few_shot_activation, few_shot_description):
        reply += f"{description}{config.GPT_SEP_TOKEN}{weight}\n"
    reply += config.GPT_END_TOKEN 
    return reply


def get_simulation_prompt(
        llm_explanation:str, 
        target_layer:str, 
        target_neuron:int, 
        few_shot_examples:dict, 
        activation_dict:dict, 
        scaling_dict:dict, 
        caption_dict:dict,
    ):
    # initialize with system prompt 
    gpt_prompt = [
        {"role": "system", "content":config.ASS_BASE_PROMPT}
    ]

    # get few-shot examples
    for few_shot_key in np.random.choice(
        list(few_shot_examples.keys()), 
        config.NUM_FEW_SHOT_ASSESSMENT, 
        replace=False
    ):
        few_shot_prompt, few_shot_activation, few_shot_description = get_neuron_assessment_prompt_top_50_50(
            explanation=few_shot_examples[few_shot_key]["explanation"],
            neuron_idx=few_shot_key,
            target_layer=target_layer,
            activation_dict=activation_dict,
            scaling_dict=scaling_dict,
            caption_dict=caption_dict,
        )

        few_shot_reply = get_few_shot_assessment_reply(
            few_shot_activation=few_shot_activation,
            few_shot_description=few_shot_description
        )

        gpt_prompt.extend(
            [
                {"role": "user", "content": few_shot_prompt},
                {"role": "assistant", "content": few_shot_reply}
            ]
        )

    # get final prompt
    final_prompt, target_activations, target_descriptions = get_neuron_assessment_prompt(
        explanation=llm_explanation,
        neuron_idx=target_neuron,
        target_layer=target_layer,
        activation_dict=activation_dict,
        scaling_dict=scaling_dict,
        caption_dict=caption_dict,
    )

    gpt_prompt.append(
        {"role": "user", "content": final_prompt}
    )

    return gpt_prompt, target_activations, target_descriptions

def generate_mock_reply(target_descriptions:list) -> str:
    reply = config.GPT_START_TOKEN + "\n"
    for description in target_descriptions:
        reply += f"{description}{config.GPT_SEP_TOKEN}10\n"
    reply += config.GPT_END_TOKEN + "\n"
    return reply


def extract_weights_from_simulation_response(
        pred_string:str, 
        target_activations:list, 
        target_descriptions:list
    ):
    # turn the pred_string into a dict of description/activation pairs
    pred_dict = {}
    for line in pred_string.split("\n"):
        if config.GPT_SEP_TOKEN in line:
            description, activation = line.split(config.GPT_SEP_TOKEN)
            pred_dict[description] = int(np.round(float(activation),0))
        else:
            print(f"No SEP token in line: {line}")

    # iterate over the target_activations/description pairs and extract the corresponding activation
    return_activations_true, return_activations_pred = [], []
    for description, activation in zip(target_descriptions, target_activations):
        if description not in pred_dict:
            # this is something that does not really tend to happend. However, if it does,
            # we just skip it
            print(f"Description {description} not in pred_dict")
            continue
        return_activations_true.append(activation)
        return_activations_pred.append(pred_dict[description])

    return return_activations_true, return_activations_pred


def compute_assessment_metric(activations_true:list, activations_pred:list) -> int:
    # convert both to pytorch tensors they currently are python lists
    activations_true = torch.tensor(activations_true, dtype=torch.float)
    activations_pred = torch.tensor(activations_pred, dtype=torch.float)

    # calculate the pearson correlation
    mean_true_activations = torch.mean(activations_true)
    mean_pred_activations = torch.mean(activations_pred)
    cov = torch.mean((activations_true - mean_true_activations) * (activations_pred - mean_pred_activations))
    std_true = torch.std(activations_true)
    std_pred = torch.std(activations_pred)
    if std_true == 0 or std_pred == 0:
        pearson = 0
    else:
        pearson = cov / (std_true * std_pred)

    return pearson * 100 # convert to percentage


def simulate_neuron_activations(
        simulation_prompt:List[dict], 
        target_activations:list, 
        target_descriptions:list
    ):
    print("Simulating activations...\n")
    # get a mock reply to estimate the number of tokens required
    mock_reply = generate_mock_reply(target_descriptions)
    # estimate the number of reply tokens
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    estimated_max_tokens = len(encoding.encode(mock_reply))

    # use gpt to simulate activations
    while True:
        try:
            response = openai.ChatCompletion.create(
                model=config.GPT_VERSION,
                messages=simulation_prompt,
                max_tokens=estimated_max_tokens,
                temperature=0.0,
                n=1,
            )
            llm_reply = response.choices[0].message.content
            break
        except openai.InvalidRequestError as exc:
            if "This model's maximum context" in str(exc):
                raise ValueError(
                        "Input text is too long for Explanation Assessment. "+\
                        "Try reducing the number of few-shot examples or caption-activation pairs."
                    )
        except Exception as exc:
            print(f"API call failed. Retrying... {exc}")
            time.sleep(2)
            continue
    
    # extract the actual weights from the output
    activations_true, activations_pred = extract_weights_from_simulation_response(
        pred_string=llm_reply,
        target_activations=target_activations,
        target_descriptions=target_descriptions,
    )

    # calculate the correlation score
    score = compute_assessment_metric(
        activations_true=activations_true, 
        activations_pred=activations_pred
    )

    return score