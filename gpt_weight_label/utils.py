import numpy as np 
from scipy import stats
import torch
import pickle, random, re, time, os

import tiktoken, openai

import config

from typing import List


def load_few_shot_dict(target_layer):
    with open(os.path.join(
        config.DATA_DIR, 
        f"explanation_dict_{target_layer.replace('.', '_')}.pkl"
    ), "rb") as f:
        few_shot_dict = pickle.load(f)
    return few_shot_dict

def load_class_names():
    # load the list of class names 
    class_names = []
    with open(os.path.join(
        config.DATA_DIR,
        'imagenet_class_idx.txt'
    ), 'r') as file:
        for line in file:
            match = re.search(r'\'(.*)\'|"(.*)"', line)
            if match:
                # Pick whichever group that matched
                name = match.group(1) if match.group(1) else match.group(2)
                class_names.append(name)
    return class_names


def scale_weights_pos(weights:torch.tensor) -> torch.tensor:
    # separate positive and negative parts
    pos_tensor = torch.clamp(weights, min=0)
    neg_tensor = torch.clamp(weights, max=0)

    # find max absolute values
    max_pos = torch.max(pos_tensor)

    # normalize
    if max_pos > 0:
        pos_tensor = pos_tensor / max_pos

    # set all negative weights to 0
    neg_tensor = torch.where(
        neg_tensor < 0,
        torch.zeros_like(neg_tensor),
        neg_tensor
    )

    # combine
    result_tensor = pos_tensor + neg_tensor
    return torch.round(result_tensor*10)

def get_weight_class_name_prompt(weights:torch.tensor, class_names:list) -> str:
    """
    Sort the list in increasing order by weights and include the top n pairs
    """
    sorted_weights, sorted_class_names = zip(*sorted(zip(weights, class_names)))


    prompt = config.GPT_START_TOKEN + "\n"
    for activation, class_name in zip(
        sorted_weights[-config.M_FEW_PRED_EXP:], 
        sorted_class_names[-config.M_FEW_PRED_EXP:]
    ):
        if activation <= 0:
            prompt += f"{class_name}{config.GPT_SEP_TOKEN}{0}\n"
        else:
            prompt += f"{class_name}{config.GPT_SEP_TOKEN}{int(activation)}\n"
    prompt += config.GPT_END_TOKEN + "\n"

    return prompt 

def few_shot_prompt_exp(few_shot_dict:dict) -> str:
    # select a random few-shot example
    idx = np.random.choice(len(few_shot_dict), size=1, replace=False)[0]
    neuron_description = few_shot_dict[idx]["neuron_description"]
    weights = few_shot_dict[idx]["weights"]
    class_name = few_shot_dict[idx]["class_names"]

    few_shot_prompt = f"Neuron {idx}\nActivations:\n"
    few_shot_prompt += get_weight_class_name_prompt(weights, class_name)
    few_shot_prompt += f"\nExplanation of Neuron {idx} behavior: the main thing this neuron does is find"

    # the first part is part of the actual prompt, so remove it here if necessary
    return few_shot_prompt, neuron_description.replace(
        "the main thing this neurond does is find", ""
    )

def get_llm_explanation_prompt(neuron_idx:int, weights:torch.tensor, class_names:str, few_shot_dict:dict):
    # set up the system prompt
    gpt_prompt = [
        {"role": "system", "content": config.BASE_PROMPT_EXPLANATION}
    ]

    # add the few-shot examples
    for i in range(config.NUM_FEW_EXP):
        prompt, reply = few_shot_prompt_exp(
            few_shot_dict=few_shot_dict
        )
        gpt_prompt.extend(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": reply}
            ]
        )

    # scale the weights between 0 and 10 
    scaled_weights = scale_weights_pos(weights)

    # get the weight/class name pairs prompt
    weight_class_name_prompt = get_weight_class_name_prompt(
        weights=scaled_weights,
        class_names=class_names
    )

    # add the prompt for the actual neuron 
    target_prompt = f"Neuron {neuron_idx}\nActivations:\n"
    target_prompt += weight_class_name_prompt
    target_prompt += f"\nExplanation of Neuron {neuron_idx} behavior: the main thing this neuron does is find "

    gpt_prompt.append(
        {"role": "user", "content": target_prompt}
    )

    return gpt_prompt, scaled_weights
    

def generate_llm_explanation(prompt:List[dict]) -> str:
    print("Generating explanation...\n")
    while True:
        try:
            response = openai.ChatCompletion.create(
                model=config.GPT_VERSION,
                messages=prompt,
                max_tokens=64,
                temperature=1.0,
                n=1,
            )
            llm_explanation = response.choices[0].message.content
            break
        except Exception as exc:
            print(f"API call failed. Retrying... {exc}")
            time.sleep(2)
            continue
    print(f"LLM explanation: {llm_explanation}\n")
    return llm_explanation

def check_CI(score_list, verbose=False):
    score_list = [s for s in score_list if not np.isnan(s)]
    sample_mean = np.mean(score_list)
    sample_std = np.std(score_list)
    sample_size = len(score_list)

    # Compute the standard error
    se = sample_std / np.sqrt(sample_size)

    # Compute the 95% confidence interval
    ci = stats.t.interval(0.9, df=sample_size-1, loc=sample_mean, scale=se)
    half_width = stats.t.ppf(0.95, df=sample_size-1) * se  

    if verbose:
        print(f"Sample mean: {sample_mean}")
        print(f"Score list: {score_list}")
        print(f"95% confidence interval for the population mean: {ci}")
        print(f"Sample size: {sample_size}")
        print(f"Half width: {half_width}\n")

    if half_width < config.ACCPETABLE_MARGIN:
        return False 
    return True

def get_simulation_options_few_shot(weights:torch.tensor, class_names:List[str]):
    """
    For the few-shot examples of the GPT (Weight-Label) version we use 75% randomly
    selected non-zero weights and 25% randomly selected zero weights.
    """
    weights_subset, class_names_subset = [], [] 

    # shuffle the weights and class names
    tmp = list(zip(weights, class_names))
    random.shuffle(tmp)
    weights, class_names = zip(*tmp)
    weights, class_names = list(weights), list(class_names)

    # first, create a list of non-zero weights and a list of zero weights
    weights_subset_nonzero, weights_subset_zero = [], []
    class_names_subset_nonzero, class_names_subset_zero = [], []

    for weight, class_name in zip(weights, class_names):
        if weight > 0:
            weights_subset_nonzero.append(weight)
            class_names_subset_nonzero.append(class_name)
        else:
            weights_subset_zero.append(weight)
            class_names_subset_zero.append(class_name)
    
    # no need to shuffle them because we shuffled the base list

    # populate the prompt list 
    qrt = config.M_FEW_SIM//4 
    for weight, class_name in zip(
        weights_subset_nonzero[:(config.M_FEW_SIM-qrt)], 
        class_names_subset_nonzero[:(config.M_FEW_SIM-qrt)]
    ):
        weights_subset.append(weight)
        class_names_subset.append(class_name)

    for weight, class_name in zip(
        weights_subset_zero[:qrt], 
        class_names_subset_zero[:qrt]
    ):
        weights_subset.append(weight)
        class_names_subset.append(class_name)

    # shuffle the subsets 
    tmp = list(zip(weights_subset, class_names_subset))
    random.shuffle(tmp)
    weights_subset, class_names_subset = zip(*tmp)
    weights_subset, class_names_subset = list(weights_subset), list(class_names_subset)

    # now create the actual prompt 
    prompt = config.GPT_START_TOKEN + "\n"
    for activation, class_name in zip(weights_subset, class_names_subset):
        prompt += f"{class_name}{config.GPT_SEP_TOKEN}{config.GPT_UNK_TOKEN}\n"
    prompt += config.GPT_END_TOKEN + "\n"
    return prompt, weights_subset, class_names_subset

def get_simulation_options_target(weights:torch.tensor, class_names:List[str]):
    weights_subset, class_names_subset = [], [] 

    # shuffle both lists 
    tmp = list(zip(weights, class_names))
    random.shuffle(tmp)
    weights, class_names = zip(*tmp)
    weights, class_names = list(weights), list(class_names)

    # get a random subset
    for weight, class_name in zip(
        weights[-config.M_FEW_SIM:], 
        class_names[-config.M_FEW_SIM:]
    ):
        weights_subset.append(weight)
        class_names_subset.append(class_name)

    # now create the actual prompt
    prompt = config.GPT_START_TOKEN + "\n"
    for activation, class_name in zip(weights_subset, class_names_subset):
        prompt += f"{class_name}{config.GPT_SEP_TOKEN}{config.GPT_UNK_TOKEN}\n"
    prompt += config.GPT_END_TOKEN + "\n"
    return prompt, weights_subset, class_names_subset

def get_simulation_prompt(
        llm_explanation:str,
        neuron_idx:int,
        scaled_weights:torch.tensor,
        class_names:List[str],
        few_shot_dict:dict
    ):
    print("Assessing explanation...\n")

    # initialize the prompt 
    simulation_prompt = [
        {"role": "system", "content": config.BASE_PROMPT_SIMULATION}
    ]

    # get few-shot examples
    for idx in np.random.choice(len(few_shot_dict), size=config.NUM_FEW_SIM, replace=False):
        neuron_description = few_shot_dict[idx]["neuron_description"].replace("the main thing this neuron does is find", "")
        weights = few_shot_dict[idx]["weights"]
        fs_class_name = few_shot_dict[idx]["class_names"]

        question_prompt = ""
        question_prompt += f"Neuron {idx}\n" 
        question_prompt += f"Explanation of neuron {idx}: the main thing this neuron does is find {neuron_description}\n"
        question_prompt += f"Activations:\n"
        fs_prompt, fs_weights, fs_class_names = get_simulation_options_few_shot(
            weights=weights, 
            class_names=fs_class_name
        )
        question_prompt += fs_prompt + "\n"

        reply_prompt = ""
        reply_prompt += config.GPT_START_TOKEN + "\n"
        for w, c in zip(fs_weights, fs_class_names):
            if w <= 0:
                reply_prompt += f"{c}{config.GPT_SEP_TOKEN}{0}\n"
            else:
                reply_prompt += f"{c}{config.GPT_SEP_TOKEN}{int(w)}\n"
        reply_prompt += config.GPT_END_TOKEN
        simulation_prompt.extend(
            [
                {"role": "user", "content": question_prompt},
                {"role": "assistant", "content": reply_prompt}
            ]
        )

    option_prompt, weights_subset, class_names_subset = get_simulation_options_target(
        weights=scaled_weights,
        class_names=class_names,
    )
    target_prompt = ""
    target_prompt += f"Neuron {neuron_idx}\n"
    target_prompt += f"Explanation of neuron {neuron_idx}: the main thing this neuron does is find {llm_explanation}\n"
    target_prompt += f"Activations:\n"
    target_prompt += option_prompt + "\n"
    simulation_prompt.append(
        {"role": "user", "content": target_prompt}
    )

    return simulation_prompt, weights_subset, class_names_subset


def estimate_tokens(class_names_subset:List[str]) -> int:
    """
    Create a mock reply to estimate the number of tokens 
    required.
    """
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

    mock_reply = ""
    mock_reply += "Neuron 0\nActivations:\n"
    mock_reply += config.GPT_START_TOKEN
    for i in range(len(class_names_subset)):
        mock_reply += f"{class_names_subset[i]}{config.GPT_SEP_TOKEN}10\n"
    mock_reply += config.GPT_END_TOKEN

    return len(encoding.encode(mock_reply))


def extract_weights_from_simulation_response(
        pred_string:str, 
        target_weights:list, 
        target_class_names:list
    ):
    # turn the pred_string into a dict of description/activation pairs
    pred_dict = {}
    for line in pred_string.split("\n"):
        if config.GPT_SEP_TOKEN in line:
            class_name, weight = line.split(config.GPT_SEP_TOKEN)
            pred_dict[class_name] = int(np.round(float(weight),0))
        else:
            print(f"No SEP token in line: {line}")

    # iterate over the target_activations/description pairs and extract the corresponding activation
    return_weights_true, return_weights_pred = [], []
    for class_name, weight in zip(target_class_names, target_weights):
        if class_name not in pred_dict:
            # this is something that does not really tend to happend. However, if it does,
            # we just skip it
            print(f"Description {class_name} not in pred_dict")
            continue
        return_weights_true.append(weight)
        return_weights_pred.append(pred_dict[class_name])

    return return_weights_true, return_weights_pred

def compute_assessment_metric(weights_true:list, weights_pred:list) -> int:
    # convert both to pytorch tensors they currently are python lists
    weights_true = torch.tensor(weights_true, dtype=torch.float)
    weights_pred = torch.tensor(weights_pred, dtype=torch.float)

    # calculate the pearson correlation
    mean_true_activations = torch.mean(weights_true)
    mean_pred_activations = torch.mean(weights_pred)
    cov = torch.mean((weights_true - mean_true_activations) * (weights_pred - mean_pred_activations))
    std_true = torch.std(weights_true)
    std_pred = torch.std(weights_pred)
    if std_true == 0 or std_pred == 0:
        pearson = 0
    else:
        pearson = cov / (std_true * std_pred)

    return pearson * 100 # convert to percentage



def simulate_neuron_weights(
        simulation_prompt:List[dict], 
        target_weights:list, 
        target_class_names:list
    ):
    # estimate the number of tokens in the reply
    estimated_max_tokens = estimate_tokens(target_class_names)

    # try to pass it through GPT
    while True:
        try:
            response = openai.ChatCompletion.create(
                model=config.GPT_VERSION,
                messages=simulation_prompt,
                max_tokens=estimated_max_tokens,
                temperature=0.0,
                n=1,
            )
            
            too_long=False
            break
        except openai.InvalidRequestError as exc:
            if "This model's maximum context" in str(exc):
                raise ValueError(
                        "Input text is too long for Explanation Simulation. "+\
                        "Try reducing the number of few-shot examples or caption-activation pairs."
                    ) 
        except Exception as exc:
            print(f"API call failed. Retrying... {exc}")
            time.sleep(2)
            continue

    # post-process the output 
    weights_true, weights_pred = extract_weights_from_simulation_response(
        pred_string=response.choices[0].message.content,
        target_weights=target_weights,
        target_class_names=target_class_names,
    )

    return compute_assessment_metric(
        weights_true=weights_true,
        weights_pred=weights_pred
    )