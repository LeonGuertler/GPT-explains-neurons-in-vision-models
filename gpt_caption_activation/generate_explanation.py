import numpy as np 
import random, torch 
import openai 

import utils, config 

# fix all seeds
np.random.seed(config.SEED)
random.seed(config.SEED)
torch.manual_seed(config.SEED)

# log into openai
openai.api_key = config.OPENAI_API_KEY

# clasification layer (torch specific name. In this case for AlexNet)
target_layer = "classifier.1"
target_neuron = 489

# load relevant data
caption_dict, activation_dict, neuron_min_max, few_shot_examples = utils.load_caption_activation_data(
    target_layer=target_layer
)
# load the individual scaling functions for each neuron (few shot + target)
scaling_dict = utils.load_scaling_functions(
    target_layer=target_layer,
    target_neuron=target_neuron,
    neuron_min_max=neuron_min_max,
)

explanation_score_list = [] 
for i in range(config.NUM_ITERS):
    print(f"[{i+1}/{config.NUM_ITERS}] for Neuron {target_neuron} of Layer {target_layer}")
    
    # generate the neuron explanation prompt
    llm_explanation_prompt = utils.get_llm_explanation_prompt(
        target_layer=target_layer,
        target_neuron=target_neuron,
        scaling_dict=scaling_dict,
        few_shot_examples=few_shot_examples,
        caption_dict=caption_dict,
        activation_dict=activation_dict,
    )

    # generate the explanation 
    llm_explanation = utils.generate_llm_explanation(
        prompt=llm_explanation_prompt,
    )

    # get the score of the explanation
    scores = []
    while len(scores) <= 2 or utils.check_CI(scores, verbose=True):
        # get the simulation prompt
        simulation_prompt, target_activations, target_descriptions = utils.get_simulation_prompt(
            llm_explanation=llm_explanation,
            target_layer=target_layer,
            target_neuron=target_neuron,
            few_shot_examples=few_shot_examples,
            activation_dict=activation_dict,
            scaling_dict=scaling_dict,
            caption_dict=caption_dict,
        )

        # simulate the neuron activations and get the correlation score
        try:
            correlation_score = utils.simulate_neuron_activations(
                simulation_prompt=simulation_prompt,
                target_activations=target_activations,
                target_descriptions=target_descriptions,
            )
        except ValueError:
            # this is thrown when the simulation prompt is too long (which should not happen often)
            continue

        scores.append(correlation_score)

    print(f"[{i+1}/{config.NUM_ITERS}] Neuron {target_neuron} Explanation: the main thing this neuron does is find {llm_explanation}")
    print(f"Average correlation score: {np.nanmean(scores)}")
    print(f"Standard deviation of correlation score: {np.nanstd(scores)}")

    explanation_score_list.append((np.nanmean(scores), llm_explanation))

# print sorted explanation_score_list
explanation_score_list.sort(key=lambda x: x[0], reverse=True)
for i, (score, explanation) in enumerate(explanation_score_list):
    print(f"{i+1}\t {explanation} ({score})")