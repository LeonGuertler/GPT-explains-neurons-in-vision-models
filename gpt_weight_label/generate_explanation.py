import numpy as np 
import torch, torchvision 
import random, time 

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

# wrap the vision model to easily extract weights
class WrappedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.alexnet(pretrained=True)

    def get_weights(self, target_layer:str):
        if target_layer == "classifier.4":
            return self.model.classifier[6].weight.detach()
        elif target_layer == "classifier.1":
            return self.model.classifier[6].weight.detach() @ self.model.classifier[4].weight.detach()

model = WrappedModel()

# load few-shot examples
few_shot_dict = utils.load_few_shot_dict(target_layer=target_layer)

# load the class names 
class_names = utils.load_class_names()


explanation_score_list = []
for i in range(config.NUM_ITERS):
    print(f"[{i+1}/{config.NUM_ITERS}] for Neuron {target_neuron} of Layer {target_layer}")

    # get the neuron weights 
    weights = model.get_weights(target_layer=target_layer).transpose(-1, 0)[target_neuron]

    # generate the neuron explanation prompt
    llm_explanation_prompt, scaled_weights = utils.get_llm_explanation_prompt(
        neuron_idx=target_neuron,
        weights=weights,
        class_names=class_names,
        few_shot_dict=few_shot_dict,
    )

    # generate the explanation
    llm_explanation = utils.generate_llm_explanation(
        prompt=llm_explanation_prompt,
    )

    # get the score of the explanation
    scores = []
    while len(scores) <= 2 or utils.check_CI(scores, verbose=True):
        # get the simulation prompt 
        simulation_prompt, weights_subset, class_names_subset = utils.get_simulation_prompt(
            llm_explanation=llm_explanation,
            neuron_idx=target_neuron,
            scaled_weights=scaled_weights,
            class_names=class_names,
            few_shot_dict=few_shot_dict,
        )

        # simulate the neuron weights and get the correlation score
        try:
            correlation_score = utils.simulate_neuron_weights(
                simulation_prompt=simulation_prompt,
                target_weights=weights_subset,
                target_class_names=class_names_subset,
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