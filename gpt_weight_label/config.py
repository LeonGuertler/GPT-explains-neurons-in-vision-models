OPENAI_API_KEY = "YOUR OPENAI KEY"
DATA_DIR = "data"

SEED = 0
NUM_ITERS = 10
ACCPETABLE_MARGIN = 5

# GPT special tokens 
GPT_START_TOKEN = "<start>"
GPT_END_TOKEN = "<end>"
GPT_SEP_TOKEN = "\t"
GPT_UNK_TOKEN = "unk"
GPT_VERSION = "gpt-3.5-turbo-0613"



# hyperparameters
NUM_FEW_EXP = 2
M_FEW_PRED_EXP = 25

NUM_FEW_SIM = 2 
M_FEW_SIM = 50
M_PRED_SIM = 50




# prompts
BASE_PROMPT_EXPLANATION = "We're studying neurons in a computer vision neural network. "+\
    "Each neuron looks for some particular thing in an image. "+\
    "Look at the classes that the neuron is positively contributing towards and summarize in a short text what the neuron is looking for (the key thing most of classes have in common). Don't list exmaples of classes. " +\
    "Keep in mind that these are images, so the information connecting them might not be related to the object itself, but rather to what can be found in images of the object (i.e. fish & water).\n"+\
    f"The activation format is token{GPT_SEP_TOKEN}weight. "+\
    "Activation values range from 0 to 10. A neuron finding what it's looking for is represented by a high positive value. The higher the value the stronger the match. "+\
    "You should not list objects, but rather try to find what they have in common."

BASE_PROMPT_SIMULATION = "We're studying neurons in a computer vision neural network. "+\
    "Each neuron looks for some particular thing in an image, and based on that influences the prediction probability for the available classes. "+\
    "Based on a short explanation of the neuron, you should try to predict the weight associated with each class." +\
    "The description need not be related to the objects themselves, but what might typically be found in the same image as the object (i.e. fish & water)." +\
    f"We present the classes as class-description{GPT_SEP_TOKEN}{GPT_UNK_TOKEN}, where it will be your task to predict weight inplace of the {GPT_UNK_TOKEN} tokens."+\
    "A neuron finding what it's looking for is represented by a high positive value. The higher the value the stronger the match.\n"
    
