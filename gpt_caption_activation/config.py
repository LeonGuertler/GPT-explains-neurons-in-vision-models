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
NUM_FEW_SHOT_EXPLANATION = 2
M_EXPLANATION = 75

NUM_FEW_SHOT_ASSESSMENT = 2
M_ASSESSMENT = 50



# prompts
EXP_BASE_PROMPT = "We\'re studying neurons in a neural network. "+\
    "Each neuron looks for some particular thing in an image. "+\
    "Look at the descriptions of images the neuron activates for and summarize in a single sentence what the neuron is looking for. "+\
    "Don\'t list examples of words.\n" + \
    f"The activation format is description{GPT_SEP_TOKEN}activation. "+\
    "Activation values range from 0 to 10. "+\
    "A neuron finding what it's looking for is represented by a non-zero activation value. "+\
    "The higher the activation value, the stronger the match.\n"

NON_ZERO_INTRO = "Same activations, but with all zeros filtered out:\n"



ASS_BASE_PROMPT = "We\'re studying neurons in a neural network. "+\
    "Each neuron looks for some particular thing in an image. "+\
    "Look at an explanation of what the neuron does, and try to predict how it will fire for each image description.\n"+\
    f"The activation format is description{GPT_SEP_TOKEN}activation, activations go from 0 to 10, {GPT_UNK_TOKEN} indicates an unknown activation. "+\
    "Most activations will be 0.\n"
