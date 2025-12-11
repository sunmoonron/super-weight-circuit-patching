MODEL_NAME = "allenai/OLMo-1B-0724-hf"

# Checkpoint revision control variable
REVISION = None

DEVICE = "cpu" # Because too poor MPS :(
DTYPE = "float32" # Broke budget type accomodations

# Coordinates from apple paper, maps directly to HF model.layers[1]
SUPERWEIGHTS = [
    {"layer": 1, "row": 1764, "col": 1710}, # Only use this value for our main experiments
    {"layer": 1, "row": 1764, "col": 8041},
]

# Define safe tensor paths
BASE_PATH = "base.safetensors"
BROKEN_PATH = "broken.safetensors"
PATCH_PATH = "patch.safetensors"
PATCHED_MERGED_PATH = "patched_merged.safetensors" # This showcases our findings!

# Define dataset / trainign settings 
MAX_LEN = 128
BATCH_SIZE = 2
NUM_EPOCHS = 2
LEARNING_RATE = 1e-3

# Simple prompts for behaviour-tests
TEST_PROMPTS = [
    "Summer is hot. Winter is",
    "Paris is in France. Tokyo is in",
    "2, 4, 6, 8,",
    "The capital of Canada is",
    "He opened the quote \" and then wrote Hello. To close it he should type",
    "List of numbers: [1, 2, 3",
    "In Python, to close a string started with ' you end with ",
    "The opposite of big is",
    "A cat is an animal. A rose is a",
    "Complete the analogy: day : night :: up :",
]