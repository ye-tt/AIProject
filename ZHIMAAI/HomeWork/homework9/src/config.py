import torch
from pathlib import Path

ROOT_DATA_DIR = Path(__file__).parent.parent
RAW_DATA_DIR= ROOT_DATA_DIR / "data"/"raw"
PROCESSED_DATA_DIR= ROOT_DATA_DIR /"data"/ "processed"
MODEL_DIR= ROOT_DATA_DIR / "model"
VOCAB_PATH= MODEL_DIR / "vocab.txt"

MAX_SEQ_LEN = 128
EMBEDED_DIM = 128
HIDDEN_DIM = 256
BATCH_SIZE = 64
LEARN_RATE = 0.001
EPOCHS = 20
NUK_TOKEN = "<nuk>"
PAD_TOKEN = "<pad>"
PAD_IDEX = 0
SOS_TOKEN = "<sos>"
ESO_TOKEN = "<eos>"
device = "cuda" if torch.cuda.is_available() else "cpu"