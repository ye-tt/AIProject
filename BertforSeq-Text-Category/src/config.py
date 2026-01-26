from pathlib import Path


ROOT_DATA_DIR = Path(__file__).parent.parent
RAW_DATA_PATH = ROOT_DATA_DIR / "data" / "raw" / "online_shopping_10_cats.csv"
PROCESSED_DATA_DIR = ROOT_DATA_DIR / "data" / "processed"
LOGS_DIR = ROOT_DATA_DIR / "logs"
MODEL_DIR = ROOT_DATA_DIR / "models"
PRE_TRAINED_DIR = ROOT_DATA_DIR / "pretrained"

# 输入的序列长度
SEQ_LEN = 128
BATCH_SIZE = 16
EMBEDDING_SIZE = 128
HIDDEN_SIZE = 256
LEARN_RATE = 1e-5
EPOCHS = 100
MAX_SEQ_LEN = 128
