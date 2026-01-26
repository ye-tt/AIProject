from pathlib import Path

ROOT_DATA_DIR = Path(__file__).parent.parent
# 路径
RAW_DATA_PATH = ROOT_DATA_DIR / "data" / "raw" / "cmn.txt"
PROCESSED_DATA_DIR = ROOT_DATA_DIR / "data" / "processed"
LOGS_DIR = ROOT_DATA_DIR / "logs"
MODEL_DIR = ROOT_DATA_DIR / "models"

# 训练参数
BATCH_SIZE = 64
LEARN_RATE = 1e-3
EPOCHS = 100
MAX_SEQ_LENGTH = 128

# 模型结构
DIM_MODEL = 128
N_HEAD = 4
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
