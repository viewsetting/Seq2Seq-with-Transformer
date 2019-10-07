SEED = 1
BATCH_SIZE = 64
MIN_FREQ = 2

HID_DIM = 256
N_LAYERS = 6
N_HEADS = 8
PF_DIM = 2048
DROPOUT = 0.1

N_EPOCH = 5
CLIP = 1
SAVE_DIR = 'best_model'
import os
MODEL_PATH = os.path.join(SAVE_DIR,'best_model seq2seq using transformer.pt')
