from transformers import *
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
INPUT_MAX_LEN = 128  # It was 50 before here. Now, 24*2+3 = 51 in total, before it would have been 50*2+3 = 103 which I feel is incorrect
SEQ_LENGTH = 60
TRAIN_BATCH_SIZE = 16
TEST_BATCH_SIZE  = 8
EPOCHS           = 30
BERT_PATH = "bert-base-uncased"
SCI_BERT_PATH = "allenai/scibert_scivocab_uncased"
BERT_TOKENIZER = BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
SCI_TOKENIZER  = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_cased", do_lower_case=True)
DROPOUT   = 0.1
WARMUP_PROPORTION = 0.1 # 10% of total
LEARNING_RATE     = 3e-5

# Project Specific
FREQUENCY = 5
YEAR = 2017
isRight = True

# Paths
MODEL_SAVED = "outputs/{}/model/state_dict_model.pt"
PREDICTIONS_PATH = "outputs/{}/predictions/final_results_and_targets.pkl"
METRICS_PATH = "outputs/{}/metric/metric.txt"

# Running Configurations
dotrain = True
modelName = "BertBase"