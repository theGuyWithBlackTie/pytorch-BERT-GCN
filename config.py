import transformers
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MAX_LEN = 50
TRAIN_BATCH_SIZE = 16
TEST_BATCH_SIZE  = 8
EPOCHS           = 30
BERT_PATH = "bert-base-uncased"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
DROPOUT   = 0.1

# Project Specific
FREQUENCY = 5
YEAR = 2017
isRight = True

# Paths
MODEL_SAVED = '/outputs/PeerRead/model/state_dict_model.pt'
PREDICTIONS_PATH = '/outputs/PeerRead/predictions/final_results_and_targets.pkl'
METRICS_PATH = '/outputs/PeerRead/metric/metric.txt'