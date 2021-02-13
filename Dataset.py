import torch
import config

class BertBaseDataset:
    def __init__(self, contextLeft, targetIndex, contextRight = None,  isRight = False):
        self.contextLeft  = contextLeft
        self.contextRight = contextRight
        self.targetIndex  = targetIndex

        # Assigning tokenizer based on model being used
        if config.modelName == "SciBert":
            self.tokenizer = config.SCI_TOKENIZER
        else:
            self.tokenizer = config.BERT_TOKENIZER
        
        self.max_len      = config.INPUT_MAX_LEN
        self.isRight      = isRight
        '''
        if(self.isRight):
            self.max_len = (config.MAX_LEN*2)+3
        else:
            self.max_len = config.MAX_LEN+2
        '''


    # This dataset class is a Map-style dataset
    def __len__(self):
        return len(self.contextLeft)

    def __getitem__(self, item):
        contextLeft = str(self.contextLeft[item])
        contextLeft = " ".join(contextLeft.split())

        contextRight = None
        if self.isRight:
            contextRight = str(self.contextRight[item])
            contextRight = " ".join(contextRight.split())

        ''' [CLS] will be automatically added by the 'tokenizer'. If right context is taken, then [SEP] will be added between left
            and right contextes
        '''
        inputs  = self.tokenizer.encode_plus(
            contextLeft,
            contextRight,
            add_special_tokens = True,
            max_length = self.max_len,
            padding = "max_length",
            pad_to_max_length = True
        )

        ids            = inputs["input_ids"]
        mask           = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(self.targetIndex[item], dtype=torch.long)
        }