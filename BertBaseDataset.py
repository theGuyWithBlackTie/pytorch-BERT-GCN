import torch

class BertBaseDataset:
    def __init__(self, contextLeft, contextRight = None, targetIndex, isRight = False):
        self.contextLeft  = contextLeft
        self.contextRight = contextRight
        self.targetIndex  = targetIndex
        self.tokenizer    = config.TOKENIZER
        self.max_len      = config.MAX_LEN
        self.isRight      = isRight



    # This dataset class is a Map-style dataset
    def __len__(self):
        return self.contenxt

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
            add_special_tokens = True
            max_length = self.max_len
            padding = True
        )

        ids            = inputs["input_ids"]
        mask           = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return(
            "ids": torch.tensor(ids, dtype=torch.long)
            "mask": torch.tensor(mask, dtype=torch.long)
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long)
            "targets": torch.tensor(self.target[item], dtype=torch.float)
        )