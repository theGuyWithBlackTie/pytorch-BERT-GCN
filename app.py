import torch
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import engine
import config
import utils
import model

def run():
    trainDataset, testDataset, labelGenerator = utils.loadDataset()

    # Making DataLoaders
    trainDataLoader = torch.utils.data.DataLoader(trainDataset, batch_size=config.TRAIN_BATCH_SIZE,shuffle=True, num_workers=1)
    testDataLoader  = torch.utils.data.DataLoader(testDataset, batch_size=config.TEST_BATCH_SIZE, num_workers=1)

    totalNOsOfLabels = len(labelGenerator.classes_)

    device = torch.device(config.DEVICE)
    citemodel = model.BERTBaseUncased(numOfLabels=totalNOsOfLabels, dropout=config.DROPOUT )
    citemodel.to(device)

    param_optimizer = list(citemodel.named_parameters())
    no_decay        = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },

        {
             "params": [
                 p for n, p in param_optimizer if any(nd in n for nd in no_decay)
             ],
             "weight_decay": 0.0,
        }
    ]

    num_train_steps = int( len(trainDataLoader) * config.EPOCHS )
    optimizer       = AdamW(optimizer_parameters, lr=3e-5)
    scheduler       = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    best_accuracy = 0
    for epoch in range(config.EPOCHS):
        engine.train(trainDataLoader, citemodel, optimizer, device, scheduler)

if __name__ == "__main__":
    run()