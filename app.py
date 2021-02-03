import os
import pickle
import torch
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import engine
import config
import utils
import model
import argparse

def run():
    trainDataset, testDataset, labelGenerator = utils.loadDataset()

    # Making DataLoaders
    trainDataLoader = torch.utils.data.DataLoader(trainDataset, batch_size=config.TRAIN_BATCH_SIZE,shuffle=True, num_workers=4)
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
    
    if config.dotrain:
        print('In Training')
        exit()
        for epoch in range(config.EPOCHS):
            loss = engine.train(trainDataLoader, citemodel, optimizer, device, scheduler)
            print("Epoch: ", epoch, " Loss: ",loss,'\n')


        # Saving the model
        os.makedirs(os.path.dirname(config.MODEL_SAVED), exist_ok=True)
        torch.save(citemodel.state_dict(), config.MODEL_SAVED)
        print('Model is saved at: ',config.MODEL_SAVED)

    '''
     Evaluating the model
    '''

    print("Loading the model")
    exit()
    #citemodel = model.BERTBaseUncased(*args, **kwargs)
    citemodel.load_state_dict(torch.load(config.MODEL_SAVED))
    outputs, targets = engine.eval(trainDataLoader,citemodel,device)
    
    # Saving the results with corresponding targets
    os.makedirs(os.path.dirname(config.PREDICTIONS_PATH), exist_ok=True)
    with open(config.PREDICTIONS_PATH, 'wb') as f:
        pickle.dump(outputs, f) # First saved the predicted outputs
        pickle.dump(targets, f) # Then saved the corresponding targets

    print('Starting Evaluation...')
    utils.metric(outputs,targets)

    

    




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--do-train', dest="doTrain", required=True, type=str, help="Enter True or False")
    args   = parser.parse_args()

    if args.doTrain.lower() == "True".lower():
        config.dotrain = True
    else:
        config.dotrain = False

    run()