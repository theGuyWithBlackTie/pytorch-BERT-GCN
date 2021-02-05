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
    
    # Defining Model
    citeModel = None
    if config.modelName == "BertBase":
        citemodel = model.BERTBaseUncased(numOfLabels=totalNOsOfLabels, dropout=config.DROPOUT )
    elif config.modelName == "SciBert":
        citemodel = model.SciBertUncased(numOfLabels=totalNOsOfLabels, dropout=config.DROPOUT)
    citemodel.to(device)

    param_optimizer = list(citemodel.named_parameters())

    '''
        There is generally no need to apply L2 penalty (i.e. weight decay) to biases and LayerNorm.weight. 
        Hence, we have following line.
    '''
    no_decay        = ["bias", "LayerNorm.weight"] # Removed "LayerNorm.bias",

    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01, # changed this from 0.001 to 0.1
        },

        {
             "params": [
                 p for n, p in param_optimizer if any(nd in n for nd in no_decay)
             ],
             "weight_decay": 0.0,
        }
    ]

    num_train_steps = int( len(trainDataLoader) * config.EPOCHS )
    optimizer       = AdamW(optimizer_parameters, lr=2e-5)
    scheduler       = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )
    
    if config.dotrain:
        print('In Training')
        for epoch in range(config.EPOCHS):
            loss = engine.train(trainDataLoader, citemodel, optimizer, device, scheduler)
            print("Epoch: ", epoch, " Loss: ",loss,'\n')


        # Saving the model
        os.makedirs(os.path.dirname(config.MODEL_SAVED.format(config.modelName)), exist_ok=True)
        torch.save(citemodel.state_dict(), config.MODEL_SAVED.format(config.modelName))
        print('Model is saved at: ',config.MODEL_SAVED.format(config.modelName))

    '''
     Evaluating the model
    '''

    print("Loading the model")
    #citemodel = model.BERTBaseUncased(*args, **kwargs)
    citemodel.load_state_dict(torch.load(config.MODEL_SAVED.format(config.modelName)))
    outputs, targets = engine.eval(testDataLoader,citemodel,device)
    
    # Saving the results with corresponding targets
    os.makedirs(os.path.dirname(config.PREDICTIONS_PATH.format(config.modelName)), exist_ok=True)
    with open(config.PREDICTIONS_PATH.format(config.modelName), 'wb') as f:
        pickle.dump(outputs, f) # First saved the predicted outputs
        pickle.dump(targets, f) # Then saved the corresponding targets

    print('Starting Evaluation...')
    utils.metric(outputs,targets)




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--do-train', dest="doTrain", required=True, type=str, help="Enter True or False")
    parser.add_argument('--model', dest="model", required=True, type=str, help="Enter \"BertBase\" | \"SciBert\"")
    args   = parser.parse_args()

    if args.doTrain.lower() == "True".lower():
        config.dotrain = True
    else:
        config.dotrain = False

    if args.model.lower() == "BertBase".lower():
        config.modelName = "BertBase"
    elif args.model.lower() == "SciBert".lower():
        config.modelName = "SciBert"

    run()