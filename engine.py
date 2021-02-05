import torch 
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F


def loss_fn(outputs, targets):
    return F.cross_entropy(outputs, targets.type(torch.int64))

    

def train(dataLoader, model, optimizer, device, scheduler):
    model.train()
    resultloss = 0
    for bi, d in tqdm(enumerate(dataLoader), total=len(dataLoader)):
        ids            = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask           = d["mask"]
        target         = d["targets"]

        ids            = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask           = mask.to(device, dtype=torch.long)
        targets        = target.to(device, dtype=torch.long)

        optimizer.zero_grad()
        output = model(ids=ids, mask=mask, token_type_ids=token_type_ids)

        loss = loss_fn(output, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        resultloss += loss
    return resultloss/len(dataLoader)

def eval(dataLoader, model, device):
    model.eval()
    finalOutputs = []
    finalTargets = []
    with torch.no_grad():
        for bi, d in tqdm(enumerate(dataLoader), total=len(dataLoader)):
            ids            = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask           = d["mask"]
            target         = d["targets"]

            ids            = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask           = mask.to(device, dtype=torch.long)
            targets        = target.to(device, dtype=torch.long)

            outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
            #print("Outputs are: \n",outputs.shape,' target.shape: ',target.shape)
            #print("F.log_softmax(outputs):\n\n",F.log_softmax(outputs)[0])
            #print("torch.exp(F.log_softmax(outputs)):\n\n",torch.exp(F.log_softmax(outputs)))

            finalOutputs.extend(F.softmax(outputs).cpu().detach().numpy().tolist())
            finalTargets.extend(target.cpu().detach().numpy().tolist())

    return finalOutputs, finalTargets

