To run this project follow the below steps:

```
1. Copy the repository code in local machine
2. Make sure dataset is available in 'data' folder.
3. Run: python app.py --do-train=True|False --model=SciBert|BertBase
```
While running first time, make sure ```--do-train-``` is set to ```True``` to train the model. 

Default ```--model``` is set to ```BertBase``` so if model name is incorrect ```BertBase``` will be set by default.

All other project configurations can be set in ```config.py``` file. Learning rate is not set to 3e-5 which is best for BertBase. For SciBert this learning rate gives poor result. Trying using 2e-5 as learning rate while using SciBert.

Currently these are the results:

```
BertBase
Recall@5 : 0.5295839431319256
Recall@10 : 0.5776709178339954
Recall@30 : 0.655446372569517
Recall@50 : 0.6966339117708551
Recall@80 : 0.74430273886682
mrr: 0.4529959038914291
```
```
SciBert
Recall@5 : 0.4633075475642902
Recall@10 : 0.5024043487351035
Recall@30 : 0.5720259251515785
Recall@50 : 0.614258833368179
Recall@80 : 0.6564917415847794
mrr: 0.4032790611791721
```

**GCN module is being implemented. Once that is done, code will be added**

Datatset *Full Context PeerRead* can be obtained from https://github.com/TeamLab/bert-gcn-for-paper-citation . Or you can contact me too.