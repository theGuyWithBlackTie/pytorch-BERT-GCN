To run this project follow the below steps:

```
1. Copy the repository code in local machine
2. Make sure dataset is available in 'data' folder.
3. Run: python app.py --do-train=True|False --model=SciBert|BertBase
```
While running first time, make sure ```--do-train-``` is set to ```True``` to train the model. 

Default ```--model``` is set to ```BertBase``` so if model name is incorrect ```BertBase``` will be set by default.

All other project configurations can be set in ```config.py``` file. Learning rate is not set to 3e-5 which is best for BertBase. For SciBert this learning rate gives poor result. Trying using 2e-5 as learning rate while using SciBert. 

**GCN module is being implemented. Once that is done, code will be added**