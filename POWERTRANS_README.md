# PowerTransformers

## Model Training
To train the model, 
1. Download our dataset from https://homes.cs.washington.edu/~msap/controllable-debiasing/ and place 
in a `data` folder
2. Do the installation as stated in README.md
3. Train the model with `train.py`. As in our ablated studies, there are two model setups
    1. Joint Objective
    ```python train.py --setup joint```
    2. Para Only 
    ```python train.py --setup para```

## Generation
To generate paraphrases with augmented agency, run `generate.py` in the following fashion.

```python generate.py --dataset <dataset> --epoch <model epoch> --setup <model setup>```

It will generate paraphrase sentences in all `pos`, `equal` and `neg` categories for each sentence

| Argument | Description |
| -------- | ----------- |
| dataset  | Dataset for generation <br> **para**: devset for paraphrase <br> **roc**: devset for ROC stories <br> **rco-test**: test set for ROC stories |
| epoch | the epoch of the model |
| setup | whether the model is trained on `joint` or `para-only` objective |
