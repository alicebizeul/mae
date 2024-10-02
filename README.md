# Principled Masked Autoencoders

In this repository we explore more principled methods for image masking 

## Installation 

In your environment of choice, install the necessary requirements

    !pip install -r requirements.txt 

Create a config file that suits your machine:

    cd ./config/user
    cp myusername_mymachine.yaml <myusername>_<mymachine>.yaml

Adjust the paths to point to the directory you would like to use for storage of results and for fetching the data

### Training
To launch experiments, you can find a good example for training at  ```./script/jobs_pcmae_random.sh``` and ```./script/jobs_eval_pcmae_random.sh``` for evaluation.

### Evaluation on linear probing
To evaluate a checkpoint, you can gain inspiration from ```./config/user/<myusername>_<mymachine>.yaml``` where runs are stored. Then the following command gives an overview of how to launch the evaluation

    EXPERIMENT="mae_cifar10"
    EPOCH=800
    RUN_TAG="$EXPERIMENT_eval_$EPOCH"
    python main.py user=<myusername>_<mymachine> experiment=$DATASET trainer=eval checkpoint=pretrained checkpoint.epoch=$EPOCH run_tag=$RUN_TAG"


### Adding pointers to key aspects of the repo 

A change in the masking strategy should be reflected in ```./dataset/dataloader.py``` file which define the image-masking pairs. The change should also be reflected in ```./model/module.py``` where each batch is masked and passed through the model
