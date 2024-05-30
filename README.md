# FedFromScratch

![header](./assets/image.png)
This repository contains the code for the project "Federated Learning from Scratch" for the course "Continual Learning" at the University of Pisa

# Project structure
```
.
├── README.md
├── histories
│   ├── history_5e10c.pkl
│   ├── history_5e10c_fashion.pkl
│   ├── history_5e128c.pkl
│   ├── history_5e20c.pkl
│   ├── history_5e5c.pkl
│   └── history_niid.pkl
├── models
│   ├── efficientnet_b0.pth
│   ├── efficientnet_cifar10.pth
│   ├── mnist_cnn_model.pth
│   ├── model5e10c_fashion.pth
│   ├── model_niid.pth
│   └── resnet18_cifar10.pth
├── requirements.txt
└── src
    ├── federated.py
    ├── finetuning.py
    ├── models.py
    ├── report.ipynb
    └── utils.py
```

# How to run the code

To run the code, you need to first install the required packages. You can do this by running the following command:

``` 
pip install -r requirements.txt
```

After that, you can run the code by running the following command:

```
python src/fedmnist.py [--help]
```

The script will train a federated model on the MNIST dataset. You can specify the number of clients, the number of epochs, the batch size, the learning rate, the optimizer, the model, and the dataset.

```bash
usage: fedmnist.py [-h] [--n_clients N_CLIENTS] [--local_epochs LOCAL_EPOCHS] [--lr LR] [--n_samples N_SAMPLES] [--device DEVICE] [--weight_decay WEIGHT_DECAY] [--iid IID] [--rounds ROUNDS] [--jobs JOBS]

FedMNIST

options:
  -h, --help            show this help message and exit
  --n_clients N_CLIENTS
                        number of clients to initialize
  --local_epochs LOCAL_EPOCHS
                        number of local epochs
  --lr LR               learning rate
  --n_samples N_SAMPLES
                        number of samples per client
  --device DEVICE       device to train on
  --weight_decay WEIGHT_DECAY
                        weight decay for optimizer
  --iid IID             whether to sample clients iid or not
  --rounds ROUNDS       number of rounds to train for
  --jobs JOBS           number of jobs to run in parallel
```

## Results

Results of the experiments are reported in the norebook `report.ipynb`.
