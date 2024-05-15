from models import MNISTConvNet, MNISTFederatedServer, get_mnist_dataloader, init_clients

import argparse, json

train, val, test = get_mnist_dataloader(64)
model  = MNISTConvNet()
server = MNISTFederatedServer(model=model, test_loader=val)

# add arguments
parser = argparse.ArgumentParser(description='FedMNIST')
parser.add_argument('--config', type=str, default='../mnist_config.json', help='path to config file')

args = parser.parse_args()

# load client configuration from path json
with open(args.config) as f:
    cfg = json.load(f)

model_cfg = cfg['client_config']


# load client configira
if __name__ == '__main__':
    clients = init_clients(
        num_clients=16, 
        local_epochs=10, 
        lr=1e-2, 
        train_dataset=train, 
        n_samples=512, 
        device='mps', 
        weight_decay=1e-6
    )
    server.start_train(clients, n_rounds=25, n_jobs=8)

    server.global_model.eval()
    server.test_loader = test
    server.test()

