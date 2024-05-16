

import matplotlib.pyplot as plt

def plot_mnist(train, width=10, height=5):
    imgs = []
    for i in range(10):
        for img, label in train:
            if label[0] == i:
                imgs.append(img[0])
                break

    #plot the images
    fig, axs = plt.subplots(2, 5, figsize=(width, height))
    for i in range(10):
        ax = axs[i//5, i%5]
        ax.imshow(imgs[i][0], cmap='gray')
        ax.set_title(f"Class {i}")
        ax.axis('off')

    fig.suptitle("MNIST dataset")
    fig.tight_layout()
    

# load config from../config.json
def load_config():
    import json
    with open("../mnist_config.json", "r") as f:
        config = json.load(f)
    return config
