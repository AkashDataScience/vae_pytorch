import torch
import numpy as np
import matplotlib.pyplot as plt

CIFAR_CLASS_NAMES= ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def save_sample_output(model, loader, device, path, is_mnist, image_no=5):
    dataiter = iter(loader)

    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)

    figure = plt.figure(figsize=(20,20))

    for index in range(1, image_no + 1):
        plt.subplot(5, 5, index)
        plt.axis('off')
        image = images[index]
        label = labels[index]
        wrong_label = torch.randint(low=0, high=10, size=(1,))[0]

        while label == wrong_label:
            wrong_label = torch.randint(low=0, high=10, size=(1,))[0]
        
        if is_mnist:
            plt.title(f'Input:{label.numpy()} Label:{wrong_label.numpy()}')
        else:
            plt.title(f'Input:{CIFAR_CLASS_NAMES[label.numpy()]} Label:{CIFAR_CLASS_NAMES[wrong_label.numpy()]}')

        input_img_lbl = image.unsqueeze(0), wrong_label.unsqueeze(0)

        with torch.no_grad():
            pred_image = model(input_img_lbl)

        pred_image = np.array(pred_image.cpu())
        pred_image = np.transpose(pred_image, (1, 2, 0))
        plt.imshow(pred_image)
    
    plt.tight_layout()
    plt.savefig(path)
    print(f"Sample output images are saved at {path}")


