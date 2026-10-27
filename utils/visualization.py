import numpy as np
import matplotlib.pyplot as plt

def denormalize(image):
    """Riporta l'immagine ai valori originali per poterla stampare."""
    image = image.to('cpu').numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image * std + mean
    image = np.clip(image, 0, 1)
    return image

def visualize_classes(dataloader, num_classes_to_show=10):
    """Visualizza un esempio per ogni classe (da completare)."""
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    classes_sampled = []
    found_classes = 0

    for i, (inputs, classes) in enumerate(dataloader):
        # TODO: Completa il tuo codice qui per l'esercizio!
        pass

    plt.show()