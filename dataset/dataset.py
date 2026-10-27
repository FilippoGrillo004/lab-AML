import os
import shutil
import requests
from zipfile import ZipFile
from io import BytesIO
import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as T

# ==========================================
# 1. FUNZIONE DAL LAB 1 (Per scaricare)
# ==========================================
def download_and_extract_tiny_imagenet(extract_path='./tiny-imagenet'):
    """Scarica ed estrae il dataset se non esiste già."""
    if os.path.exists(os.path.join(extract_path, 'tiny-imagenet-200')):
        print("Dataset già presente!")
        return

    print("Scaricamento del dataset in corso...")
    dataset_url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    response = requests.get(dataset_url)
    
    if response.status_code == 200:
        with ZipFile(BytesIO(response.content)) as zip_file:
            zip_file.extractall(extract_path)
        print('Download ed estrazione completati!')
    else:
        print("Errore nel download del dataset.")

# ==========================================
# 2. FUNZIONE DAL LAB 2 (Per sistemare la validation)
# ==========================================
def prepare_tiny_imagenet_val_folder(base_path='./tiny-imagenet/tiny-imagenet-200'):
    """Riordina la cartella di validazione per renderla compatibile con ImageFolder."""
    val_dir = os.path.join(base_path, 'val')
    val_annotations = os.path.join(val_dir, 'val_annotations.txt')
    val_images_dir = os.path.join(val_dir, 'images')
    
    # Se abbiamo già eliminato la cartella 'images', il setup è già stato fatto
    if not os.path.exists(val_images_dir):
        return

    with open(val_annotations) as f:
        for line in f:
            fn, cls, *_ = line.split('\t')
            os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
            shutil.copyfile(
                os.path.join(val_images_dir, fn), 
                os.path.join(val_dir, cls, fn)
            )

    shutil.rmtree(val_images_dir)

# ==========================================
# 3. DATALOADER AGGIORNATI (Unione Lab 1 + Lab 2)
# ==========================================
def get_dataloaders(base_path='./tiny-imagenet/tiny-imagenet-200', batch_size=32):
    """Crea i dataloader completi per il training."""
    
    # 1. Definiamo le trasformazioni
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 2. Prepariamo le cartelle di validazione usando la nuova funzione
    prepare_tiny_imagenet_val_folder(base_path)

    # 3. Creiamo i Dataset
    train_dataset = ImageFolder(root=os.path.join(base_path, 'train'), transform=transform)
    val_dataset = ImageFolder(root=os.path.join(base_path, 'val'), transform=transform)

    # 4. Creiamo i Dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader