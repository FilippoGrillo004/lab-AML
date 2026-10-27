# Importiamo le funzioni dai nostri moduli appena creati
from data.dataset import download_and_extract_tiny_imagenet, get_dataloaders
from utils.visualization import visualize_classes

def main():
    # 1. Scarica il dataset (usa un percorso temporaneo comodo su PC locale o Colab)
    # Su Colab va bene '/dataset', sul tuo PC locale meglio usare './data/dataset'
    extract_path = './dataset'
    download_and_extract_tiny_imagenet(extract_path=extract_path)

    # 2. Crea i Dataloader
    dataset_dir = f"{extract_path}/tiny-imagenet-200"
    dataloader_train, dataloader_test, train_dataset = get_dataloaders(data_dir=dataset_dir, batch_size=32)

    # 3. Stampa le statistiche
    num_classes = len(train_dataset.classes)
    num_samples = len(train_dataset)
    print(f'Number of classes: {num_classes}')
    print(f'Number of samples: {num_samples}')

    # 4. Avvia la visualizzazione (con il TODO da completare)
    visualize_classes(dataloader_train)

if __name__ == "__main__":
    main()