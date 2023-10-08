from ep_inteligentes_2023.io import mostrarLote, leImagens
import numpy as np


def main() -> None:
    

    imgs, labels = leImagens(
        [
            "./data/Lung Segmentation Data/Lung Segmentation Data/Train/COVID-19/images/*.png",
            "./data/Lung Segmentation Data/Lung Segmentation Data/Train/Non-COVID/images/*.png",
            "./data/Lung Segmentation Data/Lung Segmentation Data/Train/Normal/images/*.png",
        ],
        classes=[0, 1, 2],
    )
    
    random_sample = np.random.choice(labels.shape[0], size=16)
    
    print("Amostra aleatória:")
    mostrarLote(imgs[random_sample])
    input("Pressione enter para ver os labels")
    mostrarLote(imgs[random_sample], y=labels[random_sample])



if __name__ == "__main__":
    main()
