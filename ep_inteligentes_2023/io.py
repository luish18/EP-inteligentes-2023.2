import glob
import cv2 as cv
import numpy.typing as npt
import numpy as np
import matplotlib.pyplot as plt


def leImagens(
    wildcards: list[str], classes: list[int], nl: int = 224, nc: int = 224
) -> tuple[npt.NDArray, npt.NDArray]:
    data_label_pairs = []

    for wildcard, class_name in zip(wildcards, classes):
        img_paths = glob.glob(wildcard)

        for path in img_paths:
            data_label_pairs.append((cv.imread(path, cv.IMREAD_GRAYSCALE), class_name))

    data, labels = zip(*data_label_pairs)

    return np.array(data), np.array(labels)


def mostrarLote(x, y=None):
    class_name_map = {0: "Covid", 1: "Non-Covid", 2: "Normal"}
    fig, axes = plt.subplots(4, 4)
    axes = axes.flatten()

    for i in range(16):
        axes[i].imshow(x[i], cmap="gray")
        axes[i].axis("off")
        if y is not None:
            label = class_name_map[y[i]]
            axes[i].set_title(label)

    plt.tight_layout()
    plt.show()


