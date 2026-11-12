import numpy as np
import matplotlib.pyplot as plt
from register_image import translate_image, register_translation_ssd


def prepare_image(img):
    # si l'image a 3 dimensions, on garde un seul canal
    if img.ndim == 3:
        img = img[:, :, 0]

    # conversion en float
    img = img.astype(np.float64)

    return img


def main():
    I = plt.imread("./Data_TP2/Data/BrainMRI_1.jpg")
    I = prepare_image(I)
    J = translate_image(I, 70, -30)  # translation de I pour créer J
    
    # Enregistrement par translation
    p, q = register_translation_ssd(I, J, alpha=1e-3, max_iter=10000, tol=1e-9)
    
    print(f"Vecteur de translation optimal : (p={p}, q={q})")
    
if __name__ == "__main__":
    main()
