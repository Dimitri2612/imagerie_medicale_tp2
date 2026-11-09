import numpy as np
import argparse
import matplotlib.pyplot as plt
from similarity import ssd, cr, IM


def prepare_image(img):
    # si l'image a 3 dimensions, on garde un seul canal
    if img.ndim == 3:
        img = img[:, :, 0]

    # conversion en float
    img = img.astype(np.float64)

    return img


def parse_args():
    parser = argparse.ArgumentParser(description="Critères de similarité entre deux images")
    parser.add_argument("image1", help="Chemin vers la première image")
    parser.add_argument("image2", help="Chemin vers la deuxième image")
    parser.add_argument("--bin", type=int, default=32, help="Nombre de bins pour l'histogramme conjoint")

    return parser.parse_args()


def main():
    args = parse_args()
    I = prepare_image(plt.imread(args.image1))
    J = prepare_image(plt.imread(args.image2))

    print("Calcul des similarités entre les images " + args.image1 + " et " + args.image2)
    print("Le SSD est : ", ssd(I, J))   # SSD
    print("Le coefficient de corrélation est : ", cr(I, J))   # CR
    print("L'information mutuelle est : ", IM(I, J, args.bin))   # IM


if __name__ == "__main__":
    main()


'''
Pour tester (depuis le répertoire imagerie_medicale_tp2) :
python ./src/compare_pairs.py ./Data_TP2/Data/BrainMRI_1.jpg ./Data_TP2/Data/BrainMRI_2.jpg --bin 32
python ./src/compare_pairs.py ./Data_TP2/Data/BrainMRI_2.jpg ./Data_TP2/Data/BrainMRI_3.jpg --bin 32

python ./src/compare_pairs.py ./Data_TP2/Data/I1.png ./Data_TP2/Data/J1.png --bin 32
python ./src/compare_pairs.py ./Data_TP2/Data/I2.jpg ./Data_TP2/Data/J2.jpg --bin 32
python ./src/compare_pairs.py ./Data_TP2/Data/I3.jpg ./Data_TP2/Data/J3.jpg --bin 32
python ./src/compare_pairs.py ./Data_TP2/Data/I4.jpg ./Data_TP2/Data/J4.jpg --bin 32
python ./src/compare_pairs.py ./Data_TP2/Data/I5.jpg ./Data_TP2/Data/J5.jpg --bin 32
python ./src/compare_pairs.py ./Data_TP2/Data/I6.jpg ./Data_TP2/Data/J6.jpg --bin 32
'''
