import argparse
import matplotlib.pyplot as plt
import numpy as np
from joint_hist import joint_hist, check_hist


def prepare_image(img):
    # si l'image a 3 dimensions, on garde un seul canal
    if img.ndim == 3:
        img = img[:, :, 0]

    # conversion en float
    img = img.astype(np.float64)

    return img


def show_hist(H, nom_I, nom_J, bin):
    H_log = np.log1p(H)  # échelle logarithmique, sans problème pour les cases à 0 (calcul de log(1+ H))

    plt.figure()
    plt.imshow(H_log.T, origin='lower')  # transpose pour que l'axe des x corresponde à I et l'axe des y à J, et origin='lower' pour que le bin (0,0) soit en bas à gauche
    plt.colorbar(label="log(1 + nombre de pixels)")
    plt.xlabel(f"Intensité de I ({nom_I}), indice de bin")
    plt.ylabel(f"Intensité de J ({nom_J}), indice de bin")
    plt.title("Histogramme conjoint")
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Histogramme conjoint de deux images I et J")
    parser.add_argument("image1", help="Chemin de l'image I")
    parser.add_argument("image2", help="Chemin de l'image J")
    parser.add_argument("--bin", type=int, default=32, help="Nombre de bins")
    
    return parser.parse_args()


def main():
    args = parse_args()
    I = plt.imread(args.image1)
    J = plt.imread(args.image2)
    
    if I.ndim == 3:  # utile uniquement pour le cas où l'image est en couleur, ce qui est valable uniquement pour l'image I1.png
        print("canaux R=G :", np.allclose(I[:, :, 0], I[:, :, 1]))  # on vérifie que les canaux R et G sont égaux
        print("canaux R=B :", np.allclose(I[:, :, 0], I[:, :, 2]))  # on vérifie que les canaux R et B sont égaux
        print("alpha min/max :", I[:, :, 3].min(), I[:, :, 3].max())  # on vérifie les valeurs min et max du canal alpha

    # affichage des informations sur les images avant la conversion des types uint8 en float64
    for nom, img in (("I", I), ("J", J)):
        print(nom, img.dtype, img.shape, img.min(), img.max())
        
    I = prepare_image(I)
    J = prepare_image(J)
    
    # affichage des informations sur les images après la conversion des types uint8 en float64
    for nom, img in (("I", I), ("J", J)):
        print(nom, img.dtype, img.shape, img.min(), img.max())
    
    H = joint_hist(I, J, args.bin)
    if check_hist(H, I):
        print("L'histogramme conjoint est correct : chaque pixel a été compté exactement une fois.")
        show_hist(H, args.image1, args.image2, args.bin)
    else:
        print("Erreur : l'histogramme conjoint n'est pas correct.")

if __name__ == "__main__":
    main()
    

'''
Pour tester (depuis le répertoire imagerie_medicale_tp2) :
python ./src/show_joint_hist.py ./Data_TP2/Data/BrainMRI_1.jpg ./Data_TP2/Data/BrainMRI_2.jpg --bin 32
python ./src/show_joint_hist.py ./Data_TP2/Data/BrainMRI_2.jpg ./Data_TP2/Data/BrainMRI_3.jpg --bin 32

python ./src/show_joint_hist.py ./Data_TP2/Data/I1.png ./Data_TP2/Data/J1.png --bin 32
python ./src/show_joint_hist.py ./Data_TP2/Data/I2.jpg ./Data_TP2/Data/J2.jpg --bin 32
python ./src/show_joint_hist.py ./Data_TP2/Data/I3.jpg ./Data_TP2/Data/J3.jpg --bin 32
python ./src/show_joint_hist.py ./Data_TP2/Data/I4.jpg ./Data_TP2/Data/J4.jpg --bin 32
python ./src/show_joint_hist.py ./Data_TP2/Data/I5.jpg ./Data_TP2/Data/J5.jpg --bin 32
python ./src/show_joint_hist.py ./Data_TP2/Data/I6.jpg ./Data_TP2/Data/J6.jpg --bin 32
'''