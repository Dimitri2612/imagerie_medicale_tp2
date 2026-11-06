import numpy as np


# Regroupement des valeurs de l'image dans des bins, pour chaque pixel, on calcule l'indice du bin correspondant
def bin_indices(img, bin):
    vmin = img.min()
    vmax = img.max()
    largeur = (vmax - vmin) / bin

    idx = np.floor((img - vmin) / largeur)  # indice de bin de chaque pixel, d'après ta formule
    idx = np.clip(idx, 0, bin - 1).astype(int)  # correction pour que le max n'aille pas au bin "bin" et conversion en entiers
    return idx


# Calcul de l'histogramme conjoint de deux images I et J de même taille, avec un nombre de bins donné
def joint_hist(I, J, bin):
    
    if  I.shape != J.shape:
        raise ValueError("Les images I et J doivent avoir la même taille")
    
    idx_I = bin_indices(I, bin)
    idx_J = bin_indices(J, bin)
    H = np.zeros((bin, bin), dtype=int)
    
    np.add.at(H, (idx_I, idx_J), 1)  # ajoute +1 à la case (idx_I, idx_J) de chaque pixel, en comptant les indices répétés
    
    return H


# Vérifie que chaque pixel a été compté exactement une fois : somme de H = n * p
def check_hist(H, I):
    n, p = I.shape
    total = H.sum()
    return total == n * p  # retourne True si la somme de H est égale au nombre de pixels de l'image I, sinon False
