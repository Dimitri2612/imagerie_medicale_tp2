import numpy as np
from joint_hist import joint_hist


# Calcul de la ssd (Sum of Squared Difference) entre deux images I et J de même taille
def ssd(I, J):
    if  I.shape != J.shape:
            raise ValueError("Les images I et J doivent avoir la même taille")
    diff = I - J  # différence pixel par pixel
    result = np.sum(diff ** 2)  # somme des carrés de diff
    
    return result


# Calcul du coefficient de corrélation entre deux images I et J de même taille
def cr(I, J):
    if I.shape != J.shape:
        raise ValueError("Les images I et J doivent avoir la même taille")

    I_c = I - np.mean(I)  # I centrée : I moins sa moyenne
    J_c = J - np.mean(J)  # J centrée
    result = np.sum((I_c * J_c )) / (np.sqrt(np.sum(I_c **2))*np.sqrt(np.sum(J_c **2)))  # calcul du coefficient de corrélation
    return result


def IM(I, J, bin = 32):
    if I.shape != J.shape:
            raise ValueError("Les images I et J doivent avoir la même taille")
     
    H = joint_hist(I, J, bin)
    p = H / I.size  # histogramme normalisé : H divisé par le nombre total de pixels
    p_I = np.sum(p, axis = 1)  # marginale de I : somme de p sur les J
    p_J = np.sum(p, axis = 0)  # marginale de J : somme de p sur les I
    produit = np.outer(p_I, p_J)  # tableau du produit des marginales : p_I[i] * p_J[j] pour chaque case (i,j)

    masque = p > 0  # on ne garde que les cases non vides (sinon log(0))
    result = np.sum(p[masque] * np.log(p[masque] / produit[masque]))  # calcul de l'information mutuelle
    return result
