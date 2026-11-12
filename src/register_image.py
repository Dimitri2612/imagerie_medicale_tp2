import numpy as np
from scipy import ndimage
from similarity import ssd


# Translation de l'image I du vecteur (p, q) : p selon l'axe 0 (lignes), q selon l'axe 1 (colonnes)
def translate_image(I, p, q):
    
    return ndimage.shift(I, shift=(p, q), order=1, mode='constant', cval=0.0)  # order = 1 : interpolation bilinéaire, mode = 'constant' : pixels hors image = 0


def gradient_ssd(I_t, J):
    # I_t : image I déjà translatée du vecteur courant (p, q) ; J : image fixe
    g0, g1 = np.gradient(I_t)   # dérivées de I_t selon l'axe 0 puis l'axe 1
    diff = I_t - J              # I_t - J
    dE_dp = 2 * np.sum(g0 * diff)  # dérivée de l'énergie ssd par rapport à p
    dE_dq = 2 * np.sum(g1 * diff)  # dérivée de l'énergie ssd par rapport à q
    return dE_dp, dE_dq


def register_translation_ssd(I, J, alpha=1e-3, max_iter=1000, tol=1e-6):
    # I : image à translatée ; J : image fixe
    p, q = 0.0, 0.0  # vecteur de translation initial
    I_t = translate_image(I, p, q)  # image I translatée du vecteur (p, q)
    E_prev = ssd(I_t, J)  # énergie ssd initiale

    for i in range(max_iter):
        dE_dp, dE_dq = gradient_ssd(I_t, J)  # calcul du gradient de l'énergie ssd
        p = p - alpha * dE_dp  # mise à jour de p
        q = q - alpha * dE_dq  # mise à jour de q
        I_t = translate_image(I, p, q)  # nouvelle image translatée
        E_curr = ssd(I_t, J)  # nouvelle énergie ssd

        if abs(E_curr - E_prev) < tol:  # critère d'arrêt basé sur la variation de l'énergie
            break

        E_prev = E_curr

    return p, q
