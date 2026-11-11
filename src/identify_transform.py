import numpy as np

TOL = 1e-3   # tolérance : les matrices de l'énoncé sont arrondies à 4 décimales

# Matrices de l'énoncé (à vérifier chiffre par chiffre sur ton sujet)
M1 = np.array([[0.9045, -0.3847, -0.1840, 10.0],
               [0.2939,  0.8750, -0.3847, 10.0],
               [0.3090,  0.2939,  0.9045, 10.0],
               [0.0,     0.0,     0.0,     1.0]])

M2 = np.array([[0.0,    -0.2598,  0.1500, -3.0],
               [0.0,    -0.1500, -0.2598,  1.5],
               [0.3000,  0.0,     0.0,     0.0],
               [0.0,     0.0,     0.0,     1.0]])

M3 = np.array([[ 0.7182, -1.3727, -0.5660, 1.8115],
               [-1.9236, -4.6556, -2.5512, 0.2873],
               [-0.6426, -1.7985, -1.6285, 0.7404],
               [ 0.0,     0.0,     0.0,    1.0]])


def classify(M, tol=TOL):
    A = M[:3, :3]
    s = np.linalg.svd(A, compute_uv=False)
    det = np.linalg.det(A)

    if not np.allclose(M[3], [0, 0, 0, 1], atol=tol):
        return "pas une transformation affine (dernière ligne différente de 0 0 0 1)"
    if not np.allclose(s, s[0], atol=tol):
        return "affine générale (valeurs singulières différentes)"

    d = s.mean()
    suffixe = "" if det > 0 else " avec réflexion (det < 0)"
    if np.isclose(d, 1, atol=tol):
        return "rigide" + suffixe
    return f"similitude de rapport d = {d:.4f}" + suffixe


def diagnostic(M, nom):
    A = M[:3, :3]   # bloc 3x3
    t = M[:3, 3]    # translation
    s = np.linalg.svd(A, compute_uv=False)   # valeurs singulières (sans U ni V*)

    print(f"--- {nom} ---")
    print("dernière ligne :", M[3])
    print("translation t :", t)
    print("A.T @ A :")
    print(np.round(A.T @ A, 3))
    print("valeurs singulières :", np.round(s, 4))
    print("déterminant de A :", round(np.linalg.det(A), 4))
    print("verdict :", classify(M))
    print()


def main():
    for nom, M in (("M1", M1), ("M2", M2), ("M3", M3)):
        diagnostic(M, nom)


if __name__ == "__main__":
    main()

