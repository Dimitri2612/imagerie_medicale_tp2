import numpy as np
import matplotlib.pyplot as plt
import argparse
from grid_generator import make_grid, show_points


def to_homogeneous(points):
    n = points.shape[1]
    
    return np.vstack((points, np.ones((1, n))))  # tableau (4, N) : N le nombre de points, 4 pour les coordonnées homogènes


def from_homogeneous(points_h):
    
    return points_h[:3, :]  # revenir à (3, N) en gardant les 3 premières lignes de coordonnées


def trans_rigide(theta, omega, phi, p, q, r):
    # angles en radians
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta), -np.sin(theta)],
                   [0, np.sin(theta),  np.cos(theta)]])
    Ry = np.array([[np.cos(omega), 0, np.sin(omega)],
                   [0, 1, 0],
                   [-np.sin(omega), 0, np.cos(omega)]])
    Rz = np.array([[np.cos(phi), -np.sin(phi), 0],
                   [np.sin(phi), np.cos(phi), 0],
                   [0, 0, 1]])

    R = Rz @ Ry @ Rx

    M = np.eye(4)  # matrice identité 4x4
    M[:3, :3] = R   # le bloc de rotation
    M[:3, 3] = [p, q, r]   # la colonne de translation (p, q, r)
    
    return M


def similitude(theta, omega, phi, p, q, r, s):
    S = np.eye(4)  # matrice identité 4x4
    S[:3, :3] = s * S[:3, :3]
    
    return trans_rigide(theta, omega, phi, p, q, r) @ S  # ajout de la transformation de similitude : rotation + translation + mise à l'échelle


def parse_args():
    parser = argparse.ArgumentParser(description="Transformation rigide d'une grille 3D")
    parser.add_argument("--nx", type=int, default=20)
    parser.add_argument("--ny", type=int, default=20)
    parser.add_argument("--nz", type=int, default=3)
    parser.add_argument("--theta", type=float, default=0, help="rotation autour de x (degrés)")
    parser.add_argument("--omega", type=float, default=0, help="rotation autour de y (degrés)")
    parser.add_argument("--phi", type=float, default=0, help="rotation autour de z (degrés)")
    parser.add_argument("--p", type=float, default=0, help="translation sur x")
    parser.add_argument("--q", type=float, default=0, help="translation sur y")
    parser.add_argument("--r", type=float, default=0, help="translation sur z")
    parser.add_argument("--s", type=float, default=None, help="facteur de mise à l'échelle")
    return parser.parse_args()


def main():
    args = parse_args()
    pts = make_grid(args.nx, args.ny, args.nz)
    pts_h = to_homogeneous(pts)
    
    if args.s is None:
        M = trans_rigide(np.deg2rad(args.theta), np.deg2rad(args.omega), np.deg2rad(args.phi), args.p, args.q, args.r)
    else:
        M = similitude(np.deg2rad(args.theta), np.deg2rad(args.omega), np.deg2rad(args.phi), args.p, args.q, args.r, args.s)

    pts_t = from_homogeneous(M @ pts_h)

    ax = show_points(pts, color='k')
    show_points(pts_t, ax=ax, color='r')
    plt.show()


if __name__ == "__main__":
    main()


'''
Pour tester (depuis le répertoire imagerie_medicale_tp2) :
python ./src/transform_grid.py --theta 20 --omega 10 --phi 30 --r 30 --s 0.5
'''
