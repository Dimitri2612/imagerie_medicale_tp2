import argparse
import numpy as np
import matplotlib.pyplot as plt


def make_grid(nx, ny, nz):
    x, y, z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
    points = np.vstack((x.ravel(), y.ravel(), z.ravel())).astype(float)
    return points


def show_points(points, ax=None, color='k'):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')   # axes en 3D

    ax.scatter(points[0], points[1], points[2], c=color, s=5)     # x, y, z : les trois lignes de points

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_aspect('equal')   # même échelle sur les trois axes
    return ax


def parse_args():
    parser = argparse.ArgumentParser(description="Génération d'une grille de points 3D")
    parser.add_argument("--nx", type=int, default=20, help="Nombre de points sur l'axe x")
    parser.add_argument("--ny", type=int, default=20, help="Nombre de points sur l'axe y")
    parser.add_argument("--nz", type=int, default=4, help="Nombre de points sur l'axe z")

    return parser.parse_args()


def main():
    args = parse_args()
    pts = make_grid(args.nx, args.ny, args.nz)
    show_points(pts)
    plt.show()


if __name__ == "__main__":
    main()