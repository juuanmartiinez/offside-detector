from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from pathlib import Path
from src.lineas import puntosDeRecta
import numpy as np, cv2

COLORES = {
    "player":     "red",
    "goalkeeper": "blue",
    "referee":    "yellow",
    "ball":       "lime",
    "ata":        "cyan",
    "def":        "magenta",
    "x":          "cyan",
    "dudoso":     "gray",
}

def dibujarCajas(ax, pares, colores=COLORES, grosor=1):

        for (x1, y1, x2, y2), clase in pares:
            ax.add_patch(Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=False,
                edgecolor=colores.get(clase, "white"),
                linewidth=grosor
            ))

def dibujarAllImagenes(ds, carpeta, predicciones=None, dpi=72):

    Path(carpeta).mkdir(parents=True, exist_ok=True)


    for i in sorted(ds.id2fichero):

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(ds.imagen(i))
        ax.axis("off")

        dibujarCajas(ax, ds.cajas(i))

        if predicciones is not None:
            cajas_p = predicciones.get(str(i), {}).get("boxes", [])
            dibujarCajas(ax, [(c, "pred") for c in cajas_p], colores={"pred": "cyan"})

        fig.savefig(Path(carpeta) / f"{i:04d}.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)

def bandaCentral(caja):

    x1, y1 , x2, y2 = caja

    ancho = x2 - x1
    alto = y2 - y1

    nx1 = x1 + ancho * 0.10
    nx2 = x2 - ancho * 0.10

    ny1 = y1 + alto * 0.15
    ny2 = y1 + alto * 0.45

    return nx1, ny1, nx2, ny2

def recortesTorso(ds, imgId, cajas=None):

    img = ds.imagen(imgId)
    recortes = []

    if cajas is None:
        cajas = [c for c, k in ds.cajas(imgId)]


    for caja in cajas:
        x1, y1, x2, y2 = bandaCentral(caja)
        recortes.append(img.crop((x1, y1, x2, y2)))

    return recortes

def dibujarPuntos(ax, puntos, color="cyan", tamano=30):

    for punto in puntos:
        x, y = punto
        ax.scatter(x, y, s=tamano, c=color, zorder=3)

def dibujarLineas(ax, rectas, ancho, alto):

    for i, recta in enumerate(rectas)   :
        (xa, ya), (xb, yb) = puntosDeRecta(recta, ancho, alto)
        ax.plot([xa, xb], [ya, yb])
        ax.text((xa+xb)/2, (ya+yb)/2, str(i), color="white", fontsize=10,
        bbox=dict(fc="black", ec="none", pad=1))

def imprimirCenital(ax, imagen, H, escala=10, L=105.0, W=68.0):

    S = np.array([[escala, 0, 0],
                  [0, escala, 0],
                  [0, 0,      1]], dtype=np.float64)

    ancho, alto = int(L * escala), int(W * escala)
    cenital = cv2.warpPerspective(np.asarray(imagen), S @ H, (ancho, alto))

    ax.imshow(cenital, extent=[0, L, W, 0])
    ax.set_aspect("equal")

def dibujarCampo(ax, L=105.0, W=68.0, color="white", grosor=2):

    ax.set_facecolor("#3f8f4a")

    ax.plot([0, L, L, 0, 0], [0, 0, W, W, 0], color=color, lw=grosor)
    ax.plot([L/2, L/2], [0, W], color=color, lw=grosor)
    ax.add_patch(plt.Circle((L/2, W/2), 9.15, fill=False,
                            color=color, lw=grosor))

    for x0, signo in ((0, 1), (L, -1)):
        ax.plot([x0, x0 + signo*16.5, x0 + signo*16.5, x0],
                [13.84, 13.84, 54.16, 54.16], color=color, lw=grosor)
        ax.plot([x0, x0 + signo*5.5, x0 + signo*5.5, x0],
                [24.84, 24.84, 43.16, 43.16], color=color, lw=grosor)
        ax.scatter([x0 + signo*11], [W/2], s=18, c=color, zorder=4)

    ax.set_xlim(-5, L + 5)
    ax.set_ylim(W + 5, -5)
    ax.set_aspect("equal")

def dibujarPuntosCenital(ax, puntos, color="yellow"):

    if not len(puntos):
        return

    xs, ys = zip(*puntos)
    ax.scatter(xs, ys, s=130, c=color, ec="black", lw=1.2, zorder=5,
    label=f"en el campo ({len(puntos)})")

def dibujarEscena(ax, metros, etiquetas, xLinea=None, iDefensa=None):

    metros    = np.asarray(metros)
    etiquetas = np.asarray(etiquetas)

    dibujarCampo(ax)

    for etiqueta in ("ata", "def", "dudoso"):
        dibujarPuntosCenital(ax, metros[etiquetas == etiqueta],
                             color=COLORES.get(etiqueta, "white"))

    if xLinea is not None:
        ax.axvline(xLinea, color="yellow", lw=2)

    if iDefensa is not None:
        ax.scatter(metros[iDefensa, 0], metros[iDefensa, 1], s=260,
                   facecolors="none", edgecolors="yellow", lw=2.5, zorder=6)