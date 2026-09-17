from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from pathlib import Path

COLORES = {
    "player":     "red",
    "goalkeeper": "blue",
    "referee":    "yellow",
    "ball":       "lime",
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

    nx1 = x1 + ancho * 0.25
    nx2 = x2 - ancho * 0.25

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

