import numpy as np
import cv2

def aHSV(imagen):

    arr = np.array(imagen)[:, :, :3]

    return cv2.cvtColor(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)

def colorCesped(imagen):
    # La mediana HSV de la foto entera ES el cesped: el campo ocupa casi todo el
    # encuadre. Asi funciona con cualquier tonalidad de hierba, en vez de dar por
    # hecho que el verde vive en un rango fijo de tono.

    return np.median(aHSV(imagen).reshape(-1, 3), axis=0)

def distanciaTono(tonos, referencia):
    # El tono es circular (0..179 en OpenCV): 178 y 2 estan a 4, no a 176.

    d = np.abs(tonos.astype(np.int16) - int(referencia))

    return np.minimum(d, 180 - d)

def descriptorCamisetas(recorte, cesped, margenTono=8, satMin=40, minPix=12, minFrac=0.15):
    # OJO: el rango fijo tono=(25, 95) que habia aqui antes se comia equipos
    # enteros. Medido sobre 342 cajas de 17 fotos: una camiseta amarilla cae en
    # H=28..35 y el cesped en H=42, asi que la mascara vieja la borraba y el
    # equipo entero salia NaN (41% de las cajas). Con la mascara centrada en el
    # cesped de la propia imagen: 14%.
    #
    # margenTono es el compromiso: cuanto mas grande, mas cesped quita pero mas
    # camisetas se lleva por delante. Medido: 6 -> 12% NaN, 8 -> 14%, 10 -> 20%.

    arr = np.array(recorte)[:, :, :3]

    if arr.size == 0:
        return np.full(3, np.nan)

    hsv = aHSV(recorte)

    esCesped = (distanciaTono(hsv[:, :, 0], cesped[0]) <= margenTono) & (hsv[:, :, 1] >= satMin)

    pix = hsv[~esCesped]

    if len(pix) < minPix or len(pix) < minFrac * hsv[:, :, 0].size:
        return np.full(3, np.nan)

    h = pix[:, 0].astype(np.float32) * 2 * np.pi / 180      # OpenCV va 0..179
    s = pix[:, 1].astype(np.float32) / 255
    v = pix[:, 2].astype(np.float32) / 255

    return np.array([np.median(s * np.cos(h)),
                     np.median(s * np.sin(h)),
                     np.median(v)])

def clasificarPorSemillas(X, idAtacante, idDefensor):

    if np.isnan(X[idAtacante]).any() or np.isnan(X[idDefensor]).any():
        raise ValueError("una de las semillas no tiene pixeles utiles, pincha otro jugador")

    difAta = X - X[idAtacante]
    difDef = X - X[idDefensor]

    dAta = np.linalg.norm(difAta, axis=1)
    dDef = np.linalg.norm(difDef, axis=1)

    etiquetas = np.where(dAta < dDef, "ata", "def").astype("<U10")
    etiquetas[np.isnan(X).any(axis=1)] = "dudoso"
    
    return etiquetas
