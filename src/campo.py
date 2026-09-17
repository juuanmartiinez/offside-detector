import cv2
import numpy as np


def _aBGR(imagen):
   
    a = np.asarray(imagen)
    if a.ndim == 2:
        return cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(a[:, :, :3], cv2.COLOR_RGB2BGR)


def mascaraCampo(imagen, tono=(32, 92), satMin=45, valMin=35, cierre=15):

    hsv = cv2.cvtColor(_aBGR(imagen), cv2.COLOR_BGR2HSV)
    verde = cv2.inRange(hsv, (tono[0], satMin, valMin), (tono[1], 255, 255))

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cierre, cierre))
    verde = cv2.morphologyEx(verde, cv2.MORPH_CLOSE, k)

    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(verde, 8)
    if n <= 1:
        return np.zeros(verde.shape, bool)
    mayor = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mascara = etiquetas == mayor

    fuera = (~mascara).astype(np.uint8)
    h, w = fuera.shape
    flood = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(fuera, flood, (0, 0), 2)          
    for semilla in ((w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        if fuera[semilla[1], semilla[0]] == 1:
            cv2.floodFill(fuera, flood, semilla, 2)
    return mascara | (fuera == 1)


def estaEnCampo(mascara, punto):
    x, y = int(round(punto[0])), int(round(punto[1]))
    alto, ancho = mascara.shape
    if not (0 <= x < ancho and 0 <= y < alto):
        return False
    return bool(mascara[y, x])


def filtrarEnCampo(mascara, puntos):
    return [p for p in puntos if estaEnCampo(mascara, p)]
