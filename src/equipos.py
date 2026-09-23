import numpy as np
import cv2

def descriptorCamisetas(recorte, tono=(25, 95), satMin=40, valMin=30):

    arr = np.array(recorte)[:, :, :3]

    if arr.size == 0:
        return np.full(3, np.nan)

    hsv   = cv2.cvtColor(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
    verde = cv2.inRange(hsv, (tono[0], satMin, valMin), (tono[1], 255, 255))

    pix = hsv[verde == 0]

    if len(pix) < 12 or len(pix) < 0.20 * len(hsv.reshape(-1, 3)):
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