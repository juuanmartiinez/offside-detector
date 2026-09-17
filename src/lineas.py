import cv2
import numpy as np


def realce(imagen, kernel=13):

    a = np.asarray(imagen)
    g = cv2.cvtColor(a[:, :, :3], cv2.COLOR_RGB2GRAY) if a.ndim == 3 else a
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, k)

    return cv2.normalize(th, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _fusionar(rectas, tolAng=np.deg2rad(4), tolRho=14):

    grupos = []
    for rho, theta in rectas:
        for g in grupos:
            r0, t0 = g[0]
            dt = abs(theta - t0)
            dr = abs(rho - r0)
            if min(dt, np.pi - dt) < tolAng and dr < tolRho:
                g.append((rho, theta))
                break
            # una recta es la misma con (-rho, theta+pi)
            if min(abs(dt - np.pi), abs(dt + np.pi)) < tolAng and abs(rho + r0) < tolRho:
                g.append((-rho, theta - np.pi))
                break
        else:
            grupos.append([(rho, theta)])
    return [tuple(np.mean(g, axis=0)) for g in grupos]


def detectarLineas(imagen, mascara=None, minLargo=100, huecoMax=12, votos=45):

    th = realce(imagen)
    if mascara is not None:
        th = np.where(mascara, th, 0).astype(np.uint8)

    _, binaria = cv2.threshold(th, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    segs = cv2.HoughLinesP(binaria, 1, np.pi / 360, votos,
                           minLineLength=minLargo, maxLineGap=huecoMax)
    if segs is None:
        return []

    rectas = []
    for x1, y1, x2, y2 in segs[:, 0]:
        theta = np.arctan2(y2 - y1, x2 - x1) + np.pi / 2
        rho = x1 * np.cos(theta) + y1 * np.sin(theta)
        rectas.append((rho, theta))
    return _fusionar(rectas)


def interseccion(r1, r2):
    rho1, t1 = r1
    rho2, t2 = r2
    A = np.array([[np.cos(t1), np.sin(t1)], [np.cos(t2), np.sin(t2)]])
    if abs(np.linalg.det(A)) < 1e-6:
        return None
    x, y = np.linalg.solve(A, np.array([rho1, rho2]))
    return (float(x), float(y))


def puntosDeRecta(recta, ancho, alto):
    rho, theta = recta
    c, s = np.cos(theta), np.sin(theta)
    x0, y0 = c * rho, s * rho
    L = max(ancho, alto) * 2
    return ((x0 - L * s, y0 + L * c), (x0 + L * s, y0 - L * c))