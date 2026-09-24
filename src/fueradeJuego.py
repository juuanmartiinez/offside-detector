import numpy as np

def direccionAtaque(direccion):

    if direccion == "izquierda":
        return -1
    elif direccion == "derecha":
        return 1

    raise ValueError(f"direccion invalida: {direccion!r} (izquierda o derecha)")

def ultimoDefensa(etiquetas, sentido, metros):

    metros    = np.asarray(metros)
    etiquetas = np.asarray(etiquetas)

    defensas = etiquetas == "def"

    if not defensas.any():
        return None

    xs = metros[defensas, 0]

    indices = np.flatnonzero(defensas)      

    return indices[np.argmax(sentido * xs)]

def lineaFueraDeJuego(metros, iDefensa):

    if iDefensa is None:
        return None

    return float(np.asarray(metros)[iDefensa, 0])

def veredicto(metros, etiquetas, sentido, xLinea, tolerancia=0.5):

    if xLinea is None:
        return []

    metros    = np.asarray(metros)
    etiquetas = np.asarray(etiquetas)

    resultado = []

    for i in np.flatnonzero(etiquetas == "ata"):

        margen = sentido * (metros[i, 0] - xLinea)

        if margen > tolerancia:
            estado = "fuera de juego"
        elif margen < -tolerancia:
            estado = "habilitado"
        else:
            estado = "ajustado"

        resultado.append((int(i), float(margen), estado))

    return resultado


def dudososEnRiesgo(metros, etiquetas, sentido, xLinea):

    if xLinea is None:
        return []

    metros    = np.asarray(metros)
    etiquetas = np.asarray(etiquetas)

    dudosos = np.flatnonzero(etiquetas == "dudoso")

    return [int(i) for i in dudosos if sentido * (metros[i, 0] - xLinea) > 0]