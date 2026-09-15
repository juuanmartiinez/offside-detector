import numpy as np
from src.viz import recortesTorso
from src.equipos import colorRecorte, descriptor, clasificarEquipos

def analizarImagen(ds, imgId, cajas=None):

    if cajas is None:
            cajas = [c for c, k in ds.cajas(imgId) if k != "ball"]

    recortes = recortesTorso(ds, imgId, cajas)
    colores  = np.array([colorRecorte(r) for r in recortes])

    X = descriptor(colores)
    et = clasificarEquipos(X, k=2)

    pares = [(caja, f"e{e}") for caja, e in zip(cajas, et)]

    return pares
    

    