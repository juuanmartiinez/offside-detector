import numpy as np

from src.viz import recortesTorso
from src.equipos import descriptorCamisetas, clasificarPorSemillas

def descriptores(ds, imgId, cajas):

    recortes = recortesTorso(ds, imgId, cajas)

    return np.array([descriptorCamisetas(r) for r in recortes])


def analizarImagen(ds, imgId, cajas, iAtacante, iDefensor):

    X = descriptores(ds, imgId, cajas)
    etiquetas = clasificarPorSemillas(X[:, :1], iAtacante, iDefensor)

    return list(zip(cajas, etiquetas))
    

    