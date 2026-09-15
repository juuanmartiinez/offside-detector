import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.preprocessing import RobustScaler

def colorRecorte(recorte):

    arr = np.array(recorte)
    return np.median(arr, axis=(0,1))


def clasificarEquipos(colores, k=2, semilla=0):

    km = KMeans(n_clusters=k, n_init=10, random_state=semilla).fit(colores)

    return km.labels_

def descriptor(colores):

    cromaticidad = colores / colores.sum(axis=1, keepdims=True)
    brillo = colores.sum(axis=1, keepdims=True)
    
    return RobustScaler().fit_transform(np.hstack([cromaticidad[:, :2], brillo]))


