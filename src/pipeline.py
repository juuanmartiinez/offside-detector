import numpy as np
import torch

from src.viz import recortesTorso
from src.equipos import descriptorCamisetas, clasificarPorSemillas, colorCesped
from torchvision.transforms.functional import to_tensor
from src.geometria import calcularHomografia, puntoApoyo, aMetros
from src.campo import enCampo
from src.fueradeJuego import (direccionAtaque, ultimoDefensa, lineaFueraDeJuego,
                                veredicto, dudososEnRiesgo)

def descriptores(imagen, cajas):

    recortes = recortesTorso(imagen, cajas)

    # El color del cesped se mide UNA vez por foto, no por recorte
    cesped = colorCesped(imagen)

    return np.array([descriptorCamisetas(r, cesped) for r in recortes])


def analizarImagen(imagen, cajas, iAtacante, iDefensor):

    X = descriptores(imagen, cajas)
    etiquetas = clasificarPorSemillas(X[:, :1], iAtacante, iDefensor)

    return list(zip(cajas, etiquetas))
    
def detectar(modelo, imagen, umbral=0.5, clase=3):

    with torch.no_grad():
        pred = modelo([to_tensor(imagen)])[0]

    cajas = [b for b, l, s in zip(pred["boxes"].tolist(),
                                    pred["labels"].tolist(),
                                    pred["scores"].tolist())
            if int(l) == clase and s >= umbral]

    return cajas

def analizarJugada(imagen, cajas, correspondencias, iAtacante, iDefensor,
                   direccion, tolerancia=0.5):

    pares = analizarImagen(imagen, cajas, iAtacante, iDefensor)
    etiquetas = np.array([e for _, e in pares])

    
    H, Hinv = calcularHomografia(correspondencias)
    metros = aMetros(H, [puntoApoyo(c) for c in cajas])

    dentro    = np.array([enCampo(m) for m in metros])
    metros    = metros[dentro]
    etiquetas = etiquetas[dentro]
    cajas     = [c for c, d in zip(cajas, dentro) if d]

    sentido = direccionAtaque(direccion)
    iDef    = ultimoDefensa(etiquetas, sentido, metros)
    xLinea  = lineaFueraDeJuego(metros, iDef)

    return {
        "cajas":     cajas,        
        "metros":    metros,       
        "etiquetas": etiquetas,    
        "H":         H,
        "Hinv":      Hinv,
        "sentido":   sentido,
        "iDefensa":  iDef,
        "xLinea":    xLinea,
        "resultado": veredicto(metros, etiquetas, sentido, xLinea, tolerancia),
        "riesgo":    dudososEnRiesgo(metros, etiquetas, sentido, xLinea),
    } 