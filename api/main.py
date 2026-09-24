from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles          

from PIL import Image

from src.modelo import crearModelo

from src.pipeline import detectar, analizarJugada
from api.esquema import PeticionAnalisis

CARPETA = Path("data/subidas")
CARPETA.mkdir(parents=True, exist_ok=True)
sesiones = {}
modelo = None

@asynccontextmanager
async def lifespan(app):                                            # corre una vez al arrancar
    global modelo
    modelo = crearModelo(congelarBackbone=False, ligero=False)
    modelo.load_state_dict(torch.load("outputs/modelo_experimento_ResNet_20ep.pt",
                                      map_location="cpu"))
    modelo.eval()
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/api/imagen")
async def subirImagen(archivo: UploadFile = File()):

    if not archivo.content_type or not archivo.content_type.startswith("image/"):
        raise HTTPException(400, "el fichero tiene que ser una imagen")

    idImagen = uuid4().hex
    ruta = CARPETA / f"{idImagen}.png"

    contenido = await archivo.read()
    ruta.write_bytes(contenido)

    imagen = Image.open(ruta).convert("RGB")
    cajas = detectar(modelo, imagen)

    sesiones[idImagen] = {"ruta": ruta, "cajas": cajas}

    return {"id": idImagen,                            
            "ancho": imagen.width,
            "alto": imagen.height,
            "cajas": cajas}

@app.post("/api/analizar")
def analizar(peticion: PeticionAnalisis):

    sesion = sesiones.get(peticion.id)
    if sesion is None:
        raise HTTPException(404, "imagen no encontrada")

    cajas = sesion["cajas"]

    for i in (peticion.iAtacante, peticion.iDefensor):
        if not 0 <= i < len(cajas):
            raise HTTPException(400, f"indice de jugador fuera de rango: {i}")

    imagen = Image.open(sesion["ruta"]).convert("RGB")

    try:
        r = analizarJugada(imagen, cajas,
                           peticion.correspondencias,
                           peticion.iAtacante,
                           peticion.iDefensor,
                           peticion.direccion,
                           peticion.tolerancia)
    except ValueError as e:
        raise HTTPException(400, str(e))

    return {
        "cajas":     r["cajas"],
        "metros":    r["metros"].tolist(),
        "etiquetas": r["etiquetas"].tolist(),
        "sentido":   int(r["sentido"]),
        "iDefensa":  None if r["iDefensa"] is None else int(r["iDefensa"]),
        "xLinea":    r["xLinea"],
        "resultado": r["resultado"],
        "riesgo":    r["riesgo"],
    }

app.mount("/", StaticFiles(directory="web", html=True), name="web")