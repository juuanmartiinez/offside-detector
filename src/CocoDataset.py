import json
from pathlib import Path
from PIL import Image

def _xyxy(bbox):
        x, y, w, h = bbox
        return(x, y, x + w, y + h)

class CocoDataset:

    def __init__(self, carpeta):

        self.carpeta = Path(carpeta)

        with open(self.carpeta / "_annotations.coco.json") as f:
            self.data = json.load(f)

        self.id2clase = {}
        for c in self.data["categories"]:
            if c["id"] != 0:
                self.id2clase[c["id"]] = c["name"]

        self.id2fichero = {}
        for i in self.data["images"]:
            self.id2fichero[i["id"]] = i["file_name"]

        self.anotaciones = {}
        for a in self.data["annotations"]:
            img = a["image_id"]
            if img not in self.anotaciones:
                self.anotaciones[img] = []
            self.anotaciones[img].append(a)

    def imagen(self, img_id):
        return Image.open(self.carpeta / self.id2fichero[img_id])

    def cajas(self, img_id):

        resultado = []

        for a in self.anotaciones[img_id]:
            x, y, w, h = a["bbox"]
            clase = self.id2clase[a["category_id"]]
            caja = (x, y, x + w, y + h)

            resultado.append((caja, clase))             

        return resultado

    def cajasYEtiquetas(self, img_id):                 # usada para entrenamiento

        cajas, etiquetas = [], []
        for a in self.anotaciones[img_id]:
            cajas.append(_xyxy(a["bbox"]))
            etiquetas.append(a["category_id"])

        return cajas, etiquetas

