from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    fasterrcnn_mobilenet_v3_large_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def crearModelo(numClases=5, congelarBackbone=True, ligero=True):

    if ligero:                                                                      # Cargamos el modelo
        modelo = fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")   
    else:
        modelo = fasterrcnn_resnet50_fpn(weights="DEFAULT")

    in_features = modelo.roi_heads.box_predictor.cls_score.in_features              # Lee cuántos números le llegan a la cabeza vieja desde la capa anterior

    modelo.roi_heads.box_predictor = FastRCNNPredictor(in_features, numClases)      # Sustituir la cabeza del modelo

    if congelarBackbone:                                                            # Congelar el backbone 
        for p in modelo.backbone.parameters():
            p.requires_grad = False

    return modelo