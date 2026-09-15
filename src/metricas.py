def iou(a, b):

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    izq = max(ax1, bx1)
    arriba = max(ay1, by1)
    der = min(ax2, bx2)
    abajo = min(ay2, by2)

    ancho = der - izq
    alto =  abajo - arriba

    if ancho < 0.0 or alto < 0.0:
        return 0.0

    interseccion = ancho * alto

    area1 = (ax2 - ax1) * (ay2 - ay1)
    area2 = (bx2 - bx1) * (by2 - by1)

    union = (area1 + area2) - interseccion

    return interseccion / union

def emparejar(verdad, predichas, umbral=0.5):

    parejas = []
    usadas = set()

    for i, caja_v in enumerate(verdad):

        mejor_iou = 0.0
        mejor_j = None

        for j, caja_p in enumerate(predichas):

            if j in usadas:
                continue

            res = iou(caja_v, caja_p)

            if res > mejor_iou:
                mejor_iou = res
                mejor_j = j

        if mejor_iou >= umbral:
            parejas.append((i, mejor_j, mejor_iou))
            usadas.add(mejor_j)

    v_sin = [i for i in range(len(verdad)) if i not in [p[0] for p in parejas]]
    p_sin = [j for j in range(len(predichas)) if j not in usadas]

    return parejas, v_sin, p_sin

def evaluar(ds, predicciones, clases=("player", "goalkeeper", "referee"), umbral=0.5):

    aciertos = 0
    perdidos = 0
    falsos = 0

    for i in sorted(ds.id2fichero):
        verdad = [c for c, k in ds.cajas(i) if k in clases]
        predichas = predicciones.get(i) or predicciones.get(str(i)) or {}
        predichas = predichas.get("boxes", [])

        parejas, sinPareja, sobrantes = emparejar(verdad, predichas, umbral)

        aciertos += len(parejas)
        perdidos += len(sinPareja)
        falsos += len(sobrantes)

    recall    = aciertos / (aciertos + perdidos)
    precision = aciertos / (aciertos + falsos)

    return recall, precision

