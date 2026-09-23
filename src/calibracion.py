import matplotlib.pyplot as plt
from src.viz import dibujarCajas

PUNTOS_CAMPO = {
    # --- córners
    "corner_izq_lejano":    (0.0,    0.0),
    "corner_izq_cercano":   (0.0,   68.0),
    "corner_der_lejano":    (105.0,  0.0),
    "corner_der_cercano":   (105.0, 68.0),

    # --- área grande, donde toca la línea de meta
    "areaMeta_izq_lejana":  (0.0,   13.84),
    "areaMeta_izq_cercana": (0.0,   54.16),
    "areaMeta_der_lejana":  (105.0, 13.84),
    "areaMeta_der_cercana": (105.0, 54.16),

    # --- área grande, esquinas exteriores
    "area_izq_lejana":      (16.5,  13.84),
    "area_izq_cercana":     (16.5,  54.16),
    "area_der_lejana":      (88.5,  13.84),
    "area_der_cercana":     (88.5,  54.16),

    # --- área pequeña, donde toca la línea de meta
    "chicaMeta_izq_lejana": (0.0,   24.84),
    "chicaMeta_izq_cercana":(0.0,   43.16),
    "chicaMeta_der_lejana": (105.0, 24.84),
    "chicaMeta_der_cercana":(105.0, 43.16),

    # --- área pequeña, esquinas exteriores
    "chica_izq_lejana":     (5.5,   24.84),
    "chica_izq_cercana":    (5.5,   43.16),
    "chica_der_lejana":     (99.5,  24.84),
    "chica_der_cercana":    (99.5,  43.16),

    # --- puntos de penalti
    "penalti_izq":          (11.0,  34.0),
    "penalti_der":          (94.0,  34.0),

    # --- medio campo y círculo central
    "medio_lejano":         (52.5,   0.0),
    "medio_cercano":        (52.5,  68.0),
    "circulo_lejano":       (52.5,  24.85),
    "circulo_cercano":      (52.5,  43.15),
    "centro":               (52.5,  34.0),
}

def marcarPuntos(imagen, orden):

    for nombre in orden:
        if nombre not in PUNTOS_CAMPO:
            raise ValueError(f"'{nombre}' no está en PUNTOS_CAMPO")

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(imagen)

    pixeles = plt.ginput(len(orden), timeout=0)

    return [(pixel, PUNTOS_CAMPO[nombre]) for pixel, nombre in zip(pixeles, orden)]

def marcarJugadores(imagen, cajas):

    fig, ax = plt.subplots(figsize=(12,12))
    ax.imshow(imagen)
    dibujarCajas(ax, [(c, "x") for c in cajas], colores={"x": "cyan"})

    print(f"Pincha primero a un atacante y luego a un defensor cualquiera.")

    while True:
        pixeles = plt.ginput(2, timeout=0)

        atacante = cajaEnPunto(cajas, pixeles[0])
        defensor = cajaEnPunto(cajas, pixeles[1])

        if atacante is None or defensor is None:
            print("algun clic no cayo dentro de ninguna caja, repite")
        elif atacante == defensor:
            print("has pinchado dos veces al mismo jugador")
        else:
            break

    plt.close(fig)

    return atacante, defensor

def cajaEnPunto(cajas, punto):

    x, y = punto

    dentro = [i for i, (x1, y1, x2, y2) in enumerate(cajas)
              if x1 <= x <= x2 and y1 <= y <= y2]

    if not dentro:
        return None

    return min(dentro, key=lambda i: (cajas[i][2] - cajas[i][0]) * (cajas[i][3] - cajas[i][1]))