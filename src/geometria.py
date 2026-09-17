import numpy as np, cv2

def puntoApoyo(caja):
    x1, y1, x2, y2 = caja

    x = (x1 + x2) / 2
    y = y2 

    return (x, y)

def calcularHomografia(correspondencias):

    pixeles = [p for p, m in correspondencias]
    metros  = [m for p, m in correspondencias]

    src = np.array(pixeles, dtype=np.float32)
    dst = np.array(metros,  dtype=np.float32)

    H, _ = cv2.findHomography(src, dst)
    inversa = np.linalg.inv(H)

    return H, inversa

def aMetros(H, puntos):
    
    a = np.array(puntos, dtype=np.float32).reshape(-1, 1, 2)
    b = cv2.perspectiveTransform(a, H)
    return b.reshape(-1, 2)