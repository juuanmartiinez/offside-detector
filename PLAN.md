# Detector de fuera de juego — Plan

Pipeline, no clasificador: **detección → equipos → homografía → línea de fuera de juego**.

Este documento cubre la **Fase 1** en detalle y deja las siguientes esbozadas.
Se actualiza al cerrar cada hito.

- **Framework:** PyTorch (torchvision para detección)
- **Hardware:** CPU, sin GPU dedicada
- **Estado:** Fase 1 — hitos 1.0 a 1.4 **cerrados**. Clasificación de equipos funcionando; falta el 1.5 (porteros y árbitros)

---

## El punto de partida que condiciona todo

El detector es **preentrenado sobre COCO, sin fine-tuning**. COCO tiene una sola
clase relevante: `person`. No tiene "jugador", ni "portero", ni "árbitro".

Consecuencia: **la detección de equipos y de porteros no sale del detector.**
El detector devuelve N cajas de personas — jugadores de ambos equipos, los dos
porteros, el árbitro, los linieres, banquillos y público si entra en plano.
Todo el "quién es quién" lo resuelve un segundo paso, por apariencia.

Y ese segundo paso no es clasificación supervisada (no hay etiquetas en
inferencia): es **clustering no supervisado**. Con dos complicaciones:

- **k=2 no vale.** Mínimo cinco grupos: equipo A, equipo B, árbitro,
  portero A, portero B.
- **Los porteros son un clúster de una sola muestra.** KMeans reparte por
  varianza; un punto aislado casi siempre acaba absorbido por el grupo más
  cercano. **Detectar al portero es el problema difícil de la Fase 1**, no un
  extra que sale gratis.

---

## Fase 1 — Una sola imagen

**Terminada cuando:** dada una imagen de partido, se obtienen las cajas de todas
las personas, cada una asignada a un rol (equipo A / equipo B / portero /
árbitro / descartado), con una medida de cuánto acierta frente al ground truth.

Seis hitos. **No se pasa al siguiente sin cerrar el anterior.** En cada uno el
entregable es visual: una imagen que se pueda mirar.

| # | Hito | Entregable | Qué se revisa |
|---|---|---|---|
| 1.0 | Dataset descargado y entendido | Sé leer el formato de anotación y qué significa cada campo | Formato de caja: ¿`xyxy` o `xywh`? ¿Normalizadas o en píxeles? |
| 1.1 | Carga y visualización | Una imagen con sus cajas **verdad** pintadas | BGR/RGB, orden de ejes, que las cajas caigan donde deben |
| 1.2 | Detección COCO | La misma imagen con las cajas **predichas** | `eval()`, `no_grad()`, umbral de score, filtrado a clase `person` |
| 1.3 | Evaluación de la detección | Precision/recall a un IoU dado + lista de fallos concretos | IoU implementado a mano, matching predicción↔verdad |
| 1.4 | Recortes y descriptor de apariencia | Un mosaico con el recorte de cada persona | Qué región del recorte se usa y por qué |
| 1.5 | Clustering en roles | La imagen con cada caja coloreada por grupo | Cuántos clústeres, y qué se hace con el portero |
| 1.6 | Evaluación del rol | Aciertos frente al `goalkeeper`/`referee` del dataset | Que la métrica no engañe |

### Regla de la línea base

En 1.2, **antes de tocar nada**, correr el detector con el umbral por defecto y
mirar el desastre. Esa imagen fea es la referencia. Sin ella no se sabe si las
mejoras mejoran.

### Tres decisiones de diseño (a decidir antes de escribir código)

1. **Región del recorte (1.4).** La caja completa de un jugador incluye césped en
   las esquinas, piernas y calcetines. ¿Torso? ¿Qué fracción de la caja? ¿Qué se
   hace con el césped que se cuela?
2. **Descriptor (1.4).** Histograma de color en HSV / media de color en RGB /
   embedding de una red preentrenada. Cada uno tiene un coste y un fallo
   característico.
3. **Separación del portero (1.5).** ¿Qué lo distingue **estructuralmente** del
   resto, más allá del color?

---

## Dónde lo dejé

### PASO 2 — circuito completo validado

Los cuatro sub-pasos funcionan de punta a punta:

```
2a  Dataset + DataLoader + collate   ✓  lotes con nº de cajas distinto
2b  Cirugía del modelo               ✓  cabeza de 5 clases, backbone congelado
2c  Bucle de entrenamiento           ✓  la pérdida baja (0,96 → 0,82)
2d  Evaluación sobre test            ✓  número comparable
```

Ficheros nuevos: `src/DetectionDataset.py` (adaptador a PyTorch + `collate`),
`src/modelo.py` (`crearModelo`). El notebook `1.2-train.ipynb` tiene el bucle.

### Tabla de resultados — ir rellenando

| Experimento | épocas | batch | backbone | recall | precisión |
|---|---|---|---|---|---|
| COCO sin afinar (sobre `train`) | — | — | — | 0,700 | 0,643 |
| **Roboflow afinado (techo)** | — | — | — | **0,983** | **0,983** |
| MobileNet, prueba de humo | 1 | 2 | congelado | 0,537 | 0,357 |
| MobileNet, 15-sep | 8 (¿+8 previas?) | 4 | congelado | 0,563 | 0,509 |
| | | | | | |

**El 0,537 no es un fracaso: es una prueba de humo.** La cabeza partió de pesos
aleatorios y tuvo 149 pasos de gradiente. El detector COCO daba 0,70 con una cabeza
entrenada con millones de imágenes.

**Y la pérdida seguía bajando al acabar la época** — evidencia directa de que el
entrenamiento se cortó, no de que se estancara.

### Lectura de la tanda del 15-sep

Costes por época: 0,7630 · 0,7650 · 0,7608 · 0,7571 · 0,7367 · 0,7283 · 0,7293 · 0,7257

Caída total en 8 épocas: **0,037**. La curva está prácticamente plana, y las dos
primeras épocas incluso suben. No es ruido de una época suelta: son ocho.

Frente a la prueba de humo (1 época): **precisión 0,357 → 0,509**, pero
**recall 0,537 → 0,563**. La precisión sube mucho, el recall casi nada.

Qué significa: la cabeza ha aprendido a *afinar y filtrar* las cajas que le llegan
(por eso sube la precisión), pero **no está apareciendo ningún jugador nuevo**.
Los que faltan siguen faltando. Con el backbone congelado, las características
sobre las que se decide son fijas: si un jugador pequeño o tapado no se distingue
en esas características, más épocas de cabeza no lo van a rescatar.

**Conclusión: el techo del backbone congelado está en ~0,56 de recall.** Más épocas
así no es la palanca. La siguiente es descongelar.

⚠️ Dato pendiente de confirmar: si la celda 0 cargó pesos previos del `.pt`, el total
real de épocas es 16, no 8. Corregir la fila antes de comparar con la siguiente.

### Plan de entrenamiento, una variable cada vez

1. **Más épocas** (8-10), MobileNet, congelado. El cambio con más recorrido.
2. **`batch_size` mayor** (4 u 8) si la RAM lo permite — gradientes menos ruidosos.
3. **ResNet50** (`crearModelo(ligero=False)`).
4. **Descongelar** el backbone con `lr` bajo, **solo si hace falta**. Si congelado
   llega a 0,90, quizá no.

Guardar los pesos con nombres que digan qué son: `modelo_mobilenet_10ep.pt`.

### Dos cosas que NO se van a hacer, y por qué

**Los equipos no pueden ser clases del detector.** Una clase de detección tiene que
significar lo mismo en todas las imágenes. `player` siempre es un jugador; "equipo A"
es de lima en una foto y blanco en otra — **no hay nada en común entre partidos**.
Los roles son globales (detección supervisada); los equipos son relativos a cada
imagen (clustering). Son dos tipos de problema distintos, no un apaño.

**Los árbitros no se quitan del entrenamiento.** Si se quitaran, el modelo los
seguiría detectando —parecen personas— pero los etiquetaría como `player`, y
entrarían en la lógica del fuera de juego. Lo que se quiere es lo contrario: que los
**etiquete bien** para poder filtrarlos con una línea. *Ignorar no se enseña: se
etiqueta y se filtra.*

### Transfer learning, para no perderlo

Lo que se conserva del modelo preentrenado es **el ver**, no **el nombrar**:

- **backbone** (congelado) → convierte píxeles en descripciones útiles. No sabe de
  clases. Eso es lo caro de aprender y vale igual para fútbol.
- **cabeza vieja** → sabía nombrar 91 clases de COCO. **Se borró entera**, incluido
  "persona". Por eso el modelo arrancó cerca de cero y no en 0,70.
- **cabeza nueva** → aleatoria, aprende las 5 clases propias.

Congelar **no es** lo que re-educa (eso es sustituir la cabeza y entrenar). Congelar
es decidir que una parte **no** se re-eduque, y se hace por tres motivos distintos:

| Motivo | Qué evita |
|---|---|
| Velocidad | minutos en vez de horas en CPU |
| Sobreajuste | 298 imágenes no dan para 40M de parámetros; el modelo memorizaría |
| Proteger lo preentrenado | la cabeza aleatoria genera gradientes enormes que destrozarían el backbone |

Los dos últimos son averías **distintas**: el sobreajuste ocurre poco a poco y se ve
en que test empeora mientras train mejora; el destrozo ocurre en las primeras
iteraciones y empeora todo a la vez.

---

## RUTA PARA SEGUIR SOLO

Todo lo que sigue se puede ejecutar sin ayuda. Lo importante no es la lista de
cosas que probar: es **el método**.

### El método, que vale para todo lo demás

1. **Cambia UNA variable.** Si tocas dos y mejora, no sabes cuál lo hizo. Y si
   empeora, tampoco.
2. **Mide sobre `test`**, nunca sobre `train`. Desde que se entrena, medir sobre
   train es medir memorización.
3. **Apunta el resultado en la tabla** con sus condiciones (épocas, batch,
   backbone, congelado). Un número sin condiciones no sirve para comparar.
4. **Compara contra la fila anterior**, no contra tu impresión.
5. **No concluyas desde pocas imágenes.** Ha fallado tres veces en este proyecto:
   el recall de la imagen 20, la cromaticidad de la 137, y el 148/298 de equipos.

### Cómo leer las señales

| Lo que ves | Qué significa | Qué hacer |
|---|---|---|
| Coste por época sigue bajando al acabar | paraste pronto | más épocas |
| Coste plano en las 3 últimas épocas | ha aprendido lo que podía con esta capacidad | cambiar otra cosa (ResNet50, descongelar) |
| Coste de train baja pero recall de test baja | **sobreajuste** | aumento de datos, o parar antes |
| Todo empeora de golpe tras descongelar | destrozaste lo preentrenado | bajar mucho el `lr` |
| Recall alto y precisión baja | detecta de más | subir el umbral de score |
| Recall bajo y precisión alta | detecta de menos | bajar el umbral, o entrenar más |

### Qué probar, en orden

**1. Más épocas hasta que el coste se estanque.** Es lo más barato y lo que más
recorrido tiene ahora. Sigue subiendo mientras la última época siga bajando.

**2. ResNet50** — `crearModelo(ligero=False)`. Más capacidad. Horas en CPU: déjalo
de noche. Mismo número de épocas que tu mejor resultado con MobileNet, para que la
comparación sea de una variable.

**3. Descongelar con `lr` bajo.** Solo cuando la cabeza ya prediga bien. Carga los
pesos entrenados, pon `requires_grad = True` en el backbone, y **baja el `lr` un
orden de magnitud** (de 0,005 a 0,0005). Si todo empeora de golpe, el `lr` sigue
siendo alto.

**4. Aumento de datos.** Con 298 imágenes es la palanca clásica contra el
sobreajuste. El volteo horizontal es gratis y multiplica por dos los datos.

**Trampa:** al voltear la imagen **hay que voltear también las cajas**, o el target
deja de corresponder. Para una imagen de ancho `W`, una caja `(x1,y1,x2,y2)` pasa a
`(W-x2, y1, W-x1, y2)`. Ojo al orden: el x2 nuevo sale del x1 viejo. Va dentro de
`__getitem__`, aplicado solo en entrenamiento, nunca en test.

**5. Barrer el umbral de score** sobre el mejor modelo, como en el hito 1.3. Es
gratis (no reentrena) y mueve mucho el compromiso recall/precisión. Recuerda el
criterio del proyecto: **un jugador no detectado puede invalidar el fuera de juego;
un falso positivo se filtra después.** Protege el recall.

**6. Si el recall se atasca:** mirar las imágenes peores, como en el 1.3. Ordenar
por fallos y abrir las seis primeras. El patrón que encuentres dirá qué falta.

### Cuándo parar

Cuando dos experimentos seguidos no muevan el recall de `test`. El techo medido es
**0,983**; si llegas a 0,90 con el detector funcionando, el cuello de botella deja
de ser la detección y hay que pasar a lo siguiente.

---

## EL RESTO DEL PROYECTO, después del detector

**Fase 1, lo que queda.** Con el detector afinado devolviendo `player`,
`goalkeeper`, `referee` y `ball` etiquetados:

- Los **árbitros** se filtran con una línea. El problema de clustering desaparece.
- Los **equipos** se agrupan sobre cajas de `player` limpias — reevaluar el
  desequilibrio 148/298, que puede haber mejorado solo.
- Los **porteros** vienen identificados, pero su equipo sigue pendiente de la
  geometría.

**Fase 2 — punto de apoyo.** De cada caja, el punto donde el jugador pisa el
césped. El centro inferior de la caja es la primera aproximación, y falla cuando el
jugador corre o salta.

**Fase 3 — homografía.** Detectar líneas y puntos del campo, y calcular la matriz
que lleva de la imagen a un plano cenital. Aquí se resuelve también el equipo del
portero (el de la portería atacada defiende) y el filtro de "dentro del campo".
Referencia: el paper de SoccerNet Game State Reconstruction.

**Fase 4 — la línea.** Dirección de ataque, penúltimo defensor, comparación en
coordenadas del campo. Aquí se acumulan todos los errores anteriores.

**Fase 5 — vídeo.** El fuera de juego se juzga en el instante del pase, no en un
frame cualquiera. Detectar ese instante es un problema en sí mismo.

---

## Hallazgos del hito 1.0

Medido sobre el split `train` (298 imágenes, 7133 anotaciones).

### Formato de las anotaciones

- COCO estándar: tres listas enlazadas por identificadores — `images`, `annotations`, `categories`
- `bbox` = `[x, y, ancho, alto]` en **píxeles** (confirmado porque `ancho × alto == area`)
- Los `id` empiezan en **0**, no en 1 como el COCO original
- `categories` trae una **categoría raíz** (`id: 0`, `football-players-detection`) que **no es una clase**. Hay que filtrarla o `len(categories)` da 5 en vez de 4
- Clases reales: `1: ball`, `2: goalkeeper`, `3: player`, `4: referee`

### Estadísticas

| Clase | Cajas | Por imagen |
|---|---|---|
| `player` | 5955 | 19,98 |
| `referee` | 690 | 2,32 |
| `ball` | 258 | 0,87 |
| `goalkeeper` | 230 | 0,77 |

Porteros: **214 de 298 imágenes** tienen al menos uno (72%); 84 no tienen ninguno.
Distribución: **198 imágenes con 1 portero, 16 con 2**.

### Qué implican

**Planos muy abiertos.** ~24 cajas por imagen significa que cabe casi la plantilla
entera, luego la cámara está lejos. Consecuencia directa: hay jugadores diminutos.
La primera anotación del fichero es un `player` de **4,5 × 25 píxeles**. No es un
caso raro, es lo normal en el fondo del campo.

**El balón no siempre está.** 0,87 por imagen: en ~40 imágenes no se ve. Irrelevante
en Fase 1, decisivo en Fase 5 (detectar el instante del pase).

**El hito 1.6 es viable.** 214 imágenes con portero es ground truth de sobra para
que un porcentaje de acierto signifique algo. No hace falta recortar el alcance.

**El hito 1.5 cambia de forma.** En una imagen típica hay ~20 jugadores, ~2 árbitros
y **1 portero**: una persona entre veintitrés, un 4%. Un KMeans plano con k=4 o k=5
se traga ese punto solitario, porque veinte jugadores dominan la varianza. Así que
el 1.5 **no es "agrupar en cinco"**: es *agrupar en dos equipos y luego detectar
quién no encaja en ninguno*. Detección de anomalía, no clustering plano.

### Decisiones tomadas

- **Una sola convención de cajas en todo el proyecto: `xyxy`**, que es lo que habla
  torchvision. El ground truth se convierte **una vez, en la carga**, nunca en cada
  punto de uso.
- Reservar un split y no tocarlo hasta el final: elegir el umbral de score en el
  1.3 es ajustar un parámetro, y hacerlo sobre los mismos datos con los que luego
  se reporta es fuga de datos (`C3-Modulo-07`).
- **Pendiente para el 1.3:** decidir y justificar un **área mínima** por debajo de
  la cual un jugador no se considera detectable. Un jugador de 4 píxeles no
  participa en un fuera de juego juzgable. Que sea una decisión explícita, no un
  accidente.
- **Experimento opcional del 1.3:** forkear el dataset en Roboflow y generar una
  versión sin resize. Si sale, se tiene el mismo dataset estirado y original →
  misma imagen, mismas anotaciones, mismo detector, una sola variable cambiada
  (el aspect ratio) y una métrica que mide el efecto (el recall). Verificación de
  que el fork ha servido: `images[0]` debe dar 16:9, no cuadrado.

### Lecciones de método

- **Al contar cosas repartidas en categorías, comprobar siempre que la suma cuadre
  con el total.** Un `if/if/if/else` mal encadenado infló los árbitros de 690 a
  1178 sin dar ningún error, y los números seguían siendo creíbles. Solo la suma
  (7621 ≠ 7133) lo delató.
- **Una comprobación tiene que gritar cuando falla, no susurrar cuando acierta.**
  Un `assert` o un `else` ruidoso, no un `print` de éxito que se echa de menos.
- **Un `id` se busca, no se indexa.** Que `categories[1]` tenga `id: 1` es una
  coincidencia de este fichero, no una garantía.
- **Predecir el resultado antes de medirlo.** 230 cajas en 214 imágenes ⇒ 16 de
  sobra ⇒ 16 imágenes con dos porteros. El `Counter` lo confirmó.

---

## Hallazgos del hito 1.1

Verificado pintando las anotaciones sobre una imagen con dos porteros
(`outputs/1.1-gt.png`). Las cajas caen ajustadas: el formato COCO está bien
entendido.

**Los porteros están en los bordes horizontales de la imagen.** Uno en el borde
izquierdo, otro en el derecho y medio cortado por el marco. Tiene sentido: cada
portero está en su portería, y en un plano tan abierto las porterías caen en los
extremos. Es una **pista espacial** aprovechable en el 1.5, además del color de
la equipación.

**Hay personas en la imagen que no están anotadas, y con razón.** Personal de
banda, banquillos y público no son ninguna de las cuatro clases. Pero el detector
COCO no lo sabe: para él son `person` igual que los jugadores. Los de la grada se
salvan por resolución (dos o tres píxeles), pero **los de la banda tienen el mismo
tamaño en píxeles que muchos jugadores** y sí serán detectados.

Consecuencia para el 1.3: **esas detecciones contarán como falsos positivos sin
serlo.** El detector acierta —ahí hay una persona— pero el ground truth solo anota
jugadores. Se estaría midiendo el desajuste entre dos definiciones de "qué cuenta",
no un fallo del modelo. Sin tenerlo presente, la conclusión falsa sería "el
detector COCO tiene una precisión malísima".

**Dependencia circular entre fases.** El filtro natural sería "quedarse solo con
las personas dentro del terreno de juego", pero saber dónde está el campo es
justamente lo que se construye en la **fase 3** (homografía). Las fases no son tan
independientes como parecían en el plan.

Sustitutos baratos para la Fase 1, de más tosco a más fino — **decisión pendiente
del 1.3**:

- corte por coordenada `y` (funciona en este encuadre, se rompe al cambiarlo)
- polígono del campo dibujado a mano (honesto para una imagen, no escala)
- test de "está sobre césped": mirar el color bajo los pies de la caja
- filtro por tamaño mínimo (ya pendiente por el asunto de los jugadores diminutos)

**Lección de método:** lo que se puede confirmar con datos no se concluye mirando.
La imagen sirve para detectar errores grandes; los pequeños se cuentan.

---

## Trampas conocidas que van a aparecer

De la lista de errores ya estudiados en la bóveda. Cuando fallen, deberían
reconocerse.

| Trampa | Síntoma | Nota |
|---|---|---|
| **`model.eval()` ausente** | En torchvision los modelos de detección en modo `train` esperan `targets` y devuelven **pérdidas**, no detecciones. No salen cajas: sale un dict de losses o un error | `C4-Modulo-04` |
| **Canales invertidos** | `cv2.imread` devuelve **BGR**, el modelo espera **RGB**. **No da error**: da detecciones peores y colores de equipo mal clasificados en 1.5 | Code review, prioridad 3 |
| **Normalización duplicada** | Los detectores de torchvision normalizan internamente. Normalizar antes con medias de ImageNet lo hace dos veces. **Tampoco da error** | — |
| **`torch.no_grad()` ausente** | No rompe nada, pero gasta memoria y tiempo sin motivo en inferencia | `C3-Modulo-04` |
| **IoU y mAP** | Definidos en el glosario pero nunca calculados. El hito 1.3 es el momento | `C5-Modulo-02` |

---

## Dataset

**Elegido: `football-players-detection-3zvbc`, versión 20 (Roboflow Universe).**

- Workspace `roboflow-jvuqo`, 372 imágenes de broadcast
- Clases: `ball`, `player`, `goalkeeper`, `referee`
- Licencia CC BY 4.0
- Descarga en formato **COCO JSON** a `data/raw/`
- https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/20

**Por qué éste:** en Fase 1 no se entrena nada, así que lo que hace falta no es
dataset de entrenamiento sino **ground truth para evaluar**. Da las dos verdades
que faltan sin anotar nada a mano: dónde están las personas (para medir qué se le
escapa al detector COCO, hito 1.3) y qué caja es portero o árbitro (para medir el
clustering, hito 1.6).

**Por qué la versión 20 y no otra.** El proyecto tiene 20 versiones y casi todas
llevan preprocesado: la v19 redimensiona a 640×640 con *stretch*, la v18 a
512×512 con *stretch*. Estirar un plano panorámico de TV a un cuadrado deforma a
los jugadores, y el detector COCO —que nunca vio a nadie deformado así— rinde
peor. **La v20 no lleva resize**: conserva el aspect ratio original. Es la única
razón por la que se elige, y conviene recordarla si algún día se cambia de
versión.

Descargado desde la pestaña **Dataset** de la página del proyecto (no desde
*Deploy Model*, que sirve el modelo ya afinado por API y se saltaría toda la
Fase 1). **No versionar** (ver `.gitignore`).

### Alternativas descartadas para Fase 1

| Dataset | Por qué no ahora | Para cuándo |
|---|---|---|
| [SoccerNet Game State Reconstruction](https://www.soccer-net.org/tasks/game-state-reconstruction) | Es este proyecto entero hecho por un equipo de investigación: detección + equipo + dorsal + minimapa. Grande y con términos de descarga que aceptar | Referencia de arquitectura al llegar a la homografía, y para saber qué métricas usa gente seria. [Paper](https://arxiv.org/html/2404.11335) · [Código](https://github.com/soccernet/sn-gamestate) |
| [DFL Bundesliga (Kaggle)](https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout) | Vídeos de 30s: obliga a extraer frames y anotarlos a mano | Al pasar de imagen a secuencia |

### Referencia de implementación

[`roboflow/sports`, ejemplo soccer](https://github.com/roboflow/sports/tree/main/examples/soccer)
— resuelve los equipos con embeddings SigLIP → UMAP → KMeans. Es el extremo
pesado del espectro frente a un histograma de color. **Leer, no copiar**: sirve
para calibrar la decisión de diseño nº 2.

---

## Cómputo

Sin GPU:

- `fasterrcnn_resnet50_fpn` — unos segundos por imagen en CPU. Irrelevante para
  una imagen; minutos para evaluar sobre 100. Aceptable.
- `fasterrcnn_mobilenet_v3_large_fpn` — mismo API, mucho más rápido, si la
  evaluación se hace pesada.

---

## Fases siguientes (esbozo)

| Fase | Qué añade | Riesgo principal |
|---|---|---|
| 2 | Punto de apoyo: extraer el punto del pie de cada jugador desde su caja | El pie no está en el centro inferior de la caja cuando el jugador corre |
| 3 | Homografía: detectar líneas/puntos del campo y calcular la matriz imagen→plano cenital | Encontrar suficientes correspondencias en un plano de TV cerrado |
| 4 | Línea de fuera de juego: dirección de ataque, penúltimo defensor, comparación en coordenadas del campo | Errores de la homografía se amplifican; margen de error en cm |
| 5 | Vídeo: el instante del pase es lo que decide el fuera de juego, no un frame cualquiera | Detectar el momento del pase es un problema en sí mismo |

---

## Estructura del repositorio

```
offside-detector/
├── PLAN.md              este documento
├── requirements.txt
├── data/
│   ├── raw/             dataset descargado (no versionado)
│   └── interim/         recortes, cachés (no versionado)
├── notebooks/           exploración, un notebook por hito
├── src/                 código que se estabiliza y se reutiliza
└── outputs/             imágenes generadas, entregables visuales de cada hito
```
