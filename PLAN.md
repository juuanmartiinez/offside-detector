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

**Estado al cerrar el 16-sep:** ResNet50 descongelado, 10 épocas, **recall 0,948 ·
precisión 0,732** en `valid` con umbral 0,20. Entrenando en Colab (T4), ~1 min/época.
Pesos y dataset en Drive (`Colab Notebooks/`). `test` sigue sin tocarse.

### Próxima sesión — Fase 2: el punto de apoyo

**0. Cerrar Fase 1** (5 min). Anotar la fila del último tramo de 10 épocas en la tabla
y guardar los pesos con su nombre. Con eso el detector queda cerrado y no se vuelve.

**1. El punto de apoyo.** Una caja `(x1,y1,x2,y2)` hay que reducirla a **un punto del
suelo**, porque la homografía transforma puntos, no rectángulos. La aproximación de
trabajo es el centro del borde inferior: `((x1+x2)/2, y2)`.

Función nueva, probablemente en un `src/geometria.py`. Verificación visual obligatoria:
pintar el punto sobre las imágenes y comprobar que cae en los pies.

Casos que van a fallar y hay que ver con los ojos antes de decidir nada:
- jugador en carrera → la caja llega al pie más adelantado, no al punto de apoyo real
- jugador saltando → no hay contacto con el suelo
- portero tirado en el suelo → la caja es ancha y baja, el centro inferior no significa nada

No inventar correcciones antes de mirar cuántos casos hay. Puede que la aproximación
simple baste para la inmensa mayoría.

**2. Reencajar el clasificador de equipos.** Los hitos 1.4/1.5 se hicieron sobre cajas
*anotadas*. Ahora hay que pasarle las cajas *predichas*, que son más ruidosas. Es un
cambio pequeño en `pipeline.analizarImagen` y conviene comprobar que no se rompe.

**Recordatorio de prioridad:** el detector está resuelto (0,948). El proyecto está
parado en la geometría, no en el modelo. No volver a abrir el detector hasta tener la
línea dibujada de extremo a extremo.

### Riesgo abierto

El notebook con los cambios de Colab (`device`, rutas de Drive, traslado de tensores)
**solo existe en Colab**. Subirlo a GitHub con Archivo → Guardar una copia en GitHub,
o se pierde con la sesión.


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
| MobileNet, 15-sep | 16 | 4 | congelado | 0,563 | 0,509 |
| MobileNet descongelado (Colab T4) | 16+10 | 4 | **descongelado**, lr 0,0005 | **0,622** | **0,557** |
| MobileNet descongelado, 2º tramo | 16+20 | 4 | descongelado, lr 0,0005 | 0,657 | 0,626 |
| MobileNet descongelado, 3º tramo | 16+30 | 4 | descongelado, lr 0,0005 | 0,650 | **0,689** |

A partir de aquí se mide en **`valid`** y con **umbral 0,20** (ver barrido). Las filas
de arriba están en `test` con umbral 0,50 y **no son comparables** con las de abajo.

| Experimento (en `valid`, umbral 0,20) | épocas | recall | precisión |
|---|---|---|---|
| Descongelado, 3º tramo | 16+30 | 0,713 | 0,457 |
| Descongelado, 4º tramo | 16+40 | **0,737** | **0,490** |
| Descongelado, 5º tramo (20 ép.) | 16+60 | 0,748 | 0,527 |
| **ResNet50 descongelado** | 10 | **0,948** | **0,732** |

**Criterio de parada cumplido.** 20 épocas para +0,011 de recall (≈0,005 por cada 10,
por debajo del umbral de 0,01 que se fijó). La palanca "más épocas descongelado" está
agotada. El coste sigue bajando muy despacio (0,6564 → 0,6355) pero ya no se traduce
en recall: el modelo exprime lo que sus características le permiten ver.

Siguiente palanca: **ResNet50** (`crearModelo(ligero=False)`), un solo parámetro.
Backbone más grande y mejores características. En T4 es asumible. Mismo protocolo:
descongelado, lr 0,0005, medir en `valid` con umbral 0,20 contra la línea 0,748.
Si ResNet50 tampoco mueve nada, la siguiente es **aumento de datos** (volteo
horizontal, caja → `(W-x2, y1, W-x1, y2)`, solo en train), que ataca la causa de
fondo: 298 imágenes.
| | | | | | |

**El 0,537 no es un fracaso: es una prueba de humo.** La cabeza partió de pesos
aleatorios y tuvo 149 pasos de gradiente. El detector COCO daba 0,70 con una cabeza
entrenada con millones de imágenes.

**Y la pérdida seguía bajando al acabar la época** — evidencia directa de que el
entrenamiento se cortó, no de que se estancara.

### Verificación visual del 0,948 — superada

Notebook `1.3-prueba-ResNet.ipynb`. Cinco imágenes de `valid`, predicciones en rojo
(umbral 0,20, clases 2/3/4) y anotaciones reales en verde sobre el mismo `ax`.

**Resultado: prácticamente todo verde lleva su rojo encima.** No se aprecian huecos
— jugadores anotados sin detectar — que es lo que restaría recall. El 0,948 se
sostiene mirándolo con los ojos, no solo en la métrica.

**De dónde sale la pérdida de precisión**, mirando los rojos sin verde:

- gente en la banda y detrás de las vallas publicitarias (fotógrafos, personal,
  entrenadores). Son personas reales, pero no están anotadas como jugador/árbitro.
- algún duplicado sobre un mismo jugador, pero **muchos menos de los que parecía**
  en las imágenes de `train`.

Ni una sola caja en la grada, con miles de personas visibles. El balón tampoco se
cajea (correcto: se filtra por clase).

**Implicación:** el grueso de los falsos positivos está **fuera del terreno de juego**,
así que el filtro geométrico de la Fase 3 (homografía) los elimina sin tocar el modelo.
La precisión de 0,732 subirá sola. Y eso a su vez permitirá **bajar el umbral** para
ganar más recall sin pagar el precio que hoy se pagaría.

**Consecuencia de planificación: la Fase 1 se puede dar por cerrada.** El detector
está resuelto. Seguir afinándolo antes de tener el pipeline completo es optimizar la
pieza más intercambiable del proyecto.

⚠️ Dos veces se miraron imágenes de `train` creyendo que eran de `valid`: la primera
por la ruta, la segunda por editar la celda 0 sin re-ejecutarla. El título del gráfico
decía "valid" porque estaba escrito a mano. **Lección: que el título salga del dato,
no de una constante escrita a mano.**

### 16-sep: el backbone era el cuello de botella desde el principio

ResNet50 descongelado, lr 0,0005, **10 épocas desde cero** (cabeza aleatoria):

    1,3648 · 0,9624 · 0,8749 · 0,8200 · 0,7804 · 0,7514 · 0,7225 · 0,6970 · 0,6759 · 0,6605

En `valid`, umbral 0,20: **recall 0,948 · precisión 0,732**

Contra MobileNet con 76 épocas: 0,748 / 0,527. **+0,200 de recall en una décima parte
del entrenamiento.**

Es coherente con lo que ya sabíamos: el techo medido de Roboflow (0,983) decía que el
dataset es aprendible con alta precisión. Lo que faltaba no eran épocas ni ajustes —
era capacidad del extractor de características. MobileNetV3 es un backbone pensado
para móviles; los jugadores pequeños y tapados no se distinguen en sus características,
y por eso ninguna cantidad de entrenamiento los hacía aparecer.

**El coste seguía bajando con fuerza en la época 9** (0,6605, con caídas de ~0,02 por
época). Quedan muchas épocas útiles.

**Lo que no fue en balde de los dos días con MobileNet:** el barrido de umbral (que
valía más que 30 épocas), la separación train/valid/test, el criterio de parada, y
saber leer coste vs recall. Ese método es lo que permitió detectar en 10 épocas que
ResNet era otra liga — y lo que evitará creerse un número sin verificarlo.

Pendiente antes de dar el 0,948 por bueno: **verificación visual**. Pintar las cajas
predichas sobre varias imágenes de `valid` con `dibujarCajas` y mirarlas. Un número
alto puede venir de un fallo en la medición; los ojos no se engañan igual.

### Barrido de umbral sobre `valid` (49 imgs) — modelo 16+30 descongelado

| umbral | recall | precisión | Δrecall | Δprecisión | ganancia/coste |
|---|---|---|---|---|---|
| 0,05 | 0,797 | 0,303 | +0,026 | −0,066 | 0,39 |
| 0,10 | 0,771 | 0,369 | +0,027 | −0,047 | 0,57 |
| 0,15 | 0,744 | 0,416 | +0,031 | −0,041 | **0,76** |
| 0,20 | 0,713 | 0,457 | +0,019 | −0,033 | 0,58 |
| 0,25 | 0,694 | 0,490 | +0,018 | −0,037 | 0,49 |
| 0,30 | 0,676 | 0,527 | +0,036 | −0,071 | 0,51 |
| 0,40 | 0,640 | 0,598 | — | — | — |

**No satura.** El recall sigue subiendo hasta 0,797 con umbral 0,05. El "codo" que
parecía haber en 0,25 sobre `test` era ruido de 25 imágenes: con 49 la curva es
suave y no hay punto de inflexión que elija por ti.

La última columna es recall ganado por precisión perdida en cada escalón. **El mejor
canje está entre 0,15 y 0,20**; por debajo de 0,10 se paga mucho por poco.

**Decisión pendiente y por qué no es solo métrica:** con umbral 0,15 seis de cada
diez cajas son falsas. Esas cajas entran luego al clustering de equipos y al cálculo
de la línea. Hasta que exista el filtro geométrico de dentro/fuera del campo (Fase 3),
conviene no irse al extremo: **0,20 como valor de trabajo**, y revisarlo cuando la
homografía pueda descartar lo que cae fuera del campo. Entonces el recall alto sale
casi gratis.

Nota de método: el umbral se elige en `valid`. `test` (0,753 a umbral 0,25) quedó
contaminado al haberse usado para elegir, así que esa cifra está optimista. La
estimación honesta a 0,25 es **0,694**.

### 16-sep: el umbral de score valía más que 30 épocas

Mismo modelo (16+30 épocas, descongelado), **sin entrenar nada**, cambiando solo el
corte con el que se filtran las predicciones:

| umbral | recall | precisión |
|---|---|---|
| 0,3 | **0,733** | 0,538 |
| 0,4 | 0,699 | 0,620 |
| 0,5 | 0,650 | 0,689 |

**+0,083 de recall gratis**, más de lo que dio cualquier tramo de 10 épocas, y sin
un segundo de GPU. Esas detecciones ya existían: se estaban tirando en el filtro.

**La lección:** el umbral es una decisión de *medición*, no del modelo. Durante 30
épocas se estuvo midiendo a través de un corte fijo de 0,5 que el modelo ya había
dejado atrás — según se recalibraban sus scores, ese corte iba dejando fuera
detecciones correctas. La caída de recall del 3er tramo era eso, no sobreajuste.

**Consecuencia para la tabla:** todas las filas anteriores están medidas a 0,5. El
umbral pasa a ser una columna, no una constante escondida en el notebook.

Pendiente: bajar a 0,2 y 0,25 para ver dónde satura el recall. Para este proyecto
el recall pesa más que la precisión — un jugador que falta descoloca la línea de
fuera de juego; una caja de más la filtra después la geometría del campo.

### 16-sep: descongelar era la palanca

Migrado a Colab (Tesla T4). Época: **13 s** frente a ~50 s congelado en el portátil,
y descongelado allí no llegaba a terminar.

Verificación de la mudanza antes de tocar nada: 1 época con la configuración de ayer
(congelado, lr 0,005) → coste 0,7206, justo donde se quedó el 15-sep. Migración limpia.

Descongelado entero, lr 0,0005, 10 épocas partiendo de `modelo_16ep.pt`:

    0,7182 · 0,7169 · 0,7140 · 0,7141 · 0,7091 · 0,7038 · 0,7091 · 0,6989 · 0,6992 · 0,6947

**recall 0,563 → 0,622 · precisión 0,509 → 0,557**

Es la primera vez que el recall se mueve de verdad. Comparación del ritmo:

| | épocas | Δ recall |
|---|---|---|
| congelado | 16 | +0,026 |
| descongelado | 10 | **+0,059** |

Confirma el diagnóstico: el cuello de botella eran las características fijas del
backbone, no la cabeza. Al poder adaptarse al fútbol, empiezan a aparecer jugadores
que antes no se detectaban.

**El coste sigue bajando en la época 9** (0,6947) y **el recall de test sube a la vez**.
Las dos señales apuntan igual: quedan épocas por aprovechar. Es el único caso en que
"más épocas" está justificado.

Siguiente: seguir descongelado, en tramos de 10, **midiendo test entre tramo y tramo**.
Lo que se busca es el punto donde el coste siga bajando pero el recall de test se
gire — ese es el comienzo del sobreajuste, y con 298 imágenes y el backbone suelto
va a llegar. Sin medir cada tramo, ese punto se pasa sin verlo.

Pesos en Drive con nombre por experimento; no pisar `modelo_16ep.pt`, que respalda
la fila de congelado.

### Decisión del 15-sep: entrenar en Colab

**El problema.** Al descongelar el backbone el entrenamiento se volvió inviable en
local: el portátil (IdeaPad 3 15ALC6, Ryzen con gráficos integrados, sin CUDA) se
calentaba y una tanda no terminaba. Congelado solo se calculaban gradientes de la
cabeza; descongelado, de toda la red. Es esperable, no es un fallo.

**La decisión: Colab, con Kaggle como plan B.** Las tiradas en GPU duran minutos,
así que la ejecución en segundo plano de Kaggle (hasta 12 h) no aporta nada. Lo que
decide es dónde viven los `.pt` entre sesiones: Drive se monta con una línea y da
carpeta persistente. En Kaggle habría que versionar el notebook o subirlos como
Dataset. Si en hora punta Colab no da GPU, Kaggle (P100, más rápida que la T4).

**Lo que hay que montar, por orden:**

1. Clonar el repo en el notebook. `src/` importa tal cual — aquí se paga haberlo
   modularizado en vez de dejarlo todo en celdas.
2. Dataset a Drive una vez (39 MB) y apuntar `CocoDataset` a esa ruta. Evita
   rebajarlo de Roboflow cada vez y quita el problema de la clave.
3. **`device`** — el único cambio de código real. El bucle actual no lo tiene: hay
   que mover modelo y tensores a la GPU.
4. Medir **una** época antes de lanzar nada largo.
5. `torch.save` apuntando a Drive, y cada pocas épocas — Colab desconecta a los
   ~90 min de inactividad y borra el disco local.

**Alternativa más barata si se sigue en local:** descongelar solo las últimas capas
con `trainable_backbone_layers` (0-5, por defecto 3) en lugar de todo el backbone.
Las primeras capas detectan bordes y texturas, que valen igual para fútbol que para
COCO; lo específico del dominio está en las últimas.

⚠️ No repetir tandas con backbone congelado y lr 0,005: 16 épocas de evidencia dicen
que el recall se queda en 0,563. Cumple el criterio de parada él solo.

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

Confirmado: 16 épocas en dos tandas de 8 (guardados a las 14:22 y 14:28 del 15-sep).
La segunda tanda arrancó en 0,763 justamente porque venía entrenada.

Pesos: `outputs/modelo_16ep.pt` (= `modelo_mobilenet_8ep_b4.pt`, mismo modelo).

**Siguiente paso: descongelar Y bajar el lr, las dos cosas a la vez.** Bajar el `lr`
con el backbone congelado no haría nada: seguiría entrenando solo la cabeza, que es
justo lo que ya está agotado. El `lr` baja *porque* se descongela — al soltar los
pesos preentrenados, los gradientes de 0,005 los destrozarían.

En `crearModelo`: `congelarBackbone=False`. En el optimizer: `lr=0.0005`.
Y kernel nuevo cargando `modelo_16ep.pt`, para partir de lo ya aprendido.

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
