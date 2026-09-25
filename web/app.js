/* ---------------------------------------------------------------
   Detector de fuera de juego — cliente

   Este fichero NO sabe nada de futbol: no calcula margenes ni decide
   quien esta adelantado. Recoge clics, los manda a la API y dibuja lo
   que vuelve. Toda la logica vive en Python.
   --------------------------------------------------------------- */

const estado = {
    id: null,               // el que devuelve /api/imagen
    ancho: 0, alto: 0,      // dimensiones ORIGINALES de la imagen
    cajas: [],
    paso: 1,
    iAtacante: null,
    iDefensor: null,
    correspondencias: [],   // [[[px,py],[mx,my]], ...]
    pendiente: null,        // el clic en la foto esperando pareja en el plano
    analisis: null,         // la respuesta de /api/analizar
};

const NS = "http://www.w3.org/2000/svg";

// --- elementos, cogidos una sola vez ---
const archivo    = document.getElementById("archivo");
const foto       = document.getElementById("foto");
const capa       = document.getElementById("capa");
const zonaSubida = document.getElementById("zonaSubida");
const cargando   = document.getElementById("cargando");
const leyenda    = document.getElementById("leyendaFoto");
const metaFoto   = document.getElementById("metaFoto");

const panelPlano   = document.getElementById("panelPlano");
const plano        = document.getElementById("plano");
const puntosPlano  = document.getElementById("puntosPlano");
const metaPlano    = document.getElementById("metaPlano");

const panelCenital  = document.getElementById("panelCenital");
const lineaCenital  = document.getElementById("lineaCenital");
const puntosCenital = document.getElementById("puntosCenital");

const pasos            = document.getElementById("pasos");
const textoInstruccion = document.getElementById("textoInstruccion");
const botonesDireccion = document.getElementById("botonesDireccion");
const deshacer         = document.getElementById("deshacer");
const reiniciar        = document.getElementById("reiniciar");

const veredicto        = document.getElementById("veredicto");
const cinta            = document.getElementById("cinta");
const titular          = document.getElementById("titular");
const detalleVeredicto = document.getElementById("detalleVeredicto");
const tablaAtacantes   = document.getElementById("tablaAtacantes");
const cuerpoTabla      = document.getElementById("cuerpoTabla");

const aviso            = document.getElementById("aviso");
const tituloAviso      = document.getElementById("tituloAviso");
const textoAviso       = document.getElementById("textoAviso");
const resolverDudosos  = document.getElementById("resolverDudosos");

const COLOR = {
    ata:    "var(--ata)",
    def:    "var(--def)",
    dudoso: "var(--dudoso)",
};


/* =========================================================
   AYUDANTES
   ========================================================= */

function svgEl(nombre, atributos){
    const e = document.createElementNS(NS, nombre);
    for (const clave in atributos) e.setAttribute(clave, atributos[clave]);
    return e;
}

// Clic del raton -> coordenadas de la imagen ORIGINAL.
// clientX/clientY van respecto a la ventana; getBoundingClientRect da
// donde esta la capa en ese mismo sistema. Restando se pasa a "dentro de
// la capa", y multiplicando por (original / mostrado) a pixeles de imagen.
// No se usa offsetX: en SVG va referido al elemento que recibe el clic, y
// si pinchas sobre un <rect> te daria coordenadas de ese rectangulo.
function aImagen(e){
    const r = capa.getBoundingClientRect();

    const x = (e.clientX - r.left) * (estado.ancho / r.width);
    const y = (e.clientY - r.top)  * (estado.alto  / r.height);

    return [x, y];
}

// Clic en el plano -> metros. Aqui se usa la matriz del propio SVG en vez
// de la regla de tres: getScreenCTM conoce el viewBox y el encaje real, asi
// que funciona aunque el SVG quede con bandas por diferencia de proporcion.
function aMetros(e){
    const p = new DOMPoint(e.clientX, e.clientY);
    const q = p.matrixTransform(plano.getScreenCTM().inverse());
    return [q.x, q.y];
}

// De las cajas que contienen el punto, la de menor area.
// Con jugadores amontonados un clic cae dentro de varias: la pequeña es la
// del jugador concreto, la grande suele abarcar a medio grupo.
function cajaEnPunto(x, y){
    let mejor = null;
    let menorArea = Infinity;

    estado.cajas.forEach((caja, i) => {
        const [x1, y1, x2, y2] = caja;

        if (x1 <= x && x <= x2 && y1 <= y && y <= y2) {
            const area = (x2 - x1) * (y2 - y1);
            if (area < menorArea) { menorArea = area; mejor = i; }
        }
    });

    return mejor;
}

const unMetro = (n) => n.toFixed(2).replace(".", ",");


/* =========================================================
   RED
   ========================================================= */

async function subir(fichero){
    const datos = new FormData();
    datos.append("archivo", fichero);          // mismo nombre que el parametro del endpoint

    // sin headers: el navegador pone el Content-Type con su boundary
    const r = await fetch("/api/imagen", { method: "POST", body: datos });

    // fetch NO lanza con un 400 o un 422, solo si no hay red
    if (!r.ok) {
        console.error("fallo al subir:", r.status);
        return false;
    }

    const recibido = await r.json();

    estado.id    = recibido.id;
    estado.ancho = recibido.ancho;
    estado.alto  = recibido.alto;
    estado.cajas = recibido.cajas;

    return true;
}

async function analizar(direccion){
    const r = await fetch("/api/analizar", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            id: estado.id,
            correspondencias: estado.correspondencias,
            iAtacante: estado.iAtacante,
            iDefensor: estado.iDefensor,
            direccion: direccion,
        }),
    });

    if (!r.ok) {
        const detalle = await r.json().catch(() => ({}));
        return { bien: false, motivo: String(detalle.detail || r.status) };
    }

    estado.analisis = await r.json();
    return { bien: true, motivo: null };
}


/* =========================================================
   DIBUJO SOBRE LA FOTO
   ========================================================= */

// Aplica una homografia de 3x3 a un punto. Es lo que hacia
// cv2.perspectiveTransform en Python: el punto se escribe como [x, y, 1], se
// multiplica por la matriz y se DIVIDE por la tercera componente. Sin esa
// division no hay perspectiva, solo una transformacion afin.
// Con Hinv va de metros a pixeles; con H, de pixeles a metros.
function aplicarHomografia(M, x, y){
    const u = M[0][0]*x + M[0][1]*y + M[0][2];
    const v = M[1][0]*x + M[1][1]*y + M[1][2];
    const w = M[2][0]*x + M[2][1]*y + M[2][2];

    // w ~ 0: el punto cae en la linea del horizonte y no tiene imagen finita
    if (!isFinite(w) || Math.abs(w) < 1e-9) return null;

    return [u / w, v / w];
}

// La x en metros a la que se DIBUJA la linea. El veredicto usa a.xLinea, que
// sale del punto de apoyo (centro de la base de la caja) y le cruzaria el cuerpo
// al defensa por la mitad. Para dibujar se coge la esquina de abajo de la caja
// que queda A LA ESPALDA del defensa: el defensa mira hacia donde vienen los
// atacantes, asi que su espalda da a la porteria que defiende, o sea el lado de
// MAYOR sentido*x. Asi el jugador queda entero por delante de la linea.
function xLineaDibujo(){
    const a = estado.analisis;
    if (a.iDefensa === null || !a.H) return a.xLinea;

    const [x1, y1, x2, y2] = a.cajas[a.iDefensa];

    const esquinas = [aplicarHomografia(a.H, x1, y2),
                      aplicarHomografia(a.H, x2, y2)].filter((m) => m !== null);

    if (esquinas.length === 0) return a.xLinea;

    // "a la espalda" = mayor sentido*x, con el mismo criterio que fueradeJuego.py
    return Math.max(...esquinas.map(([mx]) => a.sentido * mx)) * a.sentido;
}

// La linea de fuera de juego sobre la foto. Una homografia manda rectas a
// rectas, asi que con los dos extremos de x = const (de banda a banda) basta:
// no hay que muestrear puntos intermedios.
function pintarLineaFoto(){
    const a = estado.analisis;
    if (!a || a.xLinea === null || !a.Hinv) return;

    const x = xLineaDibujo();

    const p1 = aplicarHomografia(a.Hinv, x, 0);
    const p2 = aplicarHomografia(a.Hinv, x, 68);
    if (p1 === null || p2 === null) return;

    const l = svgEl("line", {
        x1: p1[0], y1: p1[1],
        x2: p2[0], y2: p2[1],
        "stroke-width": 1,         // el mismo trazo que las cajas
    });
    l.style.stroke = "var(--linea)";
    capa.append(l);          // antes de las cajas: los jugadores van encima
}

function pintarCajas(){
    capa.setAttribute("viewBox", `0 0 ${estado.ancho} ${estado.alto}`);
    capa.replaceChildren();

    // Antes del analisis se pintan las cajas crudas, todas del mismo color.
    // Despues se pintan las que devolvio la API, que vienen ya filtradas a
    // las que caen dentro del campo y con su equipo.
    const a = estado.analisis;
    const cajas     = a ? a.cajas     : estado.cajas;
    const etiquetas = a ? a.etiquetas : null;

    pintarLineaFoto();

    cajas.forEach((caja, i) => {
        const [x1, y1, x2, y2] = caja;
        const equipo = etiquetas ? etiquetas[i] : "dudoso";

        const rect = svgEl("rect", {
            x: x1, y: y1,
            width:  x2 - x1,
            height: y2 - y1,
            fill: "none",
            "stroke-width": 1,
            rx: 0,
        });
        rect.style.stroke = COLOR[equipo];
        capa.append(rect);

    });

    if (!a) marcarSemillas();
    else    marcarSenalados();
}

// Paso 2: rodear los dos jugadores ya elegidos
function marcarSemillas(){
    [[estado.iAtacante, "ata"], [estado.iDefensor, "def"]].forEach(([i, equipo]) => {
        if (i === null) return;

        const [x1, y1, x2, y2] = estado.cajas[i];
        const aro = svgEl("rect", {
            x: x1 - 4, y: y1 - 4,
            width:  (x2 - x1) + 8,
            height: (y2 - y1) + 8,
            fill: "none", "stroke-width": 1.5, rx: 2,
        });
        aro.style.stroke = COLOR[equipo];
        capa.append(aro);
    });
}

// Tras el analisis: el defensa que pone la linea y los dudosos peligrosos
function marcarSenalados(){
    const a = estado.analisis;

    const aro = (i, discontinuo) => {
        const [x1, y1, x2, y2] = a.cajas[i];
        const e = svgEl("rect", {
            x: x1 - 5, y: y1 - 5,
            width:  (x2 - x1) + 10,
            height: (y2 - y1) + 10,
            fill: "none", "stroke-width": 1.5, rx: 2,
        });
        if (discontinuo) e.setAttribute("stroke-dasharray", "5 4");
        e.style.stroke = "var(--acento)";
        capa.append(e);
    };

    // El ultimo defensa NO lleva aro: ya lo senala la linea.
    // Los dudosos peligrosos si, con trazo discontinuo.
    a.riesgo.forEach((i) => aro(i, true));
}


/* =========================================================
   DIBUJO SOBRE EL CAMPO
   ========================================================= */

function pintarPuntosPlano(){
    puntosPlano.replaceChildren();

    estado.correspondencias.forEach(([, metros], i) => {
        const [x, y] = metros;

        const c = svgEl("circle", { cx: x, cy: y, r: 1.4 });
        c.style.fill = "var(--acento)";
        puntosPlano.append(c);

        const t = svgEl("text", {
            x: x, y: y - 2.4, "font-size": 3.2, "text-anchor": "middle",
            "font-family": "Barlow Condensed, sans-serif", "font-weight": 600,
        });
        t.style.fill = "var(--acento)";
        t.textContent = i + 1;
        puntosPlano.append(t);
    });

    metaPlano.textContent = `${estado.correspondencias.length} de 4 puntos`;
}

function pintarCenital(){
    const a = estado.analisis;

    lineaCenital.replaceChildren();
    puntosCenital.replaceChildren();

    if (a.xLinea !== null) {
        const l = svgEl("path", { d: `M ${a.xLinea} -3 V 71`, "stroke-width": .8,
                                  "stroke-linecap": "round" });
        l.style.stroke = "var(--linea)";
        lineaCenital.append(l);

        const t = svgEl("text", {
            x: a.xLinea, y: -4, "font-size": 3.4, "text-anchor": "middle",
            "font-family": "Barlow Condensed, sans-serif", "font-weight": 600,
        });
        t.style.fill = "var(--linea)";
        t.textContent = unMetro(a.xLinea) + " m";
        lineaCenital.append(t);
    }

    a.metros.forEach(([x, y], i) => {
        // el que marca la linea y los dudosos peligrosos llevan aro
        if (i === a.iDefensa || a.riesgo.includes(i)) {
            const aro = svgEl("circle", { cx: x, cy: y, r: 2.5, fill: "none",
                                          "stroke-width": .7 });
            if (i !== a.iDefensa) aro.setAttribute("stroke-dasharray", "1.2 1");
            aro.style.stroke = "var(--acento)";
            puntosCenital.append(aro);
        }

        const p = svgEl("circle", { cx: x, cy: y, r: 1.35, "stroke-width": .3 });
        p.style.fill   = COLOR[a.etiquetas[i]];
        p.style.stroke = "rgba(0,0,0,.5)";
        puntosCenital.append(p);
    });
}


/* =========================================================
   EL VEREDICTO
   ========================================================= */

function pintarVeredicto(){
    const a = estado.analisis;

    const fuera    = a.resultado.filter((r) => r[2] === "fuera de juego");
    const ajustado = a.resultado.filter((r) => r[2] === "ajustado");

    cinta.className   = "cinta";
    titular.className = "titular";

    if (a.xLinea === null) {
        titular.textContent = "Sin línea";
        detalleVeredicto.textContent =
            "Ningún jugador quedó clasificado como defensa, así que no hay contra qué comparar.";
        tablaAtacantes.hidden = true;
        veredicto.hidden = false;
        return;
    }

    if (fuera.length > 0) {
        cinta.classList.add("fuera");
        titular.classList.add("fuera");
        titular.textContent = fuera.length === 1
            ? "Un atacante en fuera de juego"
            : `${fuera.length} atacantes en fuera de juego`;
    } else if (ajustado.length > 0) {
        cinta.classList.add("ajustado");
        titular.classList.add("ajustado");
        titular.textContent = "Jugada ajustada";
    } else {
        titular.textContent = "Ningún atacante en fuera de juego";
    }

    // el margen mas grande: el atacante mas adelantado
    const puntas = a.resultado.map((r) => r[1]);
    const punta  = puntas.length ? Math.max(...puntas) : null;

    detalleVeredicto.innerHTML =
        `La línea la marca el defensa más retrasado, en <b>x = ${unMetro(a.xLinea)} m</b>.` +
        (punta === null ? ""
            : ` El atacante más adelantado queda a <b>${unMetro(Math.abs(punta))} m</b> ` +
              (punta >= 0 ? "por delante." : "por detrás."));

    // la tabla, del mas adelantado al menos
    cuerpoTabla.replaceChildren();
    [...a.resultado].sort((p, q) => q[1] - p[1]).forEach(([i, margen, esclase]) => {
        const fila = document.createElement("tr");

        const cj = document.createElement("td");
        cj.textContent = "#" + i;

        const cm = document.createElement("td");
        cm.textContent = (margen >= 0 ? "+" : "−") + unMetro(Math.abs(margen)) + " m";

        const ce = document.createElement("td");
        const chip = document.createElement("span");
        chip.className = "chip" +
            (esclase === "fuera de juego" ? " fuera" : esclase === "ajustado" ? " ajustado" : "");
        chip.textContent = esclase === "fuera de juego" ? "Fuera de juego"
                         : esclase === "ajustado"       ? "Ajustado"
                         : "Habilitado";
        ce.append(chip);

        fila.append(cj, cm, ce);
        cuerpoTabla.append(fila);
    });

    tablaAtacantes.hidden = a.resultado.length === 0;
    veredicto.hidden = false;

    // el aviso de los dudosos que si pueden cambiar el resultado
    if (a.riesgo.length > 0) {
        tituloAviso.textContent = a.riesgo.length === 1
            ? "Un jugador sin equipo"
            : `${a.riesgo.length} jugadores sin equipo`;

        const quienes = a.riesgo.map((i) => "#" + i).join(", ");
        const uno = a.riesgo.length === 1;
        textoAviso.textContent =
            `${quienes} ${uno ? "está" : "están"} por delante de la línea y no se pudo ` +
            `${uno ? "asignarle" : "asignarles"} equipo: ` +
            `${uno ? "su recorte no tiene" : "sus recortes no tienen"} píxeles de camiseta ` +
            "suficientes. Si fuera defensa, la línea se movería; si fuera atacante, estaría " +
            "en fuera de juego sin contarse.";

        resolverDudosos.hidden = true;   // pendiente: necesita "forzados" en la API
        aviso.hidden = false;
    } else {
        aviso.hidden = true;
    }
}


/* =========================================================
   LOS PASOS
   ========================================================= */

const INSTRUCCIONES = {
    1: "Elige una foto para empezar.",
    2: "Pincha sobre un ATACANTE.",
    3: "Pincha un punto reconocible del campo en la foto: una esquina del área, el punto de penalti, un córner.",
    4: "¿Hacia qué portería ataca el equipo del atacante que señalaste?",
    5: "Listo.",
};

// Al subir una foto nueva hay que borrar TODO lo de la jugada anterior.
// Si no, `estado.analisis` sigue vivo: pintarCajas dibuja las cajas de la foto
// vieja sobre la foto nueva, y los clics se resuelven contra estado.cajas (las
// nuevas), asi que el indice que se manda al servidor apunta a otro jugador.
function limpiarJugada(){
    estado.analisis        = null;
    estado.iAtacante       = null;
    estado.iDefensor       = null;
    estado.correspondencias = [];
    estado.pendiente       = null;

    puntosPlano.replaceChildren();
    lineaCenital.replaceChildren();
    puntosCenital.replaceChildren();
    cuerpoTabla.replaceChildren();

    metaPlano.textContent = "0 de 4 puntos";
    panelCenital.hidden   = true;
    veredicto.hidden      = true;
    aviso.hidden          = true;
    tablaAtacantes.hidden = true;
}


function irAPaso(n){
    estado.paso = n;

    pasos.querySelectorAll(".paso").forEach((li) => {
        const p = Number(li.dataset.paso);
        li.classList.toggle("hecho",  p < n);
        li.classList.toggle("activo", p === n);
    });

    textoInstruccion.textContent = INSTRUCCIONES[n];

    // la capa solo se come los clics cuando toca pinchar
    capa.classList.toggle("clicable", n === 2 || n === 3);

    panelPlano.hidden       = n !== 3;
    botonesDireccion.hidden = n !== 4;
    deshacer.hidden         = !(n === 2 || n === 3);
    reiniciar.hidden        = n === 1;
}


/* =========================================================
   EVENTOS
   ========================================================= */

archivo.addEventListener("change", async () => {
    const fichero = archivo.files[0];
    if (!fichero) return;                      // abrio el dialogo y cancelo

    limpiarJugada();

    // enseñar la foto YA: la inferencia tarda segundos
    foto.src = URL.createObjectURL(fichero);
    foto.hidden = false;

    zonaSubida.hidden = true;
    cargando.hidden = false;

    const bien = await subir(fichero);

    cargando.hidden = true;
    if (!bien) {
        textoInstruccion.textContent = "No se pudo procesar la imagen. Prueba con otra.";
        return;
    }

    leyenda.hidden = false;
    pintarCajas();

    metaFoto.textContent =
        `${estado.ancho} × ${estado.alto} px · ${estado.cajas.length} detecciones`;

    if (estado.cajas.length < 2) {
        textoInstruccion.textContent =
            `Solo se detectaron ${estado.cajas.length} jugadores. Prueba con una foto más abierta.`;
        return;
    }

    irAPaso(2);
});


capa.addEventListener("click", (e) => {
    const [x, y] = aImagen(e);

    if (estado.paso === 2) {
        const i = cajaEnPunto(x, y);
        if (i === null) {
            textoInstruccion.textContent = "Ese clic no cayó sobre ningún jugador detectado.";
            return;
        }

        if (estado.iAtacante === null) {
            estado.iAtacante = i;
            textoInstruccion.textContent = "Ahora pincha sobre un DEFENSA.";
        } else if (i === estado.iAtacante) {
            textoInstruccion.textContent = "Ese es el atacante que ya elegiste. Pincha otro.";
            return;
        } else {
            estado.iDefensor = i;
            irAPaso(3);
        }

        pintarCajas();
        return;
    }

    if (estado.paso === 3) {
        estado.pendiente = [x, y];
        textoInstruccion.textContent =
            "Ahora pincha ESE MISMO punto en el plano del campo, abajo.";
    }
});


plano.addEventListener("click", (e) => {
    if (estado.paso !== 3 || estado.pendiente === null) return;

    const [mx, my] = aMetros(e);

    estado.correspondencias.push([estado.pendiente, [mx, my]]);
    estado.pendiente = null;
    pintarPuntosPlano();

    if (estado.correspondencias.length < 4) {
        textoInstruccion.textContent =
            `Van ${estado.correspondencias.length} de 4. Pincha otro punto en la foto.`;
    } else {
        irAPaso(4);
    }
});


botonesDireccion.addEventListener("click", async (e) => {
    const boton = e.target.closest("[data-direccion]");
    if (!boton) return;

    textoInstruccion.textContent = "Calculando…";
    botonesDireccion.hidden = true;

    const { bien, motivo } = await analizar(boton.dataset.direccion);
    if (!bien) {
        if (motivo.includes("semilla")) {
            // El recorte de uno de los dos jugadores no tiene camiseta legible.
            // La direccion no tiene nada que ver: hay que volver a elegirlos.
            estado.iAtacante = null;
            estado.iDefensor = null;
            irAPaso(2);
            pintarCajas();
            textoInstruccion.textContent =
                "No se le puede leer la camiseta a uno de los dos jugadores que " +
                "elegiste. Pincha sobre un ATACANTE, a ser posible de los grandes.";
        } else {
            botonesDireccion.hidden = false;
            textoInstruccion.textContent = "No se pudo analizar: " + motivo;
        }
        return;
    }

    irAPaso(5);
    pintarCajas();
    panelCenital.hidden = false;
    pintarCenital();
    pintarVeredicto();
});


deshacer.addEventListener("click", () => {
    if (estado.paso === 2) {
        if (estado.iDefensor !== null)      estado.iDefensor = null;
        else if (estado.iAtacante !== null) estado.iAtacante = null;
        textoInstruccion.textContent = estado.iAtacante === null
            ? "Pincha sobre un ATACANTE."
            : "Pincha sobre un DEFENSA.";
        pintarCajas();
        return;
    }

    if (estado.paso === 3) {
        if (estado.pendiente !== null) {
            estado.pendiente = null;
        } else {
            estado.correspondencias.pop();
            pintarPuntosPlano();
        }
        textoInstruccion.textContent =
            `Van ${estado.correspondencias.length} de 4. Pincha un punto en la foto.`;
    }
});


reiniciar.addEventListener("click", () => location.reload());

irAPaso(1);
