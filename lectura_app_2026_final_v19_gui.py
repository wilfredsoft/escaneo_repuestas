import customtkinter as ctk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import os
import time
import sys
from pyzbar.pyzbar import decode
import re
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing
import shutil
import tempfile
import platform
import threading

# Configuración de CustomTkinter
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

# OBLIGATORIO para multiprocessing en ejecutables
if hasattr(multiprocessing, 'freeze_support'):
    multiprocessing.freeze_support()

# Detectar sistema operativo
SISTEMA_OP = platform.system()

# La redirección de stderr usada por pyzbar afecta al proceso completo. Este
# bloqueo evita que dos hilos manipulen al mismo tiempo el descriptor global.
BARCODE_STDERR_LOCK = threading.Lock()

# ============================================================
# CONFIGURACIÓN DE SENSIBILIDAD Y DEBUG
# ============================================================
# Cambiar a False cuando NO quieras generar las imágenes de inspección visual
MODO_DEBUG = True  

REF_ID_X, REF_ID_Y = 66, 35
ANCHO_REFERENCIA, ALTO_REFERENCIA = 1695, 2195
BINARIZACION_ID = 205
LLENADO_MIN_ID = 45

# Configuración específica para Sesión 1 y 2 (43 filas, burbujas pequeñas)
BINARIZACION_S12 = 225
LLENADO_MIN_S12 = 28

# Configuración específica para Sesión 3 (25 filas, burbujas grandes)
BINARIZACION_S3 = 205
LLENADO_MIN_S3 = 35

MAPA_LETRAS = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H"}
MARCAS_SESION_REL = [(1560, 156), (1560, 218), (1560, 269), (1560, 320), 
                     (1560, 438), (1560, 487), (1560, 597)]

# ============================================================
# ELIMINACIÓN DE COLOR MAGENTA
# ============================================================

def detectar_hoja_color(img_bgr):
    """
    Detecta si la hoja contiene impresión a color.

    La detección anterior dependía de un rango HSV específico para magenta.
    Las variaciones de escáner, tinta e iluminación podían dejar fuera tonos
    más claros, oscuros o saturados. Aquí se mide la diferencia entre canales,
    por lo que también se reconocen las líneas verdes del formulario.
    """
    if img_bgr is None or len(img_bgr.shape) != 3:
        return False
    
    h, w = img_bgr.shape[:2]
    zona = img_bgr[h//2:h//2+150, w//4:3*w//4]
    
    if zona.size == 0:
        return False
    
    canal_max = np.max(zona, axis=2).astype(np.int16)
    canal_min = np.min(zona, axis=2).astype(np.int16)
    diferencia_color = canal_max - canal_min

    # Se excluyen zonas oscuras para no confundir grafito o tinta negra con
    # color debido al ruido y a la compresión JPEG.
    pixeles_color = (diferencia_color >= 12) & (canal_max >= 100)
    porcentaje_color = (np.count_nonzero(pixeles_color) / pixeles_color.size) * 100

    return porcentaje_color > 2

def eliminar_magenta(img_bgr):
    """
    Suprime la impresión de color conservando grafito y tinta negra.

    El máximo de los canales BGR convierte en claro cualquier píxel que tenga
    al menos un canal claro (magenta, verde y sus variaciones). Las marcas
    negras permanecen oscuras porque sus tres canales tienen valores bajos.
    Este método evita mantener rangos HSV distintos para cada lote de hojas.
    """
    if img_bgr is None:
        return None
    
    if len(img_bgr.shape) != 3:
        return img_bgr
    
    resultado = np.max(img_bgr, axis=2).astype(np.uint8)

    # No se aplica un umbral binario global: hacerlo aquí puede borrar marcas
    # tenues de lápiz. Cada zona conserva después su umbral especializado.
    return cv2.cvtColor(resultado, cv2.COLOR_GRAY2BGR)

# ============================================================
# DETECCIÓN DE RUTAS DE RED
# ============================================================

def es_ruta_red(ruta):
    """Detecta si una ruta es de red (multiplataforma)."""
    if ruta.startswith('\\\\'):
        return True
    
    if SISTEMA_OP == 'Windows' and len(ruta) >= 3 and ruta[1] == ':':
        letra_unidad = ruta[0].upper()
        try:
            import ctypes
            unidad_tipo = ctypes.windll.kernel32.GetDriveTypeW(f"{letra_unidad}:\\")
            if unidad_tipo == 4:
                return True
        except:
            pass
    
    if SISTEMA_OP in ['Darwin', 'Linux']:
        if ruta.startswith('/Volumes/'):
            return True
        if ruta.startswith('/mnt/') or ruta.startswith('/media/'):
            return True
    
    return False

def copiar_a_local(ruta_origen, carpeta_temporal, progress_callback=None):
    """Copia todos los archivos JPEG a la carpeta temporal."""
    archivos = [f for f in os.listdir(ruta_origen) if f.lower().endswith(('.jpg', '.jpeg'))]
    total = len(archivos)
    
    for i, nombre in enumerate(archivos):
        src = os.path.join(ruta_origen, nombre)
        dst = os.path.join(carpeta_temporal, nombre)
        shutil.copy2(src, dst)
        
        if progress_callback:
            progress_callback(i + 1, total, f"Copiando: {i+1}/{total} archivos")
    
    return archivos

# ============================================================
# FUNCIONES DE PROCESAMIENTO
# ============================================================

def leer_barcode_optimizado(roi_barcode):
    """
    Lectura de código de barras con validación estricta.
    Solo permite números (0-9) y guiones (-).
    Cualquier otro carácter se reemplaza con '?'.
    """
    if roi_barcode is None or roi_barcode.size == 0:
        return ""
    
    gray = cv2.cvtColor(roi_barcode, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    processed = clahe.apply(gray)

    with BARCODE_STDERR_LOCK:
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stderr = os.dup(2)
        os.dup2(devnull, 2)

        try:
            decoded = decode(processed)
            if not decoded:
                _, thresh = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                decoded = decode(thresh)
        finally:
            os.dup2(old_stderr, 2)
            os.close(devnull)
            os.close(old_stderr)
        
    if decoded:
        barcode_text = decoded[0].data.decode('utf-8', errors='ignore')
        
        # FILTRO ESTRICTO: Solo números y guiones, resto = '?'
        cleaned = ''.join(char if (char.isdigit() or char == '-') else '?' for char in barcode_text)
        
        if len(cleaned) >= 10:
            return cleaned
    
    return ""

def obtener_valor_sesion(thresh_img, dx, dy):
    """Lógica de clasificación de sesión."""
    filled_marks = []
    for mx, my in MARCAS_SESION_REL:
        rx, ry = int(mx + dx), int(my + dy)
        roi_s = thresh_img[max(0, ry):ry+22, max(0, rx):rx+36]
        cnts, _ = cv2.findContours(roi_s, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_marks.append(any(3 < cv2.minEnclosingCircle(c)[1] < 20 for c in cnts))
    
    n = sum(filled_marks)
    if n == 4:
        if filled_marks[0] and filled_marks[1] and filled_marks[4] and filled_marks[6]: return "1;5"
        if filled_marks[0] and filled_marks[3] and filled_marks[5] and filled_marks[6]: return "2;4"
        if filled_marks[0] and filled_marks[2] and filled_marks[4] and filled_marks[6]: return "1;3"
    elif n < 4 and filled_marks[0] and filled_marks[6]:
        if filled_marks[1]: return "1;3"
        if filled_marks[2]: return "2;4"
        if filled_marks[3]: return "1;5"
    return "1;3"

def anclar_hoja_desde_bordes(img_gray):
    """Detecta el cuadro de identificación binaria desde los bordes."""
    _, thresh_anc = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY_INV)
    contornos, _ = cv2.findContours(thresh_anc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    ix, iy, iw, ih = REF_ID_X, REF_ID_Y, 634, 656
    candidatos = []
    
    W_ESPERADO = 634
    H_ESPERADO = 656
    TOLERANCIA_SIZE = 60
    DESPLAZAMIENTO_MAX = 20
    
    for cnt in contornos:
        x, y, w, h = cv2.boundingRect(cnt)
        size_ok = (abs(w - W_ESPERADO) < TOLERANCIA_SIZE) and (abs(h - H_ESPERADO) < TOLERANCIA_SIZE)
        zona_ok = (abs(x - REF_ID_X) <= DESPLAZAMIENTO_MAX) and \
                  (abs(y - REF_ID_Y) <= DESPLAZAMIENTO_MAX)
        
        if size_ok and zona_ok:
            candidatos.append((x, y, w, h))
    
    if candidatos:
        candidatos.sort(key=lambda k: abs(k[0] - REF_ID_X) + abs(k[1] - REF_ID_Y))
        mejor_x, mejor_y, mejor_w, mejor_h = candidatos[0]
        dx = mejor_x - REF_ID_X
        dy = mejor_y - REF_ID_Y
        ix, iy, iw, ih = mejor_x, mejor_y, mejor_w, mejor_h
        return dx, dy, ix, iy, iw, ih
    
    return 0, 0, REF_ID_X, REF_ID_Y, 634, 656

def corregir_orientacion(img_bgr):
    """
    Coloca la hoja en orientación vertical usando las marcas del borde.

    El formulario tiene una columna de marcas negras en el borde derecho.
    Se comparan las dos orientaciones verticales posibles y se conserva la
    que concentra más píxeles oscuros a la derecha que a la izquierda.
    """
    if img_bgr is None or len(img_bgr.shape) != 3:
        return img_bgr

    h, w = img_bgr.shape[:2]

    if w > h:
        candidatos = [
            cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE),
            cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        ]
    else:
        candidatos = [
            img_bgr,
            cv2.rotate(img_bgr, cv2.ROTATE_180)
        ]

    def puntaje_borde(imagen):
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        ancho_borde = max(20, int(gris.shape[1] * 0.06))
        borde_derecho = gris[:, -ancho_borde:]
        borde_izquierdo = gris[:, :ancho_borde]
        proporcion_derecha = np.mean(borde_derecho < 80)
        proporcion_izquierda = np.mean(borde_izquierdo < 80)
        return proporcion_derecha - proporcion_izquierda

    return max(candidatos, key=puntaje_borde)

def normalizar_resolucion(img_bgr):
    """
    Ajusta la hoja al tamaño usado por las coordenadas de calibración.

    Diferencias pequeñas de resolución se acumulan hacia las filas inferiores
    y pueden desplazar la lectura de ID y respuestas. La orientación debe estar
    corregida antes de llamar esta función.
    """
    if img_bgr is None:
        return None

    alto, ancho = img_bgr.shape[:2]
    if ancho == ANCHO_REFERENCIA and alto == ALTO_REFERENCIA:
        return img_bgr

    if ancho > ANCHO_REFERENCIA or alto > ALTO_REFERENCIA:
        interpolacion = cv2.INTER_AREA
    else:
        interpolacion = cv2.INTER_CUBIC

    return cv2.resize(
        img_bgr,
        (ANCHO_REFERENCIA, ALTO_REFERENCIA),
        interpolation=interpolacion
    )

def procesar_hoja(args):
    """Procesa una sola hoja con anclaje correcto y eliminación de magenta."""
    ruta_archivo, ruta_carpeta = args
    
    try:
        img_raw = cv2.imread(os.path.join(ruta_carpeta, ruta_archivo))
        if img_raw is None:
            return (ruta_archivo, None)
        
        # Corregir hojas horizontales o invertidas antes de calcular zonas.
        img_raw = corregir_orientacion(img_raw)

        # Todas las coordenadas están calibradas para una resolución común.
        img_raw = normalizar_resolucion(img_raw)
        
        img_h, img_w = img_raw.shape[:2]
        
        # DETECTAR Y ELIMINAR MAGENTA SI ES HOJA A COLOR
        es_color = detectar_hoja_color(img_raw)
        if es_color:
            img_raw = eliminar_magenta(img_raw)
        
        img_gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
        
        # Imagen para depuración visual (Copia a color de la original)
        img_debug = img_raw.copy() if MODO_DEBUG else None
        
        # 1. ANCLAJE (Detectar cuadro de identificación)
        dx, dy, ix, iy, iw, ih = anclar_hoja_desde_bordes(img_gray)
        
        if MODO_DEBUG:
            # Cuadro de Anclaje / Identificación (Azul)
            cv2.rectangle(img_debug, (ix, iy), (ix + iw, iy + ih), (255, 0, 0), 2)
            cv2.putText(img_debug, "ANCLAJE ID", (ix, iy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # 2. SESIÓN
        img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
        _, thresh_bin = cv2.threshold(img_blur, 100, 255, cv2.THRESH_BINARY_INV)
        sesion_val = obtener_valor_sesion(thresh_bin, dx, dy)
        
        if MODO_DEBUG:
            # Dibujar rectángulos de marcas de sesión (Amarillo)
            for mx, my in MARCAS_SESION_REL:
                rx, ry = int(mx + dx), int(my + dy)
                cv2.rectangle(img_debug, (rx, ry), (rx + 36, ry + 22), (0, 255, 255), 2)

        # 3. IDENTIFICACIÓN
        id_str = " "
        _, thresh_id = cv2.threshold(img_blur, BINARIZACION_ID, 255, cv2.THRESH_BINARY_INV)
        a_col, a_fil = (iw - 30) / 15, (ih - 100) / 10
        for c in range(15):
            mp, mf = 0, -1
            for f in range(10):
                cx, cy = int(ix + 15 + (c + 0.5) * a_col), int(iy + 100 + (f + 0.5) * a_fil)
                roi = thresh_id[max(0, cy-17):cy+17, max(0, cx-17):cx+17]
                
                # Marcas visuales de identificación (Círculos pequeños en la grilla ID)
                if MODO_DEBUG:
                    cv2.circle(img_debug, (cx, cy), 12, (200, 200, 200), 1)

                if roi.size > 0:
                    p = (cv2.countNonZero(roi) / (np.pi * 17.5**2)) * 100
                    if p > LLENADO_MIN_ID and p > mp:
                        mp, mf = p, f
            
            if MODO_DEBUG and mf != -1:
                cy_marcado = int(iy + 100 + (mf + 0.5) * a_fil)
                cx_marcado = int(ix + 15 + (c + 0.5) * a_col)
                cv2.circle(img_debug, (cx_marcado, cy_marcado), 12, (0, 255, 0), -1)

            id_str += str(mf) if mf != -1 else " "
        
        # 4. BARCODE
        x1_bc, y1_bc, x2_bc, y2_bc = int(800+dx), int(450+dy), int(1550+dx), int(600+dy)
        roi_bc = img_raw[y1_bc:y2_bc, x1_bc:x2_bc]
        barcode_fijo = leer_barcode_optimizado(roi_bc).ljust(25)[:25]
        
        if MODO_DEBUG:
            # Recuadro de ROI Barcode (Cian)
            cv2.rectangle(img_debug, (x1_bc, y1_bc), (x2_bc, y2_bc), (255, 255, 0), 2)
            cv2.putText(img_debug, f"BARCODE: {barcode_fijo.strip()}", (x1_bc, y1_bc - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # 5. RESPUESTAS CON VALORES FIJOS BASADOS EN SESIÓN
        if sesion_val == "1;5":  # Sesión 3: 25 filas, burbujas grandes
            y_b = iy + 725
            alt, fls, rd = 54.5, 25, 16
            umbral_llenado = LLENADO_MIN_S3
        else:  # Sesión 1 o 2: 43 filas, burbujas pequeñas
            y_b = iy + 673
            alt, fls, rd = 33.5, 43, 13
            umbral_llenado = LLENADO_MIN_S12
        
        res = []
        t_res = cv2.adaptiveThreshold(
            img_blur, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            51, 10
        )
        kernel = np.ones((3,3), np.uint8)
        t_res = cv2.morphologyEx(t_res, cv2.MORPH_OPEN, kernel)
        
        area_circulo = np.pi * rd**2
        ancho_col = 328 / 8
        umbral_pixeles = area_circulo * (umbral_llenado / 100)
        
        bloques_x = [12, 412, 812, 1212]
        
        for offset_x in bloques_x:
            for f in range(fls):
                marcas_validas = []
                cy = int(y_b + (f + 0.5) * alt)
                
                # Para guardar posiciones de la fila actual y dibujarlas juntas al evaluar
                coords_fila = []

                for c in range(8):
                    cx = int(ix + offset_x + (c + 0.5) * ancho_col)
                    
                    y1 = max(0, int(cy - rd))
                    y2 = min(img_h, int(cy + rd))
                    x1 = max(0, int(cx - rd))
                    x2 = min(img_w, int(cx + rd))
                    
                    coords_fila.append((cx, cy))

                    if y2 <= y1 or x2 <= x1:
                        continue
                    
                    burbuja = t_res[y1:y2, x1:x2]
                    
                    if burbuja.size == 0:
                        continue
                    
                    pixeles_negros = cv2.countNonZero(burbuja)
                    if pixeles_negros < umbral_pixeles:
                        continue
                    
                    p = (pixeles_negros / area_circulo) * 100
                    
                    if p > umbral_llenado:
                        marcas_validas.append({
                            "letra": MAPA_LETRAS[c],
                            "fill": p,
                            "burbuja": burbuja,
                            "cx": cx,
                            "cy": cy
                        })
                
                cantidad = len(marcas_validas)
                
                # Dibujado de inspección de respuestas
                if MODO_DEBUG:
                    # Mostrar la rejilla/posiciones evaluadas (Círculo o Cuadro en Gris/Marrón)
                    for cx_i, cy_i in coords_fila:
                        cv2.rectangle(img_debug, (cx_i - rd, cy_i - rd), (cx_i + rd, cy_i + rd), (180, 180, 180), 1)

                if cantidad == 0:
                    res.append(" ")
                elif cantidad == 1:
                    res.append(marcas_validas[0]["letra"])
                    if MODO_DEBUG:
                        # Marca correcta / única (VERDE)
                        m = marcas_validas[0]
                        cv2.circle(img_debug, (m["cx"], m["cy"]), rd, (0, 255, 0), -1)
                elif cantidad == 2:
                    m1 = marcas_validas[0]
                    m2 = marcas_validas[1]
                    
                    for m in marcas_validas:
                        cnts, _ = cv2.findContours(m["burbuja"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        m["area"] = max((cv2.contourArea(cnt) for cnt in cnts), default=0)
                    
                    max_fill = max(m1['fill'], m2['fill'])
                    max_area = max(m1['area'], m2['area'])
                    
                    ratio_fill = min(m1['fill'], m2['fill']) / max_fill if max_fill > 0 else 0
                    ratio_area = min(m1['area'], m2['area']) / max_area if max_area > 0 else 0
                    
                    if ratio_fill > 0.9 and ratio_area > 0.9:
                        res.append("*")
                        if MODO_DEBUG:
                            # Marca múltiple / inválida (ROJO)
                            for m in marcas_validas:
                                cv2.circle(img_debug, (m["cx"], m["cy"]), rd, (0, 0, 255), -1)
                    else:
                        if (m1['fill'] > m2['fill'] and m1['area'] >= m2['area']) or \
                           (m1['area'] > m2['area'] and m1['fill'] >= m2['fill']):
                            ganadora = m1
                            descartada = m2
                        elif (m2['fill'] > m1['fill'] and m2['area'] >= m1['area']) or \
                             (m2['area'] > m1['area'] and m2['fill'] >= m1['fill']):
                            ganadora = m2
                            descartada = m1
                        else:
                            score1 = (m1['fill'] * 0.4 + m1['area'] * 0.6)
                            score2 = (m2['fill'] * 0.4 + m2['area'] * 0.6)
                            ganadora = m1 if score1 > score2 else m2
                            descartada = m2 if ganadora == m1 else m1
                        
                        res.append(ganadora["letra"])
                        
                        if MODO_DEBUG:
                            # Ganadora Verde, Descartada Naranja/Rojo
                            cv2.circle(img_debug, (ganadora["cx"], ganadora["cy"]), rd, (0, 255, 0), -1)
                            cv2.circle(img_debug, (descartada["cx"], descartada["cy"]), rd, (0, 165, 255), -1)
                else:
                    res.append("*")
                    if MODO_DEBUG:
                        # Marcas Múltiples (ROJO)
                        for m in marcas_validas:
                            cv2.circle(img_debug, (m["cx"], m["cy"]), rd, (0, 0, 255), -1)

        while len(res) < 172:
            res.append(" ")
        
        # GUARDAR IMAGEN DE DEBUG SI EL MODO ESTÁ ACTIVADO
        if MODO_DEBUG and img_debug is not None:
            nombre_out = f"debug_{os.path.basename(ruta_archivo)}"
            cv2.imwrite(os.path.join(ruta_carpeta, nombre_out), img_debug)

        resultado = f"{sesion_val};{id_str};{barcode_fijo};{';'.join(res)}"
        return (ruta_archivo, resultado)
        
    except Exception as e:
        print(f"\n[ERROR] Procesando {ruta_archivo}: {e}")
        return (ruta_archivo, None)

def natural_sort_key(s):
    """Función auxiliar para ordenamiento natural de strings."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

# ============================================================
# CLASE DE LA INTERFAZ GRÁFICA
# ============================================================

class ScanAppGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("SCAN APP 2026 - Auto-Calibración Total")
        self.geometry("900x700")
        
        self.ruta_seleccionada = ""
        self.procesando = False
        
        self.crear_interfaz()
    
    def crear_interfaz(self):
        # Título
        title = ctk.CTkLabel(self, text=" SCAN APP 2026", 
                            font=ctk.CTkFont(size=28, weight="bold"))
        title.pack(pady=20)
        
        subtitle = ctk.CTkLabel(self, text="Auto-Calibración Total - Sin intervención humana",
                               font=ctk.CTkFont(size=14), text_color="gray")
        subtitle.pack(pady=5)
        
        # Frame de carpeta
        folder_frame = ctk.CTkFrame(self)
        folder_frame.pack(pady=10, padx=20, fill="x")
        
        ctk.CTkLabel(folder_frame, text=" Carpeta con imágenes:",
                    font=ctk.CTkFont(size=14, weight="bold")).pack(pady=5)
        
        self.folder_entry = ctk.CTkEntry(folder_frame, height=40)
        self.folder_entry.pack(pady=5, padx=10, fill="x")
        
        ctk.CTkButton(folder_frame, text=" Examinar", 
                     command=self.seleccionar_carpeta,
                     height=35, width=120).pack(pady=5)
        
        # Información de características
        info_frame = ctk.CTkFrame(self)
        info_frame.pack(pady=10, padx=20, fill="x")
        
        ctk.CTkLabel(info_frame, text=" Características Automáticas:",
                    font=ctk.CTkFont(size=14, weight="bold")).pack(pady=5)
        
        info_text = (" Detección automática de color y eliminación de magenta\n"
                     " Anclaje automático desde cuadro de identificación\n"
                     " Posicionamiento preciso basado en sesión detectada\n"
                     " Soporte para Sesión 1/2 (43 filas) y Sesión 3 (25 filas)\n"
                     " Modo Inspección/Debug visual activo para prueba de zonas")
        
        ctk.CTkLabel(info_frame, text=info_text, font=ctk.CTkFont(size=11),
                    text_color="gray", justify="left").pack(pady=5, padx=10)
        
        # Botón de procesar
        self.process_btn = ctk.CTkButton(self, text=" Iniciar Procesamiento Automático",
                                        command=self.iniciar_procesamiento,
                                        width=300,
                                        height=45,
                                        font=ctk.CTkFont(size=16, weight="bold"),
                                        fg_color="#2E86AB")
        self.process_btn.pack(pady=20)
        
        # Barra de progreso
        self.progress = ctk.CTkProgressBar(self, height=20)
        self.progress.pack(pady=10, padx=20, fill="x")
        self.progress.set(0)
        
        self.progress_label = ctk.CTkLabel(self, text="Listo para comenzar")
        self.progress_label.pack(pady=5)
        
        # Log
        ctk.CTkLabel(self, text=" Log de actividad:",
                    font=ctk.CTkFont(size=14, weight="bold")).pack(pady=5)
        
        self.log = ctk.CTkTextbox(self, height=200, state="disabled")
        self.log.pack(pady=10, padx=20, fill="both", expand=True)
        
        # Footer
        footer = ctk.CTkLabel(self, text="Desarrollado por: Ing. Wilfred Tovar",
                             text_color="gray")
        footer.pack(pady=10)
    
    def seleccionar_carpeta(self):
        carpeta = filedialog.askdirectory(title="Seleccione la carpeta con imágenes JPEG")
        if carpeta:
            self.ruta_seleccionada = carpeta
            self.folder_entry.delete(0, 'end')
            self.folder_entry.insert(0, carpeta)
            self.log_message(f" Carpeta seleccionada: {carpeta}")
    
    def log_message(self, msg):
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")
    
    def iniciar_procesamiento(self):
        if not self.ruta_seleccionada:
            messagebox.showerror("Error", "Por favor seleccione una carpeta con imágenes.")
            return
        
        if not os.path.exists(self.ruta_seleccionada):
            messagebox.showerror("Error", f"La ruta no existe: {self.ruta_seleccionada}")
            return
        
        archivos = [f for f in os.listdir(self.ruta_seleccionada) 
                   if f.lower().endswith(('.jpg', '.jpeg')) and not f.startswith('debug_')]
        if not archivos:
            messagebox.showerror("Error", "No se encontraron archivos JPEG en la carpeta.")
            return
        
        if self.procesando:
            messagebox.showwarning("Advertencia", "Ya hay un proceso en ejecución.")
            return
        
        self.procesando = True
        self.process_btn.configure(state="disabled", text=" Procesando...")
        self.log_message(f"\n Iniciando procesamiento automático de {len(archivos)} imágenes...")
        self.log_message(f" Carpeta: {self.ruta_seleccionada}")
        self.log_message(f" Modo: Auto-calibración total (sin intervención humana)")
        self.log_message(f" Detección de color: Activada")
        self.log_message(f" Anclaje automático: Activado")
        self.log_message(f" Generación imágenes Debug: {'ACTIVADO' if MODO_DEBUG else 'DESACTIVADO'}")
        
        thread = threading.Thread(target=self.procesar_archivos)
        thread.daemon = True
        thread.start()
    
    def procesar_archivos(self):
        try:
            start_time = time.time()
            es_red = es_ruta_red(self.ruta_seleccionada)
            
            if es_red:
                self.log_message("\n Detectada ruta de red. Copiando archivos localmente...")
                carpeta_temporal = tempfile.mkdtemp(prefix='scan_app_')
                self.log_message(f" Carpeta temporal: {carpeta_temporal}")
                
                def copy_progress(actual, total, msg):
                    self.progress.set(actual/total)
                    self.progress_label.configure(text=msg)
                    self.update_idletasks()
                
                archivos = copiar_a_local(self.ruta_seleccionada, carpeta_temporal, copy_progress)
                self.log_message(f" {len(archivos)} archivos copiados localmente")
                ruta_procesamiento = carpeta_temporal
            else:
                archivos = [f for f in os.listdir(self.ruta_seleccionada) 
                           if f.lower().endswith(('.jpg', '.jpeg')) and not f.startswith('debug_')]
                ruta_procesamiento = self.ruta_seleccionada
            
            self.log_message(f"\n Procesando {len(archivos)} imágenes...")
            archivos.sort(key=natural_sort_key)
            
            executor_class = None

            if SISTEMA_OP == 'Darwin':
                self.log_message(" macOS detectado - Usando procesamiento secuencial")
                usar_paralelismo = False
            elif getattr(sys, 'frozen', False):
                # PyInstaller ejecuta el script como __main__. En Windows los
                # procesos hijos no pueden importar procesar_hoja desde ese
                # módulo temporal y ProcessPoolExecutor termina abruptamente.
                num_workers = min(8, max(2, multiprocessing.cpu_count()))
                executor_class = ThreadPoolExecutor
                self.log_message(
                    f" Ejecutable detectado - Usando {num_workers} hilos en paralelo")
                usar_paralelismo = True
            else:
                num_workers = multiprocessing.cpu_count()
                executor_class = ProcessPoolExecutor
                self.log_message(f" Usando {num_workers} procesos en paralelo")
                usar_paralelismo = True
            
            resultados_dict = {}
            
            if usar_paralelismo:
                args_list = [(nombre, ruta_procesamiento) for nombre in archivos]
                
                with executor_class(max_workers=num_workers) as executor:
                    future_to_archivo = {executor.submit(procesar_hoja, args): args[0] 
                                       for args in args_list}
                    
                    for i, future in enumerate(as_completed(future_to_archivo)):
                        nombre_archivo = future_to_archivo[future]
                        try:
                            nombre, resultado = future.result()
                            if resultado is not None:
                                resultados_dict[nombre] = resultado
                            
                            progreso = (i + 1) / len(archivos)
                            self.progress.set(progreso)
                            self.progress_label.configure(
                                text=f"Procesando: {i+1}/{len(archivos)}")
                            self.log_message(f" {nombre_archivo}")
                            
                        except Exception as e:
                            self.log_message(f" Error en {nombre_archivo}: {str(e)}")
            else:
                for i, nombre in enumerate(archivos):
                    try:
                        _, resultado = procesar_hoja((nombre, ruta_procesamiento))
                        if resultado is not None:
                            resultados_dict[nombre] = resultado
                        
                        progreso = (i + 1) / len(archivos)
                        self.progress.set(progreso)
                        self.progress_label.configure(
                            text=f"Procesando: {i+1}/{len(archivos)}")
                        self.log_message(f" {nombre}")
                        
                    except Exception as e:
                        self.log_message(f" Error en {nombre}: {str(e)}")
            
            resultados = []
            for nombre in archivos:
                if nombre in resultados_dict:
                    resultados.append(resultados_dict[nombre])
            
            nombre_carpeta = os.path.basename(os.path.normpath(self.ruta_seleccionada))
            
            if es_red:
                ruta_txt_temporal = os.path.join(carpeta_temporal, f"{nombre_carpeta}.txt")
                with open(ruta_txt_temporal, "w", encoding='utf-8') as f:
                    f.write("\n".join(resultados))
                
                ruta_txt_final = os.path.join(self.ruta_seleccionada, f"{nombre_carpeta}.txt")
                shutil.copy2(ruta_txt_temporal, ruta_txt_final)
                
                # Copiar imágenes de debug de vuelta a la ruta de red si existiesen
                if MODO_DEBUG:
                    for item in os.listdir(carpeta_temporal):
                        if item.startswith('debug_'):
                            shutil.copy2(os.path.join(carpeta_temporal, item), os.path.join(self.ruta_seleccionada, item))

                try:
                    shutil.rmtree(carpeta_temporal)
                    self.log_message(" Carpeta temporal eliminada")
                except:
                    pass
            else:
                ruta_txt = os.path.join(self.ruta_seleccionada, f"{nombre_carpeta}.txt")
                with open(ruta_txt, "w", encoding='utf-8') as f:
                    f.write("\n".join(resultados))
            
            total_time = time.time() - start_time
            t_m, t_s = divmod(int(total_time), 60)
            
            self.progress.set(1.0)
            self.progress_label.configure(text="¡Proceso completado!")
            self.log_message(f"\n Proceso finalizado con éxito")
            self.log_message(f" Hojas procesadas: {len(resultados)}")
            self.log_message(f" Tiempo total: {t_m:02d}:{t_s:02d}")
            self.log_message(f" Archivo guardado: {nombre_carpeta}.txt")
            if MODO_DEBUG:
                self.log_message(" Se generaron las imágenes 'debug_*.jpg' para inspección.")
            
            messagebox.showinfo("Éxito", 
                              f"Proceso completado\n\n"
                              f"Hojas procesadas: {len(resultados)}\n"
                              f"Tiempo: {t_m:02d}:{t_s:02d}")
            
        except Exception as e:
            self.log_message(f"\n ERROR CRÍTICO: {str(e)}")
            messagebox.showerror("Error", f"Error durante el procesamiento:\n{str(e)}")
        
        finally:
            self.procesando = False
            self.process_btn.configure(state="normal", text=" Iniciar Procesamiento Automático")
            self.progress.set(0)
            self.progress_label.configure(text="Listo")
            self.folder_entry.delete(0, "end")

# ============================================================
# EJECUCIÓN DE LA APLICACIÓN
# ============================================================

if __name__ == '__main__':
    multiprocessing.freeze_support()
    app = ScanAppGUI()
    app.mainloop()
