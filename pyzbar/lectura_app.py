import cv2
import numpy as np
import os
import re
from pyzbar.pyzbar import decode

# Configuración de constantes
CONFIG = {
    "MARCAS_SESION": [
        (1560, 156, 1596, 180), (1560, 218, 1596, 239), (1560, 269, 1596, 290),
        (1560, 320, 1596, 341), (1560, 438, 1596, 459), (1560, 487, 1596, 508),
        (1560, 597, 1596, 618)
    ],
    #REGION_BARRAS = (850, 450, 1500, 600),
    "REGION_BARRAS": (450, 850, 600, 1500),  # Ajustado para la posición del código de barras
    "REGION_IDENT": (66, 35, 700, 691),
    "SECTORES_S3": [
        {"x1": 82, "y1": 760, "x2": 404, "y2": 2140, "inicio_fila": 1, "filas": 25, "alto_fila": 55},
        {"x1": 482, "y1": 760, "x2": 804, "y2": 2140, "inicio_fila": 26, "filas": 25, "alto_fila": 55},
        {"x1": 880, "y1": 760, "x2": 1202, "y2": 2140, "inicio_fila": 51, "filas": 25, "alto_fila": 55},
        {"x1": 1278, "y1": 760, "x2": 1600, "y2": 2140, "inicio_fila": 76, "filas": 25, "alto_fila": 55}
    ],
    "SECTORES_S1_S2": [
        {"x1": 82, "y1": 708, "x2": 404, "y2": 2145, "inicio_fila": 1, "filas": 43, "alto_fila": 33.4},
        {"x1": 482, "y1": 708, "x2": 804, "y2": 2145, "inicio_fila": 44, "filas": 43, "alto_fila": 33.4},
        {"x1": 882, "y1": 708, "x2": 1202, "y2": 2145, "inicio_fila": 87, "filas": 43, "alto_fila": 33.4},
        {"x1": 1278, "y1": 708, "x2": 1600, "y2": 2145, "inicio_fila": 130, "filas": 43, "alto_fila": 33.4}
    ],
    "ANCHO_CELDA": 40,
    "UMBRAL_AREA_FACTOR": 0.24,  # Configurable: porcentaje del área de la celda para considerar una marca 0.25
    "UMBRAL_BINARIO_S1_S2": 215, # 215- 225
    "UMBRAL_BINARIO_S3": 215,
    "INTENSITY_THRESHOLD": 10,
    "MAX_FILA_S1_S2": 172,
    "MAX_FILA_S3": 100,
    "IDENT_LENGTH": 15,
    "BARCODE_LENGTH": 26,
    "NUM_COLUMNAS": 8,
    "AREA_SIMILAR_THRESHOLD": 0.1  # Umbral para considerar áreas similares
}

def extract_region(image, region_coords):
    """Extrae una región de la imagen asegurando límites válidos."""
    y1, x1, y2, x2 = region_coords
    y1, y2 = max(0, y1), min(image.shape[0], y2)
    x1, x2 = max(0, x1), min(image.shape[1], x2)
    if y2 <= y1 or x2 <= x1:
        return None
    return image[y1:y2, x1:x2]

def binarize_image(image, threshold=None):
    """Convierte una imagen a escala de grises y aplica binarización."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if threshold is None:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    return binary

def detect_filled_circle_ident(x, y, radius, binary_image, start_x, start_y, cell_width, cell_height):
    """Detecta círculos rellenos en la región de identificación."""
    col = (x - start_x) // cell_width
    if col < 0 or col >= 15:
        return False, None

    max_area = 0
    best_row = None
    cell_area = cell_width * cell_height

    for row in range(10):
        y_lower = start_y + row * cell_height
        y_upper = y_lower + cell_height
        x_left = start_x + col * cell_width
        x_right = x_left + cell_width

        x_inter_left = max(x - radius, x_left)
        x_inter_right = min(x + radius, x_right)
        y_inter_lower = max(y - radius, y_lower)
        y_inter_upper = min(y + radius, y_upper)

        width = x_inter_right - x_inter_left
        height = y_inter_upper - y_inter_lower

        if width > 0 and height > 0:
            area = width * height
            if area > max_area and area >= 0.25 * cell_area:
                max_area = area
                best_row = row

    return (True, best_row) if best_row is not None else (False, None)

def process_identification(image):
    """Procesa la región de identificación fila por fila y columna por columna como en las respuestas."""
    ident_region = extract_region(image, CONFIG["REGION_IDENT"])
    if ident_region is None:
        return " " * CONFIG["IDENT_LENGTH"]

    binary_ident = binarize_image(ident_region, threshold=205)

    cell_width = 40
    cell_height = 55
    start_x = 50
    start_y = 75
    columnas = 15
    filas = 10

    marcas_por_columna = {col: [] for col in range(columnas)}

    for col in range(columnas):
        for row in range(filas):
            x_left = start_x + col * cell_width
            x_right = x_left + cell_width
            y_upper = start_y + row * cell_height
            y_lower = y_upper + cell_height

            cell = binary_ident[y_upper:y_lower, x_left:x_right]

            if cell.size == 0:
                continue

            cell_area = cell.shape[0] * cell.shape[1]
            marked_area = np.sum(cell == 255)

            if marked_area >= CONFIG["UMBRAL_AREA_FACTOR"] * cell_area:
                intensity = np.mean(cell) if cell.size > 0 else 255
                marcas_por_columna[col].append({
                    "row": row,
                    "intensity": intensity,
                    "area": marked_area
                })

    # Construir la identificación con reglas mejoradas
    identificacion = ""
    for col in range(columnas):
        marcas = marcas_por_columna[col]
        if not marcas:
            identificacion += " "  # Sin marcas
        elif len(marcas) == 1:
            identificacion += str(marcas[0]["row"])
        elif len(marcas) == 2:
            mejor = sorted(marcas, key=lambda x: (-x["intensity"], -x["area"]))[0]
            identificacion += str(mejor["row"])
        else:
            identificacion += "*"  # Más de dos marcas

    return identificacion.ljust(CONFIG["IDENT_LENGTH"])

def process_sector(binary_image, sector, global_row_offset, max_rows):
    """Procesa un sector para detectar marcas en cada fila."""
    responses = []
    cell_width = CONFIG["ANCHO_CELDA"]
    img_height, img_width = binary_image.shape

    for row in range(sector["filas"]):
        y_lower = int(row * sector["alto_fila"])
        y_upper = int(y_lower + sector["alto_fila"])
        y_lower, y_upper = max(0, y_lower), min(img_height, y_upper)
        if y_upper <= y_lower:
            continue

        marks_in_row = []
        for col in range(CONFIG["NUM_COLUMNAS"]):
            x_left = col * cell_width
            x_right = x_left + cell_width
            x_left, x_right = max(0, x_left), min(img_width, x_right)
            if x_right <= x_left:
                continue

            cell = binary_image[y_lower:y_upper, x_left:x_right]
            if cell.size == 0:
                continue

            marked_area = np.sum(cell == 255)
            cell_area = (x_right - x_left) * (y_upper - y_lower)
            if marked_area >= CONFIG["UMBRAL_AREA_FACTOR"] * cell_area:
                # Calcular la intensidad promedio (menor valor = más oscuro)
                intensity = np.mean(cell) if cell.size > 0 else 255  # 255 es el valor máximo (blanco)
                marks_in_row.append({
                    "opcion": chr(65 + col),
                    "area": marked_area,
                    "intensity": intensity,  # Nueva clave para la intensidad
                    "col": col
                })

        global_row = sector["inicio_fila"] + row
        if 1 <= global_row <= max_rows:
            responses.append((global_row - 1, marks_in_row))

    return responses

def format_responses(responses_by_row, max_rows):
    """Formatea las respuestas detectadas en el formato requerido, priorizando la marca más oscura."""
    final_responses = []
    for row in range(max_rows):
        options = responses_by_row[row]
        if not options:
            final_responses.append(f"{row + 1}-")
            continue

        if len(options) == 1:
            final_responses.append(f"{row + 1}-{options[0]['opcion']}")
            continue

        # Ordenar por intensidad (menor intensidad = más oscuro) y luego por área como criterio secundario
        sorted_options = sorted(options, key=lambda x: (-x["intensity"], -x["area"]))
        min_intensity = sorted_options[0]["intensity"]

        # Opcional: Umbral para considerar intensidades similares (por ejemplo, 10 unidades)
        
        similar_intensities = [opt for opt in sorted_options if abs(opt["intensity"] - min_intensity) <= CONFIG["INTENSITY_THRESHOLD"]]

        if len(similar_intensities) > 1:
            final_responses.append(f"{row + 1}-{sorted_options[0]['opcion']}")  # Marcas con intensidades similares se consideran ambiguas
        else:
            final_responses.append(f"{row + 1}-{sorted_options[0]['opcion']}")  # Selecciona la más oscura

    return final_responses

def leer_codigo_barras(imagen, debug=True, debug_output="debug_barcode.jpg"):
    """Lee el código de barras en la región especificada con depuración y formato."""
    # Extraer las coordenadas de la región desde CONFIG
    y1, x1, y2, x2 = CONFIG["REGION_BARRAS"]

    # Verificar que las coordenadas estén dentro de los límites de la imagen
    y1, y2 = max(0, y1), min(imagen.shape[0], y2)
    x1, x2 = max(0, x1), min(imagen.shape[1], x2)
    if y2 <= y1 or x2 <= x1:
        print("Área del código de barras fuera de los límites.")
        return " " * CONFIG["BARCODE_LENGTH"]
    
    # Extraer la región y convertir a escala de grises
    roi = imagen[y1:y2, x1:x2]
    gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
   
    # Decodificar el código de barras
    codigos = decode(gris)

    # Leer el código y aplicar formato
    if codigos:
        raw_code = codigos[0].data.decode('utf-8', errors='ignore').strip()
        #print(f"Valor crudo del código de barras: '{raw_code}'")

        # Verificar longitud mínima y formatear
        if len(raw_code) >= 19:           
            # Formatear: 099852-650-1103746110 (6-3-10)
            
            code = raw_code.ljust(CONFIG["BARCODE_LENGTH"])
            #print(f"Código de barras formateado: '{formatted_code}'")
            return code
        else:
            #print("Código de barras demasiado corto para el formato esperado.")
            return " " * CONFIG["BARCODE_LENGTH"]
    else:
        #print("Código de barras no detectado.")
        return " " * CONFIG["BARCODE_LENGTH"]

def process_image(image_path):
    """Procesa una imagen para extraer identificación, respuestas, código de barras y sesión."""
    if not os.path.exists(image_path):
        print(f"Error: {image_path} no existe.")
        return "", [], "", ""

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: No se pudo cargar {image_path}.")
        return "", [], "", ""

    print(f"Procesando: {image_path}")
    session = detect_session(image)
    # Leer el código de barras con depuración
    barcode = leer_codigo_barras(image, debug=True, debug_output=f"debug_barcode_{os.path.basename(image_path)}")
    identification = process_identification(image)

    # Configuración según la sesión
    sectors = CONFIG["SECTORES_S1_S2"] if session in ["Sesión 1", "Sesión 2"] else CONFIG["SECTORES_S3"]
    max_rows = CONFIG["MAX_FILA_S1_S2"] if session in ["Sesión 1", "Sesión 2"] else CONFIG["MAX_FILA_S3"]
    threshold = CONFIG["UMBRAL_BINARIO_S1_S2"] if session in ["Sesión 1", "Sesión 2"] else CONFIG["UMBRAL_BINARIO_S3"]

    # Procesar respuestas
    responses_by_row = [[] for _ in range(max_rows)]
    for sector in sectors:
        sector_region = extract_region(image, (sector["y1"], sector["x1"], sector["y2"], sector["x2"]))
        if sector_region is None:
            continue

        binary_sector = binarize_image(sector_region, threshold)
        sector_responses = process_sector(binary_sector, sector, sector["inicio_fila"], max_rows)
        for row_idx, marks in sector_responses:
            responses_by_row[row_idx] = marks

    responses = format_responses(responses_by_row, max_rows)
    return identification, responses, barcode, session

def detect_session(image):
    """Detecta la sesión basada en las marcas de la hoja."""
    filled_marks = []
    for x1, y1, x2, y2 in CONFIG["MARCAS_SESION"]:
        roi = extract_region(image, (y1, x1, y2, x2))
        if roi is None:
            filled_marks.append(False)
            continue

        binary_roi = binarize_image(roi, threshold=100)
        contours, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_marks.append(any(1 < cv2.minEnclosingCircle(c)[1] < 25 for c in contours))

    num_marks = sum(filled_marks)
    if num_marks > 4:
        return "Sesión 1"
    elif num_marks == 4:
        if filled_marks[0] and filled_marks[1] and filled_marks[4] and filled_marks[6]:
            return "Sesión 3"
        elif filled_marks[0] and filled_marks[3] and filled_marks[5] and filled_marks[6]:
            return "Sesión 2"
        elif filled_marks[0] and filled_marks[2] and filled_marks[4] and filled_marks[6]:
            return "Sesión 1"
        return "Sesión desconocida"
    elif num_marks < 4 and filled_marks[0] and filled_marks[6]:
        if filled_marks[1]:
            return "Sesión 1"
        elif filled_marks[2]:
            return "Sesión 2"
        elif filled_marks[3]:
            return "Sesión 3"
    return "Sesión desconocida"

def natural_sort_key(filename):
    """Función auxiliar para ordenamiento natural de nombres de archivo."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', filename)]

def process_folder():
    """Procesa todas las imágenes en una carpeta y genera un archivo de salida."""
    while True:
        folder_path = input("Ingrese la ruta de la carpeta con archivos JPEG: ")
        if os.path.isdir(folder_path):
            break  # Sale del bucle si la ruta es válida
        print(f"Error: {folder_path} no es una carpeta válida.")

    folder_name = os.path.basename(folder_path.rstrip('/\\'))
    output_file = os.path.join(folder_path, f"{folder_name}.txt")

    images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpeg', '.jpg'))]
    sorted_images = sorted(images, key=natural_sort_key)

    if not sorted_images:
        print("No se encontraron imágenes JPEG.")
        return 0

    lines = []
    for img in sorted_images:
        img_path = os.path.join(folder_path, img)
        ident, responses, barcode, session = process_image(img_path)
        session_code = "1;3" if session in ["Sesión 1", "Sesión desconocida"] else "2;4" if session == "Sesión 2" else "1;5"
        line = f"{session_code};{ident};{barcode}"
        line += ";" + ";".join(r.split('-')[1] if r.split('-')[1] and r.split('-')[1] != "*" else "*" if r.split('-')[1] == "*" else " " for r in responses)
        lines.append(line)

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        print(f"Archivo generado: {output_file}")
        return len(sorted_images)  # Retorna la cantidad de imágenes procesadas
    except Exception as e:
        print(f"Error al generar archivo: {e}")
        return 0

if __name__ == "__main__":
    try:
        print(f"\n**************** SCAN APP 2025 ****************\nVersión de OpenCV: {cv2.__version__} \nAplicación Desarrollada por: Ing. Wilfred Tovar\nVersion: 1.6 \n***********************************************\n")
    except AttributeError:
        print("Error: OpenCV no está instalado correctamente. Por favor, instálelo con 'pip install opencv-python'.")
        exit(1)

    while True:
        processed_count = process_folder()
        if processed_count > 0:
            print(f"\nFinalización exitosa: Se procesaron {processed_count} imágenes.")
        else:
            print("\nFinalización con errores: No se procesaron imágenes.")

        # Preguntar si desea procesar otra carpeta
        while True:
            response = input("\n¿Desea procesar otra carpeta? (Sí/No): ").strip().lower()
            if response in ['sí', 'si', 's']:
                print("\n")
                break  # Continúa el ciclo para procesar otra carpeta
            elif response in ['no', 'n']:
                print("Aplicación finalizada.")
                exit(0)  # Finaliza la aplicación
            else:
                print("Por favor, ingrese 'Sí' o 'No'.")