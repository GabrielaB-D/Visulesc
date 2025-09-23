# Configuración del sistema de detección de manos

# Parámetros de optimización
TARGET_WIDTH = 256
TARGET_HEIGHT = 192
SEND_EVERY_N_FRAMES = 2
NUM_HANDS = 2

# Configuración de detección
MIN_HAND_DETECTION_CONFIDENCE = 0.5
MIN_HAND_PRESENCE_CONFIDENCE = 0.5

# Configuración de visualización
LANDMARK_RADIUS = 4
CONNECTION_THICKNESS = 2
LANDMARK_COLOR = (0, 255, 0)  # Verde
CONNECTION_COLOR = (0, 150, 255)  # Naranja

# Configuración de rendimiento
FPS_CALCULATION_INTERVAL = 30
CAPTURE_BUFFER_SIZE = 1

# Rutas de archivos
MODEL_PATH = "assets/models/hand_landmarker.task"
WINDOW_NAME = "Hand Landmarker"
