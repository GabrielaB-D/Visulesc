"""
Visualizador de landmarks de manos con soporte de overlays de debug.

Características:
- Dibuja landmarks y conexiones.
- Colorea por dedo para mejor lectura.
- Modo debug para mostrar índices y coordenadas (x, y).
"""

import cv2
import mediapipe as mp
from typing import List, Tuple, Dict
try:
    from ..config.settings import *
except ImportError:
    from config.settings import *


class HandVisualizer:
    """Clase para visualizar landmarks de manos en imágenes."""
    
    def __init__(self):
        """Inicializa el visualizador."""
        self.connections = mp.solutions.hands.HAND_CONNECTIONS
        # Mapeo de índices por dedo según MediaPipe Hands
        self.finger_indices: Dict[str, List[int]] = {
            "thumb": [1, 2, 3, 4],
            "index": [5, 6, 7, 8],
            "middle": [9, 10, 11, 12],
            "ring": [13, 14, 15, 16],
            "pinky": [17, 18, 19, 20],
        }
        # Colores por dedo para overlays
        self.finger_colors: Dict[str, Tuple[int, int, int]] = {
            "thumb": (0, 255, 255),   # Amarillo
            "index": (0, 255, 0),     # Verde
            "middle": (255, 0, 0),    # Azul
            "ring": (255, 0, 255),    # Magenta
            "pinky": (0, 165, 255),   # Naranja
        }
    
    def draw_landmarks(self, image, hand_landmarks_list: List, *, debug: bool = False, show_indices: bool = False, show_coords: bool = False, mirrored: bool = False) -> None:
        """
        Dibuja landmarks y conexiones en la imagen.
        
        Args:
            image: Imagen donde dibujar
            hand_landmarks_list: Lista de landmarks de manos
            debug: Si True, activa overlays de depuración
            show_indices: Si True, muestra el índice de cada landmark
            show_coords: Si True, muestra las coordenadas (x, y) en píxeles
        """
        if not hand_landmarks_list:
            return
            
        h, w, _ = image.shape
        
        for hand_landmarks in hand_landmarks_list:
            # Convertir landmarks a coordenadas de píxeles
            if not mirrored:
                points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
            else:
                # Si la imagen está espejada, invertimos X
                points = [(int((1.0 - lm.x) * w), int(lm.y * h)) for lm in hand_landmarks]
            
            # Dibujar puntos de landmarks
            self._draw_landmark_points(image, points)
            
            # Dibujar conexiones entre landmarks
            self._draw_connections(image, points)
            
            if debug:
                self._draw_per_finger_overlays(image, points)
                if show_indices:
                    self._draw_indices(image, points)
                if show_coords:
                    self._draw_coordinates(image, points)
    
    def _draw_landmark_points(self, image, points: List[Tuple[int, int]]) -> None:
        """
        Dibuja los puntos de landmarks.
        
        Args:
            image: Imagen donde dibujar
            points: Lista de coordenadas (x, y)
        """
        for (x, y) in points:
            cv2.circle(image, (x, y), LANDMARK_RADIUS, LANDMARK_COLOR, -1)
    
    def _draw_connections(self, image, points: List[Tuple[int, int]]) -> None:
        """
        Dibuja las conexiones entre landmarks.
        
        Args:
            image: Imagen donde dibujar
            points: Lista de coordenadas (x, y)
        """
        for start, end in self.connections:
            if start < len(points) and end < len(points):
                cv2.line(image, points[start], points[end], 
                        CONNECTION_COLOR, CONNECTION_THICKNESS)

    def _draw_per_finger_overlays(self, image, points: List[Tuple[int, int]]) -> None:
        """Colorea por dedo para facilitar la lectura de la pose."""
        # Dibuja círculos con color por dedo (sin tapar excesivamente)
        for finger, indices in self.finger_indices.items():
            color = self.finger_colors.get(finger, LANDMARK_COLOR)
            for idx in indices:
                if idx < len(points):
                    x, y = points[idx]
                    cv2.circle(image, (x, y), max(2, LANDMARK_RADIUS - 1), color, -1)

    def _draw_indices(self, image, points: List[Tuple[int, int]]) -> None:
        """Muestra el índice de cada landmark al lado del punto."""
        for idx, (x, y) in enumerate(points):
            cv2.putText(
                image,
                str(idx),
                (x + 4, max(12, y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    def _draw_coordinates(self, image, points: List[Tuple[int, int]]) -> None:
        """Muestra las coordenadas (x, y) en píxeles de cada landmark."""
        for (x, y) in points:
            cv2.putText(
                image,
                f"({x},{y})",
                (x + 4, y + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.33,
                (200, 200, 0),
                1,
                cv2.LINE_AA,
            )
