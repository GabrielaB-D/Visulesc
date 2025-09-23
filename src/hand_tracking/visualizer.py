"""
Visualizador de landmarks de manos.
"""

import cv2
import mediapipe as mp
from typing import List, Tuple
try:
    from ..config.settings import *
except ImportError:
    from config.settings import *


class HandVisualizer:
    """Clase para visualizar landmarks de manos en imágenes."""
    
    def __init__(self):
        """Inicializa el visualizador."""
        self.connections = mp.solutions.hands.HAND_CONNECTIONS
    
    def draw_landmarks(self, image, hand_landmarks_list: List) -> None:
        """
        Dibuja landmarks y conexiones en la imagen.
        
        Args:
            image: Imagen donde dibujar
            hand_landmarks_list: Lista de landmarks de manos
        """
        if not hand_landmarks_list:
            return
            
        h, w, _ = image.shape
        
        for hand_landmarks in hand_landmarks_list:
            # Convertir landmarks a coordenadas de píxeles
            points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
            
            # Dibujar puntos de landmarks
            self._draw_landmark_points(image, points)
            
            # Dibujar conexiones entre landmarks
            self._draw_connections(image, points)
    
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
