"""
Sistema modular de reconocimiento de gestos específicos usando MediaPipe landmarks.
Implementa reconocimiento del gesto "A" en lenguaje de señas.
"""

import numpy as np
from typing import Optional, List, Dict, Any
import math


class GestureRecognizer:
    """
    Reconocedor modular de gestos específicos usando landmarks de MediaPipe.
    Diseñado para ser extensible y permitir agregar nuevos gestos fácilmente.
    """
    
    def __init__(self):
        """Inicializa el reconocedor de gestos."""
        self.gesture_functions = {
            'A': self._recognize_gesture_A
        }
    
    def recognize_gesture(self, landmarks: List) -> Optional[str]:
        """
        Reconoce el gesto actual basado en los landmarks de la mano.
        
        Args:
            landmarks: Lista de landmarks de MediaPipe (21 puntos por mano)
            
        Returns:
            String con el nombre del gesto reconocido o None si no se reconoce
        """
        if not landmarks or len(landmarks) < 21:
            return None
        
        # Intentar reconocer cada gesto registrado
        for gesture_name, gesture_function in self.gesture_functions.items():
            if gesture_function(landmarks):
                return gesture_name
        
        return None
    
    def _recognize_gesture_A(self, landmarks: List) -> bool:
        """
        Reconoce el gesto "A" en lenguaje de señas.
        
        Características del gesto "A":
        - Pulgar extendido hacia arriba
        - Los otros 4 dedos cerrados en puño
        - El pulgar forma un ángulo de aproximadamente 90° con el índice
        
        Args:
            landmarks: Lista de 21 landmarks de MediaPipe
            
        Returns:
            True si se reconoce el gesto "A", False en caso contrario
        """
        try:
            # Obtener puntos clave para el gesto "A"
            thumb_tip = landmarks[4]      # Punta del pulgar
            thumb_ip = landmarks[3]        # Articulación intermedia del pulgar
            thumb_mcp = landmarks[2]       # Articulación base del pulgar
            index_mcp = landmarks[5]       # Articulación base del índice
            index_pip = landmarks[6]       # Articulación intermedia del índice
            index_tip = landmarks[8]       # Punta del índice
            middle_tip = landmarks[12]     # Punta del dedo medio
            ring_tip = landmarks[16]      # Punta del anular
            pinky_tip = landmarks[20]      # Punta del meñique
            
            # 1. Verificar que el pulgar esté extendido hacia arriba
            thumb_extended = self._is_thumb_extended(thumb_tip, thumb_ip, thumb_mcp)
            
            # 2. Verificar que los otros dedos estén cerrados
            fingers_closed = self._are_fingers_closed(
                index_tip, index_pip, index_mcp,
                middle_tip, landmarks[10], landmarks[9],
                ring_tip, landmarks[14], landmarks[13],
                pinky_tip, landmarks[18], landmarks[17]
            )
            
            # 3. Verificar ángulo del pulgar con respecto al índice
            thumb_angle_ok = self._check_thumb_index_angle(
                thumb_tip, thumb_mcp, index_mcp
            )
            
            # El gesto "A" se reconoce si todas las condiciones se cumplen
            return thumb_extended and fingers_closed and thumb_angle_ok
            
        except Exception as e:
            print(f"Error reconociendo gesto A: {e}")
            return False
    
    def _is_thumb_extended(self, thumb_tip, thumb_ip, thumb_mcp) -> bool:
        """
        Verifica si el pulgar está extendido hacia arriba.
        
        Args:
            thumb_tip: Punta del pulgar
            thumb_ip: Articulación intermedia del pulgar
            thumb_mcp: Articulación base del pulgar
            
        Returns:
            True si el pulgar está extendido, False en caso contrario
        """
        # Calcular distancia entre punta y base del pulgar
        thumb_length = self._calculate_distance(thumb_tip, thumb_mcp)
        
        # Calcular distancia entre articulación intermedia y base
        thumb_ip_distance = self._calculate_distance(thumb_ip, thumb_mcp)
        
        # El pulgar está extendido si la punta está más lejos de la base
        # que la articulación intermedia (ratio > 1.2)
        if thumb_length > 0 and thumb_ip_distance > 0:
            extension_ratio = thumb_length / thumb_ip_distance
            return extension_ratio > 1.2
        
        return False
    
    def _are_fingers_closed(self, index_tip, index_pip, index_mcp,
                           middle_tip, middle_pip, middle_mcp,
                           ring_tip, ring_pip, ring_mcp,
                           pinky_tip, pinky_pip, pinky_mcp) -> bool:
        """
        Verifica si los dedos índice, medio, anular y meñique están cerrados.
        
        Args:
            index_tip, index_pip, index_mcp: Landmarks del índice
            middle_tip, middle_pip, middle_mcp: Landmarks del medio
            ring_tip, ring_pip, ring_mcp: Landmarks del anular
            pinky_tip, pinky_pip, pinky_mcp: Landmarks del meñique
            
        Returns:
            True si todos los dedos están cerrados, False en caso contrario
        """
        fingers = [
            (index_tip, index_pip, index_mcp),
            (middle_tip, middle_pip, middle_mcp),
            (ring_tip, ring_pip, ring_mcp),
            (pinky_tip, pinky_pip, pinky_mcp)
        ]
        
        closed_fingers = 0
        
        for tip, pip, mcp in fingers:
            # Calcular distancia de la punta a la base
            tip_to_base = self._calculate_distance(tip, mcp)
            
            # Calcular distancia de la articulación intermedia a la base
            pip_to_base = self._calculate_distance(pip, mcp)
            
            # El dedo está cerrado si la punta está más cerca de la base
            # que la articulación intermedia (ratio < 0.8)
            if tip_to_base > 0 and pip_to_base > 0:
                closure_ratio = tip_to_base / pip_to_base
                if closure_ratio < 0.8:
                    closed_fingers += 1
        
        # Al menos 3 de los 4 dedos deben estar cerrados
        return closed_fingers >= 3
    
    def _check_thumb_index_angle(self, thumb_tip, thumb_mcp, index_mcp) -> bool:
        """
        Verifica el ángulo entre el pulgar y el índice.
        
        Args:
            thumb_tip: Punta del pulgar
            thumb_mcp: Base del pulgar
            index_mcp: Base del índice
            
        Returns:
            True si el ángulo es apropiado para el gesto "A"
        """
        # Calcular vector del pulgar (de base a punta)
        thumb_vector = np.array([
            thumb_tip.x - thumb_mcp.x,
            thumb_tip.y - thumb_mcp.y
        ])
        
        # Calcular vector del índice (de base del pulgar a base del índice)
        index_vector = np.array([
            index_mcp.x - thumb_mcp.x,
            index_mcp.y - thumb_mcp.y
        ])
        
        # Calcular ángulo entre vectores
        angle = self._calculate_angle_between_vectors(thumb_vector, index_vector)
        
        # Para el gesto "A", el ángulo debe estar entre 60° y 120°
        return 60 <= angle <= 120
    
    def _calculate_distance(self, point1, point2) -> float:
        """
        Calcula la distancia euclidiana entre dos puntos.
        
        Args:
            point1: Primer punto (landmark)
            point2: Segundo punto (landmark)
            
        Returns:
            Distancia entre los puntos
        """
        return math.sqrt(
            (point1.x - point2.x) ** 2 + 
            (point1.y - point2.y) ** 2
        )
    
    def _calculate_angle_between_vectors(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calcula el ángulo entre dos vectores en grados.
        
        Args:
            vec1: Primer vector
            vec2: Segundo vector
            
        Returns:
            Ángulo en grados
        """
        # Calcular producto punto
        dot_product = np.dot(vec1, vec2)
        
        # Calcular magnitudes
        magnitude1 = np.linalg.norm(vec1)
        magnitude2 = np.linalg.norm(vec2)
        
        # Evitar división por cero
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        
        # Calcular coseno del ángulo
        cos_angle = dot_product / (magnitude1 * magnitude2)
        
        # Limitar cos_angle al rango [-1, 1] para evitar errores de arcoseno
        cos_angle = max(-1, min(1, cos_angle))
        
        # Convertir a grados
        angle_radians = math.acos(cos_angle)
        angle_degrees = math.degrees(angle_radians)
        
        return angle_degrees
    
    def add_gesture(self, gesture_name: str, gesture_function):
        """
        Añade un nuevo gesto al reconocedor.
        
        Args:
            gesture_name: Nombre del gesto
            gesture_function: Función que reconoce el gesto
        """
        self.gesture_functions[gesture_name] = gesture_function
        print(f"Gesto '{gesture_name}' añadido al reconocedor")
    
    def get_available_gestures(self) -> List[str]:
        """
        Retorna la lista de gestos disponibles.
        
        Returns:
            Lista de nombres de gestos
        """
        return list(self.gesture_functions.keys())


# Función de utilidad para debugging
def debug_landmarks(landmarks: List, gesture_name: str = None):
    """
    Función de utilidad para debuggear landmarks y gestos.
    
    Args:
        landmarks: Lista de landmarks
        gesture_name: Nombre del gesto reconocido (opcional)
    """
    if not landmarks:
        print("No hay landmarks disponibles")
        return
    
    print(f"\n=== DEBUG LANDMARKS ===")
    print(f"Gesto reconocido: {gesture_name}")
    print(f"Número de landmarks: {len(landmarks)}")
    
    # Mostrar algunos puntos clave
    key_points = {
        'Pulgar punta': landmarks[4],
        'Pulgar base': landmarks[2],
        'Índice punta': landmarks[8],
        'Índice base': landmarks[5],
        'Muñeca': landmarks[0]
    }
    
    for name, point in key_points.items():
        print(f"{name}: ({point.x:.3f}, {point.y:.3f})")
    
    print("=" * 30)
