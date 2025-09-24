"""
Sistema de detección de gestos con pausas para LESCO.
Detecta cuando el usuario está haciendo un gesto vs cuando está en pausa.
"""

import numpy as np
import time
from typing import List, Tuple, Optional, Callable
from collections import deque
try:
    from ..config.settings import *
except ImportError:
    from config.settings import *


class GestureDetector:
    """Detector de gestos con detección de pausas para LESCO."""
    
    def __init__(self, 
                 pause_threshold: float = 0.3,
                 min_gesture_frames: int = 10,
                 max_gesture_frames: int = 120,
                 movement_threshold: float = 0.02):
        """
        Inicializa el detector de gestos.
        
        Args:
            pause_threshold: Tiempo en segundos para considerar una pausa
            min_gesture_frames: Mínimo de frames para considerar un gesto válido
            max_gesture_frames: Máximo de frames para un gesto
            movement_threshold: Umbral de movimiento para detectar actividad
        """
        self.pause_threshold = pause_threshold
        self.min_gesture_frames = min_gesture_frames
        self.max_gesture_frames = max_gesture_frames
        self.movement_threshold = movement_threshold
        
        # Estado del detector
        self.is_recording = False
        self.gesture_frames = []
        self.last_landmarks = None
        self.last_movement_time = 0
        self.gesture_start_time = 0
        
        # Callbacks
        self.on_gesture_start: Optional[Callable] = None
        self.on_gesture_end: Optional[Callable] = None
        self.on_pause_detected: Optional[Callable] = None
    
    def process_landmarks(self, landmarks: List, timestamp: float) -> dict:
        """
        Procesa los landmarks y determina el estado del gesto.
        
        Args:
            landmarks: Lista de landmarks de MediaPipe
            timestamp: Timestamp actual
            
        Returns:
            Dict con información del estado actual
        """
        result = {
            'is_recording': self.is_recording,
            'gesture_frames': len(self.gesture_frames),
            'current_state': 'idle'
        }
        
        if not landmarks or len(landmarks) == 0:
            # No hay manos detectadas
            if self.is_recording:
                self._handle_no_hands_during_recording(timestamp)
            return result
        
        # Usar la primera mano detectada
        hand_landmarks = landmarks[0]
        
        # Calcular movimiento
        movement_detected = self._calculate_movement(hand_landmarks)
        
        if movement_detected:
            self.last_movement_time = timestamp
            
            if not self.is_recording:
                # Iniciar grabación de gesto
                self._start_gesture_recording(hand_landmarks, timestamp)
                result['current_state'] = 'gesture_started'
            else:
                # Continuar grabación
                self._continue_gesture_recording(hand_landmarks, timestamp)
                result['current_state'] = 'gesture_recording'
        else:
            # No hay movimiento significativo
            if self.is_recording:
                # Verificar si es una pausa
                pause_duration = timestamp - self.last_movement_time
                if pause_duration >= self.pause_threshold:
                    # Finalizar gesto por pausa
                    self._end_gesture_recording(timestamp, reason='pause')
                    result['current_state'] = 'gesture_ended_pause'
                else:
                    result['current_state'] = 'gesture_waiting'
            else:
                result['current_state'] = 'idle'
        
        # Verificar límite máximo de frames
        if self.is_recording and len(self.gesture_frames) >= self.max_gesture_frames:
            self._end_gesture_recording(timestamp, reason='max_frames')
            result['current_state'] = 'gesture_ended_max_frames'
        
        return result
    
    def _calculate_movement(self, landmarks) -> bool:
        """
        Calcula si hay movimiento significativo en los landmarks.
        
        Args:
            landmarks: Landmarks de la mano actual
            
        Returns:
            True si hay movimiento significativo
        """
        # Aceptar tanto listas de landmarks como objetos MediaPipe con atributo `.landmark`
        current_landmarks = getattr(landmarks, 'landmark', landmarks)
        previous_landmarks = getattr(self.last_landmarks, 'landmark', self.last_landmarks)

        if self.last_landmarks is None or previous_landmarks is None:
            # Inicializar con la lista actual de landmarks
            self.last_landmarks = current_landmarks
            return True  # Primer frame siempre cuenta como movimiento
        
        # Calcular distancia promedio entre landmarks
        total_distance = 0
        try:
            num_landmarks = min(len(current_landmarks), len(previous_landmarks))
        except TypeError:
            # Si por alguna razón no son indexables, considerar movimiento nulo
            num_landmarks = 0
        
        for i in range(num_landmarks):
            current = current_landmarks[i]
            previous = previous_landmarks[i]
            
            # Calcular distancia euclidiana normalizada
            distance = np.sqrt(
                (current.x - previous.x) ** 2 + 
                (current.y - previous.y) ** 2
            )
            total_distance += distance
        
        avg_distance = total_distance / num_landmarks if num_landmarks > 0 else 0
        
        # Actualizar landmarks anteriores con la lista normalizada
        self.last_landmarks = current_landmarks
        
        return avg_distance > self.movement_threshold
    
    def _start_gesture_recording(self, landmarks, timestamp: float):
        """Inicia la grabación de un gesto."""
        self.is_recording = True
        self.gesture_frames = []
        self.gesture_start_time = timestamp
        self.last_movement_time = timestamp
        
        # Agregar primer frame
        self.gesture_frames.append({
            'landmarks': landmarks,
            'timestamp': timestamp,
            'frame_index': 0
        })
        
        if self.on_gesture_start:
            self.on_gesture_start()
    
    def _continue_gesture_recording(self, landmarks, timestamp: float):
        """Continúa la grabación del gesto."""
        frame_index = len(self.gesture_frames)
        self.gesture_frames.append({
            'landmarks': landmarks,
            'timestamp': timestamp,
            'frame_index': frame_index
        })
    
    def _end_gesture_recording(self, timestamp: float, reason: str = 'unknown'):
        """Finaliza la grabación del gesto."""
        if not self.is_recording:
            return
        
        gesture_data = {
            'frames': self.gesture_frames.copy(),
            'duration': timestamp - self.gesture_start_time,
            'frame_count': len(self.gesture_frames),
            'end_reason': reason,
            'timestamp': timestamp
        }
        
        # Resetear estado
        self.is_recording = False
        self.gesture_frames = []
        self.last_landmarks = None
        
        if self.on_gesture_end:
            self.on_gesture_end(gesture_data)
    
    def _handle_no_hands_during_recording(self, timestamp: float):
        """Maneja el caso cuando no se detectan manos durante la grabación."""
        # Si no hay manos por más de la mitad del threshold de pausa, finalizar
        no_hands_duration = timestamp - self.last_movement_time
        if no_hands_duration >= (self.pause_threshold / 2):
            self._end_gesture_recording(timestamp, reason='no_hands')
    
    def get_current_gesture_data(self) -> Optional[dict]:
        """Retorna los datos del gesto actual si está siendo grabado."""
        if not self.is_recording or len(self.gesture_frames) == 0:
            return None
        
        return {
            'frames': self.gesture_frames.copy(),
            'duration': time.time() - self.gesture_start_time,
            'frame_count': len(self.gesture_frames),
            'is_recording': True
        }
    
    def force_end_gesture(self, reason: str = 'manual'):
        """Fuerza el final del gesto actual."""
        if self.is_recording:
            self._end_gesture_recording(time.time(), reason)
    
    def reset(self):
        """Resetea el estado del detector."""
        self.is_recording = False
        self.gesture_frames = []
        self.last_landmarks = None
        self.last_movement_time = 0
        self.gesture_start_time = 0
