"""
Interfaz de usuario para el entrenamiento de gestos.
Proporciona una interfaz visual guiada para entrenar nuevos gestos.
"""

import cv2
import numpy as np
from typing import Dict, Optional, Callable, List
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.gesture_recognition.gesture_trainer import GestureTrainer
from src.gesture_recognition.lesco_gestures import LESCOGestures


class TrainingUI:
    """Interfaz de usuario para entrenamiento de gestos."""
    
    def __init__(self, window_name: str = "Entrenamiento de Gestos"):
        """
        Inicializa la interfaz de entrenamiento.
        
        Args:
            window_name: Nombre de la ventana
        """
        self.window_name = window_name
        self.trainer = GestureTrainer()
        self.is_active = False
        self.current_gesture_name = ""
        self.training_state = "idle"  # idle, waiting_for_gesture, recording, completed
        self.instruction_text = ""
        self.progress_info = {}
        
        # Configurar callbacks del entrenador
        self.trainer.on_training_start = self._on_training_start
        self.trainer.on_sample_recorded = self._on_sample_recorded
        self.trainer.on_training_complete = self._on_training_complete
        self.trainer.on_training_error = self._on_training_error
        
        # Callbacks para el sistema principal
        self.on_training_completed: Optional[Callable] = None
        self.on_training_cancelled: Optional[Callable] = None
    
    def start_training_session(self, gesture_name: str, user_id: str = "default", samples_needed: int = 5) -> bool:
        """
        Inicia una sesión de entrenamiento.
        
        Args:
            gesture_name: Nombre del gesto a entrenar
            user_id: ID del usuario
            samples_needed: Número de muestras necesarias
            
        Returns:
            True si se inició exitosamente
        """
        if self.is_active:
            print("Ya hay una sesión de entrenamiento activa")
            return False
        
        success = self.trainer.start_training_session(gesture_name, user_id, samples_needed)
        if success:
            self.is_active = True
            self.current_gesture_name = gesture_name
            self.training_state = "waiting_for_gesture"
            self._update_instruction_text()
            print(f"Iniciando entrenamiento de '{gesture_name}'")
        
        return success
    
    def process_frame(self, frame: np.ndarray, landmarks: List) -> np.ndarray:
        """
        Procesa un frame durante el entrenamiento.
        
        Args:
            frame: Frame de la cámara
            landmarks: Landmarks detectados
            
        Returns:
            Frame procesado con información de entrenamiento
        """
        if not self.is_active:
            return frame
        
        # Procesar con el entrenador
        training_info = self.trainer.process_frame(landmarks, cv2.getTickCount() / cv2.getTickFrequency())
        
        # Crear frame de entrenamiento
        training_frame = self._create_training_frame(frame, training_info)
        
        return training_frame
    
    def _create_training_frame(self, frame: np.ndarray, training_info: Dict) -> np.ndarray:
        """
        Crea el frame de entrenamiento con información visual.
        
        Args:
            frame: Frame original
            training_info: Información del entrenamiento
            
        Returns:
            Frame con overlay de entrenamiento
        """
        # Crear copia del frame
        training_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # Dibujar fondo semi-transparente para el panel de información
        overlay = training_frame.copy()
        cv2.rectangle(overlay, (10, 10), (width - 10, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, training_frame, 0.3, 0, training_frame)
        
        # Información del entrenamiento
        y_offset = 40
        cv2.putText(training_frame, f"ENTRENAMIENTO: {self.current_gesture_name}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        y_offset += 35
        cv2.putText(training_frame, f"Muestra: {training_info.get('current_sample', 0)}/{training_info.get('target_samples', 5)}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Barra de progreso
        progress = training_info.get('progress', 0)
        bar_width = 300
        bar_height = 20
        bar_x = 20
        bar_y = y_offset + 20
        
        # Fondo de la barra
        cv2.rectangle(training_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        
        # Progreso
        progress_width = int((progress / 100) * bar_width)
        cv2.rectangle(training_frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), (0, 255, 0), -1)
        
        # Texto de progreso
        cv2.putText(training_frame, f"{progress:.1f}%", 
                   (bar_x + bar_width + 10, bar_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Estado actual
        y_offset += 60
        state = training_info.get('detection_state', 'idle')
        state_color = self._get_state_color(state)
        cv2.putText(training_frame, f"Estado: {state}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)
        
        # Instrucciones
        y_offset += 30
        cv2.putText(training_frame, self.instruction_text, 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Controles
        y_offset += 25
        cv2.putText(training_frame, "Controles: ESC - Cancelar | SPACE - Forzar siguiente muestra", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        return training_frame
    
    def _get_state_color(self, state: str) -> tuple:
        """Retorna el color apropiado para el estado."""
        color_map = {
            'idle': (128, 128, 128),           # Gris
            'gesture_started': (0, 255, 255),  # Amarillo
            'gesture_recording': (0, 255, 0),  # Verde
            'gesture_waiting': (0, 165, 255),  # Naranja
            'gesture_ended_pause': (0, 0, 255), # Rojo
            'gesture_ended_max_frames': (255, 0, 255) # Magenta
        }
        return color_map.get(state, (255, 255, 255))
    
    def _update_instruction_text(self):
        """Actualiza el texto de instrucciones según el estado."""
        if self.training_state == "waiting_for_gesture":
            self.instruction_text = "Realiza el gesto y manténlo estable por unos segundos"
        elif self.training_state == "recording":
            self.instruction_text = "Grabando gesto... mantén la posición"
        elif self.training_state == "completed":
            self.instruction_text = "Entrenamiento completado exitosamente"
        else:
            self.instruction_text = "Preparando entrenamiento..."
    
    def handle_key(self, key: int) -> bool:
        """
        Maneja las teclas presionadas durante el entrenamiento.
        
        Args:
            key: Código de la tecla presionada
            
        Returns:
            True si la tecla fue manejada
        """
        if not self.is_active:
            return False
        
        if key == 27:  # ESC
            self.cancel_training()
            return True
        elif key == ord(' '):  # SPACE
            self._force_next_sample()
            return True
        
        return False
    
    def _force_next_sample(self):
        """Fuerza el final de la muestra actual."""
        if self.is_active and self.training_state == "recording":
            self.trainer.gesture_detector.force_end_gesture("manual")
    
    def cancel_training(self):
        """Cancela la sesión de entrenamiento."""
        if self.is_active:
            self.trainer.cancel_training()
            self.is_active = False
            self.training_state = "idle"
            self.current_gesture_name = ""
            
            if self.on_training_cancelled:
                self.on_training_cancelled()
            
            print("Entrenamiento cancelado")
    
    def _on_training_start(self, info: Dict):
        """Callback cuando inicia el entrenamiento."""
        self.training_state = "waiting_for_gesture"
        self._update_instruction_text()
        print(f"Entrenamiento iniciado: {info['gesture_name']}")
    
    def _on_sample_recorded(self, info: Dict):
        """Callback cuando se graba una muestra."""
        self.training_state = "waiting_for_gesture"
        self._update_instruction_text()
        print(f"Muestra {info['sample_number']}/{info['total_samples']} completada")
    
    def _on_training_complete(self, info: Dict):
        """Callback cuando se completa el entrenamiento."""
        self.training_state = "completed"
        self._update_instruction_text()
        self.is_active = False
        
        if self.on_training_completed:
            self.on_training_completed(info)
        
        print(f"Entrenamiento completado: {info['gesture_name']}")
    
    def _on_training_error(self, error_message: str):
        """Callback cuando hay un error en el entrenamiento."""
        print(f"Error en entrenamiento: {error_message}")
        # No cancelar automáticamente, permitir que el usuario continúe
    
    def get_training_status(self) -> Dict:
        """Retorna el estado actual del entrenamiento."""
        return {
            'is_active': self.is_active,
            'gesture_name': self.current_gesture_name,
            'state': self.training_state,
            'trainer_status': self.trainer.get_training_status()
        }
    
    def get_available_gestures(self, user_id: str = "default") -> List[str]:
        """Retorna los gestos disponibles para un usuario."""
        return self.trainer.get_available_gestures(user_id)
    
    def delete_gesture(self, gesture_name: str, user_id: str = "default") -> bool:
        """Elimina un gesto del sistema."""
        return self.trainer.delete_gesture(gesture_name, user_id)