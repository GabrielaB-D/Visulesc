"""
Sistema de entrenamiento guiado para gestos de LESCO.
Proporciona una interfaz paso a paso para entrenar nuevos gestos.
"""

import os
import json
import time
from typing import List, Dict, Optional, Callable
from .gesture_detector import GestureDetector
from .gesture_classifier import GestureClassifier


class GestureTrainer:
    """Entrenador guiado de gestos para LESCO."""
    
    def __init__(self, data_dir: str = "data/gestures"):
        """
        Inicializa el entrenador de gestos.
        
        Args:
            data_dir: Directorio para almacenar datos de gestos
        """
        self.data_dir = data_dir
        self.gesture_detector = GestureDetector()
        self.classifier = GestureClassifier(data_dir)
        
        # Estado del entrenamiento
        self.is_training = False
        self.current_gesture_name = ""
        self.current_user_id = "default"
        self.training_samples = []
        self.current_sample_count = 0
        self.target_samples = 5  # Número de muestras por gesto
        
        # Callbacks para la UI
        self.on_training_start: Optional[Callable] = None
        self.on_sample_recorded: Optional[Callable] = None
        self.on_training_complete: Optional[Callable] = None
        self.on_training_error: Optional[Callable] = None
        
        # Configurar callbacks del detector
        self.gesture_detector.on_gesture_start = self._on_gesture_start
        self.gesture_detector.on_gesture_end = self._on_gesture_end
    
    def start_training_session(self, gesture_name: str, user_id: str = "default", samples_needed: int = 5):
        """
        Inicia una sesión de entrenamiento para un gesto.
        
        Args:
            gesture_name: Nombre del gesto a entrenar
            user_id: ID del usuario
            samples_needed: Número de muestras necesarias
        """
        if self.is_training:
            self._handle_error("Ya hay una sesión de entrenamiento activa")
            return False
        
        self.current_gesture_name = gesture_name
        self.current_user_id = user_id
        self.target_samples = samples_needed
        self.training_samples = []
        self.current_sample_count = 0
        self.is_training = True
        
        print(f"Iniciando entrenamiento de gesto '{gesture_name}' para usuario '{user_id}'")
        print(f"Se necesitan {samples_needed} muestras")
        
        if self.on_training_start:
            self.on_training_start({
                'gesture_name': gesture_name,
                'user_id': user_id,
                'samples_needed': samples_needed,
                'current_sample': 0
            })
        
        return True
    
    def process_frame(self, landmarks: List, timestamp: float) -> Dict:
        """
        Procesa un frame durante el entrenamiento.
        
        Args:
            landmarks: Landmarks detectados
            timestamp: Timestamp del frame
            
        Returns:
            Estado actual del entrenamiento
        """
        if not self.is_training:
            return {'status': 'not_training'}
        
        # Procesar con el detector de gestos
        detection_result = self.gesture_detector.process_landmarks(landmarks, timestamp)
        
        # Información adicional del entrenamiento
        training_info = {
            'status': 'training',
            'gesture_name': self.current_gesture_name,
            'user_id': self.current_user_id,
            'current_sample': self.current_sample_count,
            'target_samples': self.target_samples,
            'progress': (self.current_sample_count / self.target_samples) * 100,
            'detection_state': detection_result['current_state']
        }
        
        return training_info
    
    def _on_gesture_start(self):
        """Callback cuando inicia un gesto durante el entrenamiento."""
        if self.is_training:
            print(f"Muestra {self.current_sample_count + 1}: Iniciando grabación...")
    
    def _on_gesture_end(self, gesture_data: Dict):
        """Callback cuando termina un gesto durante el entrenamiento."""
        if not self.is_training:
            return
        
        # Validar que el gesto tenga suficientes frames
        if gesture_data['frame_count'] < self.gesture_detector.min_gesture_frames:
            print(f"Muestra {self.current_sample_count + 1}: Gesto muy corto, inténtalo de nuevo")
            if self.on_training_error:
                self.on_training_error(f"Gesto muy corto ({gesture_data['frame_count']} frames)")
            return
        
        # Agregar muestra
        self.training_samples.append(gesture_data)
        self.current_sample_count += 1
        
        print(f"Muestra {self.current_sample_count}/{self.target_samples} completada")
        
        if self.on_sample_recorded:
            self.on_sample_recorded({
                'sample_number': self.current_sample_count,
                'total_samples': self.target_samples,
                'gesture_data': gesture_data,
                'progress': (self.current_sample_count / self.target_samples) * 100
            })
        
        # Verificar si se completó el entrenamiento
        if self.current_sample_count >= self.target_samples:
            self._complete_training()
    
    def _complete_training(self):
        """Completa el entrenamiento del gesto."""
        print(f"Entrenamiento completado para '{self.current_gesture_name}'")
        
        # Guardar todas las muestras
        for i, sample in enumerate(self.training_samples):
            sample['sample_index'] = i
            self.classifier.add_gesture_sample(sample, self.current_gesture_name, self.current_user_id)
        
        # Entrenar el clasificador
        success = self.classifier.train_classifier(self.current_user_id)
        
        if success:
            print(f"Gesto '{self.current_gesture_name}' entrenado exitosamente")
            if self.on_training_complete:
                self.on_training_complete({
                    'gesture_name': self.current_gesture_name,
                    'user_id': self.current_user_id,
                    'samples_count': self.current_sample_count,
                    'success': True
                })
        else:
            print(f"Error entrenando el gesto '{self.current_gesture_name}'")
            if self.on_training_error:
                self.on_training_error("Error en el entrenamiento del clasificador")
        
        # Resetear estado
        self._reset_training_state()
    
    def cancel_training(self):
        """Cancela la sesión de entrenamiento actual."""
        if self.is_training:
            print("Entrenamiento cancelado")
            self._reset_training_state()
    
    def _reset_training_state(self):
        """Resetea el estado del entrenamiento."""
        self.is_training = False
        self.current_gesture_name = ""
        self.current_user_id = "default"
        self.training_samples = []
        self.current_sample_count = 0
        self.target_samples = 5
        self.gesture_detector.reset()
    
    def _handle_error(self, error_message: str):
        """Maneja errores del entrenamiento."""
        print(f"Error en entrenamiento: {error_message}")
        if self.on_training_error:
            self.on_training_error(error_message)
    
    def get_training_status(self) -> Dict:
        """Retorna el estado actual del entrenamiento."""
        return {
            'is_training': self.is_training,
            'gesture_name': self.current_gesture_name,
            'user_id': self.current_user_id,
            'current_sample': self.current_sample_count,
            'target_samples': self.target_samples,
            'progress': (self.current_sample_count / self.target_samples) * 100 if self.target_samples > 0 else 0
        }
    
    def get_available_gestures(self, user_id: str = "default") -> List[str]:
        """Retorna los gestos disponibles para un usuario."""
        return self.classifier.get_available_gestures(user_id)
    
    def delete_gesture(self, gesture_name: str, user_id: str = "default") -> bool:
        """
        Elimina un gesto del sistema.
        
        Args:
            gesture_name: Nombre del gesto a eliminar
            user_id: ID del usuario
            
        Returns:
            True si se eliminó exitosamente
        """
        gesture_file = os.path.join(self.data_dir, f"{user_id}_{gesture_name}.json")
        
        if os.path.exists(gesture_file):
            try:
                os.remove(gesture_file)
                print(f"Gesto '{gesture_name}' eliminado para usuario '{user_id}'")
                return True
            except Exception as e:
                print(f"Error eliminando gesto: {e}")
                return False
        
        return False
