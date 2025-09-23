"""
Detector de landmarks de manos usando MediaPipe.
"""

import cv2
import mediapipe as mp
import time
from typing import Optional, Callable
try:
    from ..config.settings import *
except ImportError:
    from config.settings import *


class HandDetector:
    """Clase principal para la detección de landmarks de manos."""
    
    def __init__(self, result_callback: Optional[Callable] = None):
        """
        Inicializa el detector de manos.
        
        Args:
            result_callback: Función callback para procesar resultados
        """
        self.result_callback = result_callback
        self.last_result = None
        self.frame_counter = 0
        
        # Configurar MediaPipe
        self._setup_mediapipe()
        
    def _setup_mediapipe(self):
        """Configura MediaPipe con los parámetros especificados."""
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        
        options = HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=MODEL_PATH,
                delegate=BaseOptions.Delegate.CPU
            ),
            running_mode=VisionRunningMode.LIVE_STREAM,
            num_hands=NUM_HANDS,
            min_hand_detection_confidence=MIN_HAND_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=MIN_HAND_PRESENCE_CONFIDENCE,
            result_callback=self._process_result
        )
        
        self.landmarker = HandLandmarker.create_from_options(options)
    
    def _process_result(self, result, output_image, timestamp_ms: int):
        """
        Procesa el resultado de la detección.
        
        Args:
            result: Resultado de la detección
            output_image: Imagen procesada
            timestamp_ms: Timestamp en milisegundos
        """
        self.last_result = result
        if self.result_callback:
            self.result_callback(result, output_image, timestamp_ms)
    
    def detect_async(self, image: mp.Image, timestamp_ms: int):
        """
        Realiza detección asíncrona de landmarks.
        
        Args:
            image: Imagen de MediaPipe
            timestamp_ms: Timestamp en milisegundos
        """
        if self.frame_counter % SEND_EVERY_N_FRAMES == 0:
            self.landmarker.detect_async(image, timestamp_ms)
        self.frame_counter += 1
    
    def get_last_result(self):
        """Retorna el último resultado de detección."""
        return self.last_result
    
    def close(self):
        """Cierra el detector y libera recursos."""
        if hasattr(self, 'landmarker'):
            self.landmarker.close()
