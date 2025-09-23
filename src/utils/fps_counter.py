"""
Utilidades para el sistema de detección de manos.
"""

import time
from typing import Optional

try:
    from ..config.settings import *
except ImportError:
    from config.settings import *


class FPSCounter:
    """Contador de FPS para monitorear el rendimiento."""
    
    def __init__(self, calculation_interval: int = FPS_CALCULATION_INTERVAL):
        """
        Inicializa el contador de FPS.
        
        Args:
            calculation_interval: Intervalo para calcular FPS
        """
        self.calculation_interval = calculation_interval
        self.frame_count = 0
        self.start_time = time.time()
        self.current_fps = 0.0
    
    def update(self) -> Optional[float]:
        """
        Actualiza el contador y retorna FPS si es momento de calcular.
        
        Returns:
            FPS actual si es momento de calcular, None en caso contrario
        """
        self.frame_count += 1
        
        if self.frame_count % self.calculation_interval == 0:
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            self.current_fps = self.calculation_interval / elapsed_time
            self.start_time = current_time
            return self.current_fps
        
        return None
    
    def get_fps(self) -> float:
        """Retorna el FPS actual."""
        return self.current_fps
