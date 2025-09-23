"""
Tests básicos para el sistema de detección de manos.
"""

import unittest
import sys
import os

# Añadir src al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import FPSCounter
from config.settings import TARGET_WIDTH, TARGET_HEIGHT


class TestFPSCounter(unittest.TestCase):
    """Tests para el contador de FPS."""
    
    def setUp(self):
        """Configuración inicial para cada test."""
        self.fps_counter = FPSCounter(calculation_interval=5)
    
    def test_initial_fps(self):
        """Test del FPS inicial."""
        self.assertEqual(self.fps_counter.get_fps(), 0.0)
    
    def test_fps_calculation(self):
        """Test del cálculo de FPS."""
        # Simular algunos frames
        for _ in range(5):
            fps = self.fps_counter.update()
        
        # El quinto frame debería retornar un FPS calculado
        self.assertIsNotNone(fps)
        self.assertGreater(fps, 0)


class TestConfiguration(unittest.TestCase):
    """Tests para la configuración del sistema."""
    
    def test_target_dimensions(self):
        """Test de las dimensiones objetivo."""
        self.assertEqual(TARGET_WIDTH, 256)
        self.assertEqual(TARGET_HEIGHT, 192)
        self.assertGreater(TARGET_WIDTH, 0)
        self.assertGreater(TARGET_HEIGHT, 0)


if __name__ == '__main__':
    unittest.main()
