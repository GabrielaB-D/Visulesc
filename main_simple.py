"""
Sistema de Reconocimiento de Lenguaje de Señas LESCO - Versión Simplificada
Versión que funciona sin MediaPipe para evitar problemas de DLL.
"""

import cv2
import time
import sys
import os
import numpy as np
import math

# Añadir el directorio src al path para imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils import FPSCounter
from config.settings import *

class SimpleGestureRecognizer:
    def __init__(self):
        self.gesture_functions = {'A': self._recognize_gesture_A}
        self.debug_mode = False

    def recognize_gesture(self, frame) -> str:
        """
        Reconoce gestos usando detección básica de contornos.
        Esta es una versión simplificada que no requiere MediaPipe.
        """
        # Convertir a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Aplicar filtro gaussiano
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detectar bordes
        edges = cv2.Canny(blurred, 50, 150)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Encontrar el contorno más grande (probablemente la mano)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calcular características del contorno
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        # Aproximar el contorno
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Detectar gesto A basado en características del contorno
        if self._is_gesture_A(area, len(approx), largest_contour):
            return 'A'
        
        return None

    def _is_gesture_A(self, area, vertices, contour) -> bool:
        """
        Detecta si el contorno corresponde al gesto A.
        Criterios simplificados basados en área y forma.
        """
        # El gesto A debe tener un área mínima
        if area < 5000:
            return False
        
        # El contorno debe tener cierta complejidad (no muy simple)
        if vertices < 5 or vertices > 15:
            return False
        
        # Calcular la relación de aspecto del contorno
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        
        # El gesto A debe tener una relación de aspecto razonable
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            return False
        
        return True

    def set_debug_mode(self, enabled: bool):
        """Activa o desactiva el modo debug"""
        self.debug_mode = enabled

class LESCORecognitionSystem:
    def __init__(self):
        print("Iniciando Sistema LESCO (Versión Simplificada)...")
        
        # Componentes del sistema
        self.gesture_recognizer = SimpleGestureRecognizer()
        self.fps_counter = FPSCounter()
        
        # Estado del sistema
        self.last_recognized_gesture = None
        self.gesture_stability_counter = 0
        self.GESTURE_STABILITY_THRESHOLD = 10  # Más frames para estabilidad
        self.is_running = False
        self.debug_mode = False
        self.current_user = "default"
        
        # Texto reconocido
        self.recognized_text = ""
        self.text_history = []
        
        print("Sistema LESCO inicializado correctamente")
        print("Versión simplificada - Detección básica de gestos")
        print("Controles: ESC - Salir | SPACE - Limpiar | D - Debug | H - Ayuda")

    def start(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, CAPTURE_BUFFER_SIZE)
        if not self.cap.isOpened():
            print("Error: No se pudo abrir la cámara")
            return False
        
        self.is_running = True
        return True

    def run(self):
        if not self.start(): 
            return
        
        try:
            while self.is_running and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret: 
                    break
                
                frame_resized = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
                
                # Reconocer gesto
                gesture = self.gesture_recognizer.recognize_gesture(frame_resized)
                
                # Manejar estabilidad del gesto
                if gesture == self.last_recognized_gesture:
                    self.gesture_stability_counter += 1
                else:
                    self.gesture_stability_counter = 0
                    self.last_recognized_gesture = gesture
                
                if self.gesture_stability_counter >= self.GESTURE_STABILITY_THRESHOLD and gesture:
                    print(f"✓ Gesto reconocido: {gesture}")
                    self._add_to_text(gesture)
                    self.gesture_stability_counter = 0  # Reset para evitar repeticiones

                # Crear frame de visualización
                display_frame = self._create_display_frame(frame_resized, gesture)
                
                # Mostrar frame
                cv2.imshow("Sistema LESCO - Versión Simplificada", display_frame)
                
                # Manejar teclado
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord(' '):  # SPACE
                    self._clear_text()
                elif key in [ord('d'), ord('D')]:  # Debug
                    self.debug_mode = not self.debug_mode
                    self.gesture_recognizer.set_debug_mode(self.debug_mode)
                    print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")
                elif key in [ord('h'), ord('H')]:  # Ayuda
                    self._show_help()
                
        finally:
            self.cleanup()

    def _create_display_frame(self, frame, gesture):
        """Crea el frame de visualización con información"""
        display_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # Panel de información
        panel_height = 150
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
        
        # Título
        cv2.putText(display_frame, "SISTEMA LESCO - VERSION SIMPLIFICADA", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Gesto actual
        y_offset = 50
        cv2.putText(display_frame, f"Gesto detectado: {gesture or 'Ninguno'}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Estabilidad
        y_offset += 25
        cv2.putText(display_frame, f"Estabilidad: {self.gesture_stability_counter}/{self.GESTURE_STABILITY_THRESHOLD}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Texto reconocido
        y_offset += 25
        cv2.putText(display_frame, f"Texto: {self.recognized_text}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Controles
        y_offset += 25
        cv2.putText(display_frame, "Controles: ESC-Salir | SPACE-Limpiar | D-Debug | H-Ayuda", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Debug info
        if self.debug_mode:
            y_offset += 20
            cv2.putText(display_frame, "MODO DEBUG ACTIVO", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        return display_frame

    def _add_to_text(self, gesture):
        """Añade un gesto al texto reconocido"""
        if gesture:
            if self.recognized_text:
                self.recognized_text += " " + gesture
            else:
                self.recognized_text = gesture
            
            # Añadir al historial
            self.text_history.append({
                'gesture': gesture,
                'timestamp': time.time()
            })
            
            # Mantener solo los últimos 20 gestos
            if len(self.text_history) > 20:
                self.text_history = self.text_history[-20:]

    def _clear_text(self):
        """Limpia el texto reconocido"""
        self.recognized_text = ""
        print("Texto limpiado")

    def _show_help(self):
        """Muestra la ayuda del sistema"""
        print("\n=== AYUDA DEL SISTEMA LESCO ===")
        print("Versión Simplificada - Sin MediaPipe")
        print("\nControles:")
        print("  ESC - Salir del programa")
        print("  SPACE - Limpiar texto reconocido")
        print("  D - Activar/desactivar modo debug")
        print("  H - Mostrar esta ayuda")
        print("\nGestos disponibles:")
        print("  A - Gesto básico detectado por contornos")
        print("\nNotas:")
        print("  - Esta versión usa detección básica de contornos")
        print("  - Para mejor precisión, instala MediaPipe correctamente")
        print("  - Mantén la mano bien iluminada y visible")
        print("=" * 40)

    def cleanup(self):
        print("Limpiando recursos...")
        if hasattr(self, 'cap'): 
            self.cap.release()
        cv2.destroyAllWindows()
        print("Recursos liberados correctamente")

def main():
    print("=== Sistema de Reconocimiento LESCO ===")
    print("Versión Simplificada - Sin MediaPipe\n")
    print("Esta versión funciona sin MediaPipe para evitar problemas de DLL.")
    print("Usa detección básica de contornos para reconocer gestos.")
    print("\nPara la versión completa con entrenamiento de gestos:")
    print("1. Soluciona el problema de MediaPipe")
    print("2. Ejecuta: python main.py")
    print("\nPresiona H en el programa para ver la ayuda\n")
    
    system = LESCORecognitionSystem()
    system.run()
    print("Sistema finalizado")

if __name__ == "__main__":
    main()
