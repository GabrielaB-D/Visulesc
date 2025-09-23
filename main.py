"""
Sistema de Reconocimiento de Lenguaje de Señas LESCO
Aplicación principal que integra detección de landmarks y reconocimiento básico de gestos.
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
from ui.main_window import MainWindow
from gesture_recognition.gesture_classifier import GestureClassifier
from data.user_manager import UserManager

# MediaPipe
import mediapipe as mp

class SimpleGestureRecognizer:
    def __init__(self):
        self.gesture_functions = {'A': self._recognize_gesture_A}
        self.debug_mode = False

    def recognize_gesture(self, landmarks) -> str:
        if not landmarks or len(landmarks) < 21:
            return None
        for name, func in self.gesture_functions.items():
            if func(landmarks):
                return name
        return None

    def set_debug_mode(self, enabled: bool):
        """Activa o desactiva el modo debug para mostrar información detallada"""
        self.debug_mode = enabled

    def _recognize_gesture_A(self, lm) -> bool:
        try:
            # Obtener puntos clave
            thumb_tip = lm[4]; thumb_ip = lm[3]; thumb_mcp = lm[2]
            index_mcp = lm[5]; index_tip = lm[8]
            middle_tip = lm[12]; ring_tip = lm[16]; pinky_tip = lm[20]
            wrist = lm[0]

            # Verificar orientación de la mano (palma hacia la cámara)
            hand_orientation_ok = self._check_hand_orientation(lm)
            
            # Verificar que el pulgar esté extendido
            thumb_extended = self._is_thumb_extended(thumb_tip, thumb_ip, thumb_mcp)
            
            # Verificar que los otros dedos estén cerrados
            fingers_closed = self._are_fingers_closed(
                index_tip, lm[6], index_mcp,
                middle_tip, lm[10], lm[9],
                ring_tip, lm[14], lm[13],
                pinky_tip, lm[18], lm[17]
            )
            
            # Verificar ángulo del pulgar con respecto al índice
            thumb_angle_ok = self._check_thumb_index_angle(thumb_tip, thumb_mcp, index_mcp)
            
            # Debug information
            if self.debug_mode:
                print(f"Debug Gesto A:")
                print(f"  Orientación mano: {hand_orientation_ok}")
                print(f"  Pulgar extendido: {thumb_extended}")
                print(f"  Dedos cerrados: {fingers_closed}")
                print(f"  Ángulo pulgar-índice: {thumb_angle_ok}")
            
            # El gesto "A" se reconoce si todas las condiciones se cumplen
            return hand_orientation_ok and thumb_extended and fingers_closed and thumb_angle_ok
        except Exception as e:
            if self.debug_mode:
                print(f"Error reconociendo gesto A: {e}")
            return False

    def _check_hand_orientation(self, lm) -> bool:
        """
        Verifica que la mano esté orientada correctamente (palma hacia la cámara).
        Para el gesto A, necesitamos que la palma esté visible.
        """
        try:
            # Puntos clave para determinar orientación
            wrist = lm[0]
            middle_mcp = lm[9]
            pinky_mcp = lm[17]
            
            # Calcular vector de la muñeca al dedo medio
            wrist_to_middle = np.array([middle_mcp.x - wrist.x, middle_mcp.y - wrist.y])
            
            # Calcular vector de la muñeca al meñique
            wrist_to_pinky = np.array([pinky_mcp.x - wrist.x, pinky_mcp.y - wrist.y])
            
            # Calcular el ángulo entre estos vectores
            angle = self._angle_between_vectors(wrist_to_middle, wrist_to_pinky)
            
            # Para una mano con palma hacia la cámara, el ángulo debe estar en un rango específico
            # Esto ayuda a distinguir entre palma hacia la cámara vs dorso hacia la cámara
            return 20 <= angle <= 80
            
        except Exception as e:
            if self.debug_mode:
                print(f"Error verificando orientación: {e}")
            return True  # Si hay error, asumir orientación correcta

    def _is_thumb_extended(self, tip, ip, mcp):
        """
        Verifica si el pulgar está extendido hacia arriba.
        Mejorado para ser más robusto con diferentes orientaciones.
        """
        # Calcular distancia de la punta a la base
        tip_to_base = self._distance(tip, mcp)
        
        # Calcular distancia de la articulación intermedia a la base
        ip_to_base = self._distance(ip, mcp)
        
        # El pulgar está extendido si la punta está más lejos de la base
        # que la articulación intermedia
        if tip_to_base > 0 and ip_to_base > 0:
            extension_ratio = tip_to_base / ip_to_base
            return extension_ratio > 1.1  # Reducido de 1.2 a 1.1 para ser más permisivo
        return False

    def _are_fingers_closed(self, i_tip,i_pip,i_mcp,m_tip,m_pip,m_mcp,r_tip,r_pip,r_mcp,p_tip,p_pip,p_mcp):
        """
        Verifica si los dedos índice, medio, anular y meñique están cerrados.
        Mejorado para ser más robusto.
        """
        fingers = [(i_tip,i_pip,i_mcp),(m_tip,m_pip,m_mcp),(r_tip,r_pip,r_mcp),(p_tip,p_pip,p_mcp)]
        closed = 0
        
        for tip, pip, mcp in fingers:
            tip_to_base = self._distance(tip, mcp)
            pip_to_base = self._distance(pip, mcp)
            
            if tip_to_base > 0 and pip_to_base > 0:
                closure_ratio = tip_to_base / pip_to_base
                if closure_ratio < 0.9:  # Aumentado de 0.8 a 0.9 para ser más permisivo
                    closed += 1
        
        # Al menos 3 de los 4 dedos deben estar cerrados
        return closed >= 3

    def _check_thumb_index_angle(self, tip, mcp, index_mcp):
        """
        Verifica el ángulo entre el pulgar y el índice.
        Mejorado para funcionar con orientación normal de la mano.
        """
        # Vector del pulgar (de base a punta)
        thumb_vector = np.array([tip.x - mcp.x, tip.y - mcp.y])
        
        # Vector del índice (de base del pulgar a base del índice)
        index_vector = np.array([index_mcp.x - mcp.x, index_mcp.y - mcp.y])
        
        # Calcular ángulo entre vectores
        angle = self._angle_between_vectors(thumb_vector, index_vector)
        
        # Para el gesto "A", el ángulo debe estar entre 45° y 135°
        # Rango ampliado para ser más permisivo
        return 45 <= angle <= 135

    def _distance(self,p1,p2):
        return math.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2)

    def _angle_between_vectors(self,v1,v2):
        mag1 = np.linalg.norm(v1)
        mag2 = np.linalg.norm(v2)
        if mag1==0 or mag2==0: return 0
        cos_angle = max(-1,min(1,np.dot(v1,v2)/(mag1*mag2)))
        return math.degrees(math.acos(cos_angle))


class LESCORecognitionSystem:
    def __init__(self):
        print("Iniciando Sistema LESCO...")
        
        # Componentes del sistema
        self.gesture_recognizer = SimpleGestureRecognizer()
        self.fps_counter = FPSCounter()
        self.main_window = MainWindow()
        self.classifier = GestureClassifier()
        self.user_manager = UserManager()
        
        # Estado del sistema
        self.last_recognized_gesture = None
        self.gesture_stability_counter = 0
        self.GESTURE_STABILITY_THRESHOLD = 5
        self.is_running = False
        self.debug_mode = False
        self.current_user = "default"

        # MediaPipe Hands setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        print("Sistema LESCO inicializado correctamente")
        print("Sistema completo con entrenamiento de gestos disponible")
        print("Controles: ESC - Salir | R - Reconocimiento | T - Entrenamiento | M - Gestión | H - Ayuda")

    def start(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, CAPTURE_BUFFER_SIZE)
        if not self.cap.isOpened():
            print("Error: No se pudo abrir la cámara")
            return False
        
        # Inicializar ventana principal
        if not self.main_window.start():
            print("Error: No se pudo inicializar la ventana principal")
            return False
        
        self.is_running = True
        return True

    def run(self):
        if not self.start(): return
        try:
            while self.is_running and self.cap.isOpened() and self.main_window.is_active():
                ret, frame = self.cap.read()
                if not ret: break
                
                frame_resized = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
                rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                
                # Detectar landmarks
                results = self.hands.process(rgb)
                landmarks = None
                recognized_gesture = None
                
                # Procesar resultados
                if results.multi_hand_landmarks:
                    landmarks = results.multi_hand_landmarks[0].landmark
                    
                    # Reconocer gesto usando el clasificador entrenado
                    if self.classifier.is_trained:
                        # Usar clasificador ML
                        gesture_data = {
                            'frames': [{'landmarks': landmarks}],
                            'duration': 0.1,
                            'frame_count': 1
                        }
                        prediction, confidence = self.classifier.classify_gesture(gesture_data, self.current_user)
                        if confidence > 0.7:  # Umbral de confianza
                            recognized_gesture = prediction
                    else:
                        # Usar reconocedor simple como fallback
                        gesture = self.gesture_recognizer.recognize_gesture(landmarks)
                        if gesture:
                            recognized_gesture = gesture
                    
                    # Manejar estabilidad del gesto
                    if recognized_gesture == self.last_recognized_gesture:
                        self.gesture_stability_counter += 1
                    else:
                        self.gesture_stability_counter = 0
                        self.last_recognized_gesture = recognized_gesture
                    
                    if self.gesture_stability_counter >= self.GESTURE_STABILITY_THRESHOLD and recognized_gesture:
                        print(f"✓ Gesto reconocido: {recognized_gesture}")
                        # Añadir al texto reconocido
                        self.main_window.text_display.add_text(recognized_gesture)

                # Actualizar ventana principal
                window_frame = self.main_window.update_frame(frame_resized, landmarks, recognized_gesture)
                
                # Mostrar frame
                cv2.imshow(self.main_window.window_name, window_frame)
                
                # Manejar teclado
                key = cv2.waitKey(1) & 0xFF
                if not self.main_window.handle_keyboard(key):
                    # Manejar teclas específicas del sistema principal
                    if key == ord('n') or key == ord('N'):
                        self._start_new_gesture_training()
                
        finally:
            self.cleanup()

    def _start_new_gesture_training(self):
        """Inicia el entrenamiento de un nuevo gesto."""
        if self.main_window.current_mode != "training":
            print("Debes estar en modo entrenamiento (T) para entrenar gestos")
            return
        
        print("\n=== ENTRENAMIENTO DE NUEVO GESTO ===")
        gesture_name = input("Ingresa el nombre del gesto: ").strip()
        
        if not gesture_name:
            print("Nombre de gesto no válido")
            return
        
        if gesture_name in self.classifier.get_available_gestures(self.current_user):
            print(f"El gesto '{gesture_name}' ya existe. ¿Deseas sobrescribirlo? (s/n): ", end="")
            response = input().strip().lower()
            if response != 's':
                print("Entrenamiento cancelado")
                return
        
        print(f"Iniciando entrenamiento de '{gesture_name}'...")
        print("Se necesitan 5 muestras del gesto")
        print("Realiza el gesto y manténlo estable por unos segundos")
        
        success = self.main_window.start_new_gesture_training(gesture_name)
        if success:
            print("Entrenamiento iniciado. Sigue las instrucciones en pantalla.")
        else:
            print("Error iniciando el entrenamiento")

    def _draw_debug_info(self, frame):
        """Dibuja información de debug en el frame"""
        if self.debug_mode:
            # Información del gesto actual
            y_offset = 30
            cv2.putText(frame, f"Debug Mode: ON", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            y_offset += 25
            cv2.putText(frame, f"Gesto actual: {self.last_recognized_gesture or 'Ninguno'}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            y_offset += 25
            cv2.putText(frame, f"Estabilidad: {self.gesture_stability_counter}/{self.GESTURE_STABILITY_THRESHOLD}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            y_offset += 25
            cv2.putText(frame, "Gesto A: Pulgar arriba, otros dedos cerrados", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        else:
            # Información básica
            cv2.putText(frame, f"Gesto: {self.last_recognized_gesture or 'Ninguno'}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def _show_help(self):
        """Muestra la ayuda del sistema"""
        print("\n=== AYUDA DEL SISTEMA LESCO ===")
        print("Controles:")
        print("  ESC - Salir del programa")
        print("  SPACE - Limpiar texto reconocido")
        print("  D - Activar/desactivar modo debug")
        print("  H - Mostrar esta ayuda")
        print("\nGestos disponibles:")
        print("  A - Pulgar extendido hacia arriba, otros dedos cerrados")
        print("\nConsejos:")
        print("  - Mantén la palma de la mano hacia la cámara")
        print("  - Asegúrate de que la mano esté bien iluminada")
        print("  - Mantén el gesto estable por unos segundos")
        print("  - Usa el modo debug (D) para ver información detallada")
        print("=" * 35)

    def cleanup(self):
        print("Limpiando recursos...")
        if hasattr(self,'cap'): self.cap.release()
        if hasattr(self,'main_window'): self.main_window.close()
        cv2.destroyAllWindows()
        if hasattr(self,'hands'): self.hands.close()
        print("Recursos liberados correctamente")


def main():
    print("=== Sistema de Reconocimiento LESCO ===")
    print("Sistema Completo con Entrenamiento de Gestos\n")
    print("Características implementadas:")
    print("- Reconocimiento de gestos con Machine Learning")
    print("- Entrenamiento guiado de nuevos gestos")
    print("- Interfaz visual completa")
    print("- Gestión de usuarios y gestos")
    print("- Detección mejorada del gesto 'A'")
    print("\nControles principales:")
    print("- R: Modo reconocimiento")
    print("- T: Modo entrenamiento")
    print("- M: Modo gestión")
    print("- H: Ayuda completa")
    print("\nPresiona H en el programa para ver todos los controles\n")
    
    system = LESCORecognitionSystem()
    system.run()
    print("Sistema finalizado")


if __name__ == "__main__":
    main()
