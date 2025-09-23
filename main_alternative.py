"""
Sistema de Reconocimiento de Lenguaje de Señas LESCO - Versión Alternativa
Versión que funciona sin MediaPipe, usando OpenCV y técnicas de visión por computadora.
"""

import cv2
import time
import sys
import os
import numpy as np
import math
import json
from typing import List, Dict, Optional
from datetime import datetime

# Añadir el directorio src al path para imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils import FPSCounter
from config.settings import *

class AlternativeGestureRecognizer:
    """Reconocedor de gestos alternativo usando OpenCV"""
    
    def __init__(self):
        self.debug_mode = False
        self.trained_gestures = {}
        self.load_trained_gestures()

    def recognize_gesture(self, frame) -> str:
        """Reconoce gestos usando técnicas de visión por computadora mejoradas"""
        # Detectar mano usando detección de piel
        hand_mask = self._detect_hand_region(frame)
        
        if hand_mask is None:
            return None
        
        # Encontrar contornos en la máscara de la mano
        contours, _ = cv2.findContours(hand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Encontrar el contorno más grande (la mano)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Verificar que el contorno sea suficientemente grande
        if cv2.contourArea(largest_contour) < 3000:
            return None
        
        # Extraer características del contorno
        features = self._extract_contour_features(largest_contour)
        
        # Intentar reconocer gestos entrenados primero
        for gesture_name, gesture_data in self.trained_gestures.items():
            if self._compare_features(features, gesture_data['features']):
                return gesture_name
        
        # Fallback a gestos predefinidos
        if self._is_gesture_A(features):
            return 'A'
        
        return None

    def _detect_hand_region(self, frame):
        """Detecta la región de la mano usando detección de color de piel"""
        # Convertir a HSV para mejor detección de color
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Definir rangos de color de piel en HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Crear máscara para color de piel
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Aplicar operaciones morfológicas para limpiar la máscara
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        
        # Aplicar filtro gaussiano
        skin_mask = cv2.GaussianBlur(skin_mask, (5, 5), 0)
        
        return skin_mask

    def _extract_contour_features(self, contour):
        """Extrae características de un contorno"""
        features = {}
        
        # Área y perímetro
        features['area'] = cv2.contourArea(contour)
        features['perimeter'] = cv2.arcLength(contour, True)
        
        # Rectángulo delimitador
        x, y, w, h = cv2.boundingRect(contour)
        features['aspect_ratio'] = float(w) / h
        features['extent'] = features['area'] / (w * h)
        
        # Aproximación del contorno
        epsilon = 0.02 * features['perimeter']
        approx = cv2.approxPolyDP(contour, epsilon, True)
        features['vertices'] = len(approx)
        
        # Momentos del contorno
        moments = cv2.moments(contour)
        if moments['m00'] != 0:
            features['centroid_x'] = moments['m10'] / moments['m00']
            features['centroid_y'] = moments['m01'] / moments['m00']
        else:
            features['centroid_x'] = 0
            features['centroid_y'] = 0
        
        # Convexidad
        hull = cv2.convexHull(contour)
        features['convexity'] = features['area'] / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
        
        return features

    def _compare_features(self, features1, features2, tolerance=0.2):
        """Compara dos conjuntos de características"""
        if not features1 or not features2:
            return False
        
        # Comparar características clave
        key_features = ['area', 'aspect_ratio', 'extent', 'vertices', 'convexity']
        
        for feature in key_features:
            if feature in features1 and feature in features2:
                diff = abs(features1[feature] - features2[feature])
                max_val = max(features1[feature], features2[feature])
                if max_val > 0:
                    relative_diff = diff / max_val
                    if relative_diff > tolerance:
                        return False
        
        return True

    def _is_gesture_A(self, features):
        """Detecta si las características corresponden al gesto A con criterios más estrictos"""
        if not features:
            return False
        
        # Criterios mucho más estrictos para el gesto A
        if features['area'] < 15000:  # Área mínima más grande
            return False
        
        if features['vertices'] < 8 or features['vertices'] > 12:  # Rango más estricto
            return False
        
        if features['aspect_ratio'] < 0.8 or features['aspect_ratio'] > 1.2:  # Proporción más estricta
            return False
        
        # Añadir criterio de solidez
        if features.get('solidity', 0) < 0.8:
            return False
        
        return True

    def train_gesture(self, gesture_name: str, samples: List[Dict]):
        """Entrena un nuevo gesto con muestras"""
        if not samples:
            return False
        
        # Calcular características promedio
        all_features = []
        for sample in samples:
            features = self._extract_contour_features(sample['contour'])
            all_features.append(features)
        
        # Calcular características promedio
        avg_features = {}
        for feature_name in all_features[0].keys():
            values = [f[feature_name] for f in all_features if feature_name in f]
            avg_features[feature_name] = sum(values) / len(values) if values else 0
        
        # Guardar gesto entrenado
        self.trained_gestures[gesture_name] = {
            'features': avg_features,
            'samples_count': len(samples),
            'trained_date': datetime.now().isoformat()
        }
        
        # Guardar en archivo
        self.save_trained_gestures()
        
        print(f"Gesto '{gesture_name}' entrenado con {len(samples)} muestras")
        return True

    def save_trained_gestures(self):
        """Guarda los gestos entrenados en un archivo JSON"""
        try:
            os.makedirs('data', exist_ok=True)
            with open('data/trained_gestures.json', 'w', encoding='utf-8') as f:
                json.dump(self.trained_gestures, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error guardando gestos: {e}")

    def load_trained_gestures(self):
        """Carga los gestos entrenados desde un archivo JSON"""
        try:
            if os.path.exists('data/trained_gestures.json'):
                with open('data/trained_gestures.json', 'r', encoding='utf-8') as f:
                    self.trained_gestures = json.load(f)
                print(f"Cargados {len(self.trained_gestures)} gestos entrenados")
        except Exception as e:
            print(f"Error cargando gestos: {e}")
            self.trained_gestures = {}

    def get_trained_gestures(self):
        """Retorna la lista de gestos entrenados"""
        return list(self.trained_gestures.keys())

    def delete_gesture(self, gesture_name: str):
        """Elimina un gesto entrenado"""
        if gesture_name in self.trained_gestures:
            del self.trained_gestures[gesture_name]
            self.save_trained_gestures()
            print(f"Gesto '{gesture_name}' eliminado")
            return True
        return False

    def set_debug_mode(self, enabled: bool):
        """Activa o desactiva el modo debug"""
        self.debug_mode = enabled

class GestureTrainer:
    """Entrenador de gestos simplificado"""
    
    def __init__(self, recognizer):
        self.recognizer = recognizer
        self.is_training = False
        self.current_gesture_name = ""
        self.training_samples = []
        self.target_samples = 5

    def start_training(self, gesture_name: str):
        """Inicia el entrenamiento de un gesto"""
        if self.is_training:
            print("Ya hay un entrenamiento en progreso")
            return False
        
        self.current_gesture_name = gesture_name
        self.training_samples = []
        self.is_training = True
        self.target_samples = 5
        
        print(f"Iniciando entrenamiento de '{gesture_name}'")
        print("Se necesitan 5 muestras del gesto")
        print("Realiza el gesto y manténlo estable por unos segundos")
        
        return True

    def add_sample(self, contour):
        """Añade una muestra al entrenamiento"""
        if not self.is_training:
            return False
        
        self.training_samples.append({
            'contour': contour,
            'timestamp': time.time()
        })
        
        print(f"Muestra {len(self.training_samples)}/{self.target_samples} capturada")
        
        if len(self.training_samples) >= self.target_samples:
            self.complete_training()
            return True
        
        return False

    def complete_training(self):
        """Completa el entrenamiento"""
        if not self.is_training:
            return False
        
        success = self.recognizer.train_gesture(self.current_gesture_name, self.training_samples)
        
        self.is_training = False
        self.current_gesture_name = ""
        self.training_samples = []
        
        if success:
            print(f"Entrenamiento de '{self.current_gesture_name}' completado exitosamente")
        else:
            print("Error en el entrenamiento")
        
        return success

    def cancel_training(self):
        """Cancela el entrenamiento"""
        self.is_training = False
        self.current_gesture_name = ""
        self.training_samples = []
        print("Entrenamiento cancelado")

class LESCORecognitionSystem:
    def __init__(self):
        print("Iniciando Sistema LESCO (Versión Alternativa)...")
        
        # Componentes del sistema
        self.gesture_recognizer = AlternativeGestureRecognizer()
        self.gesture_trainer = GestureTrainer(self.gesture_recognizer)
        self.fps_counter = FPSCounter()
        
        # Estado del sistema
        self.last_recognized_gesture = None
        self.gesture_stability_counter = 0
        self.GESTURE_STABILITY_THRESHOLD = 15  # Mucho más estricto
        self.last_gesture_time = 0
        self.GESTURE_COOLDOWN = 2.0  # 2 segundos entre detecciones
        self.is_running = False
        self.debug_mode = False
        self.current_mode = "recognition"  # recognition, training
        
        # Texto reconocido
        self.recognized_text = ""
        self.text_history = []
        
        print("Sistema LESCO inicializado correctamente")
        print("Versión alternativa con entrenamiento de gestos")
        print("Controles: ESC - Salir | T - Entrenamiento | R - Reconocimiento | H - Ayuda")

    def start(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, CAPTURE_BUFFER_SIZE)
        if not self.cap.isOpened():
            print("Error: No se pudo abrir la cámara")
            return False
        
        # Configurar resolución de la cámara para pantalla completa
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.is_running = True
        return True

    def run(self):
        if not self.start(): 
            return
        
        # Crear ventana en pantalla completa
        cv2.namedWindow("Sistema LESCO - Versión Alternativa", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Sistema LESCO - Versión Alternativa", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        try:
            while self.is_running and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret: 
                    break
                
                # Usar frame completo para mejor detección
                frame_full = frame.copy()
                
                # Procesar según el modo
                if self.current_mode == "training":
                    self._process_training_mode(frame_full)
                else:
                    self._process_recognition_mode(frame_full)
                
                # Crear frame de visualización
                display_frame = self._create_display_frame(frame_full)
                
                # Mostrar frame
                cv2.imshow("Sistema LESCO - Versión Alternativa", display_frame)
                
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
                elif key in [ord('t'), ord('T')]:  # Modo entrenamiento
                    self._switch_to_training_mode()
                elif key in [ord('r'), ord('R')]:  # Modo reconocimiento
                    self._switch_to_recognition_mode()
                elif key in [ord('n'), ord('N')]:  # Nuevo gesto
                    self._start_new_gesture_training()
                elif key == ord('f') or key == ord('F'):  # Toggle pantalla completa
                    self._toggle_fullscreen()
                
        finally:
            self.cleanup()

    def _process_recognition_mode(self, frame):
        """Procesa el frame en modo reconocimiento con cooldown"""
        import time
        current_time = time.time()
        
        gesture = self.gesture_recognizer.recognize_gesture(frame)
        
        # Manejar estabilidad del gesto
        if gesture == self.last_recognized_gesture:
            self.gesture_stability_counter += 1
        else:
            self.gesture_stability_counter = 0
            self.last_recognized_gesture = gesture
        
        # Solo reconocer gestos estables y con cooldown
        if (self.gesture_stability_counter >= self.GESTURE_STABILITY_THRESHOLD and 
            gesture and 
            current_time - self.last_gesture_time > self.GESTURE_COOLDOWN):
            print(f"✓ Gesto reconocido: {gesture}")
            self._add_to_text(gesture)
            self.gesture_stability_counter = 0
            self.last_gesture_time = current_time

    def _process_training_mode(self, frame):
        """Procesa el frame en modo entrenamiento"""
        if not self.gesture_trainer.is_training:
            return
        
        # Detectar mano usando detección de piel
        hand_mask = self.gesture_recognizer._detect_hand_region(frame)
        
        if hand_mask is not None:
            # Encontrar contornos en la máscara de la mano
            contours, _ = cv2.findContours(hand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > 3000:  # Área mínima
                    self.gesture_trainer.add_sample(largest_contour)

    def _create_display_frame(self, frame):
        """Crea el frame de visualización con detección de manos"""
        display_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # Detectar y visualizar la mano si está en modo debug
        if self.debug_mode:
            hand_mask = self.gesture_recognizer._detect_hand_region(frame)
            if hand_mask is not None:
                # Crear imagen de la máscara en color
                mask_colored = cv2.cvtColor(hand_mask, cv2.COLOR_GRAY2BGR)
                # Superponer la máscara en rojo
                display_frame = cv2.addWeighted(display_frame, 0.7, mask_colored, 0.3, 0)
                
                # Dibujar contornos de la mano
                contours, _ = cv2.findContours(hand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    cv2.drawContours(display_frame, [largest_contour], -1, (0, 255, 0), 2)
                    
                    # Dibujar rectángulo delimitador
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Panel de información más pequeño para no tapar la vista
        panel_height = int(height * 0.15)  # Solo 15% de la altura
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
        
        # Escalar texto más pequeño
        font_scale = min(0.6, width / 1200.0)  # Texto más pequeño
        font_thickness = max(1, int(font_scale * 1.5))
        
        # Información compacta y pequeña
        small_font = font_scale * 0.5
        small_thickness = max(1, int(font_thickness * 0.7))
        
        # Solo información esencial
        y_pos = 20
        cv2.putText(display_frame, f"LESCO - {self.current_mode.upper()}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, small_font, (0, 255, 255), small_thickness)
        
        # Información del entrenamiento (solo si está entrenando)
        if self.current_mode == "training" and self.gesture_trainer.is_training:
            y_pos += 20
            cv2.putText(display_frame, f"Entrenando: {self.gesture_trainer.current_gesture_name}", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, small_font, (255, 255, 0), small_thickness)
            y_pos += 15
            cv2.putText(display_frame, f"Muestras: {len(self.gesture_trainer.training_samples)}/{self.gesture_trainer.target_samples}", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, small_font, (255, 255, 0), small_thickness)
        
        # Gesto detectado (solo si hay uno)
        if self.last_recognized_gesture:
            y_pos += 20
            cv2.putText(display_frame, f"Gesto: {self.last_recognized_gesture}", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, small_font, (0, 255, 0), small_thickness)
        
        # Texto reconocido (solo si hay texto)
        if self.recognized_text:
            y_pos += 20
            # Truncar texto largo
            display_text = self.recognized_text[:30] + "..." if len(self.recognized_text) > 30 else self.recognized_text
            cv2.putText(display_frame, f"Texto: {display_text}", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, small_font, (255, 255, 0), small_thickness)
        
        # Controles mínimos
        y_pos += 20
        cv2.putText(display_frame, "D-Debug | T-Entrenar | R-Reconocer | N-Nuevo | F-Pantalla | ESC-Salir", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, small_font * 0.8, (150, 150, 150), small_thickness)
        
        return display_frame

    def _toggle_fullscreen(self):
        """Alterna entre pantalla completa y ventana normal"""
        current_prop = cv2.getWindowProperty("Sistema LESCO - Versión Alternativa", cv2.WND_PROP_FULLSCREEN)
        if current_prop == cv2.WINDOW_FULLSCREEN:
            cv2.setWindowProperty("Sistema LESCO - Versión Alternativa", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            print("Modo ventana normal")
        else:
            cv2.setWindowProperty("Sistema LESCO - Versión Alternativa", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            print("Modo pantalla completa")

    def _switch_to_training_mode(self):
        """Cambia al modo entrenamiento"""
        if self.current_mode != "training":
            self.current_mode = "training"
            print("Modo: Entrenamiento")
            print("Presiona N para entrenar un nuevo gesto")

    def _switch_to_recognition_mode(self):
        """Cambia al modo reconocimiento"""
        if self.current_mode != "recognition":
            self.current_mode = "recognition"
            print("Modo: Reconocimiento")

    def _start_new_gesture_training(self):
        """Inicia el entrenamiento de un nuevo gesto"""
        if self.current_mode != "training":
            print("Debes estar en modo entrenamiento (T) para entrenar gestos")
            return
        
        print("\n=== ENTRENAMIENTO DE NUEVO GESTO ===")
        gesture_name = input("Ingresa el nombre del gesto: ").strip()
        
        if not gesture_name:
            print("Nombre de gesto no válido")
            return
        
        if gesture_name in self.gesture_recognizer.get_trained_gestures():
            print(f"El gesto '{gesture_name}' ya existe. ¿Deseas sobrescribirlo? (s/n): ", end="")
            response = input().strip().lower()
            if response != 's':
                print("Entrenamiento cancelado")
                return
        
        self.gesture_trainer.start_training(gesture_name)

    def _add_to_text(self, gesture):
        """Añade un gesto al texto reconocido"""
        if gesture:
            if self.recognized_text:
                self.recognized_text += " " + gesture
            else:
                self.recognized_text = gesture
            
            self.text_history.append({
                'gesture': gesture,
                'timestamp': time.time()
            })
            
            if len(self.text_history) > 20:
                self.text_history = self.text_history[-20:]

    def _clear_text(self):
        """Limpia el texto reconocido"""
        self.recognized_text = ""
        print("Texto limpiado")

    def _show_help(self):
        """Muestra la ayuda del sistema"""
        print("\n=== AYUDA DEL SISTEMA LESCO ===")
        print("Versión Alternativa con Entrenamiento y Pantalla Completa")
        print("\nControles:")
        print("  ESC - Salir del programa")
        print("  T - Modo entrenamiento")
        print("  R - Modo reconocimiento")
        print("  N - Entrenar nuevo gesto (en modo entrenamiento)")
        print("  F - Alternar pantalla completa/ventana normal")
        print("  SPACE - Limpiar texto reconocido")
        print("  D - Activar/desactivar modo debug (muestra detección de manos)")
        print("  H - Mostrar esta ayuda")
        print("\nGestos disponibles:")
        print("  A - Gesto básico predefinido")
        print("  [Gestos entrenados] - Gestos que hayas entrenado")
        print("\nEntrenamiento:")
        print("  1. Presiona T para modo entrenamiento")
        print("  2. Presiona N para entrenar nuevo gesto")
        print("  3. Ingresa el nombre del gesto")
        print("  4. Realiza el gesto 5 veces")
        print("  5. El sistema entrena automáticamente")
        print("\nDetección de manos:")
        print("  - Presiona D para ver la detección de manos en tiempo real")
        print("  - La máscara roja muestra la región detectada como piel")
        print("  - El contorno verde muestra la forma de la mano")
        print("  - El rectángulo azul muestra el área delimitadora")
        print("=" * 60)

    def cleanup(self):
        print("Limpiando recursos...")
        if hasattr(self, 'cap'): 
            self.cap.release()
        cv2.destroyAllWindows()
        print("Recursos liberados correctamente")

def main():
    print("=== Sistema de Reconocimiento LESCO ===")
    print("Versión Alternativa con Entrenamiento y Pantalla Completa\n")
    print("Esta versión funciona sin MediaPipe y incluye:")
    print("- Detección mejorada de manos usando color de piel")
    print("- Reconocimiento de gestos con entrenamiento personalizado")
    print("- Almacenamiento persistente de gestos entrenados")
    print("- Interfaz de pantalla completa optimizada")
    print("- Visualización de detección en tiempo real (modo debug)")
    print("\nCaracterísticas principales:")
    print("- Pantalla completa automática al iniciar")
    print("- Detección robusta de manos en diferentes condiciones de luz")
    print("- Entrenamiento guiado de gestos personalizados")
    print("- Visualización de landmarks y contornos en modo debug")
    print("\nPresiona H en el programa para ver la ayuda completa")
    print("Presiona D para activar el modo debug y ver la detección de manos\n")
    
    system = LESCORecognitionSystem()
    system.run()
    print("Sistema finalizado")

if __name__ == "__main__":
    main()
