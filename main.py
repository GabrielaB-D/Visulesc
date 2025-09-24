
import cv2
import time
import sys
import os
import numpy as np
import math
from collections import deque
from typing import List, Tuple

# Popup simple para ingresar nombre de gesto en escritorio
try:
    import tkinter as tk
    from tkinter import simpledialog
except Exception:
    tk = None

# Añadir el directorio src al path para imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils import FPSCounter
from config.settings import *
from ui.main_window import MainWindow
from gesture_recognition.gesture_classifier import GestureClassifier
from data.user_manager import UserManager
from gesture_recognition.gesture_repository import GestureRepository
from gesture_recognition.realtime_matcher import RealTimeMatcher
from gesture_recognition.feature_extractor import extract_features_for_both_hands

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
        self.gesture_repo = GestureRepository()
        self.matcher = None
        
        # Estado del sistema
        self.last_recognized_gesture = None
        self.gesture_stability_counter = 0
        self.GESTURE_STABILITY_THRESHOLD = 5
        self.is_running = False
        self.debug_mode = False
        self.current_user = "default"
        # Buffer para gestos dinámicos (secuencia de landmarks normalizados)
        self.sequence_buffer = deque(maxlen=45)  # ~1.5s @30fps
        # Flags de control de UI
        self.detect_gestures = False
        # Botones de UI (OpenCV overlay)
        self.ui_buttons: List[Tuple[str, Tuple[int,int,int,int]]] = []
        # Captura dinámica manual (modo entrenador simple en main.py)
        self.dynamic_capture_active = False
        self.dynamic_capture_buffer: List[List[dict]] = []

        # MediaPipe Hands setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        # Último output por mano: [{'handedness': 'Left'/'Right', 'landmarks': [...], 'recognized': str|None}]
        self.last_hands_output = []
        self.current_display_gesture = None
        # Suavizado de gesto actual
        self.display_history = deque(maxlen=10)
        self.display_smoothed = None
        self.status_text = "Sin detección"
        self.capture_missing_counter = { 'Left': 0, 'Right': 0 }
        self.CAPTURE_MISSING_LIMIT = 5
        # Muestreo estadístico (plantilla estática con mean/std)
        self.sampling_active = False
        self.sampling_label = None
        self.sampling_target_frames = 30
        self.sampling_feat_list = []
        self.sampling_left_missing = 0
        self.sampling_right_missing = 0
        
        # Cargar plantillas existentes para matching directo
        self._reload_templates()
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
        # Configurar callbacks de mouse para botones superpuestos
        cv2.setMouseCallback(self.main_window.window_name, self._on_mouse)
        # Definir botones UI
        self._setup_ui_buttons()
        
        self.is_running = True
        return True

    def run(self):
        if not self.start(): return
        try:
            while self.is_running and self.cap.isOpened() and self.main_window.is_active():
                ret, frame = self.cap.read()
                if not ret: break
                
                frame_resized = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
                # Espejar horizontalmente para experiencia natural
                frame_resized = cv2.flip(frame_resized, 1)
                rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                # Optimización: evitar copias en MediaPipe
                rgb.flags.writeable = False
                
                # Detectar landmarks
                results = self.hands.process(rgb)
                rgb.flags.writeable = True
                landmarks = None
                recognized_gesture = None
                hands_output = []
                
                # Procesar resultados
                if results.multi_hand_landmarks:
                    # Dibujar y preparar salida por mano
                    multi_hand_landmarks = results.multi_hand_landmarks
                    multi_handedness = getattr(results, 'multi_handedness', [None] * len(multi_hand_landmarks))
                    
                    for idx, hand_lms in enumerate(multi_hand_landmarks):
                        # Handedness label
                        handed_label = None
                        if multi_handedness[idx] is not None:
                            handed_label = multi_handedness[idx].classification[0].label  # 'Left'/'Right'
                        
                        # Lista de landmarks para salida
                        lm_list = hand_lms.landmark
                        # Dibujo de landmarks en el frame (una vez por mano)
                        self.mp_drawing.draw_landmarks(
                            frame_resized, hand_lms, self.mp_hands.HAND_CONNECTIONS
                        )
                        
                        # Reconocimiento simple/matcher estático por mano si disponible
                        recognized_label = None
                        if self.matcher and self.detect_gestures:
                            norm = self.gesture_repo.normalize_landmarks(lm_list)
                            label_s, conf_s = self.matcher.match_static(norm)
                            if label_s and conf_s >= 0.9:
                                recognized_label = label_s
                        
                        # Overlay de texto para cada mano
                        # Usar el landmark 0 (wrist) como ancla
                        h, w = frame_resized.shape[:2]
                        anchor_x = int(lm_list[0].x * w)
                        anchor_y = int(lm_list[0].y * h)
                        info_text = f"{handed_label or '-'}"
                        if recognized_label:
                            info_text += f" | {recognized_label}"
                        cv2.putText(frame_resized, info_text, (anchor_x + 5, max(15, anchor_y - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        
                        hands_output.append({
                            'handedness': handed_label,
                            'landmarks': lm_list,
                            'recognized': recognized_label
                        })
                        # Actualizar gesto actual (prioriza derecha)
                        if recognized_label:
                            if handed_label == 'Right':
                                self.current_display_gesture = recognized_label
                            elif handed_label == 'Left' and self.current_display_gesture is None:
                                self.current_display_gesture = recognized_label
                    
                    # Mantener compatibilidad: usar primera mano para vías existentes
                    if len(multi_hand_landmarks) > 0:
                        landmarks = multi_hand_landmarks[0].landmark
                    
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
                    
                    # Matching directo con plantillas (estático)
                    if landmarks and self.matcher:
                        norm = self.gesture_repo.normalize_landmarks(landmarks)
                        label_s, conf_s = self.matcher.match_static(norm)
                        if label_s and conf_s >= 0.9:
                            recognized_gesture = label_s
                        # Actualizar buffer para gestos dinámicos
                        self.sequence_buffer.append(norm)
                        # Intento de matching dinámico con una ventana reciente
                        if len(self.sequence_buffer) >= 12:  # mínima duración
                            window = list(self.sequence_buffer)[-30:]  # último ~1s
                            label_d, conf_d = self.matcher.match_dynamic(window)
                            if label_d and conf_d >= 0.6:
                                recognized_gesture = label_d

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
                        # Limpiar buffer tras reconocimiento dinámico para evitar loops
                        self.sequence_buffer.clear()

                # Guardar salida por mano para acceso externo/debug
                self.last_hands_output = hands_output

                # Suavizado y estado
                # Construir features con ambas manos para posible matching estadístico
                left_lm = None; right_lm = None
                for item in hands_output:
                    if item['handedness'] == 'Left':
                        left_lm = item['landmarks']
                    elif item['handedness'] == 'Right':
                        right_lm = item['landmarks']
                feat_vec, left_present, right_present = extract_features_for_both_hands(left_lm, right_lm)
                # Missing counters
                self.capture_missing_counter['Left'] = 0 if left_present else self.capture_missing_counter['Left'] + 1
                self.capture_missing_counter['Right'] = 0 if right_present else self.capture_missing_counter['Right'] + 1

                # Smoothing del gesto actual
                current_now = None
                # 1) Intentar plantillas estáticas clásicas
                if self.matcher and self.detect_gestures and len(hands_output) > 0:
                    # usar primera mano para estático clásico
                    lm_first = hands_output[0]['landmarks']
                    norm_first = self.gesture_repo.normalize_landmarks(lm_first)
                    label_s, conf_s = self.matcher.match_static(norm_first)
                    if label_s and conf_s >= 0.9:
                        current_now = label_s
                # 2) Intentar plantilla estadística con Mahalanobis normalizado
                if self.matcher and self.detect_gestures and current_now is None and feat_vec.size > 0:
                    label_stat, conf_stat = self.matcher.match_statistical(feat_vec)
                    if conf_stat >= 0.6:
                        current_now = label_stat

                self.display_history.append(current_now or "")
                # media móvil por moda simple (mayoría de últimos N no vacíos)
                values = [v for v in self.display_history if v]
                self.display_smoothed = max(set(values), key=values.count) if values else None
                self.current_display_gesture = self.display_smoothed

                # Estado textual
                if not left_present or not right_present:
                    missing = 'izquierda' if not left_present else 'derecha'
                    self.status_text = f"Falta mano {missing}"
                elif self.current_display_gesture:
                    self.status_text = "Estable"
                else:
                    self.status_text = "Sin detección"
                
                # Actualizar ventana principal (ya con overlays dibujados); pasar landmarks de compatibilidad
                window_frame = self.main_window.update_frame(frame_resized, landmarks, recognized_gesture)

                # Mostrar gesto/letra actual en overlay destacado
                if self.current_display_gesture:
                    cv2.rectangle(window_frame, (10, 90), (210, 130), (30, 30, 30), -1)
                    cv2.putText(window_frame, f"Actual: {self.current_display_gesture}", (20, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    # Área vacía opcional sin texto (se puede omitir para ahorrar draw)
                    pass
                # Mostrar estado secundario
                cv2.putText(window_frame, f"Status: {self.status_text}", (10, 155),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

                # Muestreo estadístico en curso: agregar features y dibujar progreso/mensajes
                if self.sampling_active:
                    # Mensaje informativo
                    cv2.rectangle(window_frame, (10, 170), (540, 235), (20, 20, 20), -1)
                    msg = f"Grabando gesto '{self.sampling_label}' — mantén la pose ~1s. Nota: guardaremos media y variabilidad."
                    cv2.putText(window_frame, msg[:65], (15, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                    cv2.putText(window_frame, msg[65:130], (15, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                    # Progreso
                    done = len(self.sampling_feat_list)
                    total = self.sampling_target_frames
                    cv2.rectangle(window_frame, (15, 220), (515, 230), (60, 60, 60), 1)
                    if total > 0:
                        w = int(500 * min(1.0, done / total))
                        cv2.rectangle(window_frame, (15, 220), (15 + w, 230), (0, 200, 255), -1)
                    cv2.putText(window_frame, f"{done}/{total}", (520, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                    # Inestabilidad / manos faltantes
                    if self.sampling_left_missing > self.CAPTURE_MISSING_LIMIT or self.sampling_right_missing > self.CAPTURE_MISSING_LIMIT:
                        cv2.putText(window_frame, "Falta mano izquierda/derecha — reintenta", (15, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                    # Recolectar features de este frame
                    if feat_vec.size > 0:
                        self.sampling_feat_list.append(feat_vec)
                        # Abort por manos faltantes persistentes
                        if not left_present:
                            self.sampling_left_missing += 1
                        else:
                            self.sampling_left_missing = 0
                        if not right_present:
                            self.sampling_right_missing += 1
                        else:
                            self.sampling_right_missing = 0

                    # Finalizar si alcanza el objetivo
                    if len(self.sampling_feat_list) >= self.sampling_target_frames:
                        self._finalize_sampling()

                # Si está activo el modo de captura dinámica, agregar frame actual (primera mano si existe)
                if self.dynamic_capture_active and self.last_hands_output:
                    lm_list = self.last_hands_output[0].get('landmarks')
                    if lm_list:
                        norm = self.gesture_repo.normalize_landmarks(lm_list)
                        self.dynamic_capture_buffer.append(norm)
                        # Mostrar conteo en overlay
                        cv2.putText(window_frame, f"Dyn frames: {len(self.dynamic_capture_buffer)}", (10, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

                # Dibujar barra de botones sobre la ventana
                self._draw_buttons(window_frame)
                
                # Mostrar frame
                cv2.imshow(self.main_window.window_name, window_frame)
                
                # Manejar teclado
                key = cv2.waitKey(1) & 0xFF
                if not self.main_window.handle_keyboard(key):
                    # Manejar teclas específicas del sistema principal
                    if key == ord('n') or key == ord('N'):
                        self._start_new_gesture_training()
                    elif key == ord('g') or key == ord('G'):
                        # Guardar snapshot como gesto estático rápido
                        if landmarks:
                            name = input("Nombre del gesto estático: ").strip()
                            if name:
                                norm = self.gesture_repo.normalize_landmarks(landmarks)
                                if self.gesture_repo.save_static(name, norm, user_id=self.current_user):
                                    print(f"Gesto '{name}' guardado. Recargando plantillas...")
                                    self._reload_templates()
                    elif key == ord('j') or key == ord('J'):
                        # Toggle detección de gestos desde teclado
                        self.detect_gestures = not self.detect_gestures
                        print(f"Detección de gestos: {'ON' if self.detect_gestures else 'OFF'}")
                
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

    def _reload_templates(self):
        """Carga/recarga plantillas del repositorio para el matcher en tiempo real."""
        templates = self.gesture_repo.load_templates(user_id=self.current_user)
        self.matcher = RealTimeMatcher(templates)

    # ===================== UI Overlay (OpenCV) =====================
    def _setup_ui_buttons(self):
        # Define posiciones y tamaños de los botones en la barra superior
        x, y, h = 10, 10, 30
        spacing = 8
        labels = [
            "Manos",            # iniciar manos (ya activo)
            "Gestos ON/OFF",   # toggle
            "Entrenador",      # modo entrenamiento
            "Capturar Dyn",    # iniciar/pausar captura dinámica
            "Guardar Dyn",     # guardar gesto dinámico
            "Nuevo Estático",  # captura estática
            "Muestrear Stat",  # iniciar muestreo estadístico
            "Pausar"           # pausar/detener
        ]
        self.ui_buttons = []
        for label in labels:
            w = 140
            self.ui_buttons.append((label, (x, y, w, h)))
            x += w + spacing

    def _draw_buttons(self, frame):
        for label, (x, y, w, h) in self.ui_buttons:
            # Fondo
            cv2.rectangle(frame, (x, y), (x + w, y + h), (40, 40, 40), -1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)
            # Texto
            txt = label
            if label == "Gestos ON/OFF":
                txt = f"Gestos: {'ON' if self.detect_gestures else 'OFF'}"
            elif label == "Capturar Dyn":
                txt = f"Dyn: {'REC' if self.dynamic_capture_active else 'PAUSA'}"
            cv2.putText(frame, txt, (x + 6, y + int(h*0.7)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _on_mouse(self, event, mx, my, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for label, (x, y, w, h) in self.ui_buttons:
            if x <= mx <= x + w and y <= my <= y + h:
                self._handle_button(label)
                break

    def _handle_button(self, label: str):
        if label == "Manos":
            print("Cámara ya activa")
        elif label == "Gestos ON/OFF":
            self.detect_gestures = not self.detect_gestures
            print(f"Detección de gestos: {'ON' if self.detect_gestures else 'OFF'}")
        elif label == "Entrenador":
            # Cambiar a modo entrenamiento en la ventana
            self.main_window._switch_to_training_mode()
        elif label == "Capturar Dyn":
            # Alternar captura dinámica manual
            self.dynamic_capture_active = not self.dynamic_capture_active
            if self.dynamic_capture_active:
                self.dynamic_capture_buffer.clear()
                print("Captura dinámica: INICIADA")
            else:
                print("Captura dinámica: PAUSADA")
        elif label == "Guardar Dyn":
            # Guardar gesto dinámico si hay suficientes frames
            if len(self.dynamic_capture_buffer) < 5:
                print("Captura dinámica insuficiente (mínimo 5 frames)")
            else:
                name = self._prompt_gesture_name()
                if name:
                    if self.gesture_repo.save_dynamic(name, self.dynamic_capture_buffer, fps=30, user_id=self.current_user):
                        print(f"Gesto dinámico '{name}' guardado. Recargando plantillas...")
                        self._reload_templates()
                    self.dynamic_capture_buffer.clear()
        elif label == "Nuevo Estático":
            # Capturar primer mano y guardar como plantilla estática
            if self.last_hands_output:
                lm_list = self.last_hands_output[0].get('landmarks')
                if lm_list:
                    name = self._prompt_gesture_name()
                    if name:
                        norm = self.gesture_repo.normalize_landmarks(lm_list)
                        if self.gesture_repo.save_static(name, norm, user_id=self.current_user):
                            print(f"Gesto '{name}' guardado. Recargando plantillas...")
                            self._reload_templates()
        elif label == "Muestrear Stat":
            # Iniciar muestreo estadístico
            name = self._prompt_gesture_name()
            if name:
                self.start_gesture_capture(name)
        elif label == "Pausar":
            # Simular ESC
            self.is_running = False

    def _prompt_gesture_name(self) -> str:
        # Preferir popup Tkinter si disponible, si no usar consola
        try:
            if tk is not None:
                root = tk.Tk()
                root.withdraw()
                value = simpledialog.askstring("Nuevo gesto estático", "Nombre del gesto:")
                root.destroy()
                if value:
                    return value.strip()
        except Exception:
            pass
        try:
            return input("Nombre del gesto estático: ").strip()
        except Exception:
            return ""

    # ========= API expuesta =========
    def start_gesture_capture(self, label: str, target_frames: int = 30):
        self.sampling_active = True
        self.sampling_label = label
        self.sampling_target_frames = target_frames
        self.sampling_feat_list = []
        self.sampling_left_missing = 0
        self.sampling_right_missing = 0
        print(f"Grabando gesto '{label}' — mantén la pose durante ~1 segundo. Atención: no es posible 100% estabilidad; guardaremos media y variabilidad.")

    def abort_gesture_capture(self):
        if self.sampling_active:
            self.sampling_active = False
            self.sampling_feat_list = []
            print("Muestreo abortado")

    def save_gesture(self, metadata: dict = None):
        # Forzar finalización si está activo
        if self.sampling_active:
            self._finalize_sampling()
        else:
            print("No hay muestreo activo para guardar")

    def get_current_detection(self) -> dict:
        return {
            'current_gesture': self.current_display_gesture,
            'status': self.status_text,
            'hands': self.last_hands_output,
        }

    def _finalize_sampling(self):
        # Abort si manos faltantes persistentes
        if self.sampling_left_missing > self.CAPTURE_MISSING_LIMIT or self.sampling_right_missing > self.CAPTURE_MISSING_LIMIT:
            print("Falta mano izquierda/derecha — reposición requerida para este gesto.")
            self.abort_gesture_capture()
            return
        if not self.sampling_feat_list:
            print("No hay muestras para guardar")
            self.abort_gesture_capture()
            return
        feats = np.vstack(self.sampling_feat_list).astype(np.float32)
        mean = feats.mean(axis=0)
        std = feats.std(axis=0) + 1e-6
        # Variability score: promedio de std/|mean| en características con |mean|>epsilon
        denom = np.where(np.abs(mean) < 1e-6, 1e-6, np.abs(mean))
        variability = float(np.mean(std / denom))
        # Confianza promedio (heurística): 1 - (media |z| / 3)
        z = np.abs((feats - mean) / std)
        avg_z = float(np.mean(z))
        avg_conf = max(0.0, 1.0 - (avg_z / 3.0))
        if avg_conf < 0.6:
            print("Gesto inestable: intenta reducir movimiento o sostener la pose por más tiempo.")
            self.abort_gesture_capture()
            return
        ok = self.gesture_repo.save_statistical(
            self.sampling_label,
            mean_features=mean.tolist(),
            std_features=std.tolist(),
            num_frames=len(self.sampling_feat_list),
            variability_score=variability,
            avg_confidence=avg_conf,
            user_id=self.current_user,
        )
        if ok:
            print(f"Gesto '{self.sampling_label}' guardado (estadístico). Variabilidad: {variability:.3f}, Confianza: {avg_conf:.2f}")
            self._reload_templates()
        else:
            print("Error al guardar gesto estadístico")
        self.sampling_active = False
        self.sampling_feat_list = []

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
