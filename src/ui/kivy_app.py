"""
Kivy GUI para entrenamiento y visualización en tiempo real.

Características:
- Video en vivo con landmarks y modo debug opcional.
- Botones: Iniciar/Pausar, Capturar Frame, Guardar Gesto.
- Captura manual por barra espaciadora (mínimo 5 frames por gesto).
- Ventana emergente para nombrar el gesto al guardar.
- Lista de gestos disponibles y reconocimiento en tiempo real.

Optimizado para bajo consumo: resolución reducida, skip frames, matching ligero.
Compatible con Android via Kivy.
"""

from typing import List
import os
import cv2
import numpy as np
from collections import deque

from kivy.app import App
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput
from kivy.core.window import Window

try:
    from ..config.settings import TARGET_WIDTH, TARGET_HEIGHT, CAPTURE_BUFFER_SIZE, SEND_EVERY_N_FRAMES, TARGET_FPS, MAX_PROCESS_EVERY_N_FRAMES
except ImportError:
    from src.config.settings import TARGET_WIDTH, TARGET_HEIGHT, CAPTURE_BUFFER_SIZE

try:
    from ..hand_tracking.visualizer import HandVisualizer
    from ..gesture_recognition.gesture_repository import GestureRepository
    from ..gesture_recognition.realtime_matcher import RealTimeMatcher
except ImportError:
    from src.hand_tracking.visualizer import HandVisualizer
    from src.gesture_recognition.gesture_repository import GestureRepository
    from src.gesture_recognition.realtime_matcher import RealTimeMatcher

import mediapipe as mp


class GestureUIRoot(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation="vertical", **kwargs)
        self.image = Image(size_hint=(1, 0.75))
        controls = BoxLayout(size_hint=(1, 0.15))
        info_bar = BoxLayout(size_hint=(1, 0.10))

        self.btn_start = Button(text="Iniciar Manos")
        self.btn_gestures = Button(text="Iniciar Gestos")
        self.btn_trainer = Button(text="Modo Entrenador")
        self.btn_pause = Button(text="Pausar")
        self.btn_capture = Button(text="Capturar (SPACE)")
        self.btn_save = Button(text="Guardar Gesto")
        self.status = Label(text="Listo", size_hint=(0.5, 1))
        self.left_gesture = Label(text="Izq: -", size_hint=(0.25, 1))
        self.right_gesture = Label(text="Der: -", size_hint=(0.25, 1))
        self.current_gesture_lbl = Label(text="Actual: -", size_hint=(0.5, 1))

        controls.add_widget(self.btn_start)
        controls.add_widget(self.btn_gestures)
        controls.add_widget(self.btn_trainer)
        controls.add_widget(self.btn_pause)
        controls.add_widget(self.btn_capture)
        controls.add_widget(self.btn_save)
        controls.add_widget(self.status)

        # Barra de información (gestos por mano)
        info_bar.add_widget(self.left_gesture)
        info_bar.add_widget(self.right_gesture)
        info_bar.add_widget(self.current_gesture_lbl)

        self.add_widget(self.image)
        self.add_widget(controls)
        self.add_widget(info_bar)


class GestureKivyApp(App):
    def build(self):
        self.title = "Entrenamiento y Reconocimiento de Gestos"
        self.ui = GestureUIRoot()
        self.ui.btn_start.bind(on_release=self.on_start_camera)
        self.ui.btn_pause.bind(on_release=self.on_pause_camera)
        self.ui.btn_capture.bind(on_release=self.on_capture)
        self.ui.btn_save.bind(on_release=self.on_save)
        self.ui.btn_gestures.bind(on_release=self.on_start_gestures)
        self.ui.btn_trainer.bind(on_release=self.on_trainer_mode)
        Window.bind(on_key_down=self.on_key_down)

        # Estado
        self.user_id = "default"
        self.gesture_repo = GestureRepository()
        self.visualizer = HandVisualizer()
        self.templates = self.gesture_repo.load_templates(user_id=self.user_id)
        self.matcher = RealTimeMatcher(self.templates)
        # Buffers por mano (mínimo 5 frames para dinámicos)
        self.sequence_buffers = {
            'Left': deque(maxlen=45),
            'Right': deque(maxlen=45),
        }
        self.manual_capture_buffer: List[List[dict]] = []  # secuencia de landmarks normalizados
        self.is_running = False
        self.detect_gestures = False
        self.trainer_mode = False
        self.frame_count = 0
        self.process_every_n = SEND_EVERY_N_FRAMES
        self.last_fps_check_ts = 0.0
        self.frames_since_check = 0
        self.status_text = "Sin detección"
        self.display_history = deque(maxlen=10)

        # MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.cap = None
        self.event = None
        return self.ui

    def on_start_camera(self, *_):
        if self.is_running:
            return
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, CAPTURE_BUFFER_SIZE)
        if not self.cap.isOpened():
            self.set_status("No se pudo abrir la cámara")
            return
        self.is_running = True
        self.set_status("Cámara activa")
        self.event = Clock.schedule_interval(self.update, 0)

    def on_pause_camera(self, *_):
        self.is_running = False
        if self.event:
            self.event.cancel()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.set_status("Pausado")

    def on_start_gestures(self, *_):
        # Toggle detección de gestos
        self.detect_gestures = not self.detect_gestures
        if self.detect_gestures:
            self.ui.btn_gestures.text = "Pausar Gestos"
            self.set_status("Detección de gestos: ON")
        else:
            self.ui.btn_gestures.text = "Iniciar Gestos"
            self.set_status("Detección de gestos: OFF")

    def on_trainer_mode(self, *_):
        self.trainer_mode = not self.trainer_mode
        self.set_status(f"Modo entrenador: {'ON' if self.trainer_mode else 'OFF'}")

    def on_key_down(self, _window, key, _scancode, _codepoint, _mod):
        # Tecla SPACE
        if key == 32:
            self.on_capture()
        return True

    def on_capture(self, *_):
        # Captura manual: requiere que el último frame haya tenido landmarks válidos
        if hasattr(self, "last_norm") and self.last_norm:
            self.manual_capture_buffer.append(self.last_norm)
            self.set_status(f"Capturas: {len(self.manual_capture_buffer)}")

    def on_save(self, *_):
        if len(self.manual_capture_buffer) < 5:
            self.set_status("Captura mínima: 5 frames")
            return
        content = BoxLayout(orientation="vertical")
        ti = TextInput(hint_text="Nombre del gesto", multiline=False)
        btn_ok = Button(text="Guardar")
        content.add_widget(ti)
        content.add_widget(btn_ok)
        popup = Popup(title="Guardar Gesto", content=content, size_hint=(0.6, 0.3))

        def do_save(_):
            name = ti.text.strip()
            if not name:
                return
            ok = self.gesture_repo.save_dynamic(name, self.manual_capture_buffer, fps=30, user_id=self.user_id)
            if ok:
                self.templates = self.gesture_repo.load_templates(user_id=self.user_id)
                self.matcher = RealTimeMatcher(self.templates)
                self.manual_capture_buffer.clear()
                self.set_status(f"Gesto '{name}' guardado")
            else:
                self.set_status("Error al guardar gesto")
            popup.dismiss()

        btn_ok.bind(on_release=do_save)
        popup.open()

    def update(self, _dt):
        if not (self.cap and self.is_running):
            return
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
        # Espejar horizontalmente
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesamiento selectivo de frames
        do_process = (self.frame_count % self.process_every_n == 0)
        results = self.hands.process(rgb) if do_process else None
        self.frame_count += 1
        self.frames_since_check += 1

        recognized_left = None
        recognized_right = None
        recognized_current = None
        self.last_norm = None
        if results and results.multi_hand_landmarks:
            hands_list = results.multi_hand_landmarks
            handedness_list = results.multi_handedness if hasattr(results, 'multi_handedness') else [None] * len(hands_list)

            # Dibujar landmarks de todas las manos (imagen espejada => mirrored=True)
            self.visualizer.draw_landmarks(frame, [h.landmark for h in hands_list], debug=True, show_indices=False, show_coords=False, mirrored=True)

            for idx, hand in enumerate(hands_list):
                lm = hand.landmark
                norm = GestureRepository.normalize_landmarks(lm)
                self.last_norm = norm  # última mano válida

                # Determinar lado de la mano
                side = None
                if handedness_list[idx] is not None:
                    side = handedness_list[idx].classification[0].label  # 'Left' o 'Right'

                if self.detect_gestures and do_process:
                    # Actualizar buffer por mano
                    if side in self.sequence_buffers:
                        self.sequence_buffers[side].append(norm)

                    label = None
                    if self.matcher:
                        ls, cs = self.matcher.match_static(norm)
                        if ls and cs >= 0.9:
                            label = ls
                    # Matching dinámico por mano con mínimo 5 frames
                    hand_buf = self.sequence_buffers.get(side) if side else None
                    if not label and hand_buf is not None and len(hand_buf) >= 5 and self.matcher:
                        window = list(hand_buf)[-30:]
                        ld, cd = self.matcher.match_dynamic(window)
                        if ld and cd >= 0.6:
                            label = ld

                    if side == 'Left':
                        recognized_left = label
                    elif side == 'Right':
                        recognized_right = label
                    # Gesto actual prioritiza mano derecha, si no, izquierda
                    if label:
                        if side == 'Right':
                            recognized_current = label
                        elif side == 'Left' and recognized_current is None:
                            recognized_current = label

        # Mostrar etiquetas por mano
        if recognized_left:
            cv2.putText(frame, f"Izq: {recognized_left}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if recognized_right:
            cv2.putText(frame, f"Der: {recognized_right}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        # Estado textual
        if not (recognized_left or recognized_right):
            self.status_text = "Sin detección"
        else:
            self.status_text = "Estable"
        # Smoothing simple del gesto actual
        self.display_history.append(recognized_current or "")
        vals = [v for v in self.display_history if v]
        smoothed = max(set(vals), key=vals.count) if vals else None
        recognized_current = smoothed

        # Convertir a textura Kivy
        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt="bgr")
        texture.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")
        self.ui.image.texture = texture

        # Actualizar barra de información de gestos
        self.ui.left_gesture.text = f"Izq: {recognized_left or '-'}"
        self.ui.right_gesture.text = f"Der: {recognized_right or '-'}"
        self.ui.current_gesture_lbl.text = f"Actual: {recognized_current or '-'} | Status: {self.status_text}"

        # Ajuste adaptativo de frecuencia de procesamiento (simple control FPS)
        now = Clock.get_boottime()
        if self.last_fps_check_ts == 0:
            self.last_fps_check_ts = now
        elapsed = now - self.last_fps_check_ts
        if elapsed >= 1.0:
            fps = self.frames_since_check / elapsed
            if fps < TARGET_FPS and self.process_every_n < MAX_PROCESS_EVERY_N_FRAMES:
                self.process_every_n += 1
            elif fps > TARGET_FPS + 6 and self.process_every_n > 1:
                self.process_every_n -= 1
            self.frames_since_check = 0
            self.last_fps_check_ts = now

    def set_status(self, text: str):
        self.ui.status.text = text

    def on_stop(self):
        self.on_pause_camera()
        if hasattr(self, "hands") and self.hands:
            self.hands.close()


def run_kivy():
    GestureKivyApp().run()


