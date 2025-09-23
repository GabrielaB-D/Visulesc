"""
Ventana principal del sistema LESCO con interfaz completa.
Integra reconocimiento, entrenamiento y gestión de gestos.
"""

import cv2
import numpy as np
from typing import Dict, Optional, Callable, List
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.ui.training_ui import TrainingUI
from src.ui.text_display import TextDisplay
from src.gesture_recognition.gesture_classifier import GestureClassifier
from src.data.user_manager import UserManager


class MainWindow:
    """Ventana principal del sistema LESCO."""
    
    def __init__(self, window_name: str = "Sistema LESCO"):
        """
        Inicializa la ventana principal.
        
        Args:
            window_name: Nombre de la ventana
        """
        self.window_name = window_name
        self.is_running = False
        self.current_mode = "recognition"  # recognition, training, management
        
        # Componentes del sistema
        self.training_ui = TrainingUI()
        self.text_display = TextDisplay()
        self.classifier = GestureClassifier()
        self.user_manager = UserManager()
        
        # Estado del sistema
        self.current_user = "default"
        self.recognized_gestures = []
        self.debug_mode = False
        
        # Configurar callbacks
        self.training_ui.on_training_completed = self._on_training_completed
        self.training_ui.on_training_cancelled = self._on_training_cancelled
        
        # Información de la ventana
        self.window_width = 1200
        self.window_height = 800
        self.camera_width = 640
        self.camera_height = 480
    
    def start(self) -> bool:
        """
        Inicia la ventana principal.
        
        Returns:
            True si se inició exitosamente
        """
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            self.is_running = True
            
            print("Ventana principal iniciada")
            print("Controles:")
            print("  R - Modo reconocimiento")
            print("  T - Modo entrenamiento")
            print("  M - Modo gestión")
            print("  D - Debug")
            print("  H - Ayuda")
            print("  ESC - Salir")
            
            return True
        except Exception as e:
            print(f"Error iniciando ventana: {e}")
            return False
    
    def update_frame(self, frame: np.ndarray, landmarks: List, recognized_gesture: Optional[str] = None) -> np.ndarray:
        """
        Actualiza el frame de la ventana principal.
        
        Args:
            frame: Frame de la cámara
            landmarks: Landmarks detectados
            recognized_gesture: Gesto reconocido (opcional)
            
        Returns:
            Frame actualizado
        """
        if not self.is_running:
            return frame
        
        # Crear frame de la ventana
        window_frame = self._create_window_image(frame, landmarks, recognized_gesture)
        
        return window_frame
    
    def _create_window_image(self, camera_frame: np.ndarray, landmarks: List, recognized_gesture: Optional[str]) -> np.ndarray:
        """
        Crea la imagen completa de la ventana.
        
        Args:
            camera_frame: Frame de la cámara
            landmarks: Landmarks detectados
            recognized_gesture: Gesto reconocido
            
        Returns:
            Imagen completa de la ventana
        """
        # Crear imagen base
        window_image = np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)
        
        # Redimensionar frame de la cámara
        camera_resized = cv2.resize(camera_frame, (self.camera_width, self.camera_height))
        
        # Colocar frame de la cámara en la ventana
        window_image[50:50+self.camera_height, 50:50+self.camera_width] = camera_resized
        
        # Dibujar landmarks si están disponibles
        if landmarks:
            self._draw_landmarks_on_frame(camera_resized, landmarks)
            window_image[50:50+self.camera_height, 50:50+self.camera_width] = camera_resized
        
        # Dibujar paneles según el modo
        if self.current_mode == "training":
            training_frame = self.training_ui.process_frame(camera_resized, landmarks)
            window_image[50:50+self.camera_height, 50:50+self.camera_width] = training_frame
        
        # Dibujar panel de información
        self._draw_info_panel(window_image)
        
        # Dibujar barra de estado
        self._draw_status_bar(window_image)
        
        # Dibujar indicador de modo
        self._draw_mode_indicator(window_image)
        
        return window_image
    
    def _draw_landmarks_on_frame(self, frame: np.ndarray, landmarks: List):
        """Dibuja los landmarks en el frame."""
        if not landmarks:
            return
        
        # Dibujar puntos de landmarks
        for landmark in landmarks:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
    
    def _draw_info_panel(self, window_image: np.ndarray):
        """Dibuja el panel de información."""
        panel_x = self.camera_width + 100
        panel_y = 50
        panel_width = 400
        panel_height = 500
        
        # Fondo del panel
        cv2.rectangle(window_image, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), (40, 40, 40), -1)
        cv2.rectangle(window_image, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), (100, 100, 100), 2)
        
        # Título del panel
        cv2.putText(window_image, "INFORMACION DEL SISTEMA", 
                   (panel_x + 10, panel_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Información del usuario
        y_offset = panel_y + 60
        cv2.putText(window_image, f"Usuario: {self.current_user}", 
                   (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Modo actual
        y_offset += 25
        mode_colors = {
            "recognition": (0, 255, 0),
            "training": (0, 255, 255),
            "management": (255, 0, 255)
        }
        mode_color = mode_colors.get(self.current_mode, (255, 255, 255))
        cv2.putText(window_image, f"Modo: {self.current_mode.upper()}", 
                   (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 1)
        
        # Gestos disponibles
        y_offset += 40
        cv2.putText(window_image, "Gestos Disponibles:", 
                   (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        available_gestures = self.classifier.get_available_gestures(self.current_user)
        y_offset += 25
        
        if available_gestures:
            for i, gesture in enumerate(available_gestures[:8]):  # Mostrar máximo 8 gestos
                cv2.putText(window_image, f"  • {gesture}", 
                           (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 255, 150), 1)
                y_offset += 20
        else:
            cv2.putText(window_image, "  Ninguno", 
                       (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # Información de debug
        if self.debug_mode:
            y_offset += 30
            cv2.putText(window_image, "DEBUG INFO:", 
                       (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_offset += 25
            cv2.putText(window_image, f"Landmarks: {len(landmarks) if landmarks else 0}", 
                       (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def _draw_status_bar(self, window_image: np.ndarray):
        """Dibuja la barra de estado."""
        bar_y = self.window_height - 40
        cv2.rectangle(window_image, (0, bar_y), (self.window_width, self.window_height), (20, 20, 20), -1)
        
        # Estado del sistema
        status_text = f"Usuario: {self.current_user} | Modo: {self.current_mode} | Gestos: {len(self.classifier.get_available_gestures(self.current_user))}"
        cv2.putText(window_image, status_text, (10, bar_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # FPS (si está disponible)
        cv2.putText(window_image, "FPS: --", (self.window_width - 100, bar_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def _draw_mode_indicator(self, window_image: np.ndarray):
        """Dibuja el indicador de modo."""
        indicator_x = self.window_width - 150
        indicator_y = 50
        
        # Fondo del indicador
        cv2.rectangle(window_image, (indicator_x, indicator_y), 
                     (indicator_x + 100, indicator_y + 30), (60, 60, 60), -1)
        
        # Texto del modo
        mode_colors = {
            "recognition": (0, 255, 0),
            "training": (0, 255, 255),
            "management": (255, 0, 255)
        }
        mode_color = mode_colors.get(self.current_mode, (255, 255, 255))
        cv2.putText(window_image, self.current_mode.upper(), 
                   (indicator_x + 10, indicator_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 1)
    
    def handle_keyboard(self, key: int) -> bool:
        """
        Maneja las teclas presionadas.
        
        Args:
            key: Código de la tecla
            
        Returns:
            True si la tecla fue manejada
        """
        if key == 27:  # ESC
            self.close()
            return True
        elif key == ord('r') or key == ord('R'):  # Modo reconocimiento
            self._switch_to_recognition_mode()
            return True
        elif key == ord('t') or key == ord('T'):  # Modo entrenamiento
            self._switch_to_training_mode()
            return True
        elif key == ord('m') or key == ord('M'):  # Modo gestión
            self._switch_to_management_mode()
            return True
        elif key == ord('d') or key == ord('D'):  # Debug
            self.debug_mode = not self.debug_mode
            print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")
            return True
        elif key == ord('h') or key == ord('H'):  # Ayuda
            self._show_help()
            return True
        elif key == ord(' '):  # SPACE - Limpiar texto
            self.text_display.clear_text()
            return True
        
        # Manejar teclas específicas del modo de entrenamiento
        if self.current_mode == "training":
            return self.training_ui.handle_key(key)
        
        return False
    
    def _switch_to_recognition_mode(self):
        """Cambia al modo de reconocimiento."""
        if self.current_mode != "recognition":
            self.current_mode = "recognition"
            print("Modo: Reconocimiento")
    
    def _switch_to_training_mode(self):
        """Cambia al modo de entrenamiento."""
        if self.current_mode != "training":
            self.current_mode = "training"
            print("Modo: Entrenamiento")
            print("Presiona 'N' para entrenar un nuevo gesto")
    
    def _switch_to_management_mode(self):
        """Cambia al modo de gestión."""
        if self.current_mode != "management":
            self.current_mode = "management"
            print("Modo: Gestión")
            self._show_management_menu()
    
    def _show_management_menu(self):
        """Muestra el menú de gestión."""
        print("\n=== MENU DE GESTION ===")
        print("1. Listar gestos disponibles")
        print("2. Eliminar gesto")
        print("3. Crear nuevo usuario")
        print("4. Cambiar usuario")
        print("5. Ver estadísticas")
        print("Presiona el número correspondiente o ESC para salir")
    
    def _show_help(self):
        """Muestra la ayuda del sistema."""
        print("\n=== AYUDA DEL SISTEMA LESCO ===")
        print("Controles principales:")
        print("  R - Modo reconocimiento (reconocer gestos)")
        print("  T - Modo entrenamiento (entrenar nuevos gestos)")
        print("  M - Modo gestión (gestionar gestos y usuarios)")
        print("  D - Activar/desactivar modo debug")
        print("  H - Mostrar esta ayuda")
        print("  SPACE - Limpiar texto reconocido")
        print("  ESC - Salir del programa")
        print("\nEn modo entrenamiento:")
        print("  N - Entrenar nuevo gesto")
        print("  ESC - Cancelar entrenamiento")
        print("  SPACE - Forzar siguiente muestra")
        print("\nEn modo gestión:")
        print("  1-5 - Opciones del menú")
        print("=" * 40)
    
    def start_new_gesture_training(self, gesture_name: str) -> bool:
        """
        Inicia el entrenamiento de un nuevo gesto.
        
        Args:
            gesture_name: Nombre del gesto
            
        Returns:
            True si se inició exitosamente
        """
        if self.current_mode != "training":
            print("Debes estar en modo entrenamiento para entrenar gestos")
            return False
        
        return self.training_ui.start_training_session(gesture_name, self.current_user)
    
    def _on_training_completed(self, info: Dict):
        """Callback cuando se completa el entrenamiento."""
        print(f"Entrenamiento completado: {info['gesture_name']}")
        # Recargar gestos disponibles
        self.classifier = GestureClassifier()
    
    def _on_training_cancelled(self):
        """Callback cuando se cancela el entrenamiento."""
        print("Entrenamiento cancelado")
    
    def close(self):
        """Cierra la ventana."""
        self.is_running = False
        cv2.destroyAllWindows()
        print("Ventana cerrada")
    
    def is_active(self) -> bool:
        """Retorna si la ventana está activa."""
        return self.is_running