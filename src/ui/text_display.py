"""
Panel de visualización de texto reconocido.
Maneja la visualización y gestión del texto generado por los gestos.
"""

import cv2
import numpy as np
from typing import List, Optional, Callable
from datetime import datetime


class TextDisplay:
    """Panel de visualización de texto reconocido."""
    
    def __init__(self, max_history: int = 50):
        """
        Inicializa el panel de texto.
        
        Args:
            max_history: Número máximo de elementos en el historial
        """
        self.current_text = ""
        self.text_history = []
        self.max_history = max_history
        
        # Callbacks
        self.on_text_update: Optional[Callable] = None
        self.on_text_clear: Optional[Callable] = None
    
    def add_text(self, text: str):
        """
        Añade texto al display actual.
        
        Args:
            text: Texto a añadir
        """
        if text and text.strip():
            if self.current_text:
                self.current_text += " " + text.strip()
            else:
                self.current_text = text.strip()
            
            # Añadir al historial
            self._add_to_history(text.strip())
            
            if self.on_text_update:
                self.on_text_update(self.current_text)
    
    def set_text(self, text: str):
        """
        Establece el texto actual.
        
        Args:
            text: Nuevo texto
        """
        self.current_text = text.strip() if text else ""
        
        if self.on_text_update:
            self.on_text_update(self.current_text)
    
    def clear_text(self):
        """Limpia el texto actual."""
        self.current_text = ""
        
        if self.on_text_clear:
            self.on_text_clear()
    
    def _add_to_history(self, text: str):
        """Añade texto al historial."""
        history_entry = {
            'text': text,
            'timestamp': datetime.now(),
            'formatted_time': datetime.now().strftime("%H:%M:%S")
        }
        
        self.text_history.append(history_entry)
        
        # Mantener solo el historial más reciente
        if len(self.text_history) > self.max_history:
            self.text_history = self.text_history[-self.max_history:]
    
    def draw_text_panel(self, frame: np.ndarray, panel_x: int, panel_y: int, 
                       panel_width: int, panel_height: int) -> np.ndarray:
        """
        Dibuja el panel de texto en el frame.
        
        Args:
            frame: Frame donde dibujar
            panel_x: Posición X del panel
            panel_y: Posición Y del panel
            panel_width: Ancho del panel
            panel_height: Alto del panel
            
        Returns:
            Frame con el panel dibujado
        """
        # Fondo del panel
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), (30, 30, 30), -1)
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), (100, 100, 100), 2)
        
        # Título del panel
        cv2.putText(frame, "TEXTO RECONOCIDO", 
                   (panel_x + 10, panel_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Texto actual
        y_offset = panel_y + 50
        if self.current_text:
            # Dibujar texto con wrap
            wrapped_text = self._wrap_text(self.current_text, panel_width - 20)
            for line in wrapped_text:
                cv2.putText(frame, line, (panel_x + 10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                y_offset += 20
        else:
            cv2.putText(frame, "Ningún texto reconocido", 
                       (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # Separador
        y_offset += 20
        cv2.line(frame, (panel_x + 10, y_offset), (panel_x + panel_width - 10, y_offset), (100, 100, 100), 1)
        
        # Historial reciente
        y_offset += 30
        cv2.putText(frame, "Historial:", 
                   (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        y_offset += 25
        # Mostrar últimos 5 elementos del historial
        recent_history = self.text_history[-5:] if len(self.text_history) > 5 else self.text_history
        for entry in reversed(recent_history):
            if y_offset + 20 > panel_y + panel_height - 10:
                break
            
            # Formato: [HH:MM:SS] texto
            history_text = f"[{entry['formatted_time']}] {entry['text']}"
            wrapped_history = self._wrap_text(history_text, panel_width - 20)
            
            for line in wrapped_history:
                if y_offset + 20 > panel_y + panel_height - 10:
                    break
                cv2.putText(frame, line, (panel_x + 10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
                y_offset += 15
        
        return frame
    
    def _wrap_text(self, text: str, max_width: int) -> List[str]:
        """
        Envuelve el texto en líneas que no excedan el ancho máximo.
        
        Args:
            text: Texto a envolver
            max_width: Ancho máximo en píxeles
            
        Returns:
            Lista de líneas envueltas
        """
        if not text:
            return []
        
        # Aproximación simple: asumir que cada carácter tiene ~10 píxeles de ancho
        chars_per_line = max_width // 10
        
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line + " " + word) <= chars_per_line:
                if current_line:
                    current_line += " " + word
                else:
                    current_line = word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        return lines
    
    def get_current_text(self) -> str:
        """Retorna el texto actual."""
        return self.current_text
    
    def get_history(self) -> List[dict]:
        """Retorna el historial completo."""
        return self.text_history.copy()
    
    def export_text(self, filename: str) -> bool:
        """
        Exporta el texto actual a un archivo.
        
        Args:
            filename: Nombre del archivo
            
        Returns:
            True si se exportó exitosamente
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Texto reconocido - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n\n")
                f.write(self.current_text + "\n\n")
                
                if self.text_history:
                    f.write("Historial:\n")
                    f.write("-" * 30 + "\n")
                    for entry in self.text_history:
                        f.write(f"[{entry['formatted_time']}] {entry['text']}\n")
            
            print(f"Texto exportado a {filename}")
            return True
        except Exception as e:
            print(f"Error exportando texto: {e}")
            return False
    
    def get_statistics(self) -> dict:
        """Retorna estadísticas del texto."""
        total_words = len(self.current_text.split()) if self.current_text else 0
        total_chars = len(self.current_text) if self.current_text else 0
        history_count = len(self.text_history)
        
        return {
            'current_text_length': total_chars,
            'current_word_count': total_words,
            'history_entries': history_count,
            'last_update': self.text_history[-1]['timestamp'] if self.text_history else None
        }