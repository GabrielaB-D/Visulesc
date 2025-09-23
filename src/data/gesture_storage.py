"""
Sistema de gestión de datos para el sistema de LESCO.
Maneja el almacenamiento y carga de gestos, usuarios y configuraciones.
"""

import os
import json
import time
from typing import Dict, List, Optional, Any
from datetime import datetime


class GestureStorage:
    """Sistema de almacenamiento de gestos en formato JSON."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Inicializa el sistema de almacenamiento.
        
        Args:
            data_dir: Directorio base para los datos
        """
        self.data_dir = data_dir
        self.gestures_dir = os.path.join(data_dir, "gestures")
        self.users_dir = os.path.join(data_dir, "users")
        self.config_dir = os.path.join(data_dir, "config")
        
        # Crear directorios si no existen
        os.makedirs(self.gestures_dir, exist_ok=True)
        os.makedirs(self.users_dir, exist_ok=True)
        os.makedirs(self.config_dir, exist_ok=True)
    
    def save_gesture(self, gesture_data: Dict[str, Any], gesture_name: str, user_id: str) -> bool:
        """
        Guarda un gesto en formato JSON.
        
        Args:
            gesture_data: Datos del gesto
            gesture_name: Nombre del gesto
            user_id: ID del usuario
            
        Returns:
            True si se guardó exitosamente
        """
        try:
            filename = f"{user_id}_{gesture_name}.json"
            filepath = os.path.join(self.gestures_dir, filename)
            
            # Preparar datos del gesto
            gesture_entry = {
                'gesture_name': gesture_name,
                'user_id': user_id,
                'timestamp': time.time(),
                'created_date': datetime.now().isoformat(),
                'data': gesture_data,
                'metadata': {
                    'frame_count': gesture_data.get('frame_count', 0),
                    'duration': gesture_data.get('duration', 0),
                    'end_reason': gesture_data.get('end_reason', 'unknown')
                }
            }
            
            # Cargar gestos existentes o crear lista vacía
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    gestures = json.load(f)
            else:
                gestures = []
            
            # Añadir nuevo gesto
            gestures.append(gesture_entry)
            
            # Guardar archivo
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(gestures, f, indent=2, ensure_ascii=False)
            
            print(f"Gesto '{gesture_name}' guardado para usuario '{user_id}'")
            return True
            
        except Exception as e:
            print(f"Error guardando gesto: {e}")
            return False
    
    def load_gestures(self, user_id: str, gesture_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Carga gestos de un usuario.
        
        Args:
            user_id: ID del usuario
            gesture_name: Nombre específico del gesto (opcional)
            
        Returns:
            Lista de gestos cargados
        """
        gestures = []
        
        try:
            if gesture_name:
                # Cargar gesto específico
                filename = f"{user_id}_{gesture_name}.json"
                filepath = os.path.join(self.gestures_dir, filename)
                
                if os.path.exists(filepath):
                    with open(filepath, 'r', encoding='utf-8') as f:
                        gestures = json.load(f)
            else:
                # Cargar todos los gestos del usuario
                for filename in os.listdir(self.gestures_dir):
                    if filename.startswith(f"{user_id}_") and filename.endswith(".json"):
                        filepath = os.path.join(self.gestures_dir, filename)
                        
                        with open(filepath, 'r', encoding='utf-8') as f:
                            user_gestures = json.load(f)
                            gestures.extend(user_gestures)
            
            return gestures
            
        except Exception as e:
            print(f"Error cargando gestos: {e}")
            return []
    
    def delete_gesture(self, gesture_name: str, user_id: str) -> bool:
        """
        Elimina un gesto específico.
        
        Args:
            gesture_name: Nombre del gesto
            user_id: ID del usuario
            
        Returns:
            True si se eliminó exitosamente
        """
        try:
            filename = f"{user_id}_{gesture_name}.json"
            filepath = os.path.join(self.gestures_dir, filename)
            
            if os.path.exists(filepath):
                os.remove(filepath)
                print(f"Gesto '{gesture_name}' eliminado para usuario '{user_id}'")
                return True
            else:
                print(f"Gesto '{gesture_name}' no encontrado para usuario '{user_id}'")
                return False
                
        except Exception as e:
            print(f"Error eliminando gesto: {e}")
            return False
    
    def get_user_gestures(self, user_id: str) -> List[str]:
        """
        Obtiene la lista de gestos disponibles para un usuario.
        
        Args:
            user_id: ID del usuario
            
        Returns:
            Lista de nombres de gestos
        """
        gestures = []
        
        try:
            for filename in os.listdir(self.gestures_dir):
                if filename.startswith(f"{user_id}_") and filename.endswith(".json"):
                    # Extraer nombre del gesto
                    gesture_name = filename.replace(f"{user_id}_", "").replace(".json", "")
                    gestures.append(gesture_name)
            
            return gestures
            
        except Exception as e:
            print(f"Error obteniendo gestos del usuario: {e}")
            return []
    
    def get_gesture_statistics(self, user_id: str) -> Dict[str, Any]:
        """
        Obtiene estadísticas de los gestos de un usuario.
        
        Args:
            user_id: ID del usuario
            
        Returns:
            Diccionario con estadísticas
        """
        gestures = self.load_gestures(user_id)
        
        if not gestures:
            return {
                'total_gestures': 0,
                'total_samples': 0,
                'gesture_names': [],
                'average_samples_per_gesture': 0
            }
        
        gesture_names = set()
        total_samples = 0
        
        for gesture in gestures:
            gesture_names.add(gesture['gesture_name'])
            total_samples += 1
        
        return {
            'total_gestures': len(gesture_names),
            'total_samples': total_samples,
            'gesture_names': list(gesture_names),
            'average_samples_per_gesture': total_samples / len(gesture_names) if gesture_names else 0
        }


class UserManager:
    """Gestor de usuarios del sistema."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Inicializa el gestor de usuarios.
        
        Args:
            data_dir: Directorio base para los datos
        """
        self.data_dir = data_dir
        self.users_file = os.path.join(data_dir, "users", "users.json")
        self.current_user = "default"
        
        # Crear archivo de usuarios si no existe
        self._initialize_users_file()
    
    def _initialize_users_file(self):
        """Inicializa el archivo de usuarios con el usuario por defecto."""
        if not os.path.exists(self.users_file):
            default_users = {
                "default": {
                    "name": "Usuario por Defecto",
                    "created_date": datetime.now().isoformat(),
                    "last_active": datetime.now().isoformat(),
                    "gestures_count": 0,
                    "preferences": {
                        "pause_threshold": 0.3,
                        "min_gesture_frames": 10,
                        "max_gesture_frames": 120,
                        "movement_threshold": 0.02
                    }
                }
            }
            
            os.makedirs(os.path.dirname(self.users_file), exist_ok=True)
            with open(self.users_file, 'w', encoding='utf-8') as f:
                json.dump(default_users, f, indent=2, ensure_ascii=False)
    
    def create_user(self, user_id: str, name: str) -> bool:
        """
        Crea un nuevo usuario.
        
        Args:
            user_id: ID único del usuario
            name: Nombre del usuario
            
        Returns:
            True si se creó exitosamente
        """
        try:
            # Cargar usuarios existentes
            users = self.load_users()
            
            if user_id in users:
                print(f"Usuario '{user_id}' ya existe")
                return False
            
            # Crear nuevo usuario
            new_user = {
                "name": name,
                "created_date": datetime.now().isoformat(),
                "last_active": datetime.now().isoformat(),
                "gestures_count": 0,
                "preferences": {
                    "pause_threshold": 0.3,
                    "min_gesture_frames": 10,
                    "max_gesture_frames": 120,
                    "movement_threshold": 0.02
                }
            }
            
            users[user_id] = new_user
            
            # Guardar usuarios
            with open(self.users_file, 'w', encoding='utf-8') as f:
                json.dump(users, f, indent=2, ensure_ascii=False)
            
            print(f"Usuario '{user_id}' creado exitosamente")
            return True
            
        except Exception as e:
            print(f"Error creando usuario: {e}")
            return False
    
    def load_users(self) -> Dict[str, Any]:
        """
        Carga todos los usuarios.
        
        Returns:
            Diccionario con información de usuarios
        """
        try:
            with open(self.users_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error cargando usuarios: {e}")
            return {}
    
    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene información de un usuario específico.
        
        Args:
            user_id: ID del usuario
            
        Returns:
            Información del usuario o None si no existe
        """
        users = self.load_users()
        return users.get(user_id)
    
    def update_user_activity(self, user_id: str):
        """
        Actualiza la última actividad de un usuario.
        
        Args:
            user_id: ID del usuario
        """
        try:
            users = self.load_users()
            
            if user_id in users:
                users[user_id]["last_active"] = datetime.now().isoformat()
                
                with open(self.users_file, 'w', encoding='utf-8') as f:
                    json.dump(users, f, indent=2, ensure_ascii=False)
                    
        except Exception as e:
            print(f"Error actualizando actividad del usuario: {e}")
    
    def set_current_user(self, user_id: str) -> bool:
        """
        Establece el usuario actual.
        
        Args:
            user_id: ID del usuario
            
        Returns:
            True si el usuario existe
        """
        users = self.load_users()
        
        if user_id in users:
            self.current_user = user_id
            self.update_user_activity(user_id)
            return True
        else:
            print(f"Usuario '{user_id}' no encontrado")
            return False
    
    def get_current_user(self) -> str:
        """Retorna el ID del usuario actual."""
        return self.current_user
    
    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """
        Obtiene las preferencias de un usuario.
        
        Args:
            user_id: ID del usuario
            
        Returns:
            Diccionario con preferencias
        """
        user = self.get_user(user_id)
        if user:
            return user.get("preferences", {})
        return {}
    
    def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> bool:
        """
        Actualiza las preferencias de un usuario.
        
        Args:
            user_id: ID del usuario
            preferences: Nuevas preferencias
            
        Returns:
            True si se actualizaron exitosamente
        """
        try:
            users = self.load_users()
            
            if user_id in users:
                users[user_id]["preferences"].update(preferences)
                
                with open(self.users_file, 'w', encoding='utf-8') as f:
                    json.dump(users, f, indent=2, ensure_ascii=False)
                
                return True
            else:
                print(f"Usuario '{user_id}' no encontrado")
                return False
                
        except Exception as e:
            print(f"Error actualizando preferencias: {e}")
            return False
