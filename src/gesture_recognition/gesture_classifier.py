"""
Sistema híbrido de clasificación de gestos para LESCO.
Combina Machine Learning para gestos estáticos y DTW para gestos dinámicos.
"""

import numpy as np
import json
import os
from typing import List, Dict, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
try:
    from ..config.settings import *
except ImportError:
    from config.settings import *


class GestureClassifier:
    """Clasificador híbrido de gestos para LESCO."""
    
    def __init__(self, data_dir: str = "data/gestures"):
        """
        Inicializa el clasificador de gestos.
        
        Args:
            data_dir: Directorio donde se almacenan los datos de gestos
        """
        self.data_dir = data_dir
        self.static_classifier = None
        self.dynamic_classifier = None
        self.scaler = StandardScaler()
        self.gesture_labels = []
        self.is_trained = False
        
        # Parámetros para DTW
        self.dtw_window = 10
        self.dtw_threshold = 0.8
        
        # Crear directorio si no existe
        os.makedirs(data_dir, exist_ok=True)
    
    def add_gesture_sample(self, gesture_data: Dict, label: str, user_id: str = "default"):
        """
        Añade una muestra de gesto al conjunto de entrenamiento.
        
        Args:
            gesture_data: Datos del gesto (frames con landmarks)
            label: Etiqueta del gesto (ej: "HOLA", "GRACIAS")
            user_id: ID del usuario
        """
        gesture_file = os.path.join(self.data_dir, f"{user_id}_{label}.json")
        
        # Cargar gestos existentes o crear lista vacía
        if os.path.exists(gesture_file):
            with open(gesture_file, 'r', encoding='utf-8') as f:
                gestures = json.load(f)
        else:
            gestures = []
        
        # Procesar landmarks para características
        features = self._extract_features(gesture_data['frames'])
        
        # Crear entrada del gesto
        gesture_entry = {
            'features': features,
            'frames': gesture_data['frames'],
            'duration': gesture_data['duration'],
            'frame_count': gesture_data['frame_count'],
            'timestamp': gesture_data['timestamp'],
            'label': label,
            'user_id': user_id
        }
        
        gestures.append(gesture_entry)
        
        # Guardar gesto
        with open(gesture_file, 'w', encoding='utf-8') as f:
            json.dump(gestures, f, indent=2, ensure_ascii=False)
        
        print(f"Gesto '{label}' añadido para usuario '{user_id}'")
    
    def _extract_features(self, frames: List[Dict]) -> np.ndarray:
        """
        Extrae características de los frames del gesto.
        
        Args:
            frames: Lista de frames con landmarks
            
        Returns:
            Array de características extraídas
        """
        if not frames:
            return np.array([])
        
        # Características estáticas (posición promedio de landmarks)
        static_features = []
        
        # Características dinámicas (movimiento entre frames)
        dynamic_features = []
        
        # Procesar cada frame
        landmarks_sequence = []
        for frame in frames:
            landmarks = frame['landmarks']
            landmarks_sequence.append(landmarks)
            
            # Extraer características estáticas del frame
            frame_static = self._extract_static_features(landmarks)
            static_features.append(frame_static)
        
        # Calcular características dinámicas
        dynamic_features = self._extract_dynamic_features(landmarks_sequence)
        
        # Combinar características
        all_features = np.concatenate([
            np.mean(static_features, axis=0),  # Promedio de características estáticas
            dynamic_features  # Características dinámicas
        ])
        
        return all_features
    
    def _extract_static_features(self, landmarks) -> np.ndarray:
        """
        Extrae características estáticas de un frame.
        
        Args:
            landmarks: Landmarks de MediaPipe
            
        Returns:
            Array de características estáticas
        """
        features = []
        
        # Posiciones de landmarks clave (puntas de dedos, muñeca, etc.)
        key_landmarks = [0, 4, 8, 12, 16, 20]  # Muñeca y puntas de dedos
        
        for idx in key_landmarks:
            if idx < len(landmarks):
                lm = landmarks[idx]
                features.extend([lm.x, lm.y, lm.z])
            else:
                features.extend([0, 0, 0])  # Padding si no hay suficientes landmarks
        
        # Distancias entre landmarks clave
        if len(landmarks) >= 21:  # MediaPipe tiene 21 landmarks por mano
            # Distancia muñeca a puntas de dedos
            wrist = landmarks[0]
            for finger_tip in [4, 8, 12, 16, 20]:
                tip = landmarks[finger_tip]
                distance = np.sqrt(
                    (wrist.x - tip.x) ** 2 + 
                    (wrist.y - tip.y) ** 2 + 
                    (wrist.z - tip.z) ** 2
                )
                features.append(distance)
        
        return np.array(features)
    
    def _extract_dynamic_features(self, landmarks_sequence: List) -> np.ndarray:
        """
        Extrae características dinámicas de la secuencia de landmarks.
        
        Args:
            landmarks_sequence: Secuencia de landmarks
            
        Returns:
            Array de características dinámicas
        """
        if len(landmarks_sequence) < 2:
            return np.zeros(10)  # Retornar características vacías
        
        features = []
        
        # Velocidad promedio de movimiento
        velocities = []
        for i in range(1, len(landmarks_sequence)):
            prev_landmarks = landmarks_sequence[i-1]
            curr_landmarks = landmarks_sequence[i]
            
            velocity = self._calculate_velocity(prev_landmarks, curr_landmarks)
            velocities.append(velocity)
        
        if velocities:
            features.extend([
                np.mean(velocities),  # Velocidad promedio
                np.std(velocities),   # Desviación estándar de velocidad
                np.max(velocities),   # Velocidad máxima
                np.min(velocities)    # Velocidad mínima
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # Dirección de movimiento dominante
        directions = self._calculate_movement_directions(landmarks_sequence)
        features.extend(directions)
        
        # Duración normalizada
        duration = len(landmarks_sequence) / 30.0  # Asumiendo 30 FPS
        features.append(duration)
        
        return np.array(features)
    
    def _calculate_velocity(self, prev_landmarks, curr_landmarks) -> float:
        """Calcula la velocidad promedio entre dos frames."""
        if len(prev_landmarks) != len(curr_landmarks):
            return 0
        
        total_distance = 0
        for i in range(len(prev_landmarks)):
            prev = prev_landmarks[i]
            curr = curr_landmarks[i]
            
            distance = np.sqrt(
                (curr.x - prev.x) ** 2 + 
                (curr.y - prev.y) ** 2 + 
                (curr.z - prev.z) ** 2
            )
            total_distance += distance
        
        return total_distance / len(prev_landmarks)
    
    def _calculate_movement_directions(self, landmarks_sequence: List) -> List[float]:
        """Calcula las direcciones de movimiento dominantes."""
        if len(landmarks_sequence) < 2:
            return [0, 0, 0, 0, 0, 0]
        
        # Usar la muñeca como punto de referencia
        wrist_positions = []
        for landmarks in landmarks_sequence:
            if len(landmarks) > 0:
                wrist_positions.append([landmarks[0].x, landmarks[0].y])
        
        if len(wrist_positions) < 2:
            return [0, 0, 0, 0, 0, 0]
        
        # Calcular desplazamientos
        displacements = []
        for i in range(1, len(wrist_positions)):
            dx = wrist_positions[i][0] - wrist_positions[i-1][0]
            dy = wrist_positions[i][1] - wrist_positions[i-1][1]
            displacements.append([dx, dy])
        
        # Calcular estadísticas de dirección
        displacements = np.array(displacements)
        
        return [
            np.mean(displacements[:, 0]),  # Movimiento horizontal promedio
            np.std(displacements[:, 0]),    # Variabilidad horizontal
            np.mean(displacements[:, 1]),   # Movimiento vertical promedio
            np.std(displacements[:, 1]),    # Variabilidad vertical
            np.max(np.abs(displacements[:, 0])),  # Máximo movimiento horizontal
            np.max(np.abs(displacements[:, 1]))   # Máximo movimiento vertical
        ]
    
    def train_classifier(self, user_id: str = "default"):
        """
        Entrena el clasificador con los gestos disponibles.
        
        Args:
            user_id: ID del usuario para entrenar
        """
        print(f"Entrenando clasificador para usuario '{user_id}'...")
        
        # Cargar todos los gestos del usuario
        gestures_data = self._load_user_gestures(user_id)
        
        if not gestures_data:
            print(f"No hay gestos disponibles para el usuario '{user_id}'")
            return False
        
        # Preparar datos para entrenamiento
        X = []
        y = []
        
        for gesture_data in gestures_data:
            features = gesture_data['features']
            label = gesture_data['label']
            
            X.append(features)
            y.append(label)
        
        if len(X) < 2:
            print("Se necesitan al menos 2 gestos para entrenar")
            return False
        
        X = np.array(X)
        y = np.array(y)
        
        # Normalizar características
        X_scaled = self.scaler.fit_transform(X)
        
        # Dividir datos para validación
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Entrenar clasificador
        self.static_classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        
        self.static_classifier.fit(X_train, y_train)
        
        # Evaluar rendimiento
        y_pred = self.static_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Precisión del clasificador: {accuracy:.2f}")
        
        # Guardar modelo entrenado
        model_file = os.path.join(self.data_dir, f"{user_id}_model.pkl")
        joblib.dump({
            'classifier': self.static_classifier,
            'scaler': self.scaler,
            'labels': list(set(y))
        }, model_file)
        
        self.gesture_labels = list(set(y))
        self.is_trained = True
        
        return True
    
    def _load_user_gestures(self, user_id: str) -> List[Dict]:
        """Carga todos los gestos de un usuario."""
        gestures_data = []
        
        # Buscar archivos de gestos del usuario
        for filename in os.listdir(self.data_dir):
            if filename.startswith(f"{user_id}_") and filename.endswith(".json"):
                filepath = os.path.join(self.data_dir, filename)
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        gestures = json.load(f)
                        gestures_data.extend(gestures)
                except Exception as e:
                    print(f"Error cargando {filename}: {e}")
        
        return gestures_data
    
    def classify_gesture(self, gesture_data: Dict, user_id: str = "default") -> Tuple[str, float]:
        """
        Clasifica un gesto usando el modelo entrenado.
        
        Args:
            gesture_data: Datos del gesto a clasificar
            user_id: ID del usuario
            
        Returns:
            Tupla (predicción, confianza)
        """
        if not self.is_trained:
            # Intentar cargar modelo pre-entrenado
            if not self._load_trained_model(user_id):
                return "NO_TRAINED", 0.0
        
        # Extraer características
        features = self._extract_features(gesture_data['frames'])
        
        if len(features) == 0:
            return "NO_FEATURES", 0.0
        
        # Normalizar características
        features_scaled = self.scaler.transform([features])
        
        # Clasificar
        prediction = self.static_classifier.predict(features_scaled)[0]
        
        # Calcular confianza
        probabilities = self.static_classifier.predict_proba(features_scaled)[0]
        confidence = np.max(probabilities)
        
        return prediction, confidence
    
    def _load_trained_model(self, user_id: str) -> bool:
        """Carga un modelo pre-entrenado."""
        model_file = os.path.join(self.data_dir, f"{user_id}_model.pkl")
        
        if not os.path.exists(model_file):
            return False
        
        try:
            model_data = joblib.load(model_file)
            self.static_classifier = model_data['classifier']
            self.scaler = model_data['scaler']
            self.gesture_labels = model_data['labels']
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Error cargando modelo: {e}")
            return False
    
    def get_available_gestures(self, user_id: str = "default") -> List[str]:
        """Retorna la lista de gestos disponibles para un usuario."""
        gestures = set()
        
        for filename in os.listdir(self.data_dir):
            if filename.startswith(f"{user_id}_") and filename.endswith(".json"):
                # Extraer nombre del gesto del nombre del archivo
                gesture_name = filename.replace(f"{user_id}_", "").replace(".json", "")
                gestures.add(gesture_name)
        
        return list(gestures)
