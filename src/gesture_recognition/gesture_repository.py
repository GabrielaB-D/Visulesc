"""
Repositorio de gestos basado en JSON con landmarks normalizados.

Formato JSON por archivo (un gesto con múltiples muestras):
[
  {
    "gesture_name": "HOLA",
    "user_id": "default",
    "type": "static" | "dynamic",
    "timestamp": 1690000000.0,
    "fps": 30,                        # solo para dinámicos
    "landmarks": [                    # para estáticos: 21 puntos
      {"x": 0.1, "y": -0.2, "z": 0.0},
      ... (21)
    ],
    "sequence": [                     # para dinámicos: lista de frames
      [ {"x": ..., "y": ..., "z": ...}, ... (21) ],
      ...
    ]
  }
]
"""

import os
import json
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class GestureRepository:
    """Gestiona guardado y carga de plantillas de gestos normalizados."""

    def __init__(self, gestures_dir: str = "data/gestures") -> None:
        self.gestures_dir = gestures_dir
        os.makedirs(self.gestures_dir, exist_ok=True)

    # -------- Normalización --------
    @staticmethod
    def normalize_landmarks(landmarks) -> List[Dict[str, float]]:
        """
        Normaliza landmarks 3D:
        - Traslada al origen usando la muñeca (índice 0)
        - Escala por la distancia muñeca-medio_mcp (índice 9) para invarianza de tamaño
        - Mantiene la proporción en x,y,z
        """
        if not landmarks or len(landmarks) < 21:
            return []

        wrist = landmarks[0]
        ref = landmarks[9]  # middle_mcp como escala robusta de la palma
        scale = np.linalg.norm([
            ref.x - wrist.x,
            ref.y - wrist.y,
            ref.z - wrist.z,
        ])
        if scale == 0:
            scale = 1.0

        normalized = []
        for lm in landmarks:
            nx = (lm.x - wrist.x) / scale
            ny = (lm.y - wrist.y) / scale
            nz = (lm.z - wrist.z) / scale
            normalized.append({"x": float(nx), "y": float(ny), "z": float(nz)})
        return normalized

    # -------- Guardado --------
    def save_static(self, gesture_name: str, landmarks_norm: List[Dict[str, float]], *, user_id: str = "default") -> bool:
        try:
            path = os.path.join(self.gestures_dir, f"{user_id}_{gesture_name}.json")
            entry = {
                "gesture_name": gesture_name,
                "user_id": user_id,
                "type": "static",
                "timestamp": time.time(),
                "landmarks": landmarks_norm,
            }
            data: List[Dict[str, Any]] = []
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            data.append(entry)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error guardando gesto estático: {e}")
            return False

    def save_dynamic(self, gesture_name: str, sequence_norm: List[List[Dict[str, float]]], *, fps: int = 30, user_id: str = "default") -> bool:
        try:
            path = os.path.join(self.gestures_dir, f"{user_id}_{gesture_name}.json")
            entry = {
                "gesture_name": gesture_name,
                "user_id": user_id,
                "type": "dynamic",
                "timestamp": time.time(),
                "fps": fps,
                "sequence": sequence_norm,
            }
            data: List[Dict[str, Any]] = []
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            data.append(entry)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error guardando gesto dinámico: {e}")
            return False

    def save_statistical(self, gesture_name: str, *, mean_features: List[float], std_features: List[float], num_frames: int, variability_score: float, avg_confidence: float, user_id: str = "default", feature_type: str = "static_stats") -> bool:
        """Guarda una plantilla estadística (media/desviación) para gestos estáticos muestreados."""
        try:
            path = os.path.join(self.gestures_dir, f"{user_id}_{gesture_name}.json")
            entry = {
                "gesture_name": gesture_name,
                "user_id": user_id,
                "type": feature_type,
                "timestamp": time.time(),
                "mean_features": mean_features,
                "std_features": std_features,
                "num_frames": num_frames,
                "variability_score": variability_score,
                "avg_confidence": avg_confidence,
            }
            data: List[Dict[str, Any]] = []
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            data.append(entry)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error guardando gesto estadístico: {e}")
            return False

    # -------- Carga --------
    def load_templates(self, *, user_id: str = "default") -> List[Dict[str, Any]]:
        templates: List[Dict[str, Any]] = []
        try:
            for filename in os.listdir(self.gestures_dir):
                if not filename.endswith(".json"):
                    continue
                if not filename.startswith(f"{user_id}_"):
                    continue
                with open(os.path.join(self.gestures_dir, filename), "r", encoding="utf-8") as f:
                    items = json.load(f)
                    for it in items:
                        if it.get("type") in ("static", "dynamic"):
                            templates.append(it)
        except Exception as e:
            print(f"Error cargando plantillas: {e}")
        return templates


