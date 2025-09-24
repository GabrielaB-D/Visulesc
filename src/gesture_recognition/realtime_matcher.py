"""
Emparejador en tiempo real para gestos estáticos y dinámicos.

Estrategia:
- Estáticos: distancia media entre landmarks normalizados a varias plantillas.
- Dinámicos: DTW ligero sobre trayectorias de puntos claves (muñeca y puntas).

Optimizado para bajo consumo: selección de subconjunto de puntos, reducción de dimensión.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np


KeyIndices = [0, 4, 8, 12, 16, 20]  # muñeca + puntas


def _flatten_landmarks(landmarks_norm: List[Dict[str, float]]) -> np.ndarray:
    vec: List[float] = []
    for idx in KeyIndices:
        if idx < len(landmarks_norm):
            lm = landmarks_norm[idx]
            vec.extend([lm["x"], lm["y"], lm.get("z", 0.0)])
        else:
            vec.extend([0.0, 0.0, 0.0])
    return np.asarray(vec, dtype=np.float32)


def _sequence_to_array(sequence_norm: List[List[Dict[str, float]]]) -> np.ndarray:
    frames = []
    for landmarks in sequence_norm:
        frames.append(_flatten_landmarks(landmarks))
    return np.vstack(frames) if frames else np.zeros((0, len(KeyIndices) * 3), dtype=np.float32)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _dtw_distance(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    """DTW simple con costo L2. Devuelve distancia normalizada."""
    n, m = len(seq_a), len(seq_b)
    if n == 0 or m == 0:
        return 1e9
    # Limitar ventana Sakoe-Chiba (10% del max largo) para rendimiento
    w = max(1, int(0.1 * max(n, m)))
    inf = 1e12
    dtw = np.full((n + 1, m + 1), inf, dtype=np.float32)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        j_start = max(1, i - w)
        j_end = min(m, i + w)
        for j in range(j_start, j_end + 1):
            cost = np.linalg.norm(seq_a[i - 1] - seq_b[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    # Normalizar por longitud del camino
    return float(dtw[n, m] / (n + m))


class RealTimeMatcher:
    def __init__(self, templates: List[Dict], *, static_similarity_threshold: float = 0.9, dtw_max_distance: float = 0.12):
        self.static_templates = [t for t in templates if t.get("type") == "static"]
        self.dynamic_templates = [t for t in templates if t.get("type") == "dynamic"]
        self.stat_templates = [t for t in templates if t.get("type") == "static_stats"]
        self.static_similarity_threshold = static_similarity_threshold
        self.dtw_max_distance = dtw_max_distance

    def match_static(self, current_norm: List[Dict[str, float]]) -> Tuple[Optional[str], float]:
        if not self.static_templates:
            return None, 0.0
        query = _flatten_landmarks(current_norm)
        best_label, best_sim = None, -1.0
        for t in self.static_templates:
            cand = _flatten_landmarks(t["landmarks"]) if isinstance(t.get("landmarks"), list) else None
            if cand is None:
                continue
            sim = _cosine_similarity(query, cand)
            if sim > best_sim:
                best_sim = sim
                best_label = t.get("gesture_name")
        if best_sim >= self.static_similarity_threshold:
            return best_label, best_sim
        return None, best_sim

    def match_statistical(self, feat_vec: np.ndarray) -> Tuple[Optional[str], float]:
        """Compara contra plantillas con media/std usando una distancia normalizada.
        Retorna (label, confianza).
        """
        if not self.stat_templates or feat_vec.size == 0:
            return None, 0.0
        best_label, best_score = None, 1e9
        for t in self.stat_templates:
            mean = np.asarray(t.get("mean_features", []), dtype=np.float32)
            std = np.asarray(t.get("std_features", []), dtype=np.float32)
            if mean.size == 0 or mean.size != feat_vec.size:
                continue
            std_safe = np.where(std <= 1e-6, 1e-6, std)
            z = (feat_vec - mean) / std_safe
            # puntuación: media de |z| (menor es mejor)
            score = float(np.mean(np.abs(z)))
            if score < best_score:
                best_score = score
                best_label = t.get("gesture_name")
        # Convertir score a pseudo-confianza
        # score 0.0 -> 1.0; score >= 3.0 -> 0.0
        conf = max(0.0, 1.0 - (best_score / 3.0))
        return best_label, conf

    def match_dynamic(self, sequence_norm: List[List[Dict[str, float]]]) -> Tuple[Optional[str], float]:
        if not self.dynamic_templates or not sequence_norm:
            return None, 0.0
        query = _sequence_to_array(sequence_norm)
        best_label, best_score = None, 1e9
        for t in self.dynamic_templates:
            ref = _sequence_to_array(t.get("sequence", []))
            if ref.size == 0:
                continue
            dist = _dtw_distance(query, ref)
            if dist < best_score:
                best_score = dist
                best_label = t.get("gesture_name")
        if best_score <= self.dtw_max_distance:
            # Convertir distancia a pseudo-confianza
            confidence = max(0.0, 1.0 - best_score / self.dtw_max_distance)
            return best_label, confidence
        return None, 0.0


