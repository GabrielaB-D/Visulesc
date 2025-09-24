"""
Extracción de características por frame usando una o dos manos.

Características (vector compacto):
- Distancias relativas muñeca→puntas (5)
- Ángulos entre dedos (pulgar-índice, índice-medio, medio-anular, anular-meñique) (4)
- Orientación de palma (ángulo muñeca→mcp_medio vs muñeca→mcp_meñique) (1)
- Si hay dos manos: distancia entre muñecas y delta X/Y (2)
Total por mano: 10; con relación entre manos: +2 = 22 en caso de dos manos.
"""

from typing import List, Optional, Tuple
import numpy as np


def _angle(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    c = np.clip(float(np.dot(v1, v2) / (n1 * n2)), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def _norm_landmarks(lms) -> Optional[np.ndarray]:
    if lms is None or len(lms) < 21:
        return None
    # centrar en muñeca y escalar por distancia muñeca→mcp_medio (índice 9)
    wrist = lms[0]
    ref = lms[9]
    scale = np.linalg.norm([ref.x - wrist.x, ref.y - wrist.y, ref.z - wrist.z])
    if scale == 0:
        scale = 1.0
    pts = np.array([[lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z] for lm in lms], dtype=np.float32)
    pts /= scale
    return pts


def _per_hand_features(pts: np.ndarray) -> np.ndarray:
    # Distancias muñeca→puntas
    wrist = pts[0]
    tips = [4, 8, 12, 16, 20]
    dists = [np.linalg.norm(pts[i] - wrist) for i in tips]
    # Ángulos entre dedos usando vectores muñeca→punta
    vecs = [pts[i] - wrist for i in tips]
    angs = [
        _angle(vecs[0], vecs[1]),
        _angle(vecs[1], vecs[2]),
        _angle(vecs[2], vecs[3]),
        _angle(vecs[3], vecs[4]),
    ]
    # Orientación de palma: muñeca→mcp_medio vs muñeca→mcp_meñique
    v_m = pts[9] - wrist
    v_p = pts[17] - wrist
    palm_ori = _angle(v_m, v_p)
    return np.array(dists + angs + [palm_ori], dtype=np.float32)


def extract_features_for_both_hands(left_lms, right_lms) -> Tuple[np.ndarray, bool, bool]:
    """Devuelve (features, left_present, right_present)."""
    left_pts = _norm_landmarks(left_lms) if left_lms is not None else None
    right_pts = _norm_landmarks(right_lms) if right_lms is not None else None
    left_present = left_pts is not None
    right_present = right_pts is not None

    feats = []
    if left_present:
        feats.append(_per_hand_features(left_pts))
    else:
        feats.append(np.zeros(10, dtype=np.float32))
    if right_present:
        feats.append(_per_hand_features(right_pts))
    else:
        feats.append(np.zeros(10, dtype=np.float32))

    # Relación entre manos
    if left_present and right_present:
        lw, rw = left_pts[0], right_pts[0]
        dx, dy = rw[0] - lw[0], rw[1] - lw[1]
        dist_wrists = np.linalg.norm([dx, dy])
        rel = np.array([dist_wrists, dx], dtype=np.float32)
    else:
        rel = np.zeros(2, dtype=np.float32)

    return (np.concatenate([feats[0], feats[1], rel]), left_present, right_present)


