"""
Extracción de features enfocados en el pulgar para letras/gestos.

Features por frame:
- angle_thumb_index: ángulo entre v_thumb (tip-mcp) e índice (tip-mcp)
- d_thumb_index: distancia normalizada entre thumb_tip e index_tip
- angle_thumb_palm: ángulo entre v_thumb y normal de la palma
- thumb_curvature: ángulo entre (mcp→ip) y (ip→tip)
- z_depth_thumb: z relativa del thumb_tip respecto a muñeca e index_tip

Normalización de escala por hand_size = ||wrist - middle_mcp||.
"""

from typing import List, Tuple
import numpy as np


IDX = {
    'wrist': 0,
    'thumb_cmc': 1,
    'thumb_mcp': 2,
    'thumb_ip': 3,
    'thumb_tip': 4,
    'index_mcp': 5,
    'index_tip': 8,
    'middle_mcp': 9,
    'pinky_mcp': 17,
}


def _vec(a, b) -> np.ndarray:
    return np.array([b.x - a.x, b.y - a.y, getattr(b, 'z', 0.0) - getattr(a, 'z', 0.0)], dtype=np.float32)


def _angle(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    c = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def extract_thumb_features(lm_list) -> Tuple[np.ndarray, float]:
    """Devuelve (features, hand_size)."""
    try:
        w = lm_list[IDX['wrist']]
        m_mcp = lm_list[IDX['middle_mcp']]
        hand_size = float(np.linalg.norm(_vec(w, m_mcp)))
        if hand_size <= 1e-6:
            hand_size = 1.0

        t_mcp = lm_list[IDX['thumb_mcp']]
        t_ip = lm_list[IDX['thumb_ip']]
        t_tip = lm_list[IDX['thumb_tip']]
        i_mcp = lm_list[IDX['index_mcp']]
        i_tip = lm_list[IDX['index_tip']]
        p_mcp = lm_list[IDX['pinky_mcp']]

        v_thumb = _vec(t_mcp, t_tip)
        v_index = _vec(i_mcp, i_tip)

        angle_thumb_index = _angle(v_thumb, v_index)
        d_thumb_index = float(np.linalg.norm(_vec(i_tip, t_tip))) / hand_size

        # normal de la palma con (index_mcp - wrist) x (pinky_mcp - wrist)
        n_palm = np.cross(_vec(w, i_mcp), _vec(w, p_mcp))
        angle_thumb_palm = _angle(v_thumb, n_palm)

        # curvatura del pulgar
        v_mcp_ip = _vec(t_mcp, t_ip)
        v_ip_tip = _vec(t_ip, t_tip)
        thumb_curvature = _angle(v_mcp_ip, v_ip_tip)

        # z relativa
        z_thumb = getattr(t_tip, 'z', 0.0)
        z_wrist = getattr(w, 'z', 0.0)
        z_index = getattr(i_tip, 'z', 0.0)
        z_depth_thumb = float(z_thumb - (z_wrist + z_index) / 2.0)

        feats = np.array([
            angle_thumb_index,
            d_thumb_index,
            angle_thumb_palm,
            thumb_curvature,
            z_depth_thumb,
        ], dtype=np.float32)
        return feats, hand_size
    except Exception:
        return np.zeros(5, dtype=np.float32), 1.0


