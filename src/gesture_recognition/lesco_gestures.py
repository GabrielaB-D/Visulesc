"""
Gestos base de LESCO (Lengua de Señas Costarricense).
Define gestos comunes y sus características para el sistema de reconocimiento.
"""

from typing import Dict, List, Any


class LESCOGestures:
    """Gestos base de LESCO para el sistema de reconocimiento."""
    
    # Gestos básicos comunes en LESCO
    BASIC_GESTURES = {
        "HOLA": {
            "description": "Saludo básico",
            "complexity": "simple",
            "movement_type": "static",
            "duration_range": (0.5, 2.0),
            "key_landmarks": [0, 4, 8, 12, 16, 20],  # Muñeca y puntas de dedos
            "common_variations": ["hola", "hello", "saludo"]
        },
        
        "GRACIAS": {
            "description": "Expresión de agradecimiento",
            "complexity": "medium",
            "movement_type": "dynamic",
            "duration_range": (1.0, 3.0),
            "key_landmarks": [0, 4, 8, 12, 16, 20],
            "common_variations": ["gracias", "thank you", "merci"]
        },
        
        "SÍ": {
            "description": "Afirmación",
            "complexity": "simple",
            "movement_type": "static",
            "duration_range": (0.3, 1.5),
            "key_landmarks": [0, 4, 8, 12, 16, 20],
            "common_variations": ["sí", "yes", "ok", "okay"]
        },
        
        "NO": {
            "description": "Negación",
            "complexity": "simple",
            "movement_type": "dynamic",
            "duration_range": (0.5, 2.0),
            "key_landmarks": [0, 4, 8, 12, 16, 20],
            "common_variations": ["no", "not", "nope"]
        },
        
        "POR_FAVOR": {
            "description": "Solicitud cortés",
            "complexity": "medium",
            "movement_type": "dynamic",
            "duration_range": (1.0, 2.5),
            "key_landmarks": [0, 4, 8, 12, 16, 20],
            "common_variations": ["por favor", "please", "favor"]
        },
        
        "PERDÓN": {
            "description": "Disculpa",
            "complexity": "medium",
            "movement_type": "dynamic",
            "duration_range": (1.0, 2.5),
            "key_landmarks": [0, 4, 8, 12, 16, 20],
            "common_variations": ["perdón", "sorry", "disculpa"]
        },
        
        "AYUDA": {
            "description": "Solicitud de ayuda",
            "complexity": "medium",
            "movement_type": "dynamic",
            "duration_range": (1.0, 3.0),
            "key_landmarks": [0, 4, 8, 12, 16, 20],
            "common_variations": ["ayuda", "help", "socorro"]
        },
        
        "AGUA": {
            "description": "Solicitud de agua",
            "complexity": "simple",
            "movement_type": "static",
            "duration_range": (0.5, 2.0),
            "key_landmarks": [0, 4, 8, 12, 16, 20],
            "common_variations": ["agua", "water", "beber"]
        },
        
        "COMIDA": {
            "description": "Solicitud de comida",
            "complexity": "simple",
            "movement_type": "static",
            "duration_range": (0.5, 2.0),
            "key_landmarks": [0, 4, 8, 12, 16, 20],
            "common_variations": ["comida", "food", "comer"]
        },
        
        "BAÑO": {
            "description": "Solicitud de ir al baño",
            "complexity": "simple",
            "movement_type": "static",
            "duration_range": (0.5, 2.0),
            "key_landmarks": [0, 4, 8, 12, 16, 20],
            "common_variations": ["baño", "bathroom", "wc"]
        }
    }
    
    # Letras del alfabeto manual costarricense
    ALPHABET_GESTURES = {
        "A": {
            "description": "Letra A",
            "complexity": "simple",
            "movement_type": "static",
            "duration_range": (0.3, 1.0),
            "key_landmarks": [0, 4, 8, 12, 16, 20],
            "hand_shape": "fist_with_thumb_up"
        },
        
        "B": {
            "description": "Letra B",
            "complexity": "simple",
            "movement_type": "static",
            "duration_range": (0.3, 1.0),
            "key_landmarks": [0, 4, 8, 12, 16, 20],
            "hand_shape": "all_fingers_extended"
        },
        
        "C": {
            "description": "Letra C",
            "complexity": "simple",
            "movement_type": "static",
            "duration_range": (0.3, 1.0),
            "key_landmarks": [0, 4, 8, 12, 16, 20],
            "hand_shape": "c_shape"
        },
        
        "D": {
            "description": "Letra D",
            "complexity": "simple",
            "movement_type": "static",
            "duration_range": (0.3, 1.0),
            "key_landmarks": [0, 4, 8, 12, 16, 20],
            "hand_shape": "index_finger_extended"
        },
        
        "E": {
            "description": "Letra E",
            "complexity": "simple",
            "movement_type": "static",
            "duration_range": (0.3, 1.0),
            "key_landmarks": [0, 4, 8, 12, 16, 20],
            "hand_shape": "fist_with_thumb_bent"
        }
    }
    
    @classmethod
    def get_gesture_info(cls, gesture_name: str) -> Dict[str, Any]:
        """
        Obtiene información de un gesto específico.
        
        Args:
            gesture_name: Nombre del gesto
            
        Returns:
            Diccionario con información del gesto o None si no existe
        """
        gesture_name_upper = gesture_name.upper()
        
        # Buscar en gestos básicos
        if gesture_name_upper in cls.BASIC_GESTURES:
            return cls.BASIC_GESTURES[gesture_name_upper]
        
        # Buscar en alfabeto
        if gesture_name_upper in cls.ALPHABET_GESTURES:
            return cls.ALPHABET_GESTURES[gesture_name_upper]
        
        return None
    
    @classmethod
    def get_all_gestures(cls) -> Dict[str, Dict[str, Any]]:
        """
        Obtiene todos los gestos disponibles.
        
        Returns:
            Diccionario con todos los gestos
        """
        all_gestures = {}
        all_gestures.update(cls.BASIC_GESTURES)
        all_gestures.update(cls.ALPHABET_GESTURES)
        return all_gestures
    
    @classmethod
    def get_gestures_by_complexity(cls, complexity: str) -> List[str]:
        """
        Obtiene gestos filtrados por complejidad.
        
        Args:
            complexity: Nivel de complejidad ('simple', 'medium', 'complex')
            
        Returns:
            Lista de nombres de gestos
        """
        gestures = []
        all_gestures = cls.get_all_gestures()
        
        for name, info in all_gestures.items():
            if info.get('complexity') == complexity:
                gestures.append(name)
        
        return gestures
    
    @classmethod
    def get_gestures_by_movement_type(cls, movement_type: str) -> List[str]:
        """
        Obtiene gestos filtrados por tipo de movimiento.
        
        Args:
            movement_type: Tipo de movimiento ('static', 'dynamic')
            
        Returns:
            Lista de nombres de gestos
        """
        gestures = []
        all_gestures = cls.get_all_gestures()
        
        for name, info in all_gestures.items():
            if info.get('movement_type') == movement_type:
                gestures.append(name)
        
        return gestures
    
    @classmethod
    def get_training_recommendations(cls, gesture_name: str) -> Dict[str, Any]:
        """
        Obtiene recomendaciones para entrenar un gesto específico.
        
        Args:
            gesture_name: Nombre del gesto
            
        Returns:
            Diccionario con recomendaciones de entrenamiento
        """
        gesture_info = cls.get_gesture_info(gesture_name)
        
        if not gesture_info:
            return {
                'error': f'Gesto "{gesture_name}" no encontrado'
            }
        
        recommendations = {
            'gesture_name': gesture_name,
            'samples_needed': 5,  # Por defecto
            'training_tips': [],
            'common_mistakes': [],
            'validation_criteria': {}
        }
        
        # Ajustar número de muestras según complejidad
        complexity = gesture_info.get('complexity', 'simple')
        if complexity == 'simple':
            recommendations['samples_needed'] = 3
        elif complexity == 'medium':
            recommendations['samples_needed'] = 5
        else:  # complex
            recommendations['samples_needed'] = 7
        
        # Consejos específicos según el tipo de movimiento
        movement_type = gesture_info.get('movement_type', 'static')
        if movement_type == 'static':
            recommendations['training_tips'].extend([
                "Mantén la mano en una posición fija",
                "Asegúrate de que todos los dedos estén en la posición correcta",
                "No muevas la mano durante el gesto"
            ])
        else:  # dynamic
            recommendations['training_tips'].extend([
                "Realiza el movimiento completo del gesto",
                "Mantén una velocidad constante",
                "Termina el gesto en la posición final correcta"
            ])
        
        # Criterios de validación
        duration_range = gesture_info.get('duration_range', (0.5, 2.0))
        recommendations['validation_criteria'] = {
            'min_duration': duration_range[0],
            'max_duration': duration_range[1],
            'min_frames': 10,
            'max_frames': 120
        }
        
        return recommendations
