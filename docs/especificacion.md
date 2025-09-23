# Especificación del Proyecto - Sistema de Reconocimiento de Lenguaje de Señas LESCO

## 1. Descripción General

Este proyecto implementa un sistema completo de reconocimiento de gestos de Lengua de Señas Costarricense (LESCO) utilizando MediaPipe y técnicas de Machine Learning. El sistema está diseñado para funcionar en tiempo real, detectar gestos complejos con pausas, y proporcionar una interfaz intuitiva para entrenar nuevos gestos y mostrar texto reconocido.

## 2. Objetivos del Proyecto

### Objetivo Principal
Desarrollar un sistema robusto y eficiente para el reconocimiento de gestos de LESCO en tiempo real, con capacidades de entrenamiento guiado y gestión multi-usuario.

### Objetivos Específicos
- Reconocer gestos complejos de LESCO con palabras completas
- Detectar gestos con pausas automáticas entre movimientos
- Proporcionar interfaz de entrenamiento guiado para nuevos gestos
- Mantener compatibilidad con el archivo `main.py` existente
- Crear una arquitectura modular que permita agregar gestos sin reescribir el sistema
- Alcanzar precisión media-alta (75-90%) en el reconocimiento

## 3. Especificaciones Técnicas

### 3.1 Requisitos de Hardware
- **CPU**: Procesador x86/x64 compatible
- **RAM**: Mínimo 4GB, recomendado 8GB
- **Cámara**: Webcam USB compatible con OpenCV
- **Sistema Operativo**: Windows 10/11, Linux, macOS

### 3.2 Requisitos de Software
- **Python**: 3.12 (requerido por MediaPipe, NO MODIFICABLE)
- **OpenCV**: 4.8.1.78
- **MediaPipe**: 0.10.7
- **scikit-learn**: 1.3.2 (para clasificación ML)
- **NumPy**: 1.24.3
- **joblib**: 1.3.2 (para persistencia de modelos)

### 3.3 Archivos Críticos del Sistema
- **hand_landmarker.task**: Modelo MediaPipe (NO MODIFICAR más allá de su uso)
- **main.py**: Punto de entrada principal (MANTENER COMPATIBILIDAD)
- **requirements.txt**: Dependencias específicas con versiones

### 3.4 Parámetros de Configuración

#### Resolución de Procesamiento
- **Resolución objetivo**: 256x192 píxeles
- **Justificación**: Balance entre precisión y rendimiento en tiempo real

#### Detección de Gestos LESCO
- **Número máximo de manos**: 2
- **Umbral de pausa**: 0.3 segundos
- **Frames mínimos por gesto**: 10
- **Frames máximos por gesto**: 120
- **Umbral de movimiento**: 0.02

#### Clasificación de Gestos
- **Confianza mínima**: 0.5
- **Muestras por gesto**: 5 (entrenamiento)
- **Algoritmo**: Random Forest + DTW (híbrido)
- **Precisión objetivo**: 75-90%

#### Optimización de Rendimiento
- **Procesamiento de frames**: Cada 2 frames (SEND_EVERY_N = 2)
- **Buffer de captura**: Reducido a 1 frame
- **Modo de ejecución**: LIVE_STREAM para procesamiento asíncrono

## 4. Arquitectura del Sistema

### 4.1 Componentes Principales

```
src/
├── gesture_recognition/          # Sistema de reconocimiento LESCO
│   ├── gesture_detector.py       # Detección de gestos con pausas
│   ├── gesture_classifier.py     # Clasificación híbrida ML + DTW
│   ├── gesture_trainer.py        # Entrenamiento guiado paso a paso
│   ├── lesco_gestures.py         # Gestos base de LESCO
│   └── __init__.py
├── ui/                           # Interfaz de usuario completa
│   ├── main_window.py            # Ventana principal OpenCV
│   ├── training_ui.py            # Interfaz de entrenamiento guiado
│   ├── text_display.py           # Panel de visualización de texto
│   └── __init__.py
├── data/                         # Gestión de datos y persistencia
│   ├── gesture_storage.py        # Almacenamiento JSON de gestos
│   ├── user_manager.py           # Gestión multi-usuario
│   └── __init__.py
├── hand_tracking/                # Detección optimizada de landmarks
│   ├── detector.py               # Detector MediaPipe optimizado
│   ├── visualizer.py             # Visualización de landmarks
│   └── __init__.py
├── utils/                        # Utilidades del sistema
│   ├── fps_counter.py            # Medición de rendimiento
│   └── __init__.py
├── config/                       # Configuración centralizada
│   ├── settings.py               # Parámetros configurables
│   └── __init__.py
└── main.py                       # Sistema principal integrado
```

### 4.2 Funciones Principales del Sistema

#### 4.2.1 Detección de Landmarks en Tiempo Real
- **Componente**: `hand_tracking/detector.py`
- **Tecnología**: MediaPipe con modelo `hand_landmarker.task`
- **Funcionalidad**: 
  - Detección de hasta 2 manos simultáneamente
  - Procesamiento asíncrono para baja latencia
  - Optimización de resolución (256x192)
  - Callbacks para procesamiento en tiempo real

#### 4.2.2 Reconocimiento de Gestos de Señas LESCO
- **Componente**: `gesture_recognition/gesture_classifier.py`
- **Tecnología**: Random Forest + DTW (Dynamic Time Warping)
- **Funcionalidad**:
  - Clasificación híbrida para gestos estáticos y dinámicos
  - Extracción de características estáticas y dinámicas
  - Precisión objetivo 75-90%
  - Soporte para gestos complejos con pausas

#### 4.2.3 Interfaz de Administración de Gestos
- **Componente**: `gesture_recognition/gesture_trainer.py` + `ui/training_ui.py`
- **Funcionalidad**:
  - Entrenamiento guiado paso a paso
  - Validación automática de gestos
  - Gestión de múltiples muestras por gesto
  - Interfaz visual con instrucciones en tiempo real
  - Sistema multi-usuario con perfiles separados

#### 4.2.4 Interfaz de Usuario para Mostrar Texto
- **Componente**: `ui/main_window.py` + `ui/text_display.py`
- **Tecnología**: OpenCV para interfaz de escritorio
- **Funcionalidad**:
  - Ventana principal con paneles informativos
  - Visualización de texto reconocido en tiempo real
  - Historial de gestos reconocidos
  - Indicadores de estado y progreso
  - Controles de teclado para navegación

### 4.3 Flujo de Procesamiento

1. **Captura de Video**: OpenCV captura frames de la cámara
2. **Detección de Landmarks**: MediaPipe procesa landmarks usando `hand_landmarker.task`
3. **Detección de Gestos**: Sistema detecta inicio/fin de gestos con pausas
4. **Clasificación**: Algoritmo híbrido clasifica el gesto detectado
5. **Visualización**: Renderizado de landmarks y texto reconocido
6. **Persistencia**: Almacenamiento de gestos en formato JSON

## 5. Especificaciones de Rendimiento

### 5.1 Métricas Objetivo
- **FPS**: Mínimo 25 FPS en hardware estándar
- **Latencia**: Máximo 50ms desde captura hasta reconocimiento
- **Precisión**: 75-90% en gestos entrenados
- **Memoria**: Uso eficiente de RAM (< 2GB)

### 5.2 Optimizaciones Implementadas
- Redimensionamiento de frames para reducir carga computacional
- Procesamiento asíncrono con callbacks
- Buffer reducido para menor latencia
- Procesamiento selectivo de frames (cada 2 frames)
- Algoritmos eficientes de extracción de características
- Persistencia JSON para acceso rápido a datos

## 6. Reglas y Restricciones del Sistema

### 6.1 Reglas Fundamentales

#### 6.1.1 Funcionamiento en Tiempo Real
- **REQUERIDO**: El sistema debe procesar y reconocer gestos en tiempo real
- **IMPLEMENTACIÓN**: 
  - Procesamiento asíncrono con MediaPipe LIVE_STREAM
  - Detección de gestos con pausas automáticas
  - Clasificación inmediata al completar gestos
  - Visualización en tiempo real de resultados

#### 6.1.2 Arquitectura Modular
- **REQUERIDO**: Sistema modular que permita agregar gestos sin reescribir código
- **IMPLEMENTACIÓN**:
  - Separación clara de responsabilidades en módulos
  - Sistema de plugins para nuevos gestos
  - Configuración centralizada en `src/config/settings.py`
  - Interfaz abstracta para clasificadores

#### 6.1.3 Agregar Gestos Sin Reescritura
- **REQUERIDO**: Capacidad de entrenar nuevos gestos sin modificar código base
- **IMPLEMENTACIÓN**:
  - Sistema de entrenamiento guiado (`gesture_trainer.py`)
  - Almacenamiento dinámico en JSON
  - Carga automática de gestos al iniciar
  - Interfaz de administración integrada

### 6.2 Restricciones Técnicas

#### 6.2.1 Versión de Python
- **RESTRICCIÓN**: NO CAMBIAR versión de Python (debe ser 3.12)
- **JUSTIFICACIÓN**: MediaPipe requiere Python 3.12 específicamente
- **IMPLEMENTACIÓN**: 
  - `requirements.txt` especifica Python 3.12
  - Documentación clara sobre esta restricción
  - Validación de versión en `main.py`

#### 6.2.2 Archivo hand_landmarker.task
- **RESTRICCIÓN**: NO MODIFICAR el archivo `hand_landmarker.task` más allá de su uso
- **JUSTIFICACIÓN**: Es el modelo pre-entrenado de MediaPipe
- **IMPLEMENTACIÓN**:
  - Archivo se usa únicamente como referencia en `MODEL_PATH`
  - No se modifica el contenido del archivo
  - Se mantiene en `assets/models/` para organización

#### 6.2.3 Compatibilidad con main.py
- **RESTRICCIÓN**: MANTENER compatibilidad con el archivo `main.py` existente
- **JUSTIFICACIÓN**: Preservar funcionalidad base del sistema
- **IMPLEMENTACIÓN**:
  - `main.py` actúa como punto de entrada principal
  - Integración con módulos existentes de `hand_tracking`
  - Mantenimiento de la estructura de callbacks
  - Preservación de la interfaz de usuario básica

## 7. Casos de Uso y Aplicaciones

### 7.1 Aplicaciones Principales
- **Comunicación LESCO**: Traducción de gestos a texto en tiempo real
- **Educación**: Herramienta de aprendizaje para estudiantes de LESCO
- **Accesibilidad**: Interfaz de comunicación para personas sordas
- **Investigación**: Análisis de patrones de movimiento en LESCO
- **Desarrollo**: Base para aplicaciones más complejas de reconocimiento

### 7.2 Escenarios de Uso Específicos

#### 7.2.1 Entrenamiento de Nuevos Gestos
1. Usuario activa modo entrenamiento (tecla T)
2. Sistema solicita nombre del gesto
3. Usuario realiza el gesto múltiples veces
4. Sistema valida y almacena las muestras
5. Clasificador se entrena automáticamente
6. Gesto queda disponible para reconocimiento

#### 7.2.2 Reconocimiento en Tiempo Real
1. Usuario activa modo reconocimiento (tecla R)
2. Usuario realiza gestos entrenados
3. Sistema detecta gestos con pausas automáticas
4. Clasificador identifica el gesto
5. Texto aparece en pantalla inmediatamente
6. Historial se mantiene para referencia

#### 7.2.3 Gestión Multi-usuario
1. Sistema mantiene perfiles separados por usuario
2. Cada usuario tiene sus propios gestos entrenados
3. Configuraciones personalizables por usuario
4. Datos almacenados en formato JSON portable

## 8. Tecnologías Utilizadas

### 8.1 Tecnologías Principales

#### 8.1.1 Python 3.12
- **Propósito**: Lenguaje principal del sistema
- **Restricción**: Versión fija requerida por MediaPipe
- **Uso**: Desarrollo de toda la aplicación

#### 8.1.2 MediaPipe
- **Propósito**: Detección de landmarks de manos
- **Archivo**: `hand_landmarker.task` (modelo pre-entrenado)
- **Uso**: Procesamiento de video en tiempo real
- **Restricción**: No modificar el archivo del modelo

#### 8.1.3 OpenCV
- **Propósito**: Captura de video y interfaz de usuario
- **Versión**: 4.8.1.78
- **Uso**: 
  - Captura de frames de cámara
  - Interfaz de ventana principal
  - Visualización de landmarks
  - Renderizado de texto

#### 8.1.4 scikit-learn
- **Propósito**: Algoritmos de Machine Learning
- **Versión**: 1.3.2
- **Uso**: 
  - Random Forest para clasificación
  - Extracción de características
  - Validación cruzada

### 8.2 Tecnologías de Soporte

#### 8.2.1 NumPy
- **Propósito**: Operaciones matemáticas y arrays
- **Versión**: 1.24.3
- **Uso**: Procesamiento de datos de landmarks

#### 8.2.2 joblib
- **Propósito**: Persistencia de modelos ML
- **Versión**: 1.3.2
- **Uso**: Guardado y carga de clasificadores entrenados

#### 8.2.3 JSON
- **Propósito**: Almacenamiento de datos
- **Uso**: 
  - Gestos entrenados
  - Configuraciones de usuario
  - Metadatos del sistema

## 9. Limitaciones y Consideraciones

### 9.1 Limitaciones Técnicas
- **Dependencia de iluminación**: Requiere iluminación adecuada para detección precisa
- **Visibilidad de manos**: Manos deben estar completamente visibles en el frame
- **Variaciones entre usuarios**: Cada usuario debe entrenar sus propios gestos
- **Rendimiento**: Depende del hardware disponible (CPU principalmente)

### 9.2 Limitaciones de LESCO
- **Complejidad de gestos**: Algunos gestos complejos pueden requerir múltiples entrenamientos
- **Variaciones regionales**: LESCO puede tener variaciones según la región
- **Contexto**: Algunos gestos pueden requerir contexto adicional para interpretación

### 9.3 Consideraciones de Usabilidad
- **Curva de aprendizaje**: Usuarios necesitan aprender el sistema de entrenamiento
- **Calibración**: Puede requerir ajuste de parámetros según el entorno
- **Consistencia**: Usuarios deben mantener consistencia en sus gestos

## 10. Plan de Desarrollo y Mantenimiento

### 10.1 Fases Completadas ✅
1. **Fase 1**: Estructura base y detección de landmarks
2. **Fase 2**: Sistema de reconocimiento de gestos
3. **Fase 3**: Interfaz de entrenamiento guiado
4. **Fase 4**: Sistema multi-usuario y persistencia
5. **Fase 5**: Integración completa y optimización

### 10.2 Mantenimiento Continuo
- **Monitoreo de rendimiento**: Verificación de FPS y latencia
- **Actualización de gestos**: Añadir nuevos gestos según necesidades
- **Optimización**: Mejoras continuas en algoritmos
- **Documentación**: Mantenimiento de documentación actualizada

### 10.3 Próximas Mejoras Planificadas
- Reconocimiento de gestos con ambas manos
- Integración con sistemas de traducción
- Interfaz web para configuración remota
- Algoritmos de aprendizaje continuo

## 11. Estándares de Código y Documentación

### 11.1 Convenciones de Desarrollo
- **PEP 8**: Estilo de código Python estándar
- **Docstrings**: Documentación completa de funciones y clases
- **Type hints**: Anotaciones de tipos para mejor legibilidad
- **Modularización**: Separación clara de responsabilidades
- **Naming**: Convenciones consistentes para variables y funciones

### 11.2 Estructura de Archivos
- **Código fuente**: Organizado en `src/` por módulos especializados
- **Documentación**: Mantenida en `docs/` con especificaciones técnicas
- **Tests**: Implementados en `tests/` para validación
- **Recursos**: Almacenados en `assets/` (modelos, imágenes)
- **Configuración**: Centralizada en `src/config/`

### 11.3 Gestión de Versiones
- **Control de versiones**: Git con ramas para features
- **Documentación**: Actualizada con cada cambio significativo
- **Releases**: Etiquetado semántico de versiones
- **Changelog**: Registro de cambios y mejoras

---

## Resumen Ejecutivo

El **Sistema de Reconocimiento de Lenguaje de Señas LESCO** es una aplicación completa desarrollada en Python 3.12 que utiliza MediaPipe y técnicas de Machine Learning para reconocer gestos de LESCO en tiempo real. 

### Características Clave:
- ✅ **Detección en tiempo real** de landmarks usando `hand_landmarker.task`
- ✅ **Reconocimiento de gestos complejos** con sistema híbrido ML + DTW
- ✅ **Interfaz de entrenamiento guiado** para agregar nuevos gestos
- ✅ **Sistema multi-usuario** con persistencia JSON
- ✅ **Arquitectura modular** que permite extensión sin reescritura
- ✅ **Compatibilidad mantenida** con `main.py` existente

### Tecnologías Principales:
- **Python 3.12** (versión fija requerida)
- **MediaPipe** con modelo `hand_landmarker.task`
- **OpenCV** para interfaz y captura de video
- **scikit-learn** para algoritmos de Machine Learning
- **JSON** para almacenamiento de datos

### Restricciones Respetadas:
- ✅ Python 3.12 no modificable
- ✅ `hand_landmarker.task` usado sin modificación
- ✅ Compatibilidad con `main.py` preservada
- ✅ Funcionamiento en tiempo real garantizado
- ✅ Arquitectura modular implementada

*Documento de especificación v2.0 - Sistema LESCO Visulesc*
