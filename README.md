# Visulesc - Sistema de Reconocimiento de Lenguaje de Señas LESCO

Un sistema completo y avanzado para el reconocimiento de gestos de Lengua de Señas Costarricense (LESCO) utilizando MediaPipe y Machine Learning.

## 🎯 Características Principales

### ✅ Sistema Completo Implementado
- **Detección Multi-mano**: Detecta hasta 2 manos simultáneamente
- **Reconocimiento de Gestos Complejos**: Palabras completas de LESCO con pausas
- **Sistema Híbrido de Clasificación**: Combina ML (Random Forest) y DTW para máxima precisión
- **Entrenamiento Guiado**: Interfaz paso a paso para agregar nuevos gestos
- **Sistema Multi-usuario**: Soporte para múltiples usuarios con perfiles personalizados
- **Interfaz Intuitiva**: Ventana OpenCV con paneles informativos y controles

### 🔧 Funcionalidades Técnicas
- **Detección con Pausas**: Sistema híbrido que detecta gestos y pausas automáticamente
- **Precisión Media-Alta**: Objetivo de 75-90% de precisión
- **Persistencia JSON**: Almacenamiento eficiente de gestos y configuraciones
- **Arquitectura Modular**: Código organizado y extensible
- **Monitoreo en Tiempo Real**: FPS, estado de detección y métricas de rendimiento

## 📋 Requisitos del Sistema

### Software
- **Python**: 3.12 (requerido por MediaPipe)
- **OpenCV**: 4.8.1.78
- **MediaPipe**: 0.10.7
- **scikit-learn**: 1.3.2
- **NumPy**: 1.24.3

### Hardware
- **CPU**: Procesador x86/x64 moderno
- **RAM**: 4GB mínimo, 8GB recomendado
- **Cámara**: Webcam USB compatible con OpenCV
- **Sistema Operativo**: Windows 10/11, Linux, macOS

## 🚀 Instalación y Configuración

### 1. Clonar el Repositorio
```bash
git clone <url-del-repositorio>
cd Visulesc
```

### 2. Crear Entorno Virtual
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 3. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 4. Verificar Instalación
```bash
python main.py
```

## 🎮 Uso del Sistema

### Ejecución Básica
```bash
python main.py
```

### Controles Principales
- **T**: Modo entrenamiento
- **R**: Modo reconocimiento
- **S**: Configuración
- **N**: Nuevo gesto (en modo entrenamiento)
- **L**: Listar gestos disponibles
- **C**: Cancelar entrenamiento
- **ESC**: Salir
- **SPACE**: Limpiar texto

### Flujo de Trabajo Típico

#### 1. Entrenar Nuevos Gestos
1. Presiona **T** para activar modo entrenamiento
2. Presiona **N** para crear un nuevo gesto
3. Ingresa el nombre del gesto (ej: "HOLA")
4. Sigue las instrucciones en pantalla
5. Realiza el gesto varias veces según se indique
6. El sistema entrena automáticamente el clasificador

#### 2. Reconocer Gestos
1. Presiona **R** para activar modo reconocimiento
2. Realiza gestos entrenados
3. El texto reconocido aparece en tiempo real
4. Presiona **SPACE** para limpiar el texto

## 📁 Estructura del Proyecto

```
Visulesc/
├── src/                           # Código fuente modular
│   ├── gesture_recognition/       # Sistema de reconocimiento
│   │   ├── gesture_detector.py   # Detección con pausas
│   │   ├── gesture_classifier.py # Clasificación híbrida ML+DTW
│   │   ├── gesture_trainer.py    # Entrenamiento guiado
│   │   └── lesco_gestures.py     # Gestos base de LESCO
│   ├── ui/                        # Interfaz de usuario
│   │   ├── main_window.py         # Ventana principal OpenCV
│   │   ├── training_ui.py         # Interfaz de entrenamiento
│   │   └── text_display.py        # Panel de texto
│   ├── data/                      # Gestión de datos
│   │   ├── gesture_storage.py     # Almacenamiento JSON
│   │   └── user_manager.py        # Gestión de usuarios
│   ├── hand_tracking/             # Detección de landmarks
│   │   ├── detector.py           # Detector optimizado
│   │   └── visualizer.py         # Visualización
│   ├── utils/                     # Utilidades
│   │   └── fps_counter.py         # Medición de rendimiento
│   └── config/                    # Configuración
│       └── settings.py            # Parámetros del sistema
├── data/                          # Datos del sistema
│   ├── gestures/                  # Gestos entrenados (JSON)
│   └── users/                     # Perfiles de usuarios
├── docs/                          # Documentación
│   ├── especificacion.md          # Especificación técnica
│   └── tareas.md                 # Plan de desarrollo
├── tests/                         # Tests unitarios
├── assets/                        # Recursos
│   └── models/                    # Modelos MediaPipe
├── main.py                       # Aplicación principal
├── requirements.txt              # Dependencias
└── README.md                    # Este archivo
```

## ⚙️ Configuración Avanzada

### Parámetros del Sistema (`src/config/settings.py`)

```python
# Detección de gestos
PAUSE_THRESHOLD = 0.3              # Tiempo para considerar pausa
MIN_GESTURE_FRAMES = 10            # Mínimo frames por gesto
MAX_GESTURE_FRAMES = 120           # Máximo frames por gesto
MOVEMENT_THRESHOLD = 0.02          # Umbral de movimiento

# Clasificación
MIN_CONFIDENCE = 0.5              # Confianza mínima para reconocimiento
SAMPLES_PER_GESTURE = 5           # Muestras por gesto en entrenamiento

# Rendimiento
TARGET_WIDTH = 256                # Resolución de procesamiento
TARGET_HEIGHT = 192
SEND_EVERY_N_FRAMES = 2           # Procesamiento cada N frames
```

### Gestos Base de LESCO

El sistema incluye gestos comunes de LESCO:

**Gestos Básicos:**
- HOLA, GRACIAS, SÍ, NO
- POR_FAVOR, PERDÓN, AYUDA
- AGUA, COMIDA, BAÑO

**Alfabeto Manual:**
- A, B, C, D, E (expandible)

## 🔬 Arquitectura Técnica

### Sistema de Detección
1. **Captura**: OpenCV captura frames de la cámara
2. **Landmarks**: MediaPipe detecta landmarks de manos
3. **Detección**: Sistema detecta gestos vs pausas
4. **Clasificación**: ML + DTW clasifica el gesto
5. **Visualización**: Renderizado en tiempo real

### Flujo de Entrenamiento
1. **Inicio**: Usuario inicia sesión de entrenamiento
2. **Grabación**: Sistema graba múltiples muestras
3. **Extracción**: Características estáticas y dinámicas
4. **Entrenamiento**: Random Forest con validación
5. **Persistencia**: Guardado en JSON para reutilización

### Gestión de Datos
- **JSON**: Formato ligero y portable
- **Multi-usuario**: Perfiles separados por usuario
- **Versionado**: Timestamps y metadatos
- **Backup**: Fácil respaldo y restauración

## 📊 Rendimiento y Optimización

### Métricas Objetivo
- **FPS**: Mínimo 25 FPS en hardware estándar
- **Latencia**: Máximo 50ms desde captura hasta reconocimiento
- **Precisión**: 75-90% en gestos entrenados
- **Memoria**: Uso eficiente de RAM

### Optimizaciones Implementadas
- Redimensionamiento de frames para reducir carga
- Procesamiento asíncrono con callbacks
- Buffer reducido para menor latencia
- Procesamiento selectivo de frames
- Algoritmos eficientes de extracción de características

## 🐛 Solución de Problemas

### Problemas Comunes

**Error de cámara no encontrada**
```bash
# Verificar dispositivos disponibles
python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).isOpened()])"
```

**FPS bajo**
- Reduce `TARGET_WIDTH` y `TARGET_HEIGHT` en settings.py
- Aumenta `SEND_EVERY_N_FRAMES`
- Verifica iluminación del entorno

**Detección imprecisa**
- Mejora la iluminación
- Ajusta `MOVEMENT_THRESHOLD` en settings.py
- Verifica que las manos estén completamente visibles

**Error de entrenamiento**
- Verifica que hay suficientes muestras (mínimo 3)
- Asegúrate de que los gestos sean consistentes
- Revisa los logs de error en consola

### Logs y Debugging

Para habilitar logging detallado:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🚀 Próximas Mejoras

### Funcionalidades Planificadas
- [ ] Reconocimiento de emociones en gestos
- [ ] Traducción automática LESCO ↔ Español
- [ ] Interfaz web para configuración remota
- [ ] Exportación de conversaciones
- [ ] Integración con aplicaciones externas
- [ ] Reconocimiento de gestos con ambas manos

### Mejoras Técnicas
- [ ] Optimización de algoritmos de clasificación
- [ ] Soporte para múltiples cámaras
- [ ] Procesamiento en GPU
- [ ] Algoritmos de aprendizaje continuo
- [ ] Validación cruzada automática

## 🤝 Contribución

### Cómo Contribuir
1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -am 'Añadir nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

### Estándares de Desarrollo
- Seguir PEP 8 para estilo de código
- Añadir docstrings a funciones y clases
- Incluir tests para nuevas funcionalidades
- Actualizar documentación según sea necesario
- Usar type hints para mejor legibilidad

### Áreas de Contribución
- **Gestos de LESCO**: Añadir nuevos gestos al sistema base
- **Algoritmos**: Mejorar precisión de clasificación
- **UI/UX**: Mejorar la interfaz de usuario
- **Documentación**: Traducir o mejorar documentación
- **Testing**: Añadir tests unitarios e integración

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## 🙏 Agradecimientos

- **MediaPipe Team**: Por la biblioteca de detección de landmarks
- **OpenCV Community**: Por el procesamiento de imágenes
- **Comunidad LESCO**: Por el conocimiento del lenguaje de señas costarricense
- **Python Community**: Por las herramientas de desarrollo

## 📞 Soporte y Contacto

### Reportar Problemas
- Abre un [Issue](https://github.com/tu-usuario/Visulesc/issues) en GitHub
- Incluye información del sistema y pasos para reproducir

### Solicitar Funcionalidades
- Usa el template de Feature Request
- Describe el caso de uso y beneficios

### Comunidad
- Únete a nuestro Discord/Slack
- Participa en discusiones sobre LESCO y tecnología

---

**Visulesc** - Conectando el mundo a través de las manos 🖐️

*Desarrollado con ❤️ para la comunidad LESCO de Costa Rica*