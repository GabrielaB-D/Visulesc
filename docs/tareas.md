# Plan de Construcción - Sistema de Reconocimiento LESCO

## 📋 Checklist de Desarrollo Paso a Paso

### 🎯 Objetivo General
Construir un sistema completo de reconocimiento de gestos de Lengua de Señas Costarricense (LESCO) que funcione en tiempo real, sea modular y permita agregar nuevos gestos sin reescribir código.

---

## 📁 FASE 1: CONFIGURACIÓN INICIAL Y ESTRUCTURA BASE

### 1.1 Configuración del Entorno de Desarrollo
- [ ] **Instalar Python 3.12** (versión específica requerida por MediaPipe)
- [ ] **Crear entorno virtual** con `python -m venv venv`
- [ ] **Activar entorno virtual** (`venv\Scripts\activate` en Windows)
- [ ] **Crear archivo requirements.txt** con dependencias específicas
- [ ] **Instalar dependencias** con `pip install -r requirements.txt`
- [ ] **Verificar instalación** de MediaPipe, OpenCV y scikit-learn
- [ ] **Crear archivo .gitignore** para proyecto Python + MediaPipe

### 1.2 Estructura de Directorios
- [ ] **Crear directorio raíz** del proyecto (`Visulesc/`)
- [ ] **Crear directorio src/** para código fuente
- [ ] **Crear directorio src/hand_tracking/** para detección de landmarks
- [ ] **Crear directorio src/gesture_recognition/** para sistema de gestos
- [ ] **Crear directorio src/ui/** para interfaces de usuario
- [ ] **Crear directorio src/data/** para gestión de datos
- [ ] **Crear directorio src/utils/** para utilidades
- [ ] **Crear directorio src/config/** para configuración
- [ ] **Crear directorio data/gestures/** para gestos entrenados
- [ ] **Crear directorio data/users/** para perfiles de usuarios
- [ ] **Crear directorio assets/models/** para modelos MediaPipe
- [ ] **Crear directorio docs/** para documentación
- [ ] **Crear directorio tests/** para pruebas unitarias

### 1.3 Archivos de Configuración Base
- [ ] **Crear src/config/settings.py** con parámetros configurables
- [ ] **Definir constantes** de resolución (TARGET_WIDTH, TARGET_HEIGHT)
- [ ] **Definir parámetros** de detección (PAUSE_THRESHOLD, MIN_GESTURE_FRAMES)
- [ ] **Definir parámetros** de clasificación (MIN_CONFIDENCE, SAMPLES_PER_GESTURE)
- [ ] **Crear src/config/__init__.py** para importaciones
- [ ] **Crear archivos __init__.py** en todos los directorios de módulos

---

## 🖐️ FASE 2: SISTEMA DE DETECCIÓN DE LANDMARKS

### 2.1 Detector de Landmarks Optimizado
- [ ] **Crear src/hand_tracking/detector.py** con clase HandDetector
- [ ] **Implementar inicialización** con parámetros de MediaPipe
- [ ] **Configurar BaseOptions** con modelo `hand_landmarker.task`
- [ ] **Configurar HandLandmarkerOptions** con parámetros optimizados
- [ ] **Implementar detección asíncrona** con LIVE_STREAM mode
- [ ] **Añadir callback** para procesamiento de resultados
- [ ] **Implementar método close()** para liberar recursos
- [ ] **Añadir manejo de errores** para casos sin detección

### 2.2 Visualizador de Landmarks
- [ ] **Crear src/hand_tracking/visualizer.py** con clase HandVisualizer
- [ ] **Implementar draw_landmarks()** para renderizar landmarks
- [ ] **Implementar draw_landmark_points()** para puntos individuales
- [ ] **Implementar draw_connections()** para conexiones entre landmarks
- [ ] **Configurar colores** y tamaños desde settings.py
- [ ] **Manejar múltiples manos** simultáneamente
- [ ] **Optimizar renderizado** para tiempo real

### 2.3 Utilidades de Rendimiento
- [ ] **Crear src/utils/fps_counter.py** con clase FPSCounter
- [ ] **Implementar cálculo de FPS** cada N frames
- [ ] **Añadir método update()** para actualización continua
- [ ] **Implementar get_fps()** para obtener FPS actual
- [ ] **Configurar intervalo** desde settings.py
- [ ] **Crear src/utils/__init__.py** para importaciones

---

## 🎭 FASE 3: SISTEMA DE RECONOCIMIENTO DE GESTOS

### 3.1 Detector de Gestos con Pausas
- [ ] **Crear src/gesture_recognition/gesture_detector.py**
- [ ] **Implementar clase GestureDetector** con parámetros configurables
- [ ] **Implementar process_landmarks()** para análisis de movimiento
- [ ] **Implementar _calculate_movement()** para detectar actividad
- [ ] **Implementar _start_gesture_recording()** para iniciar grabación
- [ ] **Implementar _continue_gesture_recording()** para continuar grabación
- [ ] **Implementar _end_gesture_recording()** para finalizar gesto
- [ ] **Añadir callbacks** on_gesture_start, on_gesture_end
- [ ] **Implementar detección de pausas** con umbral configurable
- [ ] **Añadir validación** de frames mínimos/máximos

### 3.2 Clasificador Híbrido ML + DTW
- [ ] **Crear src/gesture_recognition/gesture_classifier.py**
- [ ] **Implementar clase GestureClassifier** con Random Forest
- [ ] **Implementar _extract_features()** para características estáticas
- [ ] **Implementar _extract_static_features()** para landmarks clave
- [ ] **Implementar _extract_dynamic_features()** para movimiento
- [ ] **Implementar _calculate_velocity()** para velocidad promedio
- [ ] **Implementar _calculate_movement_directions()** para direcciones
- [ ] **Implementar add_gesture_sample()** para añadir muestras
- [ ] **Implementar train_classifier()** para entrenar modelo
- [ ] **Implementar classify_gesture()** para clasificar gestos
- [ ] **Añadir persistencia** con joblib para modelos
- [ ] **Implementar validación cruzada** para evaluación

### 3.3 Entrenador de Gestos Guiado
- [ ] **Crear src/gesture_recognition/gesture_trainer.py**
- [ ] **Implementar clase GestureTrainer** con interfaz guiada
- [ ] **Implementar start_training_session()** para iniciar entrenamiento
- [ ] **Implementar process_frame()** para procesar durante entrenamiento
- [ ] **Implementar _on_gesture_start()** callback para inicio
- [ ] **Implementar _on_gesture_end()** callback para finalización
- [ ] **Implementar _complete_training()** para completar sesión
- [ ] **Añadir validación** de muestras suficientes
- [ ] **Implementar cancel_training()** para cancelar sesión
- [ ] **Añadir callbacks** para UI (on_training_complete, on_training_error)

### 3.4 Gestos Base de LESCO
- [ ] **Crear src/gesture_recognition/lesco_gestures.py**
- [ ] **Definir BASIC_GESTURES** con gestos comunes (HOLA, GRACIAS, SÍ, NO)
- [ ] **Definir ALPHABET_GESTURES** con letras básicas (A, B, C, D, E)
- [ ] **Implementar get_gesture_info()** para obtener información
- [ ] **Implementar get_all_gestures()** para listar todos
- [ ] **Implementar get_gestures_by_complexity()** para filtrar
- [ ] **Implementar get_training_recommendations()** para consejos
- [ ] **Añadir metadatos** de complejidad y tipo de movimiento
- [ ] **Crear src/gesture_recognition/__init__.py** para importaciones

---

## 🖥️ FASE 4: INTERFACES DE USUARIO

### 4.1 Ventana Principal OpenCV
- [ ] **Crear src/ui/main_window.py** con clase MainWindow
- [ ] **Implementar inicialización** con parámetros de ventana
- [ ] **Implementar start()** para iniciar ventana
- [ ] **Implementar update_frame()** para actualizar display
- [ ] **Implementar _create_window_image()** para composición
- [ ] **Implementar _draw_info_panel()** para panel lateral
- [ ] **Implementar _draw_status_bar()** para barra de estado
- [ ] **Implementar _draw_mode_indicator()** para indicador de modo
- [ ] **Implementar _handle_keyboard()** para controles de teclado
- [ ] **Implementar _mouse_callback()** para eventos de mouse
- [ ] **Añadir métodos** set_recognized_text(), clear_text()
- [ ] **Implementar close()** para cerrar ventana

### 4.2 Interfaz de Entrenamiento
- [ ] **Crear src/ui/training_ui.py** con clase TrainingUI
- [ ] **Implementar start_training_session()** para iniciar sesión
- [ ] **Implementar process_frame()** para procesar durante entrenamiento
- [ ] **Implementar _create_training_frame()** para frame de entrenamiento
- [ ] **Implementar _draw_training_overlay()** para overlay de información
- [ ] **Implementar _draw_instructions()** para instrucciones visuales
- [ ] **Implementar _get_state_color()** para colores de estado
- [ ] **Implementar handle_key()** para controles de teclado
- [ ] **Implementar cancel_training()** para cancelar
- [ ] **Implementar complete_training()** para completar
- [ ] **Añadir callbacks** para comunicación con sistema principal

### 4.3 Panel de Visualización de Texto
- [ ] **Crear src/ui/text_display.py** con clase TextDisplay
- [ ] **Implementar add_text()** para añadir texto reconocido
- [ ] **Implementar set_text()** para establecer texto
- [ ] **Implementar clear_text()** para limpiar texto
- [ ] **Implementar draw_text_panel()** para renderizar panel
- [ ] **Implementar _draw_wrapped_text()** para texto con wrap
- [ ] **Implementar get_current_text()** para obtener texto actual
- [ ] **Implementar get_history()** para obtener historial
- [ ] **Implementar export_text()** para exportar a archivo
- [ ] **Implementar get_statistics()** para estadísticas
- [ ] **Añadir callbacks** on_text_update, on_text_clear
- [ ] **Crear src/ui/__init__.py** para importaciones

---

## 💾 FASE 5: GESTIÓN DE DATOS Y PERSISTENCIA

### 5.1 Almacenamiento de Gestos
- [ ] **Crear src/data/gesture_storage.py** con clase GestureStorage
- [ ] **Implementar inicialización** con directorio de datos
- [ ] **Implementar save_gesture()** para guardar gestos en JSON
- [ ] **Implementar load_gestures()** para cargar gestos de usuario
- [ ] **Implementar delete_gesture()** para eliminar gestos
- [ ] **Implementar get_user_gestures()** para listar gestos disponibles
- [ ] **Implementar get_gesture_statistics()** para estadísticas
- [ ] **Añadir validación** de formato JSON
- [ ] **Implementar manejo de errores** para archivos corruptos

### 5.2 Gestión de Usuarios
- [ ] **Crear src/data/user_manager.py** con clase UserManager
- [ ] **Implementar inicialización** con archivo de usuarios
- [ ] **Implementar _initialize_users_file()** para usuario por defecto
- [ ] **Implementar create_user()** para crear nuevos usuarios
- [ ] **Implementar load_users()** para cargar todos los usuarios
- [ ] **Implementar get_user()** para obtener usuario específico
- [ ] **Implementar update_user_activity()** para actualizar actividad
- [ ] **Implementar set_current_user()** para cambiar usuario actual
- [ ] **Implementar get_user_preferences()** para obtener preferencias
- [ ] **Implementar update_user_preferences()** para actualizar preferencias
- [ ] **Crear src/data/__init__.py** para importaciones

---

## 🔗 FASE 6: INTEGRACIÓN Y SISTEMA PRINCIPAL

### 6.1 Sistema Principal Integrado
- [ ] **Actualizar main.py** con clase LESCORecognitionSystem
- [ ] **Implementar inicialización** de todos los componentes
- [ ] **Implementar _setup_callbacks()** para conectar componentes
- [ ] **Implementar start()** para iniciar sistema completo
- [ ] **Implementar run()** para bucle principal
- [ ] **Implementar _process_frame()** para procesamiento completo
- [ ] **Implementar _process_recognition_mode()** para modo reconocimiento
- [ ] **Implementar _process_training_mode()** para modo entrenamiento
- [ ] **Implementar _on_gesture_detected()** callback para gestos detectados
- [ ] **Implementar _on_mode_change()** callback para cambio de modo
- [ ] **Implementar _on_key_press()** callback para teclas
- [ ] **Implementar _start_new_gesture_training()** para nuevo gesto
- [ ] **Implementar _list_available_gestures()** para listar gestos
- [ ] **Implementar cleanup()** para liberar recursos

### 6.2 Integración de Componentes
- [ ] **Conectar HandDetector** con GestureDetector
- [ ] **Conectar GestureDetector** con GestureClassifier
- [ ] **Conectar GestureTrainer** con TrainingUI
- [ ] **Conectar MainWindow** con TextDisplay
- [ ] **Conectar UserManager** con GestureStorage
- [ ] **Configurar callbacks** entre todos los componentes
- [ ] **Implementar manejo de errores** global
- [ ] **Añadir logging** para debugging

---

## 🧪 FASE 7: TESTING Y VALIDACIÓN

### 7.1 Tests Unitarios
- [ ] **Crear tests/test_basic.py** con tests básicos
- [ ] **Implementar TestFPSCounter** para contador de FPS
- [ ] **Implementar TestConfiguration** para configuración
- [ ] **Crear tests/test_gesture_detector.py** para detector
- [ ] **Crear tests/test_gesture_classifier.py** para clasificador
- [ ] **Crear tests/test_gesture_trainer.py** para entrenador
- [ ] **Crear tests/test_ui_components.py** para interfaces
- [ ] **Crear tests/test_data_management.py** para datos
- [ ] **Implementar tests de integración** para sistema completo

### 7.2 Validación del Sistema
- [ ] **Probar detección de landmarks** con diferentes usuarios
- [ ] **Probar entrenamiento de gestos** con gestos básicos LESCO
- [ ] **Probar reconocimiento** con gestos entrenados
- [ ] **Probar interfaz de usuario** con todos los controles
- [ ] **Probar persistencia** de datos y usuarios
- [ ] **Probar rendimiento** (FPS, latencia, memoria)
- [ ] **Probar casos edge** (sin manos, iluminación pobre, etc.)

---

## 📚 FASE 8: DOCUMENTACIÓN Y FINALIZACIÓN

### 8.1 Documentación Técnica
- [ ] **Actualizar README.md** con instrucciones completas
- [ ] **Completar docs/especificacion.md** con detalles técnicos
- [ ] **Crear docs/instalacion.md** con guía de instalación
- [ ] **Crear docs/uso.md** con guía de usuario
- [ ] **Crear docs/desarrollo.md** con guía para desarrolladores
- [ ] **Añadir docstrings** a todas las funciones y clases
- [ ] **Crear ejemplos** de uso del sistema

### 8.2 Optimización Final
- [ ] **Optimizar parámetros** de configuración
- [ ] **Ajustar umbrales** de detección y clasificación
- [ ] **Mejorar rendimiento** de algoritmos críticos
- [ ] **Reducir uso de memoria** donde sea posible
- [ ] **Optimizar visualización** para mejor experiencia
- [ ] **Añadir validaciones** adicionales de entrada

### 8.3 Preparación para Producción
- [ ] **Crear script de instalación** automatizado
- [ ] **Configurar logging** para producción
- [ ] **Añadir manejo de errores** robusto
- [ ] **Crear archivos de configuración** de ejemplo
- [ ] **Preparar datos de ejemplo** con gestos LESCO básicos
- [ ] **Crear guía de troubleshooting** para problemas comunes

---

## ✅ CRITERIOS DE COMPLETACIÓN

### Funcionalidades Obligatorias
- [ ] **Detección en tiempo real** de landmarks funciona correctamente
- [ ] **Reconocimiento de gestos** con precisión > 75%
- [ ] **Entrenamiento guiado** permite agregar nuevos gestos
- [ ] **Interfaz de usuario** muestra texto reconocido
- [ ] **Sistema multi-usuario** funciona con perfiles separados
- [ ] **Persistencia JSON** guarda y carga datos correctamente

### Restricciones Respetadas
- [ ] **Python 3.12** utilizado sin modificación
- [ ] **hand_landmarker.task** usado sin alteración
- [ ] **main.py** mantiene compatibilidad
- [ ] **Arquitectura modular** permite extensión
- [ ] **Funcionamiento en tiempo real** garantizado

### Calidad del Código
- [ ] **PEP 8** seguido en todo el código
- [ ] **Docstrings** completos en funciones y clases
- [ ] **Type hints** añadidos donde sea apropiado
- [ ] **Manejo de errores** implementado
- [ ] **Tests unitarios** pasan correctamente

---

## 🚀 ORDEN DE EJECUCIÓN RECOMENDADO

1. **Completar FASE 1** completamente antes de continuar
2. **Completar FASE 2** para tener detección básica funcionando
3. **Completar FASE 3** para tener reconocimiento básico
4. **Completar FASE 4** para tener interfaces funcionales
5. **Completar FASE 5** para tener persistencia de datos
6. **Completar FASE 6** para tener sistema integrado
7. **Completar FASE 7** para validar funcionamiento
8. **Completar FASE 8** para documentación final

---

## 📋 NOTAS IMPORTANTES PARA DESARROLLADORES

### Restricciones Críticas
- **NO CAMBIAR** la versión de Python (debe ser 3.12)
- **NO MODIFICAR** el archivo `hand_landmarker.task`
- **MANTENER** compatibilidad con `main.py` existente
- **GARANTIZAR** funcionamiento en tiempo real

### Buenas Prácticas
- **Usar verbos de acción** en nombres de métodos
- **Implementar manejo de errores** en todas las funciones críticas
- **Añadir docstrings** a todas las funciones públicas
- **Seguir PEP 8** en todo el código
- **Probar cada componente** antes de integrar

### Orden de Desarrollo
- **Una fase a la vez** - no saltar entre fases
- **Completar todos los items** de una fase antes de continuar
- **Probar funcionalidad** después de cada componente
- **Documentar cambios** importantes

---

*Plan de construcción v1.0 - Sistema LESCO Visulesc*
*Creado basado en especificación técnica del proyecto*
