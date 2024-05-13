# ergotool
Rula hip and albow assesment

Este script de Python utiliza la solución de pose de MediaPipe para estimar y analizar posturas humanas a través de videos, enfocándose en la evaluación del riesgo ergonómico según el método RULA (Rapid Upper Limb Assessment). Calcula ángulos específicos de las articulaciones del cuerpo (codo y cadera) para determinar posturas que podrían representar riesgos ergonómicos.
Funciones

    Detección de Pose: Utiliza MediaPipe Pose para detectar puntos de referencia del cuerpo en videos.
    Cálculo de Ángulos: Calcula los ángulos del codo y la cadera usando la ley de cosenos, lo cual ayuda en la evaluación ergonómica.
    Conteo de Eventos: Registra la cantidad de veces que ciertas posturas de riesgo ocurren, basadas en los umbrales de ángulos definidos.
    Visualización en Tiempo Real: Muestra el video procesado con anotaciones visuales de ángulos y líneas de articulaciones.
    Estadísticas de Riesgo: Calcula y visualiza el tiempo pasado en posturas de riesgo como porcentajes del tiempo total del video.

Limitaciones

    Dependencia de la Calidad del Video: La precisión de la detección de pose depende significativamente de la calidad del video y la iluminación.
    Posible Sesgo de Ángulo: Los cálculos de ángulo pueden estar sujetos a errores si los puntos de referencia detectados no son precisos.
    Rendimiento en Tiempo Real: Dependiendo de la resolución del video y la capacidad del hardware, el procesamiento en tiempo real puede experimentar latencia.

Estado del Proyecto

Este proyecto está actualmente en desarrollo y se actualizará continuamente para mejorar la precisión y la eficiencia del análisis de pose. Las contribuciones y sugerencias son bienvenidas.
Licencia

Este trabajo está basado en investigaciones y desarrollos de omes-va.com y está licenciado bajo Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0). Puedes usar y modificar este código para fines no comerciales, siempre que des crédito a los autores originales y fuentes.
