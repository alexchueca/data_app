# ⚽ Football Analytics Pro

Aplicación web interactiva de análisis futbolístico desarrollada con Python, Dash y Machine Learning.

---

## 📊 Descripción

**Football Analytics Pro** democratiza el acceso a herramientas profesionales de análisis de fútbol para clubes amateur. Permite explorar estadísticas de 10,754 jugadores de 374 equipos mediante visualizaciones interactivas y algoritmos de Machine Learning.

**Objetivo:** Proporcionar a entrenadores y directores deportivos las mismas herramientas analíticas que tienen clubes profesionales, pero a costo accesible.

---

## ✨ Características Principales

### Análisis de Datos
- **Dashboard interactivo** con 25 tipos de visualizaciones configurables
- **Comparación inteligente** entre jugadores (adapta métricas según posición)
- **Análisis por equipos** con métricas detalladas por posición
- **Análisis de rendimiento** con filtros avanzados (20,000+ combinaciones)

### Machine Learning
- **Predicción de valor** de mercado con Random Forest
- **Clustering de estilos** de juego (6 perfiles identificados)
- **Detección de gangas** mediante Isolation Forest
- **Sistema de recomendación** con Cosine Similarity optimizado

---

## 🛠️ Tecnologías

- **Backend:** Python 3.11, Dash 2.18, Flask
- **Visualización:** Plotly 5.24, Dash Bootstrap Components
- **Machine Learning:** scikit-learn (Random Forest, K-Means, Isolation Forest, Cosine Similarity)
- **Datos:** Pandas, NumPy
- **Deployment:** Gunicorn, Render

---

## 📦 Dataset

- **10,754 jugadores** de las principales ligas europeas
- **374 equipos** incluidos
- **22 variables** estadísticas por jugador
- Fuentes: FBref, Transfermarkt, API-Football

---

## 🎯 Casos de Uso

### Director Deportivo
Buscar delanteros jóvenes eficientes filtrando por edad, goles/partido y valor de mercado.

### Entrenador
Preparar partidos analizando fortalezas y debilidades del rival por posición.

### Analista
Identificar patrones y estilos de juego mediante clustering de jugadores similares.

### Scout
Detectar oportunidades de fichaje (jugadores infravalorados) con el detector de gangas.

---

## 📁 Estructura
```
data_app/
├── app.py                  # Aplicación principal (1,817 líneas)
├── requirements.txt        # Dependencias Python
├── Procfile               # Configuración deployment
├── render.yaml            # Configuración Render
├── data/
│   ├── final_data.csv     # Dataset (10,754 jugadores)
│   └── *.pkl              # 4 modelos ML entrenados (28.6MB)
└── assets/
    └── styles.css         # Estilos personalizados
```

---

## 🔧 Optimizaciones Técnicas

### Clustering Mejorado
Rediseñado para clasificar por **estilo de juego** (usando métricas por 90 minutos) en lugar de por volumen de participación, resultando en 6 perfiles claramente diferenciados.

### Recomendación Optimizada
Reducción del 99.4% en tamaño del modelo (de 548MB a 3.8MB) mediante precálculo de top-50 similitudes por jugador, manteniendo precisión completa.

### Comparación Inteligente
Detecta automáticamente porteros vs jugadores de campo y adapta las métricas mostradas (goles concedidos vs goles marcados).

---

## 🎓 Proyecto Académico

**Asignatura:** Desarrollo de Aplicaciones para Visualización de Datos  
**Profesor:** David Martín  
**Estudiante:** Alejandro Chueca Manzanero  
**Universidad:** Universidad Pontificia Comillas
**Curso:** 2025-2026  

---

## 📊 Estadísticas del Proyecto

- **Líneas de código:** ~1,800
- **Páginas implementadas:** 8
- **Visualizaciones diferentes:** 47 tipos
- **Modelos ML:** 4 algoritmos
- **Tamaño total:** ~30MB (dataset + modelos)
- **Combinaciones de análisis:** 20,000+

---

## 📄 Licencia

Proyecto desarrollado con fines académicos como parte del Trabajo Final de la asignatura Desarrollo de Aplicaciones para Visualización de Datos.

---

## 🔗 Enlaces

- **Repositorio:** https://github.com/alexchueca/data_app
- **Aplicación desplegada:** [Disponible tras presentación]

---

**Democratizando el análisis profesional de fútbol para clubes amateur** ⚽
