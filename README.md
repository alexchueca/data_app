# ⚽ Football Analytics Pro

**Aplicación web profesional de análisis futbolístico con Machine Learning**

🌐 **Demo en vivo:** [Despliega en Render siguiendo la guía]

---

## 📊 Descripción

Football Analytics Pro es una aplicación web interactiva desarrollada con Python y Dash que democratiza el acceso a herramientas de análisis profesional de fútbol para clubes amateur.

**Dataset:** 10,754 jugadores de 374 equipos con estadísticas completas.

---

## 🚀 Características

### Páginas de Análisis (4):
- **Dashboard** - 25 visualizaciones configurables con 4 gráficos independientes
- **Comparación** - Comparación inteligente entre jugadores (adapta métricas según posición)
- **Equipos** - Análisis individual y comparativo de equipos por posición
- **Rendimiento** - Análisis avanzado con 3 filtros y 22 visualizaciones (20,736 combinaciones)

### Machine Learning (4):
- **Valuación** - Predicción de valor de mercado con Random Forest
- **Estilos** - Clustering K-Means con 6 estilos de juego específicos
- **Gangas** - Detección de oportunidades con Isolation Forest (36 jugadores)
- **Similares** - Recomendación con Cosine Similarity optimizado (99.4% reducción)

---

## 🛠️ Tecnologías

- **Backend:** Python 3.11, Dash 2.18, Flask
- **Visualización:** Plotly 5.24, Dash Bootstrap Components
- **ML:** scikit-learn 1.5 (Random Forest, K-Means, Isolation Forest, Cosine Similarity)
- **Datos:** Pandas 2.2, NumPy 1.26
- **Deployment:** Gunicorn, Render

---

## 📦 Instalación Local

### 1. Clonar repositorio
```bash
git clone https://github.com/alexchueca/football-analytics-pro.git
cd football-analytics-pro
```

### 2. Crear entorno virtual
```bash
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Ejecutar aplicación
```bash
python app.py
```

### 5. Abrir en navegador
```
http://localhost:8050
```

---

## 🌐 Despliegue en Render

### Archivos necesarios (ya incluidos):
- ✅ `Procfile` - Comando de inicio con timeout
- ✅ `render.yaml` - Configuración automática
- ✅ `runtime.txt` - Python 3.11.9
- ✅ `requirements.txt` - Todas las dependencias

### Pasos para desplegar:

1. **Crear cuenta en Render**
   - Ve a https://render.com
   - Sign up con GitHub

2. **Conectar repositorio**
   - New Web Service
   - Connect GitHub repository
   - Selecciona: `alexchueca/football-analytics-pro`

3. **Configurar (automático con render.yaml)**
   - Render detecta `render.yaml`
   - Configura automáticamente todo
   - Click "Create Web Service"

4. **Esperar despliegue**
   - Primera vez: 5-10 minutos
   - Render construye e inicia automáticamente

5. **Acceder a URL**
   - Render proporciona: `https://football-analytics-pro.onrender.com`
   - Primera carga: ~30 segundos (plan Free)

**Plan Free:**
- Aplicación hiberna tras 15 min sin uso
- Primera carga después de hibernación: 30-60 seg
- Luego funciona normalmente

---

## 📊 Estadísticas del Proyecto

- **Jugadores:** 10,754
- **Equipos:** 374
- **Posiciones:** 16
- **Páginas:** 8 optimizadas
- **Modelos ML:** 4 algoritmos
- **Visualizaciones:** 47 tipos diferentes
- **Líneas de código:** ~1,800
- **Tamaño dataset:** 1.7MB
- **Tamaño modelos:** 28.6MB total

---

## 🎯 Casos de Uso

### 1. Director Deportivo - Buscar fichaje
**Objetivo:** Encontrar delantero joven y eficiente
```
Página: Rendimiento
Filtros:
- Equipo: Todos
- Posición: Delanteros
- Partidos: 20+
Analizar: Goles/partido, edad, valor, versatilidad
```

### 2. Entrenador - Preparar partido
**Objetivo:** Analizar rival
```
Página: Equipos > Comparación
Equipo 1: Mi equipo
Equipo 2: Rival
Ver: Mejores por posición lado a lado
```

### 3. Analista - Identificar estilos
**Objetivo:** Clasificar jugadores por estilo
```
Página: Estilos ML
Ver: Scatter plot con 6 clusters
Filtrar: Por estilo específico
```

---

## 📁 Estructura del Proyecto

```
football-analytics-pro/
├── app.py                              (1,817 líneas - aplicación principal)
├── requirements.txt                    (8 dependencias)
├── Procfile                            (gunicorn con timeout)
├── render.yaml                         (configuración Render)
├── runtime.txt                         (Python 3.11.9)
├── .gitignore                          (archivos ignorados)
│
├── assets/
│   └── styles.css                      (estilos personalizados)
│
└── data/
    ├── final_data.csv                  (10,754 jugadores, 1.7MB)
    ├── model_valuation.pkl             (23MB - Random Forest)
    ├── model_clustering.pkl            (742KB - K-Means mejorado)
    ├── model_anomaly.pkl               (1.1MB - Isolation Forest)
    ├── model_recommendation_optimized.pkl (3.8MB - Cosine Similarity)
    ├── scaler_valuation.pkl            (952B)
    └── features_valuation.pkl          (191B)
```

---

## 🔧 Optimizaciones Técnicas

### 1. Clustering Mejorado
- **Antes:** Agrupación por volumen de juego (titulares vs suplentes)
- **Ahora:** Agrupación por estilo real usando métricas por 90 minutos
- **Resultado:** 6 perfiles específicos (Goleadores, Delanteros Completos, etc.)

### 2. Recomendación Optimizada
- **Antes:** Matriz completa de similitud (548MB en memoria)
- **Ahora:** Top-50 precalculado por jugador (3.8MB en disco)
- **Reducción:** 99.4% manteniendo precisión completa

### 3. Comparación Inteligente
- Detecta automáticamente porteros vs jugadores de campo
- **Porteros:** Goles concedidos, porterías a cero
- **Campo:** Goles, asistencias, contribución

---

## 🔍 Verificación

Después de instalar, verifica que todo funciona:

```bash
# Ver estado de archivos
python verify_setup.py

# Verificar modelos ML
python verificar_modelos.py

# Diagnóstico completo
python diagnostico.py
```

**Salida esperada:**
```
📊 Cargando dataset histórico...
✅ 10754 jugadores - 374 equipos
✅ Predicción Valor
✅ Clustering
✅ Gangas
✅ Recomendación
Dash is running on http://0.0.0.0:8050/
```

---

## 📚 Documentación Adicional

- **DEPLOYMENT.md** - Guía detallada de despliegue
- **QUICKSTART.md** - Inicio rápido 5 minutos
- **PROJECT_SUMMARY.md** - Resumen técnico completo
- **PRESENTATION_GUIDE.md** - Guía para presentaciones académicas

---

## 🎓 Proyecto Académico

**Asignatura:** Desarrollo de Aplicaciones para Visualización de Datos (DAVD)  
**Profesor:** David Martín  
**Estudiante:** Alejandro Chueca Manzanero  
**Fecha:** Noviembre 2025  
**Versión:** 6.7 FINAL

---

## 🚀 Características Destacadas

- ✅ **Dataset Consolidado** - Un único dataset histórico para consistencia total
- ✅ **Clustering Inteligente** - Por estilo de juego, no por volumen
- ✅ **Optimización Extrema** - 99.4% reducción en modelo de recomendación
- ✅ **Adaptabilidad** - Métricas cambian según contexto (porteros vs campo)
- ✅ **Sin Redundancia** - 8 páginas únicas y potentes
- ✅ **20,000+ Combinaciones** - Análisis infinito en página Rendimiento
- ✅ **Producción Ready** - Sin errores, desplegable inmediatamente

---

## 📄 Licencia

Este proyecto ha sido desarrollado con fines académicos como parte de la asignatura DAVD.

---

## 🤝 Contacto

- **GitHub Proyecto:** https://github.com/alexchueca/football-analytics-pro
- **GitHub Estudiante:** [@alexchueca](https://github.com/alexchueca)

---

**⚽ Democratizando el análisis profesional de fútbol para clubes amateur**

**Versión:** 6.7 FINAL  
**Estado:** ✅ LISTO PARA PRODUCCIÓN  
**Páginas:** 8 optimizadas  
**Modelos ML:** 4 algoritmos  
**Dataset:** 10,754 jugadores
