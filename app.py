# Football Analytics Pro - Dataset Histórico Completo
# 9 Páginas Funcionales

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import pickle

# ==================== INICIALIZACIÓN ====================
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        'https://use.fontawesome.com/releases/v6.0.0/css/all.css'
    ],
    suppress_callback_exceptions=True
)

server = app.server
app.title = "Football Analytics Pro"

COLORS = {
    'primary': '#1e3a8a',
    'secondary': '#10b981',
    'accent': '#f59e0b',
    'danger': '#ef4444',
    'dark': '#1f2937'
}

# ==================== CARGAR DATOS ====================
print("📊 Cargando dataset histórico...")
df = pd.read_csv('data/final_data.csv')
df.columns = df.columns.str.strip()
df = df.copy()  # Evitar ChainedAssignment warnings

# Calcular métricas
df['goles_totales'] = (df['goals'] * df['appearance']).round()
df['asistencias_totales'] = (df['assists'] * df['appearance']).round()
df['contribucion_total'] = df['goles_totales'] + df['asistencias_totales']
df['goles_por_partido'] = (df['goles_totales'] / df['appearance']).fillna(0).replace([np.inf, -np.inf], 0)
df['asistencias_por_partido'] = (df['asistencias_totales'] / df['appearance']).fillna(0).replace([np.inf, -np.inf], 0)
df['minutos_por_partido'] = (df['minutes played'] / df['appearance']).fillna(0).replace([np.inf, -np.inf], 0)

print(f"✅ {len(df)} jugadores - {df['team'].nunique()} equipos")

# ==================== CARGAR MODELOS ML ====================
try:
    with open('data/model_valuation.pkl', 'rb') as f:
        ml_model = pickle.load(f)
    with open('data/scaler_valuation.pkl', 'rb') as f:
        ml_scaler = pickle.load(f)
    with open('data/features_valuation.pkl', 'rb') as f:
        ml_features = pickle.load(f)
    ML_ENABLED = True
    print("✅ Predicción Valor")
except:
    ML_ENABLED = False
    ml_model, ml_scaler, ml_features = None, None, None
    print("⚠️ Predicción no disponible")

try:
    with open('data/model_clustering.pkl', 'rb') as f:
        clustering_model = pickle.load(f)
    CLUSTERING_ENABLED = True
    print("✅ Clustering")
except:
    CLUSTERING_ENABLED = False
    clustering_model = None
    print("⚠️ Clustering no disponible")

try:
    with open('data/model_anomaly.pkl', 'rb') as f:
        anomaly_model = pickle.load(f)
    ANOMALY_ENABLED = True
    print("✅ Gangas")
except:
    ANOMALY_ENABLED = False
    anomaly_model = None
    print("⚠️ Gangas no disponible")

try:
    with open('data/model_recommendation_optimized.pkl', 'rb') as f:
        recommendation_model = pickle.load(f)
    RECOMMENDATION_ENABLED = True
    print("✅ Recomendación")
except:
    RECOMMENDATION_ENABLED = False
    recommendation_model = None
    print("⚠️ Recomendación no disponible")

# ==================== NAVBAR ====================
navbar = dbc.Navbar(
    dbc.Container([
        html.Div([
            html.I(className="fas fa-futbol fa-2x me-3", style={'color': COLORS['accent']}),
            html.Span("Football Analytics Pro", className="fs-4 fw-bold text-white")
        ], className="d-flex align-items-center mb-2"),
        dbc.Nav([
            dbc.NavItem(dbc.NavLink("Dashboard", href="/", className="text-white")),
            dbc.NavItem(dbc.NavLink("Comparar", href="/comparison", className="text-white")),
            dbc.NavItem(dbc.NavLink("Equipos", href="/teams", className="text-white")),
            dbc.NavItem(dbc.NavLink("Rendimiento", href="/performance", className="text-white")),
            dbc.NavItem(dbc.NavLink("💰 Valuación", href="/valuation", className="text-warning")),
            dbc.NavItem(dbc.NavLink("🎨 Estilos", href="/clustering", className="text-info")),
            dbc.NavItem(dbc.NavLink("💎 Gangas", href="/bargains", className="text-success")),
            dbc.NavItem(dbc.NavLink("🔍 Similares", href="/recommend", className="text-primary")),
        ], navbar=True, className="ms-auto flex-wrap")
    ], fluid=True),
    color=COLORS['dark'],
    dark=True,
    className="mb-4"
)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    dbc.Container([html.Div(id='page-content')], fluid=True)
])

# ==================== PÁGINA 1: DASHBOARD ====================
def create_home():
    return html.Div([
        html.H1("⚽ Dashboard General", className="mb-4"),
        
        dbc.Row([
            dbc.Col([dbc.Card([dbc.CardBody([
                html.I(className="fas fa-users fa-2x mb-2", style={'color': COLORS['primary']}),
                html.H3(f"{len(df):,}"),
                html.P("Jugadores", className="text-muted mb-0")
            ])], className="text-center shadow-sm")], width=6, lg=3, className="mb-4"),
            
            dbc.Col([dbc.Card([dbc.CardBody([
                html.I(className="fas fa-shield-alt fa-2x mb-2", style={'color': COLORS['secondary']}),
                html.H3(f"{df['team'].nunique()}"),
                html.P("Equipos", className="text-muted mb-0")
            ])], className="text-center shadow-sm")], width=6, lg=3, className="mb-4"),
            
            dbc.Col([dbc.Card([dbc.CardBody([
                html.I(className="fas fa-futbol fa-2x mb-2", style={'color': COLORS['accent']}),
                html.H3(f"{int(df['goles_totales'].sum()):,}"),
                html.P("Goles Totales", className="text-muted mb-0")
            ])], className="text-center shadow-sm")], width=6, lg=3, className="mb-4"),
            
            dbc.Col([dbc.Card([dbc.CardBody([
                html.I(className="fas fa-hands-helping fa-2x mb-2", style={'color': COLORS['danger']}),
                html.H3(f"{int(df['asistencias_totales'].sum()):,}"),
                html.P("Asistencias Totales", className="text-muted mb-0")
            ])], className="text-center shadow-sm")], width=6, lg=3, className="mb-4"),
        ]),
        
        dbc.Row([
            dbc.Col([
                html.Label("Mínimo Partidos:"),
                dcc.Slider(id='home-matches', min=0, max=50, step=5, value=10,
                          marks={0:'0', 10:'10', 20:'20', 30:'30', 40:'40', 50:'50+'},
                          tooltip={"placement": "bottom", "always_visible": True})
            ], width=12, className="mb-4"),
        ]),
        
        # Gráfico 1 (Top izquierda)
        dbc.Row([
            dbc.Col([dbc.Card([
                dbc.CardHeader([
                    dbc.Row([
                        dbc.Col(html.H5("Gráfico 1", className="mb-0"), width="auto"),
                        dbc.Col([
                            dcc.Dropdown(
                                id='graph1-type',
                                options=[
                                    {'label': '⚽ Top Goleadores', 'value': 'top_scorers'},
                                    {'label': '🎯 Top Asistentes', 'value': 'top_assisters'},
                                    {'label': '⭐ Top Contribución', 'value': 'top_contribution'},
                                    {'label': '🏃 Top Minutos', 'value': 'top_minutes'},
                                    {'label': '⚡ Goles por Partido', 'value': 'goals_per_game'},
                                    {'label': '💰 Top Valiosos', 'value': 'top_value'},
                                    {'label': '🔥 Top Partidos', 'value': 'top_appearances'},
                                ],
                                value='top_scorers',
                                clearable=False,
                                style={'minWidth': '200px'}
                            )
                        ], width=True)
                    ], align="center")
                ]),
                dbc.CardBody([dcc.Graph(id='home-graph1')])
            ], className="shadow-sm")], width=12, lg=6, className="mb-4"),
            
            # Gráfico 2 (Top derecha)
            dbc.Col([dbc.Card([
                dbc.CardHeader([
                    dbc.Row([
                        dbc.Col(html.H5("Gráfico 2", className="mb-0"), width="auto"),
                        dbc.Col([
                            dcc.Dropdown(
                                id='graph2-type',
                                options=[
                                    {'label': '⚽ Top Goleadores', 'value': 'top_scorers'},
                                    {'label': '🎯 Top Asistentes', 'value': 'top_assisters'},
                                    {'label': '⭐ Top Contribución', 'value': 'top_contribution'},
                                    {'label': '🏃 Top Minutos', 'value': 'top_minutes'},
                                    {'label': '⚡ Goles por Partido', 'value': 'goals_per_game'},
                                    {'label': '💰 Top Valiosos', 'value': 'top_value'},
                                    {'label': '🔥 Top Partidos', 'value': 'top_appearances'},
                                ],
                                value='top_assisters',
                                clearable=False,
                                style={'minWidth': '200px'}
                            )
                        ], width=True)
                    ], align="center")
                ]),
                dbc.CardBody([dcc.Graph(id='home-graph2')])
            ], className="shadow-sm")], width=12, lg=6, className="mb-4"),
        ]),
        
        # Gráfico 3 (Bottom izquierda)
        dbc.Row([
            dbc.Col([dbc.Card([
                dbc.CardHeader([
                    dbc.Row([
                        dbc.Col(html.H5("Gráfico 3", className="mb-0"), width="auto"),
                        dbc.Col([
                            dcc.Dropdown(
                                id='graph3-type',
                                options=[
                                    {'label': '📊 Distribución Posiciones', 'value': 'positions'},
                                    {'label': '🏆 Distribución Equipos (Top 15)', 'value': 'teams'},
                                    {'label': '🎂 Distribución Edades', 'value': 'ages'},
                                    {'label': '💵 Distribución Valores', 'value': 'values'},
                                    {'label': '📈 Goles por Posición', 'value': 'goals_position'},
                                    {'label': '🎯 Asistencias por Posición', 'value': 'assists_position'},
                                ],
                                value='positions',
                                clearable=False,
                                style={'minWidth': '250px'}
                            )
                        ], width=True)
                    ], align="center")
                ]),
                dbc.CardBody([dcc.Graph(id='home-graph3')])
            ], className="shadow-sm")], width=12, lg=6, className="mb-4"),
            
            # Gráfico 4 (Bottom derecha)
            dbc.Col([dbc.Card([
                dbc.CardHeader([
                    dbc.Row([
                        dbc.Col(html.H5("Gráfico 4", className="mb-0"), width="auto"),
                        dbc.Col([
                            dcc.Dropdown(
                                id='graph4-type',
                                options=[
                                    {'label': '⚽ Goles vs Asistencias', 'value': 'goals_assists'},
                                    {'label': '💰 Edad vs Valor', 'value': 'age_value'},
                                    {'label': '🏃 Partidos vs Minutos', 'value': 'games_minutes'},
                                    {'label': '⚡ Goles vs Minutos', 'value': 'goals_minutes'},
                                    {'label': '🎯 Asistencias vs Minutos', 'value': 'assists_minutes'},
                                    {'label': '📊 Valor vs Contribución', 'value': 'value_contribution'},
                                ],
                                value='goals_assists',
                                clearable=False,
                                style={'minWidth': '250px'}
                            )
                        ], width=True)
                    ], align="center")
                ]),
                dbc.CardBody([dcc.Graph(id='home-graph4')])
            ], className="shadow-sm")], width=12, lg=6, className="mb-4"),
        ])
    ])

# ==================== PÁGINA 2: COMPARAR ====================
def create_comparison():
    options = [{'label': f"{row['name']} ({row['team']}) - {row['position']}", 'value': idx} 
               for idx, row in df.iterrows()]
    
    return html.Div([
        html.H1("⚖️ Comparar Jugadores", className="mb-4"),
        dbc.Row([
            dbc.Col([
                html.Label("Jugador 1:", className="fw-bold"),
                dcc.Dropdown(id='compare-p1', options=options, placeholder="Seleccionar...")
            ], width=12, md=6, className="mb-4"),
            dbc.Col([
                html.Label("Jugador 2:", className="fw-bold"),
                dcc.Dropdown(id='compare-p2', options=options, placeholder="Seleccionar...")
            ], width=12, md=6, className="mb-4"),
        ]),
        html.Div(id='compare-content')
    ])

# ==================== PÁGINA 3: ANÁLISIS POR EQUIPOS ====================
def create_teams():
    return html.Div([
        html.H1("🏟️ Análisis por Equipos", className="mb-4"),
        
        # Sección 1: Análisis Individual de Equipo
        dbc.Card([
            dbc.CardHeader(html.H4("📊 Análisis Individual", className="mb-0")),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Selecciona Equipo:", className="fw-bold"),
                        dcc.Dropdown(
                            id='teams-dropdown',
                            options=[{'label': team, 'value': team} for team in sorted(df['team'].unique())],
                            value=df['team'].value_counts().index[0],
                            clearable=False
                        )
                    ], width=12, className="mb-4")
                ]),
                html.Div(id='teams-content')
            ])
        ], className="mb-4 shadow-sm"),
        
        # Sección 2: Comparación de Equipos
        dbc.Card([
            dbc.CardHeader(html.H4("⚖️ Comparación de Equipos", className="mb-0")),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Equipo 1:", className="fw-bold"),
                        dcc.Dropdown(
                            id='team-compare-1',
                            options=[{'label': team, 'value': team} for team in sorted(df['team'].unique())],
                            value=df['team'].value_counts().index[0] if len(df['team'].unique()) > 0 else None,
                            clearable=False
                        )
                    ], width=12, md=6, className="mb-3"),
                    dbc.Col([
                        html.Label("Equipo 2:", className="fw-bold"),
                        dcc.Dropdown(
                            id='team-compare-2',
                            options=[{'label': team, 'value': team} for team in sorted(df['team'].unique())],
                            value=df['team'].value_counts().index[1] if len(df['team'].unique()) > 1 else None,
                            clearable=False
                        )
                    ], width=12, md=6, className="mb-3"),
                ]),
                html.Div(id='teams-comparison-content')
            ])
        ], className="shadow-sm")
    ])


# ==================== PÁGINA 4: ANÁLISIS DE RENDIMIENTO ====================
def create_performance():
    # Opciones de posición categorizadas
    position_options = [
        {'label': '🌍 Todas las Posiciones', 'value': 'all'},
        {'label': '🧤 Porteros', 'value': 'Goalkeeper'},
        {'label': '🛡️ Defensores', 'value': 'Defender'},
        {'label': '🎯 Mediocentros', 'value': 'Midfield'},
        {'label': '⚽ Delanteros', 'value': 'Attack'}
    ]
    
    return html.Div([
        html.H1("📊 Análisis de Rendimiento", className="mb-4"),
        
        # Filtros avanzados
        dbc.Card([
            dbc.CardHeader(html.H5("🔍 Filtros", className="mb-0")),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Equipo:", className="fw-bold"),
                        dcc.Dropdown(
                            id='perf-team',
                            options=[{'label': '🌍 Todos los Equipos', 'value': 'all'}] + 
                                   [{'label': team, 'value': team} for team in sorted(df['team'].unique())],
                            value='all',
                            clearable=False
                        )
                    ], width=12, md=4, className="mb-3"),
                    
                    dbc.Col([
                        html.Label("Posición:", className="fw-bold"),
                        dcc.Dropdown(
                            id='perf-position',
                            options=position_options,
                            value='all',
                            clearable=False
                        )
                    ], width=12, md=4, className="mb-3"),
                    
                    dbc.Col([
                        html.Label("Mínimo Partidos:", className="fw-bold"),
                        dcc.Slider(
                            id='perf-matches',
                            min=5,
                            max=50,
                            step=5,
                            value=15,
                            marks={5:'5', 15:'15', 25:'25', 35:'35', 50:'50'},
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ], width=12, md=4, className="mb-3"),
                ])
            ])
        ], className="mb-4 shadow-sm"),
        
        # Gráficos configurables (2x2)
        dbc.Row([
            # Gráfico 1 (Top izquierda)
            dbc.Col([dbc.Card([
                dbc.CardHeader([
                    dbc.Row([
                        dbc.Col(html.H5("Gráfico 1", className="mb-0"), width="auto"),
                        dbc.Col([
                            dcc.Dropdown(
                                id='perf-graph1-type',
                                options=[],  # Se llenarán dinámicamente
                                value='goals_per_game',
                                clearable=False,
                                style={'minWidth': '200px'}
                            )
                        ], width=True)
                    ], align="center")
                ]),
                dbc.CardBody([dcc.Graph(id='perf-graph1')])
            ], className="shadow-sm")], width=12, lg=6, className="mb-4"),
            
            # Gráfico 2 (Top derecha)
            dbc.Col([dbc.Card([
                dbc.CardHeader([
                    dbc.Row([
                        dbc.Col(html.H5("Gráfico 2", className="mb-0"), width="auto"),
                        dbc.Col([
                            dcc.Dropdown(
                                id='perf-graph2-type',
                                options=[],  # Se llenarán dinámicamente
                                value='assists_per_game',
                                clearable=False,
                                style={'minWidth': '200px'}
                            )
                        ], width=True)
                    ], align="center")
                ]),
                dbc.CardBody([dcc.Graph(id='perf-graph2')])
            ], className="shadow-sm")], width=12, lg=6, className="mb-4"),
        ]),
        
        dbc.Row([
            # Gráfico 3 (Bottom izquierda)
            dbc.Col([dbc.Card([
                dbc.CardHeader([
                    dbc.Row([
                        dbc.Col(html.H5("Gráfico 3", className="mb-0"), width="auto"),
                        dbc.Col([
                            dcc.Dropdown(
                                id='perf-graph3-type',
                                options=[],  # Se llenarán dinámicamente
                                value='contribution',
                                clearable=False,
                                style={'minWidth': '200px'}
                            )
                        ], width=True)
                    ], align="center")
                ]),
                dbc.CardBody([dcc.Graph(id='perf-graph3')])
            ], className="shadow-sm")], width=12, lg=6, className="mb-4"),
            
            # Gráfico 4 (Bottom derecha)
            dbc.Col([dbc.Card([
                dbc.CardHeader([
                    dbc.Row([
                        dbc.Col(html.H5("Gráfico 4", className="mb-0"), width="auto"),
                        dbc.Col([
                            dcc.Dropdown(
                                id='perf-graph4-type',
                                options=[],  # Se llenarán dinámicamente
                                value='value_market',
                                clearable=False,
                                style={'minWidth': '200px'}
                            )
                        ], width=True)
                    ], align="center")
                ]),
                dbc.CardBody([dcc.Graph(id='perf-graph4')])
            ], className="shadow-sm")], width=12, lg=6, className="mb-4"),
        ])
    ])

# ==================== PÁGINA 6: VALUACIÓN ML ====================
def create_valuation():
    if not ML_ENABLED:
        return dbc.Alert("Modelo de predicción no disponible", color="warning")
    
    options = [{'label': f"{row['name']} ({row['team']}) - {row['position']}", 'value': idx} 
               for idx, row in df[df['current_value'] > 0].iterrows()]
    
    return html.Div([
        html.H1("💰 Predicción de Valor", className="mb-2"),
        html.P("Predice el valor de mercado basándose en estadísticas de rendimiento", className="lead mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Selecciona un Jugador")),
                    dbc.CardBody([
                        dcc.Dropdown(id='val-player', options=options, placeholder="Buscar..."),
                        html.Div(id='val-result', className="mt-4")
                    ])
                ], className="shadow-sm")
            ], width=12, lg=8, className="mb-4"),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Información del Modelo")),
                    dbc.CardBody([
                        html.P([
                            html.Strong("Algoritmo: "), "Random Forest", html.Br(),
                            html.Strong("Features: "), f"{len(ml_features)}", html.Br(),
                            html.Strong("Jugadores entrenamiento: "), "~9,000", html.Br(),
                        ])
                    ])
                ], className="shadow-sm")
            ], width=12, lg=4, className="mb-4"),
        ])
    ])

# ==================== PÁGINA 7: CLUSTERING ====================
def create_clustering():
    if not CLUSTERING_ENABLED:
        return dbc.Alert("Modelo de clustering no disponible", color="warning")
    
    data = clustering_model['data']
    labels = clustering_model['labels']
    
    return html.Div([
        html.H1("🎨 Estilos de Juego", className="mb-2"),
        html.P(f"{len(data)} jugadores agrupados en 6 estilos usando K-Means", className="lead mb-4"),
        
        dbc.Row([
            dbc.Col([dbc.Card([
                dbc.CardHeader(html.H5("Visualización 2D")),
                dbc.CardBody([dcc.Graph(id='cluster-scatter', style={'height': '600px'})])
            ], className="shadow-sm")], width=12, lg=8, className="mb-4"),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Filtros")),
                    dbc.CardBody([
                        html.Label("Estilo:", className="fw-bold"),
                        dcc.Dropdown(id='cluster-filter',
                            options=[{'label': 'Todos', 'value': 'all'}] + 
                                   [{'label': labels[i], 'value': i} for i in range(6)],
                            value='all', clearable=False, className="mb-3"),
                        html.Label("Valor Máximo (M€):", className="fw-bold"),
                        dcc.Slider(id='cluster-value', min=0, max=100, value=100,
                                  marks={0:'0', 25:'25', 50:'50', 75:'75', 100:'100+'},
                                  tooltip={"placement": "bottom", "always_visible": True})
                    ])
                ], className="shadow-sm mb-4"),
                dbc.Card([
                    dbc.CardHeader(html.H5("Estilos Identificados")),
                    dbc.CardBody([html.Div([
                        html.Div([html.Strong(labels[i]), html.Br(), 
                                 html.Small(f"{len(data[data['cluster']==i])} jugadores", className="text-muted")],
                                className="mb-3") for i in range(6)
                    ])])
                ], className="shadow-sm")
            ], width=12, lg=4)
        ]),
        
        dbc.Row([dbc.Col([dbc.Card([
            dbc.CardHeader(html.H5("Top 30 Jugadores por Valor")),
            dbc.CardBody([html.Div(id='cluster-table')])
        ], className="shadow-sm")], width=12)])
    ])

# ==================== PÁGINA 8: GANGAS ====================
def create_bargains():
    if not ANOMALY_ENABLED:
        return dbc.Alert("Modelo de gangas no disponible", color="warning")
    
    gangas = anomaly_model['gangas']
    
    return html.Div([
        html.H1("💎 Gangas del Mercado", className="mb-2"),
        html.P(f"{len(gangas)} jugadores infravalorados detectados", className="lead mb-4"),
        
        dbc.Row([
            dbc.Col([dbc.Card([dbc.CardBody([
                html.I(className="fas fa-gem fa-2x mb-2", style={'color': COLORS['accent']}),
                html.H3(f"{len(gangas)}"),
                html.P("Gangas Detectadas", className="text-muted mb-0")
            ])], className="text-center shadow-sm")], width=4, className="mb-4"),
            
            dbc.Col([dbc.Card([dbc.CardBody([
                html.I(className="fas fa-chart-line fa-2x mb-2", style={'color': COLORS['secondary']}),
                html.H3(f"€{gangas['current_value'].median()/1e6:.1f}M"),
                html.P("Precio Mediano", className="text-muted mb-0")
            ])], className="text-center shadow-sm")], width=4, className="mb-4"),
            
            dbc.Col([dbc.Card([dbc.CardBody([
                html.I(className="fas fa-star fa-2x mb-2", style={'color': COLORS['primary']}),
                html.H3(f"{gangas['ratio'].mean():.0f}x"),
                html.P("Ratio Promedio", className="text-muted mb-0")
            ])], className="text-center shadow-sm")], width=4, className="mb-4"),
        ]),
        
        dbc.Row([
            dbc.Col([dbc.Card([
                dbc.CardHeader(html.H5("Top 30 Mejores Gangas")),
                dbc.CardBody([dcc.Graph(id='bargains-chart', style={'height': '700px'})])
            ], className="shadow-sm")], width=12, lg=8, className="mb-4"),
            
            dbc.Col([dbc.Card([
                dbc.CardHeader(html.H5("Cómo Funciona")),
                dbc.CardBody([
                    html.P([
                        html.Strong("Algoritmo: "), "Isolation Forest", html.Br(),
                        html.Strong("Método: "), "Detecta outliers positivos", html.Br(),
                        html.Strong("Ratio: "), "Rendimiento / Precio", html.Br(),
                    ]),
                    html.Hr(),
                    html.Small([
                        html.Strong("Fórmula Rendimiento:"), html.Br(),
                        "(Goles × 3) + (Asist × 2) + (Partidos × 0.5)"
                    ], className="text-muted")
                ])
            ], className="shadow-sm")], width=12, lg=4, className="mb-4"),
        ])
    ])

# ==================== PÁGINA 9: RECOMENDACIÓN ====================
def create_recommend():
    if not RECOMMENDATION_ENABLED:
        return dbc.Alert("Modelo de recomendación no disponible", color="warning")
    
    players = recommendation_model['players_data']
    options = [{'label': f"{row['name']} ({row['team']}) - {row['position']}", 'value': idx} 
               for idx, row in players.iterrows()]
    
    return html.Div([
        html.H1("🔍 Recomendador", className="mb-2"),
        html.P(f"Encuentra jugadores similares entre {len(players)} opciones", className="lead mb-4"),
        
        dbc.Row([dbc.Col([dbc.Card([
            dbc.CardHeader(html.H5("🎯 Selecciona un Jugador")),
            dbc.CardBody([
                dcc.Dropdown(id='rec-player', options=options, 
                            placeholder="Buscar por nombre o equipo...", className="mb-3"),
                dbc.Button("🔍 Buscar Similares", id='rec-btn', 
                          color="primary", size="lg", className="w-100")
            ])
        ], className="shadow-sm")], width=12, className="mb-4")]),
        
        html.Div(id='rec-results')
    ])

# ==================== CALLBACKS ====================

@app.callback(Output('page-content', 'children'), Input('url', 'pathname'))
def display_page(path):
    if path == '/comparison': return create_comparison()
    elif path == '/teams': return create_teams()
    elif path == '/performance': return create_performance()
    elif path == '/valuation': return create_valuation()
    elif path == '/clustering': return create_clustering()
    elif path == '/bargains': return create_bargains()
    elif path == '/recommend': return create_recommend()
    else: return create_home()

@app.callback(
    [Output('home-graph1', 'figure'), Output('home-graph2', 'figure'),
     Output('home-graph3', 'figure'), Output('home-graph4', 'figure')],
    [Input('home-matches', 'value'), Input('graph1-type', 'value'),
     Input('graph2-type', 'value'), Input('graph3-type', 'value'),
     Input('graph4-type', 'value')]
)
def update_home(min_matches, g1_type, g2_type, g3_type, g4_type):
    df_f = df[df['appearance'] >= min_matches]
    
    def create_graph(graph_type):
        # Gráficos tipo TOP (barras horizontales)
        if graph_type == 'top_scorers':
            top = df_f.nlargest(15, 'goles_totales')
            fig = px.bar(top, y='name', x='goles_totales', orientation='h',
                        color='goles_totales', color_continuous_scale='Reds',
                        labels={'goles_totales': 'Goles', 'name': 'Jugador'},
                        hover_data=['team', 'appearance'])
            fig.update_layout(showlegend=False, yaxis={'categoryorder': 'total ascending'})
            
        elif graph_type == 'top_assisters':
            top = df_f.nlargest(15, 'asistencias_totales')
            fig = px.bar(top, y='name', x='asistencias_totales', orientation='h',
                        color='asistencias_totales', color_continuous_scale='Blues',
                        labels={'asistencias_totales': 'Asistencias', 'name': 'Jugador'},
                        hover_data=['team', 'appearance'])
            fig.update_layout(showlegend=False, yaxis={'categoryorder': 'total ascending'})
            
        elif graph_type == 'top_contribution':
            top = df_f.nlargest(15, 'contribucion_total')
            fig = px.bar(top, y='name', x='contribucion_total', orientation='h',
                        color='contribucion_total', color_continuous_scale='Greens',
                        labels={'contribucion_total': 'Goles + Asistencias', 'name': 'Jugador'},
                        hover_data=['team', 'goles_totales', 'asistencias_totales'])
            fig.update_layout(showlegend=False, yaxis={'categoryorder': 'total ascending'})
            
        elif graph_type == 'top_minutes':
            top = df_f.nlargest(15, 'minutes played')
            fig = px.bar(top, y='name', x='minutes played', orientation='h',
                        color='minutes played', color_continuous_scale='Purples',
                        labels={'minutes played': 'Minutos', 'name': 'Jugador'},
                        hover_data=['team', 'appearance'])
            fig.update_layout(showlegend=False, yaxis={'categoryorder': 'total ascending'})
            
        elif graph_type == 'goals_per_game':
            top = df_f.nlargest(15, 'goles_por_partido')
            fig = px.bar(top, y='name', x='goles_por_partido', orientation='h',
                        color='goles_por_partido', color_continuous_scale='Oranges',
                        labels={'goles_por_partido': 'Goles/Partido', 'name': 'Jugador'},
                        hover_data=['team', 'goles_totales', 'appearance'])
            fig.update_layout(showlegend=False, yaxis={'categoryorder': 'total ascending'})
            
        elif graph_type == 'top_value':
            top = df_f[df_f['current_value'] > 0].nlargest(15, 'current_value')
            fig = px.bar(top, y='name', x='current_value', orientation='h',
                        color='current_value', color_continuous_scale='YlOrRd',
                        labels={'current_value': 'Valor (€)', 'name': 'Jugador'},
                        hover_data=['team', 'position'])
            fig.update_layout(showlegend=False, yaxis={'categoryorder': 'total ascending'})
            fig.update_xaxes(tickformat='.0s')
            
        elif graph_type == 'top_appearances':
            top = df_f.nlargest(15, 'appearance')
            fig = px.bar(top, y='name', x='appearance', orientation='h',
                        color='appearance', color_continuous_scale='Teal',
                        labels={'appearance': 'Partidos', 'name': 'Jugador'},
                        hover_data=['team', 'minutes played'])
            fig.update_layout(showlegend=False, yaxis={'categoryorder': 'total ascending'})
        
        # Gráficos de DISTRIBUCIÓN
        elif graph_type == 'positions':
            pos_counts = df_f['position'].value_counts().head(10)
            fig = px.pie(values=pos_counts.values, names=pos_counts.index, hole=0.3)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            
        elif graph_type == 'teams':
            team_counts = df_f['team'].value_counts().head(15)
            fig = px.bar(x=team_counts.values, y=team_counts.index, orientation='h',
                        labels={'x': 'Jugadores', 'y': 'Equipo'})
            fig.update_layout(showlegend=False, yaxis={'categoryorder': 'total ascending'})
            
        elif graph_type == 'ages':
            fig = px.histogram(df_f, x='age', nbins=20, 
                             labels={'age': 'Edad', 'count': 'Jugadores'},
                             color_discrete_sequence=['steelblue'])
            
        elif graph_type == 'values':
            df_val = df_f[df_f['current_value'] > 0]
            fig = px.histogram(df_val, x='current_value', nbins=30,
                             labels={'current_value': 'Valor (€)', 'count': 'Jugadores'},
                             color_discrete_sequence=['orange'])
            fig.update_xaxes(tickformat='.0s')
            
        elif graph_type == 'goals_position':
            pos_goals = df_f.groupby('position')['goles_totales'].sum().nlargest(10)
            fig = px.bar(x=pos_goals.values, y=pos_goals.index, orientation='h',
                        labels={'x': 'Goles Totales', 'y': 'Posición'},
                        color=pos_goals.values, color_continuous_scale='Reds')
            fig.update_layout(showlegend=False, yaxis={'categoryorder': 'total ascending'})
            
        elif graph_type == 'assists_position':
            pos_assists = df_f.groupby('position')['asistencias_totales'].sum().nlargest(10)
            fig = px.bar(x=pos_assists.values, y=pos_assists.index, orientation='h',
                        labels={'x': 'Asistencias Totales', 'y': 'Posición'},
                        color=pos_assists.values, color_continuous_scale='Blues')
            fig.update_layout(showlegend=False, yaxis={'categoryorder': 'total ascending'})
        
        # Gráficos de SCATTER (comparaciones)
        elif graph_type == 'goals_assists':
            sample = df_f.sample(min(500, len(df_f)))
            fig = px.scatter(sample, x='goles_totales', y='asistencias_totales',
                           hover_name='name', hover_data=['team', 'position'],
                           color='position', size='appearance',
                           labels={'goles_totales': 'Goles', 'asistencias_totales': 'Asistencias'})
            
        elif graph_type == 'age_value':
            sample = df_f[df_f['current_value'] > 0].sample(min(500, len(df_f[df_f['current_value'] > 0])))
            fig = px.scatter(sample, x='age', y='current_value',
                           hover_name='name', hover_data=['team', 'position'],
                           color='position', size='contribucion_total',
                           labels={'age': 'Edad', 'current_value': 'Valor (€)'})
            fig.update_yaxes(tickformat='.0s')
            
        elif graph_type == 'games_minutes':
            sample = df_f.sample(min(500, len(df_f)))
            fig = px.scatter(sample, x='appearance', y='minutes played',
                           hover_name='name', hover_data=['team', 'position'],
                           color='position', size='contribucion_total',
                           labels={'appearance': 'Partidos', 'minutes played': 'Minutos'})
            
        elif graph_type == 'goals_minutes':
            sample = df_f[df_f['goles_totales'] > 0].sample(min(500, len(df_f[df_f['goles_totales'] > 0])))
            fig = px.scatter(sample, x='minutes played', y='goles_totales',
                           hover_name='name', hover_data=['team', 'position'],
                           color='position', size='appearance',
                           labels={'minutes played': 'Minutos', 'goles_totales': 'Goles'})
            
        elif graph_type == 'assists_minutes':
            sample = df_f[df_f['asistencias_totales'] > 0].sample(min(500, len(df_f[df_f['asistencias_totales'] > 0])))
            fig = px.scatter(sample, x='minutes played', y='asistencias_totales',
                           hover_name='name', hover_data=['team', 'position'],
                           color='position', size='appearance',
                           labels={'minutes played': 'Minutos', 'asistencias_totales': 'Asistencias'})
            
        elif graph_type == 'value_contribution':
            sample = df_f[df_f['current_value'] > 0].sample(min(500, len(df_f[df_f['current_value'] > 0])))
            fig = px.scatter(sample, x='contribucion_total', y='current_value',
                           hover_name='name', hover_data=['team', 'position'],
                           color='position', size='appearance',
                           labels={'contribucion_total': 'Goles + Asistencias', 'current_value': 'Valor (€)'})
            fig.update_yaxes(tickformat='.0s')
        
        else:
            fig = px.scatter(title="Selecciona un tipo de gráfico")
        
        fig.update_layout(template='plotly_white', margin=dict(l=20, r=20, t=40, b=20))
        return fig
    
    return create_graph(g1_type), create_graph(g2_type), create_graph(g3_type), create_graph(g4_type)

@app.callback(
    Output('compare-content', 'children'),
    [Input('compare-p1', 'value'), Input('compare-p2', 'value')]
)
def update_comparison(p1_idx, p2_idx):
    if p1_idx is None or p2_idx is None:
        return dbc.Alert("Selecciona dos jugadores", color="info")
    
    p1, p2 = df.iloc[p1_idx], df.iloc[p2_idx]
    
    # Detectar si son porteros
    es_portero_p1 = p1['position'] == 'Goalkeeper'
    es_portero_p2 = p2['position'] == 'Goalkeeper'
    ambos_porteros = es_portero_p1 and es_portero_p2
    
    # Crear cards con información básica
    if ambos_porteros:
        # Stats para porteros
        card1_body = html.P([
            html.Strong("Equipo: "), p1['team'], html.Br(),
            html.Strong("Posición: "), p1['position'], html.Br(),
            html.Strong("Edad: "), f"{p1['age']:.0f}", html.Br(),
            html.Strong("Partidos: "), f"{int(p1['appearance'])}", html.Br(),
            html.Strong("Goles Concedidos: "), f"{int(p1['goals conceded']) if p1['goals conceded'] > 0 else 0}", html.Br(),
            html.Strong("Porterías a Cero: "), f"{int(p1['clean sheets']) if p1['clean sheets'] > 0 else 0}", html.Br(),
            html.Strong("Tarjetas: "), f"{int(p1['yellow cards'])}🟨 {int(p1['red cards'])}🟥", html.Br(),
            html.Strong("Valor: "), f"€{p1['current_value']:,.0f}", html.Br(),
        ])
        card2_body = html.P([
            html.Strong("Equipo: "), p2['team'], html.Br(),
            html.Strong("Posición: "), p2['position'], html.Br(),
            html.Strong("Edad: "), f"{p2['age']:.0f}", html.Br(),
            html.Strong("Partidos: "), f"{int(p2['appearance'])}", html.Br(),
            html.Strong("Goles Concedidos: "), f"{int(p2['goals conceded']) if p2['goals conceded'] > 0 else 0}", html.Br(),
            html.Strong("Porterías a Cero: "), f"{int(p2['clean sheets']) if p2['clean sheets'] > 0 else 0}", html.Br(),
            html.Strong("Tarjetas: "), f"{int(p2['yellow cards'])}🟨 {int(p2['red cards'])}🟥", html.Br(),
            html.Strong("Valor: "), f"€{p2['current_value']:,.0f}", html.Br(),
        ])
    else:
        # Stats para jugadores de campo
        card1_body = html.P([
            html.Strong("Equipo: "), p1['team'], html.Br(),
            html.Strong("Posición: "), p1['position'], html.Br(),
            html.Strong("Edad: "), f"{p1['age']:.0f}", html.Br(),
            html.Strong("Partidos: "), f"{int(p1['appearance'])}", html.Br(),
            html.Strong("Goles: "), f"{int(p1['goles_totales'])}", html.Br(),
            html.Strong("Asistencias: "), f"{int(p1['asistencias_totales'])}", html.Br(),
            html.Strong("Contribución: "), f"{int(p1['contribucion_total'])}", html.Br(),
            html.Strong("Tarjetas: "), f"{int(p1['yellow cards'])}🟨 {int(p1['red cards'])}🟥", html.Br(),
            html.Strong("Valor: "), f"€{p1['current_value']:,.0f}", html.Br(),
        ])
        card2_body = html.P([
            html.Strong("Equipo: "), p2['team'], html.Br(),
            html.Strong("Posición: "), p2['position'], html.Br(),
            html.Strong("Edad: "), f"{p2['age']:.0f}", html.Br(),
            html.Strong("Partidos: "), f"{int(p2['appearance'])}", html.Br(),
            html.Strong("Goles: "), f"{int(p2['goles_totales'])}", html.Br(),
            html.Strong("Asistencias: "), f"{int(p2['asistencias_totales'])}", html.Br(),
            html.Strong("Contribución: "), f"{int(p2['contribucion_total'])}", html.Br(),
            html.Strong("Tarjetas: "), f"{int(p2['yellow cards'])}🟨 {int(p2['red cards'])}🟥", html.Br(),
            html.Strong("Valor: "), f"€{p2['current_value']:,.0f}", html.Br(),
        ])
    
    cards = dbc.Row([
        dbc.Col([dbc.Card([
            dbc.CardHeader(html.H5(p1['name'])),
            dbc.CardBody([card1_body])
        ], color="primary", outline=True)], width=6, className="mb-4"),
        
        dbc.Col([dbc.Card([
            dbc.CardHeader(html.H5(p2['name'])),
            dbc.CardBody([card2_body])
        ], color="info", outline=True)], width=6, className="mb-4"),
    ])
    
    # Crear gráfico comparativo según el tipo de jugador
    if ambos_porteros:
        # Gráfico para porteros
        comp_data = pd.DataFrame({
            'Métrica': ['Goles Concedidos', 'Porterías a Cero', 'Partidos', 'Tarjetas'],
            p1['name']: [
                p1['goals conceded'] if p1['goals conceded'] > 0 else 0,
                p1['clean sheets'] if p1['clean sheets'] > 0 else 0,
                p1['appearance'],
                p1['yellow cards'] + p1['red cards']
            ],
            p2['name']: [
                p2['goals conceded'] if p2['goals conceded'] > 0 else 0,
                p2['clean sheets'] if p2['clean sheets'] > 0 else 0,
                p2['appearance'],
                p2['yellow cards'] + p2['red cards']
            ]
        })
    else:
        # Gráfico para jugadores de campo
        comp_data = pd.DataFrame({
            'Métrica': ['Goles', 'Asistencias', 'Contribución', 'Tarjetas'],
            p1['name']: [
                p1['goles_totales'],
                p1['asistencias_totales'],
                p1['contribucion_total'],
                p1['yellow cards'] + p1['red cards']
            ],
            p2['name']: [
                p2['goles_totales'],
                p2['asistencias_totales'],
                p2['contribucion_total'],
                p2['yellow cards'] + p2['red cards']
            ]
        })
    
    fig = px.bar(comp_data, x='Métrica', y=[p1['name'], p2['name']], barmode='group',
                color_discrete_sequence=['#3498db', '#e74c3c'])
    fig.update_layout(template='plotly_white', margin=dict(l=20, r=20, t=40, b=20))
    chart = dbc.Card([dbc.CardBody([dcc.Graph(figure=fig)])], className="shadow-sm")
    
    return html.Div([cards, chart])

@app.callback(
    Output('teams-content', 'children'),
    Input('teams-dropdown', 'value')
)
def update_teams(team):
    team_df = df[df['team'] == team]
    
    # Stats Cards (8 métricas)
    stats = dbc.Row([
        dbc.Col([dbc.Card([dbc.CardBody([
            html.I(className="fas fa-users fa-2x mb-2", style={'color': COLORS['primary']}),
            html.H5("Jugadores"),
            html.H3(f"{len(team_df)}")
        ])], className="text-center shadow-sm")], width=6, md=3, className="mb-4"),
        
        dbc.Col([dbc.Card([dbc.CardBody([
            html.I(className="fas fa-futbol fa-2x mb-2", style={'color': COLORS['accent']}),
            html.H5("Goles"),
            html.H3(f"{int(team_df['goles_totales'].sum())}")
        ])], className="text-center shadow-sm")], width=6, md=3, className="mb-4"),
        
        dbc.Col([dbc.Card([dbc.CardBody([
            html.I(className="fas fa-hands-helping fa-2x mb-2", style={'color': COLORS['secondary']}),
            html.H5("Asistencias"),
            html.H3(f"{int(team_df['asistencias_totales'].sum())}")
        ])], className="text-center shadow-sm")], width=6, md=3, className="mb-4"),
        
        dbc.Col([dbc.Card([dbc.CardBody([
            html.I(className="fas fa-euro-sign fa-2x mb-2", style={'color': COLORS['danger']}),
            html.H5("Valor Total"),
            html.H3(f"€{team_df['current_value'].sum()/1e6:.1f}M")
        ])], className="text-center shadow-sm")], width=6, md=3, className="mb-4"),
        
        dbc.Col([dbc.Card([dbc.CardBody([
            html.I(className="fas fa-birthday-cake fa-2x mb-2", style={'color': '#9b59b6'}),
            html.H5("Edad Promedio"),
            html.H3(f"{team_df['age'].mean():.1f}")
        ])], className="text-center shadow-sm")], width=6, md=3, className="mb-4"),
        
        dbc.Col([dbc.Card([dbc.CardBody([
            html.I(className="fas fa-trophy fa-2x mb-2", style={'color': '#f39c12'}),
            html.H5("Partidos Totales"),
            html.H3(f"{int(team_df['appearance'].sum())}")
        ])], className="text-center shadow-sm")], width=6, md=3, className="mb-4"),
        
        dbc.Col([dbc.Card([dbc.CardBody([
            html.I(className="fas fa-exclamation-triangle fa-2x mb-2", style={'color': '#e74c3c'}),
            html.H5("Tarjetas"),
            html.H3(f"{int(team_df['yellow cards'].sum())}🟨 {int(team_df['red cards'].sum())}🟥")
        ])], className="text-center shadow-sm")], width=6, md=3, className="mb-4"),
        
        dbc.Col([dbc.Card([dbc.CardBody([
            html.I(className="fas fa-star fa-2x mb-2", style={'color': '#e67e22'}),
            html.H5("Contribución"),
            html.H3(f"{int(team_df['contribucion_total'].sum())}")
        ])], className="text-center shadow-sm")], width=6, md=3, className="mb-4"),
    ])
    
    # Mejores Jugadores por POSICIÓN REAL
    def get_best_by_position(position_filter, metric, metric_name):
        filtered = team_df[team_df['position'].str.contains(position_filter, case=False, na=False)]
        if len(filtered) > 0:
            best = filtered.nlargest(1, metric).iloc[0]
            return html.P([
                html.Strong(best['name']), html.Br(),
                f"{int(best[metric]) if metric in ['goles_totales', 'asistencias_totales', 'appearance'] else best[metric]:.1f} {metric_name}",
                html.Br(),
                html.Small(best['position'], className="text-muted")
            ])
        return html.P("N/A")
    
    # Mejor portero
    gk_df = team_df[team_df['position'] == 'Goalkeeper']
    if len(gk_df) > 0:
        best_gk = gk_df.nlargest(1, 'clean sheets').iloc[0] if gk_df['clean sheets'].sum() > 0 else gk_df.nlargest(1, 'appearance').iloc[0]
        gk_card = html.P([
            html.Strong(best_gk['name']), html.Br(),
            f"{int(best_gk['clean sheets'])} porterías a cero" if best_gk['clean sheets'] > 0 else f"{int(best_gk['appearance'])} partidos",
            html.Br(),
            html.Small("Goalkeeper", className="text-muted")
        ])
    else:
        gk_card = html.P("N/A")
    
    best_players_by_position = html.Div([
        html.H4("⭐ Mejores Jugadores por Posición", className="mt-4 mb-3"),
        dbc.Row([
            # Mejor Portero
            dbc.Col([dbc.Card([
                dbc.CardHeader(html.H6("🧤 Mejor Portero", className="mb-0")),
                dbc.CardBody([gk_card])
            ], className="shadow-sm")], width=6, md=3, className="mb-3"),
            
            # Mejor Defensor
            dbc.Col([dbc.Card([
                dbc.CardHeader(html.H6("🛡️ Mejor Defensor", className="mb-0")),
                dbc.CardBody([get_best_by_position('Defender', 'appearance', 'partidos')])
            ], className="shadow-sm")], width=6, md=3, className="mb-3"),
            
            # Mejor Mediocentro
            dbc.Col([dbc.Card([
                dbc.CardHeader(html.H6("🎯 Mejor Medio", className="mb-0")),
                dbc.CardBody([get_best_by_position('Midfield|midfield', 'contribucion_total', 'G+A')])
            ], className="shadow-sm")], width=6, md=3, className="mb-3"),
            
            # Mejor Delantero
            dbc.Col([dbc.Card([
                dbc.CardHeader(html.H6("⚽ Mejor Delantero", className="mb-0")),
                dbc.CardBody([get_best_by_position('Attack|Forward|Striker', 'goles_totales', 'goles')])
            ], className="shadow-sm")], width=6, md=3, className="mb-3"),
        ])
    ])
    
    # Mejores jugadores por categoría general
    best_players_section = html.Div([
        html.H4("🏆 Destacados del Equipo", className="mt-4 mb-3"),
        dbc.Row([
            # Mejor Goleador
            dbc.Col([dbc.Card([
                dbc.CardHeader(html.H6("🔴 Máximo Goleador", className="mb-0")),
                dbc.CardBody([
                    html.P([
                        html.Strong(team_df.nlargest(1, 'goles_totales')['name'].iloc[0] if len(team_df) > 0 else "N/A"), html.Br(),
                        f"{int(team_df.nlargest(1, 'goles_totales')['goles_totales'].iloc[0]) if len(team_df) > 0 else 0} goles",
                        html.Br(),
                        html.Small(team_df.nlargest(1, 'goles_totales')['position'].iloc[0] if len(team_df) > 0 else "", className="text-muted")
                    ])
                ])
            ], className="shadow-sm")], width=6, md=3, className="mb-3"),
            
            # Mejor Asistente
            dbc.Col([dbc.Card([
                dbc.CardHeader(html.H6("🔵 Máximo Asistente", className="mb-0")),
                dbc.CardBody([
                    html.P([
                        html.Strong(team_df.nlargest(1, 'asistencias_totales')['name'].iloc[0] if len(team_df) > 0 else "N/A"), html.Br(),
                        f"{int(team_df.nlargest(1, 'asistencias_totales')['asistencias_totales'].iloc[0]) if len(team_df) > 0 else 0} asistencias",
                        html.Br(),
                        html.Small(team_df.nlargest(1, 'asistencias_totales')['position'].iloc[0] if len(team_df) > 0 else "", className="text-muted")
                    ])
                ])
            ], className="shadow-sm")], width=6, md=3, className="mb-3"),
            
            # Más Valioso
            dbc.Col([dbc.Card([
                dbc.CardHeader(html.H6("💰 Más Valioso", className="mb-0")),
                dbc.CardBody([
                    html.P([
                        html.Strong(team_df.nlargest(1, 'current_value')['name'].iloc[0] if len(team_df[team_df['current_value'] > 0]) > 0 else "N/A"), html.Br(),
                        f"€{team_df.nlargest(1, 'current_value')['current_value'].iloc[0]/1e6:.1f}M" if len(team_df[team_df['current_value'] > 0]) > 0 else "€0",
                        html.Br(),
                        html.Small(team_df.nlargest(1, 'current_value')['position'].iloc[0] if len(team_df[team_df['current_value'] > 0]) > 0 else "", className="text-muted")
                    ])
                ])
            ], className="shadow-sm")], width=6, md=3, className="mb-3"),
            
            # Más Experimentado
            dbc.Col([dbc.Card([
                dbc.CardHeader(html.H6("🏆 Más Experimentado", className="mb-0")),
                dbc.CardBody([
                    html.P([
                        html.Strong(team_df.nlargest(1, 'appearance')['name'].iloc[0] if len(team_df) > 0 else "N/A"), html.Br(),
                        f"{int(team_df.nlargest(1, 'appearance')['appearance'].iloc[0]) if len(team_df) > 0 else 0} partidos",
                        html.Br(),
                        html.Small(team_df.nlargest(1, 'appearance')['position'].iloc[0] if len(team_df) > 0 else "", className="text-muted")
                    ])
                ])
            ], className="shadow-sm")], width=6, md=3, className="mb-3"),
        ])
    ])
    
    # Gráficos
    top_scorers = team_df.nlargest(10, 'goles_totales')
    fig1 = px.bar(top_scorers, x='name', y='goles_totales', color='goles_totales',
                  color_continuous_scale='Reds', title="Top 10 Goleadores")
    fig1.update_layout(showlegend=False, template='plotly_white', xaxis_tickangle=-45)
    
    top_assisters = team_df.nlargest(10, 'asistencias_totales')
    fig2 = px.bar(top_assisters, x='name', y='asistencias_totales', color='asistencias_totales',
                  color_continuous_scale='Blues', title="Top 10 Asistentes")
    fig2.update_layout(showlegend=False, template='plotly_white', xaxis_tickangle=-45)
    
    pos_dist = team_df['position'].value_counts()
    fig3 = px.pie(values=pos_dist.values, names=pos_dist.index, title="Distribución por Posición", hole=0.4)
    fig3.update_traces(textposition='inside', textinfo='percent+label')
    fig3.update_layout(template='plotly_white')
    
    # Gráfico de edad vs valor
    team_val = team_df[team_df['current_value'] > 0]
    fig4 = px.scatter(team_val, x='age', y='current_value', size='contribucion_total',
                     hover_name='name', hover_data=['position', 'goles_totales', 'asistencias_totales'],
                     title="Edad vs Valor de Mercado", color='position')
    fig4.update_layout(template='plotly_white')
    fig4.update_yaxes(tickformat='.0s')
    
    charts = dbc.Row([
        dbc.Col([dbc.Card([dbc.CardBody([dcc.Graph(figure=fig1)])], className="shadow-sm")], 
                width=12, lg=6, className="mb-4"),
        dbc.Col([dbc.Card([dbc.CardBody([dcc.Graph(figure=fig2)])], className="shadow-sm")],
                width=12, lg=6, className="mb-4"),
        dbc.Col([dbc.Card([dbc.CardBody([dcc.Graph(figure=fig3)])], className="shadow-sm")], 
                width=12, lg=6, className="mb-4"),
        dbc.Col([dbc.Card([dbc.CardBody([dcc.Graph(figure=fig4)])], className="shadow-sm")],
                width=12, lg=6, className="mb-4"),
    ])
    
    return html.Div([stats, best_players_by_position, best_players_section, charts])

@app.callback(
    Output('teams-comparison-content', 'children'),
    [Input('team-compare-1', 'value'), Input('team-compare-2', 'value')]
)
def update_teams_comparison(team1, team2):
    if team1 is None or team2 is None or team1 == team2:
        return dbc.Alert("Selecciona dos equipos diferentes para comparar", color="info")
    
    t1_df = df[df['team'] == team1]
    t2_df = df[df['team'] == team2]
    
    # Comparación de stats principales
    comp_stats = dbc.Row([
        # Equipo 1
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5(team1, className="text-center")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([html.P([html.Strong("Jugadores:"), html.Br(), f"{len(t1_df)}"])], width=6),
                        dbc.Col([html.P([html.Strong("Goles:"), html.Br(), f"{int(t1_df['goles_totales'].sum())}"])], width=6),
                        dbc.Col([html.P([html.Strong("Asistencias:"), html.Br(), f"{int(t1_df['asistencias_totales'].sum())}"])], width=6),
                        dbc.Col([html.P([html.Strong("Contribución:"), html.Br(), f"{int(t1_df['contribucion_total'].sum())}"])], width=6),
                        dbc.Col([html.P([html.Strong("Valor:"), html.Br(), f"€{t1_df['current_value'].sum()/1e6:.1f}M"])], width=6),
                        dbc.Col([html.P([html.Strong("Edad Media:"), html.Br(), f"{t1_df['age'].mean():.1f}"])], width=6),
                        dbc.Col([html.P([html.Strong("Tarjetas:"), html.Br(), f"{int(t1_df['yellow cards'].sum())}🟨 {int(t1_df['red cards'].sum())}🟥"])], width=6),
                        dbc.Col([html.P([html.Strong("Partidos:"), html.Br(), f"{int(t1_df['appearance'].sum())}"])], width=6),
                    ])
                ])
            ], color="primary", outline=True)
        ], width=12, md=6, className="mb-4"),
        
        # Equipo 2
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5(team2, className="text-center")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([html.P([html.Strong("Jugadores:"), html.Br(), f"{len(t2_df)}"])], width=6),
                        dbc.Col([html.P([html.Strong("Goles:"), html.Br(), f"{int(t2_df['goles_totales'].sum())}"])], width=6),
                        dbc.Col([html.P([html.Strong("Asistencias:"), html.Br(), f"{int(t2_df['asistencias_totales'].sum())}"])], width=6),
                        dbc.Col([html.P([html.Strong("Contribución:"), html.Br(), f"{int(t2_df['contribucion_total'].sum())}"])], width=6),
                        dbc.Col([html.P([html.Strong("Valor:"), html.Br(), f"€{t2_df['current_value'].sum()/1e6:.1f}M"])], width=6),
                        dbc.Col([html.P([html.Strong("Edad Media:"), html.Br(), f"{t2_df['age'].mean():.1f}"])], width=6),
                        dbc.Col([html.P([html.Strong("Tarjetas:"), html.Br(), f"{int(t2_df['yellow cards'].sum())}🟨 {int(t2_df['red cards'].sum())}🟥"])], width=6),
                        dbc.Col([html.P([html.Strong("Partidos:"), html.Br(), f"{int(t2_df['appearance'].sum())}"])], width=6),
                    ])
                ])
            ], color="info", outline=True)
        ], width=12, md=6, className="mb-4"),
    ])
    
    # Mejores jugadores por posición de cada equipo
    def get_best_player_card(team_df, position_name, position_filter, metric, color):
        filtered = team_df[team_df['position'].str.contains(position_filter, case=False, na=False)]
        if len(filtered) > 0:
            best = filtered.nlargest(1, metric).iloc[0]
            return dbc.Card([
                dbc.CardBody([
                    html.H6(best['name'], className="mb-2"),
                    html.P([
                        f"{int(best[metric]) if metric in ['goles_totales', 'asistencias_totales', 'appearance', 'clean sheets'] else best[metric]:.1f}",
                        html.Br(),
                        html.Small(best['position'], className="text-muted")
                    ], className="mb-0")
                ])
            ], style={'border-left': f'4px solid {color}'}, className="mb-2")
        return dbc.Card([
            dbc.CardBody([html.P("No disponible", className="text-muted mb-0")])
        ], className="mb-2")
    
    # Porteros
    t1_gk = t1_df[t1_df['position'] == 'Goalkeeper']
    t2_gk = t2_df[t2_df['position'] == 'Goalkeeper']
    
    t1_gk_metric = 'clean sheets' if len(t1_gk) > 0 and t1_gk['clean sheets'].sum() > 0 else 'appearance'
    t2_gk_metric = 'clean sheets' if len(t2_gk) > 0 and t2_gk['clean sheets'].sum() > 0 else 'appearance'
    
    best_by_position = html.Div([
        html.H4("⭐ Comparación de Mejores Jugadores por Posición", className="mt-4 mb-3"),
        dbc.Row([
            # Porteros
            dbc.Col([
                html.H6("🧤 Porteros"),
                dbc.Row([
                    dbc.Col([
                        html.P(team1, className="text-muted small mb-1"),
                        get_best_player_card(t1_df, "Portero", "Goalkeeper", t1_gk_metric, "#3498db")
                    ], width=6),
                    dbc.Col([
                        html.P(team2, className="text-muted small mb-1"),
                        get_best_player_card(t2_df, "Portero", "Goalkeeper", t2_gk_metric, "#e74c3c")
                    ], width=6),
                ])
            ], width=12, lg=6, className="mb-4"),
            
            # Defensores
            dbc.Col([
                html.H6("🛡️ Defensores"),
                dbc.Row([
                    dbc.Col([
                        html.P(team1, className="text-muted small mb-1"),
                        get_best_player_card(t1_df, "Defensor", "Defender", "appearance", "#3498db")
                    ], width=6),
                    dbc.Col([
                        html.P(team2, className="text-muted small mb-1"),
                        get_best_player_card(t2_df, "Defensor", "Defender", "appearance", "#e74c3c")
                    ], width=6),
                ])
            ], width=12, lg=6, className="mb-4"),
            
            # Mediocentros
            dbc.Col([
                html.H6("🎯 Mediocentros"),
                dbc.Row([
                    dbc.Col([
                        html.P(team1, className="text-muted small mb-1"),
                        get_best_player_card(t1_df, "Medio", "Midfield|midfield", "contribucion_total", "#3498db")
                    ], width=6),
                    dbc.Col([
                        html.P(team2, className="text-muted small mb-1"),
                        get_best_player_card(t2_df, "Medio", "Midfield|midfield", "contribucion_total", "#e74c3c")
                    ], width=6),
                ])
            ], width=12, lg=6, className="mb-4"),
            
            # Delanteros
            dbc.Col([
                html.H6("⚽ Delanteros"),
                dbc.Row([
                    dbc.Col([
                        html.P(team1, className="text-muted small mb-1"),
                        get_best_player_card(t1_df, "Delantero", "Attack|Forward|Striker", "goles_totales", "#3498db")
                    ], width=6),
                    dbc.Col([
                        html.P(team2, className="text-muted small mb-1"),
                        get_best_player_card(t2_df, "Delantero", "Attack|Forward|Striker", "goles_totales", "#e74c3c")
                    ], width=6),
                ])
            ], width=12, lg=6, className="mb-4"),
        ])
    ])
    
    # Gráfico comparativo general
    comp_data = pd.DataFrame({
        'Métrica': ['Jugadores', 'Goles', 'Asistencias', 'Contribución', 'Valor (M€)', 'Tarjetas'],
        team1: [
            len(t1_df),
            int(t1_df['goles_totales'].sum()),
            int(t1_df['asistencias_totales'].sum()),
            int(t1_df['contribucion_total'].sum()),
            t1_df['current_value'].sum()/1e6,
            int(t1_df['yellow cards'].sum() + t1_df['red cards'].sum())
        ],
        team2: [
            len(t2_df),
            int(t2_df['goles_totales'].sum()),
            int(t2_df['asistencias_totales'].sum()),
            int(t2_df['contribucion_total'].sum()),
            t2_df['current_value'].sum()/1e6,
            int(t2_df['yellow cards'].sum() + t2_df['red cards'].sum())
        ]
    })
    
    fig_comp = px.bar(comp_data, x='Métrica', y=[team1, team2], barmode='group',
                      color_discrete_sequence=['#3498db', '#e74c3c'],
                      title="Comparación General")
    fig_comp.update_layout(template='plotly_white', legend_title_text='Equipo')
    
    # Top 5 goleadores de cada equipo
    t1_top = t1_df.nlargest(5, 'goles_totales')[['name', 'goles_totales', 'position']]
    t2_top = t2_df.nlargest(5, 'goles_totales')[['name', 'goles_totales', 'position']]
    
    fig_scorers = go.Figure()
    fig_scorers.add_trace(go.Bar(
        name=team1,
        y=t1_top['name'],
        x=t1_top['goles_totales'],
        orientation='h',
        marker_color='#3498db'
    ))
    fig_scorers.add_trace(go.Bar(
        name=team2,
        y=t2_top['name'],
        x=t2_top['goles_totales'],
        orientation='h',
        marker_color='#e74c3c'
    ))
    fig_scorers.update_layout(
        title="Top 5 Goleadores por Equipo",
        barmode='group',
        template='plotly_white',
        yaxis={'categoryorder': 'total ascending'}
    )
    
    # Distribución por posición
    t1_pos = t1_df['position'].value_counts().head(8)
    t2_pos = t2_df['position'].value_counts().head(8)
    
    fig_pos = go.Figure()
    fig_pos.add_trace(go.Bar(
        name=team1,
        x=t1_pos.index,
        y=t1_pos.values,
        marker_color='#3498db'
    ))
    fig_pos.add_trace(go.Bar(
        name=team2,
        x=t2_pos.index,
        y=t2_pos.values,
        marker_color='#e74c3c'
    ))
    fig_pos.update_layout(
        title="Jugadores por Posición",
        barmode='group',
        template='plotly_white',
        xaxis_tickangle=-45
    )
    
    charts = dbc.Row([
        dbc.Col([dbc.Card([dbc.CardBody([dcc.Graph(figure=fig_comp)])], className="shadow-sm")],
                width=12, className="mb-4"),
        dbc.Col([dbc.Card([dbc.CardBody([dcc.Graph(figure=fig_scorers)])], className="shadow-sm")],
                width=12, lg=6, className="mb-4"),
        dbc.Col([dbc.Card([dbc.CardBody([dcc.Graph(figure=fig_pos)])], className="shadow-sm")],
                width=12, lg=6, className="mb-4"),
    ])
    
    return html.Div([comp_stats, best_by_position, charts])
    
    fig_pos = go.Figure()
    fig_pos.add_trace(go.Bar(
        name=team1,
        x=t1_pos.index,
        y=t1_pos.values,
        marker_color='#3498db'
    ))
    fig_pos.add_trace(go.Bar(
        name=team2,
        x=t2_pos.index,
        y=t2_pos.values,
        marker_color='#e74c3c'
    ))
    fig_pos.update_layout(
        title="Jugadores por Posición",
        barmode='group',
        template='plotly_white',
        xaxis_tickangle=-45
    )
    
    charts = dbc.Row([
        dbc.Col([dbc.Card([dbc.CardBody([dcc.Graph(figure=fig_comp)])], className="shadow-sm")],
                width=12, className="mb-4"),
        dbc.Col([dbc.Card([dbc.CardBody([dcc.Graph(figure=fig_scorers)])], className="shadow-sm")],
                width=12, lg=6, className="mb-4"),
        dbc.Col([dbc.Card([dbc.CardBody([dcc.Graph(figure=fig_pos)])], className="shadow-sm")],
                width=12, lg=6, className="mb-4"),
    ])
    
    return html.Div([comp_stats, charts])

# Callback para actualizar opciones de dropdowns según posición
@app.callback(
    [Output('perf-graph1-type', 'options'), Output('perf-graph2-type', 'options'),
     Output('perf-graph3-type', 'options'), Output('perf-graph4-type', 'options')],
    Input('perf-position', 'value')
)
def update_perf_dropdown_options(position):
    # Opciones para jugadores de campo
    field_options = [
        {'label': '⚽ Goles por Partido', 'value': 'goals_per_game'},
        {'label': '🎯 Asistencias por Partido', 'value': 'assists_per_game'},
        {'label': '⭐ Contribución por Partido', 'value': 'contribution_per_game'},
        {'label': '🔴 Top Goleadores', 'value': 'top_scorers'},
        {'label': '🔵 Top Asistentes', 'value': 'top_assisters'},
        {'label': '🟢 Top Contribución', 'value': 'contribution'},
        {'label': '💰 Valor de Mercado', 'value': 'value_market'},
        {'label': '🏃 Minutos por Partido', 'value': 'minutes_per_game'},
        {'label': '🟨 Tarjetas Amarillas', 'value': 'yellow_cards'},
        {'label': '🟥 Tarjetas Rojas', 'value': 'red_cards'},
        {'label': '📊 Edad Promedio', 'value': 'age_dist'},
        {'label': '📈 Goles vs Asistencias', 'value': 'goals_assists_scatter'},
    ]
    
    # Opciones para porteros
    gk_options = [
        {'label': '🧤 Porterías a Cero', 'value': 'clean_sheets'},
        {'label': '🥅 Goles Concedidos', 'value': 'goals_conceded'},
        {'label': '📊 Porterías a Cero por Partido', 'value': 'clean_sheets_per_game'},
        {'label': '📉 Goles Concedidos por Partido', 'value': 'goals_conceded_per_game'},
        {'label': '💰 Valor de Mercado', 'value': 'value_market'},
        {'label': '🏃 Minutos por Partido', 'value': 'minutes_per_game'},
        {'label': '🟨 Tarjetas Amarillas', 'value': 'yellow_cards'},
        {'label': '🟥 Tarjetas Rojas', 'value': 'red_cards'},
        {'label': '📊 Edad Promedio', 'value': 'age_dist'},
        {'label': '📈 Porterías vs Goles Concedidos', 'value': 'clean_goals_scatter'},
    ]
    
    # Decidir qué opciones mostrar según la posición
    if position == 'Goalkeeper':
        options = gk_options
    else:
        options = field_options
    
    return options, options, options, options

@app.callback(
    [Output('perf-graph1', 'figure'), Output('perf-graph2', 'figure'),
     Output('perf-graph3', 'figure'), Output('perf-graph4', 'figure')],
    [Input('perf-team', 'value'), Input('perf-position', 'value'),
     Input('perf-matches', 'value'), Input('perf-graph1-type', 'value'),
     Input('perf-graph2-type', 'value'), Input('perf-graph3-type', 'value'),
     Input('perf-graph4-type', 'value')]
)
def update_performance(team, position, min_matches, g1_type, g2_type, g3_type, g4_type):
    # Filtrar datos
    df_f = df[df['appearance'] >= min_matches].copy()
    
    # Filtro por equipo
    if team != 'all':
        df_f = df_f[df_f['team'] == team]
    
    # Filtro por posición
    if position == 'Goalkeeper':
        df_f = df_f[df_f['position'] == 'Goalkeeper']
    elif position == 'Defender':
        df_f = df_f[df_f['position'].str.contains('Defender', case=False, na=False)]
    elif position == 'Midfield':
        df_f = df_f[df_f['position'].str.contains('Midfield|midfield', case=False, na=False)]
    elif position == 'Attack':
        df_f = df_f[df_f['position'].str.contains('Attack|Forward|Striker', case=False, na=False)]
    
    def create_perf_graph(graph_type):
        if len(df_f) == 0:
            fig = px.scatter(title="No hay datos con los filtros seleccionados")
            fig.update_layout(template='plotly_white')
            return fig
        
        # Gráficos para jugadores de campo
        if graph_type == 'goals_per_game':
            top = df_f.nlargest(20, 'goles_por_partido')
            fig = px.bar(top, y='name', x='goles_por_partido', orientation='h',
                        color='goles_por_partido', color_continuous_scale='Reds',
                        labels={'goles_por_partido': 'Goles/Partido', 'name': 'Jugador'},
                        hover_data=['team', 'position', 'goles_totales'])
            fig.update_layout(showlegend=False, yaxis={'categoryorder': 'total ascending'})
            
        elif graph_type == 'assists_per_game':
            top = df_f.nlargest(20, 'asistencias_por_partido')
            fig = px.bar(top, y='name', x='asistencias_por_partido', orientation='h',
                        color='asistencias_por_partido', color_continuous_scale='Blues',
                        labels={'asistencias_por_partido': 'Asistencias/Partido', 'name': 'Jugador'},
                        hover_data=['team', 'position', 'asistencias_totales'])
            fig.update_layout(showlegend=False, yaxis={'categoryorder': 'total ascending'})
            
        elif graph_type == 'contribution_per_game':
            df_f['contrib_per_game'] = df_f['contribucion_total'] / df_f['appearance']
            top = df_f.nlargest(20, 'contrib_per_game')
            fig = px.bar(top, y='name', x='contrib_per_game', orientation='h',
                        color='contrib_per_game', color_continuous_scale='Greens',
                        labels={'contrib_per_game': 'Contribución/Partido', 'name': 'Jugador'},
                        hover_data=['team', 'position', 'contribucion_total'])
            fig.update_layout(showlegend=False, yaxis={'categoryorder': 'total ascending'})
            
        elif graph_type == 'top_scorers':
            top = df_f.nlargest(20, 'goles_totales')
            fig = px.bar(top, y='name', x='goles_totales', orientation='h',
                        color='goles_totales', color_continuous_scale='Reds',
                        labels={'goles_totales': 'Goles Totales', 'name': 'Jugador'},
                        hover_data=['team', 'position', 'appearance'])
            fig.update_layout(showlegend=False, yaxis={'categoryorder': 'total ascending'})
            
        elif graph_type == 'top_assisters':
            top = df_f.nlargest(20, 'asistencias_totales')
            fig = px.bar(top, y='name', x='asistencias_totales', orientation='h',
                        color='asistencias_totales', color_continuous_scale='Blues',
                        labels={'asistencias_totales': 'Asistencias Totales', 'name': 'Jugador'},
                        hover_data=['team', 'position', 'appearance'])
            fig.update_layout(showlegend=False, yaxis={'categoryorder': 'total ascending'})
            
        elif graph_type == 'contribution':
            top = df_f.nlargest(20, 'contribucion_total')
            fig = px.bar(top, y='name', x='contribucion_total', orientation='h',
                        color='contribucion_total', color_continuous_scale='Greens',
                        labels={'contribucion_total': 'Goles + Asistencias', 'name': 'Jugador'},
                        hover_data=['team', 'position', 'appearance'])
            fig.update_layout(showlegend=False, yaxis={'categoryorder': 'total ascending'})
            
        elif graph_type == 'value_market':
            df_val = df_f[df_f['current_value'] > 0]
            if len(df_val) > 0:
                top = df_val.nlargest(20, 'current_value')
                fig = px.bar(top, y='name', x='current_value', orientation='h',
                            color='current_value', color_continuous_scale='YlOrRd',
                            labels={'current_value': 'Valor (€)', 'name': 'Jugador'},
                            hover_data=['team', 'position'])
                fig.update_layout(showlegend=False, yaxis={'categoryorder': 'total ascending'})
                fig.update_xaxes(tickformat='.0s')
            else:
                fig = px.scatter(title="No hay datos de valor de mercado")
                
        elif graph_type == 'minutes_per_game':
            top = df_f.nlargest(20, 'minutos_por_partido')
            fig = px.bar(top, y='name', x='minutos_por_partido', orientation='h',
                        color='minutos_por_partido', color_continuous_scale='Purples',
                        labels={'minutos_por_partido': 'Minutos/Partido', 'name': 'Jugador'},
                        hover_data=['team', 'position', 'minutes played'])
            fig.update_layout(showlegend=False, yaxis={'categoryorder': 'total ascending'})
            
        elif graph_type == 'yellow_cards':
            top = df_f.nlargest(20, 'yellow cards')
            fig = px.bar(top, y='name', x='yellow cards', orientation='h',
                        color='yellow cards', color_continuous_scale='YlOrRd',
                        labels={'yellow cards': 'Tarjetas Amarillas', 'name': 'Jugador'},
                        hover_data=['team', 'position', 'appearance'])
            fig.update_layout(showlegend=False, yaxis={'categoryorder': 'total ascending'})
            
        elif graph_type == 'red_cards':
            df_red = df_f[df_f['red cards'] > 0]
            if len(df_red) > 0:
                top = df_red.nlargest(20, 'red cards')
                fig = px.bar(top, y='name', x='red cards', orientation='h',
                            color='red cards', color_continuous_scale='Reds',
                            labels={'red cards': 'Tarjetas Rojas', 'name': 'Jugador'},
                            hover_data=['team', 'position', 'appearance'])
                fig.update_layout(showlegend=False, yaxis={'categoryorder': 'total ascending'})
            else:
                fig = px.scatter(title="No hay jugadores con tarjetas rojas")
                
        elif graph_type == 'age_dist':
            fig = px.histogram(df_f, x='age', nbins=15,
                             labels={'age': 'Edad', 'count': 'Jugadores'},
                             color_discrete_sequence=['steelblue'])
            fig.update_layout(showlegend=False)
            
        elif graph_type == 'goals_assists_scatter':
            sample = df_f.sample(min(200, len(df_f)))
            fig = px.scatter(sample, x='goles_totales', y='asistencias_totales',
                           hover_name='name', hover_data=['team', 'position', 'appearance'],
                           color='position', size='contribucion_total',
                           labels={'goles_totales': 'Goles', 'asistencias_totales': 'Asistencias'})
        
        # Gráficos para porteros
        elif graph_type == 'clean_sheets':
            df_cs = df_f[df_f['clean sheets'] > 0]
            if len(df_cs) > 0:
                top = df_cs.nlargest(20, 'clean sheets')
                fig = px.bar(top, y='name', x='clean sheets', orientation='h',
                            color='clean sheets', color_continuous_scale='Greens',
                            labels={'clean sheets': 'Porterías a Cero', 'name': 'Portero'},
                            hover_data=['team', 'appearance'])
                fig.update_layout(showlegend=False, yaxis={'categoryorder': 'total ascending'})
            else:
                fig = px.scatter(title="No hay datos de porterías a cero")
                
        elif graph_type == 'goals_conceded':
            df_gc = df_f[df_f['goals conceded'] > 0]
            if len(df_gc) > 0:
                top = df_gc.nsmallest(20, 'goals conceded')  # Los MEJORES conceden MENOS
                fig = px.bar(top, y='name', x='goals conceded', orientation='h',
                            color='goals conceded', color_continuous_scale='Reds_r',
                            labels={'goals conceded': 'Goles Concedidos', 'name': 'Portero'},
                            hover_data=['team', 'appearance'])
                fig.update_layout(showlegend=False, yaxis={'categoryorder': 'total descending'})
            else:
                fig = px.scatter(title="No hay datos de goles concedidos")
                
        elif graph_type == 'clean_sheets_per_game':
            df_cs = df_f[df_f['clean sheets'] > 0]
            if len(df_cs) > 0:
                df_cs = df_cs.copy()
                df_cs['cs_per_game'] = df_cs['clean sheets'] / df_cs['appearance']
                top = df_cs.nlargest(20, 'cs_per_game')
                fig = px.bar(top, y='name', x='cs_per_game', orientation='h',
                            color='cs_per_game', color_continuous_scale='Greens',
                            labels={'cs_per_game': 'Porterías a Cero/Partido', 'name': 'Portero'},
                            hover_data=['team', 'clean sheets', 'appearance'])
                fig.update_layout(showlegend=False, yaxis={'categoryorder': 'total ascending'})
            else:
                fig = px.scatter(title="No hay datos de porterías a cero")
                
        elif graph_type == 'goals_conceded_per_game':
            df_gc = df_f[df_f['goals conceded'] > 0]
            if len(df_gc) > 0:
                df_gc = df_gc.copy()
                df_gc['gc_per_game'] = df_gc['goals conceded'] / df_gc['appearance']
                top = df_gc.nsmallest(20, 'gc_per_game')  # Los MEJORES conceden MENOS por partido
                fig = px.bar(top, y='name', x='gc_per_game', orientation='h',
                            color='gc_per_game', color_continuous_scale='Reds_r',
                            labels={'gc_per_game': 'Goles Concedidos/Partido', 'name': 'Portero'},
                            hover_data=['team', 'goals conceded', 'appearance'])
                fig.update_layout(showlegend=False, yaxis={'categoryorder': 'total descending'})
            else:
                fig = px.scatter(title="No hay datos de goles concedidos")
                
        elif graph_type == 'clean_goals_scatter':
            df_gk = df_f[(df_f['clean sheets'] > 0) | (df_f['goals conceded'] > 0)]
            if len(df_gk) > 0:
                fig = px.scatter(df_gk, x='clean sheets', y='goals conceded',
                               hover_name='name', hover_data=['team', 'appearance'],
                               size='appearance', color='team',
                               labels={'clean sheets': 'Porterías a Cero', 'goals conceded': 'Goles Concedidos'})
            else:
                fig = px.scatter(title="No hay datos para comparar")
        
        else:
            fig = px.scatter(title="Selecciona un tipo de gráfico")
        
        fig.update_layout(template='plotly_white', margin=dict(l=20, r=20, t=40, b=20))
        return fig
    
    return create_perf_graph(g1_type), create_perf_graph(g2_type), create_perf_graph(g3_type), create_perf_graph(g4_type)

@app.callback(Output('val-result', 'children'), Input('val-player', 'value'))
def predict_value(player_idx):
    if player_idx is None or not ML_ENABLED:
        return ""
    
    player = df.iloc[player_idx]
    
    # Preparar features
    player_features = {}
    for feat in ml_features:
        if feat in player.index:
            player_features[feat] = player[feat]
        else:
            player_features[feat] = 0
    
    X = pd.DataFrame([player_features])
    X_scaled = ml_scaler.transform(X)
    prediction = ml_model.predict(X_scaled)[0]
    
    real_value = player['current_value']
    diff = prediction - real_value
    diff_pct = (diff / real_value * 100) if real_value > 0 else 0
    
    return html.Div([
        dbc.Alert([
            html.H4(player['name'], className="alert-heading"),
            html.Hr(),
            html.P([
                html.Strong("Valor Real: "), f"€{real_value:,.0f}", html.Br(),
                html.Strong("Predicción: "), f"€{prediction:,.0f}", html.Br(),
                html.Strong("Diferencia: "), f"€{abs(diff):,.0f} ({abs(diff_pct):.1f}%)", html.Br(),
            ])
        ], color="success" if abs(diff_pct) < 20 else "warning")
    ])

@app.callback(
    [Output('cluster-scatter', 'figure'), Output('cluster-table', 'children')],
    [Input('cluster-filter', 'value'), Input('cluster-value', 'value')]
)
def update_clustering(cluster, value):
    if not CLUSTERING_ENABLED:
        return {}, ""
    
    data = clustering_model['data'].copy()
    labels = clustering_model['labels']
    
    if cluster != 'all':
        data = data[data['cluster'] == cluster]
    if value < 100:
        data = data[data['current_value'] <= value * 1e6]
    
    data['label'] = data['cluster'].map({i: labels[i] for i in range(6)})
    
    fig = px.scatter(data, x='pca_x', y='pca_y', color='label', hover_name='name',
                    hover_data={'team': True, 'goles_totales': True, 'current_value': '€:,.0f',
                               'pca_x': False, 'pca_y': False, 'label': False})
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    fig.update_layout(height=600, showlegend=True)
    
    table = dbc.Table.from_dataframe(
        data.nlargest(30, 'current_value')[['name', 'team', 'position', 'goles_totales', 
                                            'asistencias_totales', 'current_value']].rename(columns={
            'name': 'Jugador', 'team': 'Equipo', 'position': 'Pos',
            'goles_totales': 'G', 'asistencias_totales': 'A', 'current_value': 'Valor (€)'
        }),
        striped=True, bordered=True, hover=True, size='sm'
    ) if len(data) > 0 else html.P("Sin jugadores")
    
    return fig, table

@app.callback(Output('bargains-chart', 'figure'), Input('url', 'pathname'))
def update_bargains(path):
    if path != '/bargains' or not ANOMALY_ENABLED:
        return {}
    
    gangas = anomaly_model['gangas'].head(30)
    fig = px.bar(gangas, x='ratio', y='name', orientation='h', color='ratio',
                 color_continuous_scale='Viridis',
                 hover_data={'team': True, 'goles_totales': True, 'current_value': '€:,.0f'})
    fig.update_layout(showlegend=False, yaxis={'categoryorder': 'total ascending'}, height=700)
    return fig

@app.callback(
    Output('rec-results', 'children'),
    [Input('rec-btn', 'n_clicks')],
    [State('rec-player', 'value')]
)
def recommend(n_clicks, player_idx):
    if not n_clicks or player_idx is None or not RECOMMENDATION_ENABLED:
        return ""
    
    players = recommendation_model['players_data']
    selected = players.iloc[player_idx]
    
    if 'top_indices' in recommendation_model:
        top_idx = recommendation_model['top_indices'][player_idx][1:11]
        similarities = recommendation_model['top_scores'][player_idx][1:11]
    else:
        sim_matrix = recommendation_model['similarity_matrix']
        sims = sim_matrix[player_idx]
        top_idx = np.argsort(sims)[-11:-1][::-1]
        similarities = sims[top_idx]
    
    player_card = dbc.Card([
        dbc.CardHeader(html.H4([html.I(className="fas fa-user me-2"), "Jugador Seleccionado"])),
        dbc.CardBody([html.Div([
            html.H3(selected['name'], className="mb-2"),
            html.P([
                html.Strong("Equipo: "), selected['team'], html.Br(),
                html.Strong("Posición: "), selected['position'], html.Br(),
                html.Strong("Goles: "), f"{int(selected['goles_totales'])}", html.Br(),
                html.Strong("Asistencias: "), f"{int(selected['asistencias_totales'])}", html.Br(),
                html.Strong("Valor: "), f"€{selected['current_value']:,.0f}"
            ])
        ])])
    ], className="shadow-sm mb-4", color="primary", outline=True)
    
    similar_cards = []
    for rank, (idx, sim) in enumerate(zip(top_idx, similarities), 1):
        s = players.iloc[idx]
        sim_pct = sim * 100
        
        color = "success" if sim_pct >= 95 else ("info" if sim_pct >= 90 else "secondary")
        
        card = dbc.Col([dbc.Card([
            dbc.CardHeader(html.H6(f"#{rank} - {sim_pct:.1f}% similar", className="text-white"),
                          style={'backgroundColor': COLORS['primary']}),
            dbc.CardBody([
                html.H5(s['name'], className="mb-2"),
                html.P([
                    html.I(className="fas fa-shield-alt me-1"), s['team'], html.Br(),
                    html.I(className="fas fa-user-tag me-1"), s['position'], html.Br(),
                    html.Strong("G:"), f"{int(s['goles_totales'])} ",
                    html.Strong("A:"), f"{int(s['asistencias_totales'])}", html.Br(),
                    html.Strong("€"), f"{s['current_value']/1e6:.1f}M"
                ], className="small")
            ])
        ], className="shadow-sm h-100", outline=True, color=color)], 
        width=12, md=6, lg=4, xl=3, className="mb-3")
        similar_cards.append(card)
    
    return html.Div([
        player_card,
        html.H3([html.I(className="fas fa-users me-2"), "Jugadores Similares"], className="mb-3 mt-4"),
        dbc.Row(similar_cards)
    ])

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8050)
