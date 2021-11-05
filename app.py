import pathlib
import dash
import datetime as dt
import pandas as pd
from dash.dependencies import Input, Output, State, ClientsideFunction
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
import plotly.graph_objects as go
import numpy as np
import requests
import json

from scipy import stats as sps
from scipy.interpolate import interp1d
from scipy.integrate import odeint

# -----------------------------
# Load static data
# -----------------------------

PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()
CONFIG_PATH = PATH.joinpath("config").resolve()
GAMMA_RT = 1/7     #1/serial interval 
RT_MAX = 12    #every possible value of Rt
rt_range = np.linspace(0, RT_MAX, RT_MAX*100+1)

date_format = '%Y-%m-%dT%H:%M:%S'               #DB date format
dpr_format = '%Y-%m-%d'                         #Date Picker Range format
sns_format = '%m/%d/%Y'

provincias = pd.read_csv(DATA_PATH.joinpath('provincias.csv'))

df_conf = pd.read_csv(DATA_PATH.joinpath('ops-conf.csv'))

df_conf['confirmado']=df_conf['confirmado'].astype(int)
df_conf['prov']=df_conf['prov'].astype(int)
df_conf['fecha_confirmado'] = pd.to_datetime(df_conf['fecha_confirmado'], format=date_format)

df_def = pd.read_csv(DATA_PATH.joinpath('ops-def.csv'))
df_def['fallecido']=df_def['fallecido'].astype(int)
df_def['prov']=df_def['prov'].astype(int)
df_def['fecha_fallecido'] = pd.to_datetime(df_def['fecha_fallecido'], format=date_format)

df_sns = pd.read_csv(DATA_PATH.joinpath('sns.csv'))
df_sns['fecha'] = pd.to_datetime(df_sns['fecha'], format=sns_format)

# -----------------------------
# Declare APP
# -----------------------------
app = dash.Dash(__name__)
server = app.server
app.title = 'Calculadora Epidemiológica - República Dominicana'

# -----------------------------
# App layout section
# -----------------------------
app.layout = html.Div(
    [
        # empty Div to trigger javascript file for graph resizing
        # Logos y titulo del dashboard
        html.Div(
            [
                html.Div(
                    [
                        html.A(
                            html.Img(
                                src=app.get_asset_url("Logo_SaludPublica.png"),
                                style={
                                    "height": "150px",
                                    "width": "auto",
                                    "margin-bottom": "25px",
                                },
                            ),
                            href="https://www.msp.gob.do/",
                        )
                        
                    ],
                    className="one-third column",
                    style={'textAlign': 'left'},
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H4(
                                    "Calculadora Epidemiológica para la República Dominicana",
                                    style={
                                        "margin-bottom": "0px",
                                        "textAlign": "center"
                                    },
                                ),
                            ]
                        )
                    ],
                    className="one-third column",
                    style={'textAlign': 'center'},
                ),
                html.Div(
                    [
                        html.A(
                            html.Img(
                                src=app.get_asset_url("Logo_DIGEPI.png"),
                                style={
                                    "height": "150px",
                                    "width": "auto",
                                    "margin-bottom": "25px",
                                },
                            ),
                            href="http://www.digepisalud.gob.do/",
                        )
                    ],
                    className="one-third column",
                    style={'textAlign': 'right'},
                ),
            ],
            id="menu",
            className="row flex-display",
            style={"margin-bottom": "25px"},
        ),

        #Botones de selección
        html.Div(
            [
                html.Div(
                    [
                        html.P("Rango de Fecha: ", className="control_label"),
                        dcc.DatePickerRange(
                                id="time_range",
                                display_format="DD/MMM/YYYY ",
                                min_date_allowed=dt.datetime(2020, 2, 1),
                                day_size=50,
                                start_date_placeholder_text="Fecha de Inicio",
                                end_date_placeholder_text="Fecha Final",
                                start_date=dt.datetime(2020, 2, 1),
                                end_date = dt.datetime(2021, 10, 31),
                            ),
                        html.Br(),

                        html.P("Provincia: ", className="control_label"),
                        dcc.Dropdown(
                            id='provincia_dropdown',
                            options=[{'label':name_prov, 'value':id_prov} for name_prov, id_prov in zip(provincias["provincia"], provincias["id"])],
                            value=0,
                            multi=False,
                            clearable=False
                        ),
                        html.Br(),

                        html.P("Agrupación por:", className="control_label"),
                        dcc.RadioItems(
                                id="group_type_selector",
                                options=[
                                    {"label": "Día", "value": "D"},
                                    {"label": "Semana", "value": "W"},
                                    {"label": "Mes", "value": "M"},
                                ],
                                value="W",
                                labelStyle={"display": "inline-block"},
                                className="dcc_control",
                            ),
                        html.Br(),

                        html.P("Filtrar por sexo:", className="control_label"),
                        dcc.RadioItems(
                                id="sex_selector",
                                options=[
                                    {"label": "Ambos", "value": "T"},
                                    {"label": "Femenino", "value": "F"},
                                    {"label": "Masculino", "value": "M"},
                                ],
                                value="T",
                                labelStyle={"display": "inline-block"},
                                className="dcc_control",
                            ),                        
                    ],

                    className="pretty_container three columns",
                ),
                html.Div(
                    [
                        dcc.Tabs(
                            [
                                dcc.Tab(label='Casos confirmados', children=[
                                    dcc.Graph(id="acumulados_cumsum")
                                ],
                                ),
                                dcc.Tab(label='Defunciones', children=[
                                    dcc.Graph(id="def_cumsum")
                                ],
                                ),
                                dcc.Tab(label='Número de Reproducción Efectivo', children=[
                                    dcc.Graph(id="rt")
                                ],
                                ),
                                dcc.Tab(label='Tasa de Positividad', children=[
                                    dcc.Graph(id="positivity_rate")
                                ],
                                ),
                                dcc.Tab(label='Tasa de Letalidad', children=[
                                    dcc.Graph(id="letality_rate")
                                ],
                                ),
                                dcc.Tab(label='Uso de camas para Covid 19', children=[
                                    dcc.Graph(id="uso_camas")
                                ],
                                ),
                                dcc.Tab(label='Uso de camas UCI para Covid 19', children=[
                                    dcc.Graph(id="uso_uci")
                                ],
                                ),
                                dcc.Tab(label='Uso de ventiladores para Covid 19', children=[
                                    dcc.Graph(id="uso_ventiladores")
                                ],
                                ),
                            ]
                        )

                    ],
                    className="pretty_container nine columns",

                ),

                
            ],
            className="row flex-display",
        ),
        
        #Modelo SEIR
        html.Div(
            [
                html.Div(
                    [
                        dcc.Tabs(
                            [
                                dcc.Tab(label='Estimación de casos por fracción de la población', children=[
                                    dcc.Graph(id="seir_pob")
                                ],
                                ),
                                dcc.Tab(label='Estimación de casos acumulados por fracción de la población', children=[
                                    dcc.Graph(id="seir_cum")
                                ],
                                ),
                                dcc.Tab(label='Estimación de casos por restricción de movilidad', children=[
                                    dcc.Graph(id="seir_mov")
                                ],
                                ),
                                dcc.Tab(label='Estimación de casos por fracción de la población con restricción', children=[
                                    dcc.Graph(id="seir_pobmov")
                                ],
                                ),
                                dcc.Tab(label='Estimación de casos acumulados por fracción de la población con restricción', children=[
                                    dcc.Graph(id="seir_cummov")
                                ],
                                ),
                            ]
                        )
                    ],
                    className="pretty_container nine columns",
                ),
                html.Div(
                    [
                        html.P("Días de estimación: ", className="control_label"),
                        dcc.Slider(
                            id='estimation_date',
                            min=0,
                            max=550,
                            step=1,
                            value=100,
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                        html.Br(),
                        html.P("Período medio de incubación: ", className="control_label"),
                        dcc.Slider(
                            id='incubation_period',
                            min=0,
                            max=45,
                            step=1,
                            value=5,
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                        html.Br(),

                        html.P("Duración media de la enfermedad: ", className="control_label"),
                        dcc.Slider(
                            id='illness_duration',
                            min=0,
                            max=45,
                            step=1,
                            value=7,
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                        html.Br(),

                        html.P("Valores de Rt a evaluar", className="control_label"),
                        dcc.RangeSlider(
                            id='rt_rs',
                            min=0,
                            max=6,
                            step=0.1,
                            value=[1.6,3],
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                        html.Br(),

                        html.P("Escenarios de control de movilidad", className="control_label"),
                        dcc.RangeSlider(
                            id='eta',
                            min=0,
                            max=1,
                            step=0.1,
                            value=[0.1,0.3],
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                        html.Br(),             
                    ],

                    className="pretty_container three columns",
                ),
                
            ],
            className="row flex-display",
        ),

        #Guardar info importante
        dcc.Store(id='intermediate-value'),

    ],
    id="mainContainer",
    style={"display": "flex", "flex-direction": "column"},

    
)

# -----------------------------
# Function section
# -----------------------------

def filter_confirmado_data(df,group_type_selector, province_selector, sex_selector, start_time, end_time):    
    #filter by province
    if province_selector >0:
        df = df.loc[df['prov'] == province_selector]

    #filter by sex
    if sex_selector != 'T':
        df = df.loc[df['sexo'] == sex_selector]
    
    #Group by period
    df_filter = df.groupby([pd.Grouper(key='fecha_confirmado', freq=group_type_selector)]).agg({'confirmado':'sum', 'cedula':'count'}).reset_index()
    df_filter['cum'] = df_filter['confirmado'].cumsum()
    df_filter['cum_casos'] = df_filter['cedula'].cumsum()
    df_filter['positividad']= (df_filter['cum']/df_filter['cum_casos'])*100

    #filter by time
    df_filter =  df_filter.loc[(df_filter['fecha_confirmado']>= start_time) & (df_filter['fecha_confirmado']<= end_time)]

    return df_filter

def filter_death_data(df,group_type_selector, province_selector, sex_selector, start_time, end_time):    
    #filter by province
    if province_selector >0:
        df = df.loc[df['prov'] == province_selector]

    #filter by sex
    if sex_selector != 'T':
        df = df.loc[df['sexo'] == sex_selector]
    
    #Group by period
    df_filter = df.groupby([pd.Grouper(key='fecha_fallecido', freq=group_type_selector)]).agg({'fallecido':'sum'}).reset_index()
    df_filter['cum_death'] = df_filter['fallecido'].cumsum()

    #filter by time
    df_filter =  df_filter.loc[(df_filter['fecha_fallecido']>= start_time) & (df_filter['fecha_fallecido']<= end_time)]

    return df_filter

def filter_sns_data(df,group_type_selector, start_time, end_time):    
  
    #Group by period
    if group_type_selector !='D':
        df_filter = df.groupby([pd.Grouper(key='fecha', freq=group_type_selector)]).agg({'camas_ocupadas':'mean','uci_ocupadas':'mean','ventiladores_ocupados':'mean','capacidad_camas':'mean','capacidad_uci':'mean','capacidad_ventiladores':'mean'}).reset_index()
    
    #filter by time
    df_filter =  df_filter.loc[(df_filter['fecha']>= start_time) & (df_filter['fecha']<= end_time)]

    df_filter['tcamas']=(df_filter['camas_ocupadas']/df_filter['capacidad_camas'])*100
    df_filter['tuci']=(df_filter['uci_ocupadas']/df_filter['capacidad_uci'])*100
    df_filter['tventiladores']=(df_filter['ventiladores_ocupados']/df_filter['capacidad_ventiladores'])*100

    return df_filter

def graficar_marker(df, title, marker, marker_date):
        
        layout = dict(
            autosize=True,
            automargin=True,
            margin=dict(l=50, r=50, b=20, t=40),
            hovermode="closest",
            plot_bgcolor="#F1F1F1",
            paper_bgcolor="#F1F1F1",
            legend=dict(font=dict(size=10), orientation="h"),
            title=title,
            yaxis=dict(
                rangemode='nonnegative'
            )
        )
        data =[
            dict(
                type="scatter",
                mode="lines+markers",
                x=df[marker_date],
                y=df[marker],
                line=dict(shape="spline", smoothing="2", color='#346888'),
            ),
        ]    
        figure = dict(data=data, layout=layout)
        
        return figure

def get_posteriors(sr, sigma=0.15):
    lam = sr[:-1].values * np.exp(GAMMA_RT * (rt_range[:,None] -1))

    likelihoods = pd.DataFrame( data = sps.poisson.pmf(sr[1:].values, lam), index = rt_range, columns = sr.index[1:])   #The likelihood for each day

    process_matrix = sps.norm(loc=rt_range, scale=sigma).pdf(rt_range[:, None]) #Gaus matrix
    process_matrix /= process_matrix.sum(axis=0)                                #Normalized sothat it adds to 1

    prior = np.ones_like(rt_range)/len(rt_range)    #Initial prior
    prior /= prior.sum()

    posteriors = pd.DataFrame(index = rt_range, columns = sr.index, data={sr.index[0]:prior})   #posterior for each day

    log_likelihood = 0.0

    for prev_day, current_day in zip(sr.index[:-1], sr.index[1:]):
        current_prior = process_matrix @ posteriors[prev_day]   #Matrix multiplication for Current prior 

        numerator = likelihoods[current_day] * current_prior    #Bayes' Rule numerator: P(k|Rt)P(Rt)
        denominator = np.sum(numerator)                         #Bayes' Rule denominator: P(k)
        posteriors[current_day] = numerator/denominator         #Bayes' Rule

        log_likelihood += np.log(denominator)

    return posteriors, log_likelihood

def hdi(pmf, p=.9, debug=False):

    if(isinstance(pmf, pd.DataFrame)):          #recursive call on the columns
        return pd.DataFrame([hdi(pmf[col],p=p) for col in pmf], index=pmf.columns)

    cumsum = np.cumsum(pmf.values)

    total_p = cumsum - cumsum[:, None]   #NxN matrix of the total probability mass for each low and high

    lows, highs = (total_p > p).nonzero() #all indices where total_p > p

    try:
        best = (highs - lows).argmin()      #Find the highest density 
        low = pmf.index[lows[best]]
        high = pmf.index[highs[best]]
    except:
        low = pmf.index[0]
        high = pmf.index[-1]

    return pd.Series([low, high],
                     index=[f'Low_{p*100:.0f}',
                            f'High_{p*100:.0f}'])

def seir_f2(t, y, beta, sigma, gamma, a, b):
    s, e, i, r = y
    return np.array([-beta * i * s + a,
                     -sigma * e + beta * i * s  + b, 
                     -gamma * i + sigma * e, 
                     gamma * i - (a + b)])

def F(x, t, gamma, sigma, R0):
    """
    Time derivative of the state vector.

        * x is the state vector (array_like)
        * t is time (scalar)
        * R0 is the effective transmission rate, defaulting to a constant

    """
    s, e, i = x

    # New exposure of susceptibles
    beta = R0(t) * gamma if callable(R0) else R0 * gamma
    ne = beta * s * i

    # Time derivatives
    ds = - ne
    de = ne - sigma * e
    di = sigma * e - gamma * i

    return ds, de, di

def solve_path(R0, t_vec, x_init, gamma, sigma):
    """
    Solve for i(t) and c(t) via numerical integration,
    given the time path for R0.

    """
    G = lambda x, t: F(x, t, gamma, sigma, R0)
    s_path, e_path, i_path = odeint(G, x_init, t_vec).transpose()

    c_path = 1 - s_path - e_path       # cumulative cases
    return i_path, c_path

def graficar_seir(paths, labels, title, times):
        
        layout = dict(
            autosize=True,
            automargin=True,
            margin=dict(l=50, r=50, b=20, t=40),
            hovermode="closest",
            plot_bgcolor="#F1F1F1",
            paper_bgcolor="#F1F1F1",
            legend=dict(font=dict(size=10), orientation="h"),
            title=title,
            yaxis=dict(
                rangemode='nonnegative'
            )
        )

        colors = ["#004c6d", "#346888", "#5886a5", "#7aa6c2", "#9dc6e0", "#c1e7ff"]
        data = []

        for path, label, color in zip(paths, labels, colors):
            data.append(
                dict(
                    type="scatter",
                    mode="lines",
                    name=label,
                    x=times,
                    y=path,
                    line=dict(shape="spline", smoothing="2", color=color),

                )
            )   
        figure = dict(data=data, layout=layout)
        
        return figure

def R0_mitigating(t, r0, η, r_bar):
    R0 = r0 * np.exp(- η * t) + (1 - np.exp(- η * t)) * r_bar
    return R0
# -----------------------------
# Callbacks section
# -----------------------------
@app.callback(
    [
        Output('acumulados_cumsum', 'figure'),
        Output('def_cumsum', 'figure'),
        Output('rt', 'figure'),
        Output('positivity_rate', 'figure'),
        Output('letality_rate', 'figure'),
        Output('uso_camas', 'figure'),
        Output('uso_uci', 'figure'),
        Output('uso_ventiladores', 'figure'),
        Output('intermediate-value','data'),
    ], 
    [
        Input('time_range', 'start_date'),
        Input('time_range', 'end_date'),
        Input('provincia_dropdown', 'value'),
        Input('group_type_selector', 'value'),
        Input('sex_selector', 'value'),
    ]
)
def update_cumsum(start_date, end_date, provincia, group_type, sex):
    confirmado = filter_confirmado_data(df_conf,group_type,provincia,sex,start_date,end_date)
    death = filter_death_data(df_def,group_type,provincia,sex,start_date,end_date)

    confirmado['smoothed'] = confirmado['confirmado'].rolling(7, win_type='gaussian', min_periods=1, center=True).mean(std=2).round()
    fecha = confirmado.set_index('fecha_confirmado')
    posteriors, log_likelihood = get_posteriors(fecha['smoothed'], 0.25)
    hdis= hdi(posteriors)
    most_likely = posteriors.idxmax().rename('ML')
    most_likely = most_likely.reset_index()

    letalidad = pd.merge(left=confirmado, right=death, left_on='fecha_confirmado', right_on='fecha_fallecido',)
    letalidad['letalidad']=letalidad['cum_death']/letalidad['cum']*100

    sns = filter_sns_data(df_sns,group_type, start_date, end_date)

    results =[]
    results.append(graficar_marker(confirmado,'Casos confirmados acumulado (personas)','cum','fecha_confirmado'))
    results.append(graficar_marker(death,'Muertes acumuladas (personas)','cum_death','fecha_fallecido'))
    results.append(graficar_marker(most_likely,'Número de Reproducción Efectivo (Rt)','ML','fecha_confirmado'))
    results.append(graficar_marker(confirmado,'Tasa de positividad (%)','positividad','fecha_confirmado'))
    results.append(graficar_marker(letalidad,'Tasa de letalidad (%)','letalidad','fecha_confirmado'))
    results.append(graficar_marker(sns,'Tasa ocupación de camas para Covid-19 (%)','tcamas','fecha'))
    results.append(graficar_marker(sns,'Tasa ocupación de UCI para Covid-19 (%)','tuci','fecha'))
    results.append(graficar_marker(sns,'Tasa ocupación de ventiladores para Covid-19 (%)','tventiladores','fecha'))

    data = {'cum':str(confirmado['cum'].max()), 'cum_casos':str(confirmado['cum_casos'].max())}

    results.append(json.dumps(data))

    return results

@app.callback(
    [
        Output('seir_pob', 'figure'),
        Output('seir_cum', 'figure'),
        Output('seir_mov', 'figure'),
        Output('seir_pobmov', 'figure'),
        Output('seir_cummov', 'figure'),
    ], 
    [
        Input('intermediate-value', 'data'),
        Input('estimation_date', 'value'),
        Input('incubation_period', 'value'),
        Input('illness_duration', 'value'),
        Input('provincia_dropdown', 'value'),
        Input('rt_rs', 'value'),
        Input('eta', 'value'),

        
    ]
)
def update_seir(data, t_len, incubation_period, illness_duration, provincia, rt, eta):

    pop_size = provincias['poblacion'].iloc[provincia]

    gamma = 1/incubation_period
    sigma = 1/illness_duration

    if data:
        df = json.loads(data)
    else:
        df = {'cum':385000, 'cum_casos':2000000}

    i_0 = int(df['cum'])/pop_size
    e_0 = int(df['cum_casos'])/pop_size
    s_0 = 1 - i_0 - e_0

    x_0 = s_0, e_0, i_0

    t_vec = np.linspace(0, t_len, 1000)
    R0_vals = np.linspace(rt[0], rt[1], 6)
    labels = [f'Rt = {r:.2f}' for r in R0_vals]
    i_paths, c_paths = [], []

    for r in R0_vals:
        i_path, c_path = solve_path(r, t_vec, x_0, gamma, sigma)
        i_paths.append(i_path)
        c_paths.append(c_path)

    results = []
    results.append(graficar_seir(i_paths, labels, 'Cantidad de casos por fracción de población', t_vec))
    results.append(graficar_seir(c_paths, labels, 'Cantidad de casos por fracción de población', t_vec))

    eta_vals = np.linspace(eta[0], eta[1], 6)
    labels = [f'η = {r:.2f}' for r in eta_vals]

    etas =[]
    for i in eta_vals:
        etas.append(R0_mitigating(t_vec,rt[0], i, rt[1]))

    results.append(graficar_seir(etas, labels, 'Mitigación por diferentes tasas de restricción de movilidad', t_vec))

    i_paths, c_paths = [], []
    for i in eta_vals:
        r = lambda t: R0_mitigating(t,rt[0], i, rt[1])
        i_path, c_path = solve_path(r, t_vec, x_0, gamma, sigma)
        i_paths.append(i_path)
        c_paths.append(c_path)
    
    results.append(graficar_seir(i_paths, labels, 'Cantidad de casos por fracción de población con restriccion', t_vec))
    results.append(graficar_seir(c_paths, labels, 'Cantidad de casos por fracción de población con restriccion', t_vec))

    return results



# -----------------------------
# Main
# -----------------------------
if __name__ == '__main__':
    app.run_server(debug=True)