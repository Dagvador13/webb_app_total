import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import base64
import datetime
import io
import numpy as np

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash import no_update
import dash_table
import virtualenv

# --------------------------------------------------

# Back end functions

# Transforming the data given by the file path, retrieving all necessary infos
def data_transformation(file_path):

    # Reading the file
    data = pd.read_csv(file_path, delimiter=';')

    # Sorting the values by perm and then by height
    data.sort_values(by=['Perm (mD)','Gross_Height (m)'])

    # Filling the zero values with 0 just in case
    data.fillna(0, inplace=True)

    # Adding the two missing columns
    data['Delta_Vert_Hz_NoCutoff'] = data[data.columns[11]] - data[data.columns[13]]
    data['Delta_Vert_Hz_Cutoff'] = data[data.columns[12]] - data[data.columns[14]]

    # Capping the values to -10 et 10 for the Delta
    data['Delta_Vert_Hz_Cutoff'] = data['Delta_Vert_Hz_Cutoff'].where(data['Delta_Vert_Hz_Cutoff'] > -10, -10)
    data['Delta_Vert_Hz_NoCutoff'] = data['Delta_Vert_Hz_NoCutoff'].where(data['Delta_Vert_Hz_NoCutoff'] > -10, -10)
    data['Delta_Vert_Hz_Cutoff'] = data['Delta_Vert_Hz_Cutoff'].where(data['Delta_Vert_Hz_Cutoff'] < 10, 10)
    data['Delta_Vert_Hz_NoCutoff'] = data['Delta_Vert_Hz_NoCutoff'].where(data['Delta_Vert_Hz_NoCutoff'] < 10, 10)

    # Parameters for the matrix
    perm = np.array(data[data.columns[0]])
    height = np.array(data[data.columns[1]])

    # Getting the two units (the one for EUR which are 4 first columns
    # and the one for the costs which are the 6 last columns)
    unit_EUR = data.columns[7].split('(')[1].replace(')', '')
    unit_COST = data.columns[11].split('(')[1].replace(')', '')


    # Setting the possible targets
    EUR_Vert_NoCutoff = np.array(data[data.columns[7]])
    EUR_Vert_Cutoff = np.array(data[data.columns[8]])
    EUR_Hz_NoCutoff = np.array(data[data.columns[9]])
    EUR_Hz_Cutoff = np.array(data[data.columns[10]])
    Cost_Vert_NoCutoff = np.array(data[data.columns[11]])
    Cost_Vert_Cutoff = np.array(data[data.columns[12]])
    Cost_Hz_NoCutoff = np.array(data[data.columns[13]])
    Cost_Hz_Cutoff = np.array(data[data.columns[14]])
    Delta_Vert_Hz_NoCutoff = np.array(data[data.columns[15]])
    Delta_Vert_Hz_Cutoff = np.array(data[data.columns[16]])

    # Establishing number of different permeability and gross height
    unique_perm = np.unique(perm)
    unique_height = np.unique(height)

    # Setting the Z matrix size according to previous step
    Z_size = (len(unique_perm), len(unique_height))

    # Calculating all Z matrixes for each target
    Z_EUR_Vert_NoCutoff = EUR_Vert_NoCutoff.reshape(Z_size[0], Z_size[1])
    Z_EUR_Vert_Cutoff = EUR_Vert_Cutoff.reshape(Z_size[0], Z_size[1])
    Z_EUR_Hz_NoCutoff = EUR_Hz_NoCutoff.reshape(Z_size[0], Z_size[1])
    Z_EUR_Hz_Cutoff = EUR_Hz_Cutoff.reshape(Z_size[0], Z_size[1])
    Z_Cost_Vert_NoCutoff = Cost_Vert_NoCutoff.reshape(Z_size[0], Z_size[1])
    Z_Cost_Vert_Cutoff = Cost_Vert_Cutoff.reshape(Z_size[0], Z_size[1])
    Z_Cost_Hz_NoCutoff = Cost_Hz_NoCutoff.reshape(Z_size[0], Z_size[1])
    Z_Cost_Hz_Cutoff = Cost_Hz_NoCutoff.reshape(Z_size[0], Z_size[1])
    Z_Delta_Vert_Hz_NoCutoff = Delta_Vert_Hz_NoCutoff.reshape(Z_size[0], Z_size[1])
    Z_Delta_Vert_Hz_Cutoff = Delta_Vert_Hz_Cutoff.reshape(Z_size[0], Z_size[1])

    # Assigning Z_matrixes to values in a dictionnary
    # Assigning lists also containing the units of each column
    Z_matrixes = {
        1: [Z_EUR_Vert_NoCutoff, unit_EUR],
        2: [Z_EUR_Vert_Cutoff, unit_EUR],
        3: [Z_EUR_Hz_NoCutoff, unit_EUR],
        4: [Z_EUR_Hz_Cutoff, unit_EUR],
        5: [Z_Cost_Vert_NoCutoff, unit_COST],
        6: [Z_Cost_Vert_Cutoff, unit_COST],
        7: [Z_Cost_Hz_NoCutoff, unit_COST],
        8: [Z_Cost_Hz_Cutoff, unit_COST],
        9: [Z_Delta_Vert_Hz_NoCutoff, unit_COST],
        10: [Z_Delta_Vert_Hz_Cutoff, unit_COST]
    }

    # Retrieving max and min_values of prod for each of the 4 columns (production)
    # It will be of use to set min and max on the slider_vmax
    Prod_matrixes = {
        1: [data[data.columns[7]].min(), data[data.columns[7]].max()],
        2: [data[data.columns[8]].min(), data[data.columns[8]].max()],
        3: [data[data.columns[9]].min(), data[data.columns[9]].max()],
        4: [data[data.columns[10]].min(), data[data.columns[10]].max()]
    }

    # Also retrieving columns names for later labels of graphs
    columns = data.columns[7:]

    return Z_matrixes, unique_height, unique_perm, columns, Prod_matrixes

def marks (value_min, value_max):
    middle = int((value_min+value_max)/2)
    value_min_middle = int((value_min+middle)/2)
    value_max_middle = int((value_max+middle)/2)
    return({int(value_min)+1: '{}'.format(int(value_min)+1),
            value_min_middle: '{}'.format(value_min_middle),
            middle: '{}'.format(middle),
            value_max_middle: '{}'.format(value_max_middle),
            int(value_max): '{}'.format(int(value_max))})

# rescale function for centering the white color on zero in the ratio contours
def rescale(v_min, v_max):
    return (abs(v_min/(abs(v_min)+abs(v_max))))

# Marks for sliders
marks_vmax = {0:'0', 2.5:'2.5', 5:'5', 7.5:'7.5', 10:'10'}
marks_vmin = {-10:'-10', -7.5:'-7.5', -5:'-5', -2.5:'-2.5', 0:'0'}
marks_granularity = {0:'0', 0.25:'0.25', 0.5:'0.5', 0.75:'0.75', 1:'1'}

# --------------------------------------------------

# CSS components

# the style arguments for the sidebar.
SIDEBAR_STYLE = {
    'position': 'fixed',
    'top': 0,
    'left': 0,
    'bottom': 0,
    'width': '20%',
    'padding': '20px 10px',
    'background-color': '#f8f9fa'
}

VALUES_STYLE = {
    'margin-left': '25%',
    'width': '10%',
    'top': 0,
    'padding': '20px 10px'
}

# the style arguments for the main content page.
CONTENT_STYLE = {
    'margin-left': '25%',
    'margin-right': '5%',
    'top': 0,
    'padding': '20px 10px'
}

TEXT_STYLE = {
    'textAlign': 'center',
    'color': '#191970'
}

CARD_TEXT_STYLE = {
    'textAlign': 'center',
    'color': '#0074D9'
}

BUTTON_STYLE =  {
  'background-color': '#D1D2D3',
  'border': None,
  'color': 'white',
  'padding': '1px 2px',
  'textAlign': 'center',
  'text-decoration': None,
  'display': 'inline-block',
  'font-size': '16px',
  'height': '30px',
  'width': '100px'
}
# --------------------------------------------------

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

# Controls in the sidebar
controls = dbc.FormGroup(
    [   html.P('Filepath', style={
        'textAlign':'center'
    }),

        # Text Area for entering the file path
        dbc.Textarea(
        id='text_area',
        placeholder='Enter file path here',
        style={
            'width': '100%',
            'height': 50,
            'border-radius': '5px',
            'box-shadow': '1px 1px 1px',
            'margin-bottom': '10px'
        },
        value=None
    ),

        # Output of the text area
        html.Div(id='text_area_output', children={}),

        #dcc.Input(style={"margin-top": "5px"}),

        # Button that triggers retrieving the text area value
        html.Div([dbc.Button('Submit', id='text_area_button', n_clicks=0, size='sm', outline=True, block=True)],
                 ),

        html.Hr(),

        html.P('Select data to display', style={
            'textAlign': 'center'
        }),

        # Dropdown menu to select the data to display
        # Values are set to 1-10, to match with the dictionnaty keys of
        # the Z_matrix extracted from data_transformation function
        dcc.Dropdown(
            id='dropdown',
            options=[
                {'label': 'EUR Vertical No Cutoff', 'value': 1},
                {'label': 'EUR Vertical Cutoff', 'value': 2},
                {'label': 'EUR Horizontal No Cutoff', 'value': 3},
                {'label': 'EUR Horizontal Cutoff', 'value': 4},
                {'label': 'TECH COST Vertical No Cutoff', 'value': 5},
                {'label': 'TECH COST Vertical Cutoff', 'value': 6},
                {'label': 'TECH COST Horizontal No Cutoff', 'value': 7},
                {'label': 'TECH COST Horizontal Cutoff', 'value': 8},
                {'label': 'DELTA Vert - Hz No Cutoff', 'value': 9},
                {'label': 'DELTA Vert - Hz Cutoff', 'value': 10}

            ],
            value=9,  # default value
            multi=False
        ),

        html.Hr(),

        html.Br(),
        html.P('V min', style={
            'textAlign': 'center'
        }),

        # First slider for v_min
        dcc.Slider(
            id='slider_vmin',
            min=-10,  # Range of -10 to 0
            max=0,
            step=0.5,
            value=0,
            marks=marks_vmin,
            disabled=False,  # By default the slider is not disabled
            updatemode='drag'  # Updating the values while dragging
        ),

        html.P(
            id='value_slider_vmin',
            children={}, style={
                'textAlign': 'center'
            }
        ),

        html.Hr(),

        html.Br(),
        html.P('V max', style={
            'textAlign': 'center'
        }),

        # Slider for v_max
        dcc.Slider(
            id='slider_vmax',
            min=0,  # Range of 0-10
            max=10,
            step=0.5,
            value=10,
            marks=marks_vmax,
            updatemode='drag'
        ),

        html.P(
            id='value_slider_vmax',
            children={}, style={
                'textAlign': 'center'
            }
        ),

        html.Hr(),

        html.Br(),
        html.P('Granularity', style={
            'textAlign': 'center'
        }),

        # Slider for granularity
        dcc.Slider(
            id='slider_granularity',
            min=0,
            max=1,
            step=0.05,
            value=0.5,
            marks=marks_granularity,
            updatemode='drag'
        ),

        html.P(
            id='value_slider_granularity',
            children={}, style={
                'textAlign': 'center'
            }
        ),

        html.Hr()

    ]
)

# Grouping controls and parameters in an html.div and applying SIDEBAR_STYLE
sidebar = html.Div(
    [
        html.H2('Parameters', style=TEXT_STYLE),
        html.Hr(),
        controls
    ],
    style=SIDEBAR_STYLE,
)

content_second_row = dbc.Col([
    dbc.Row(
        dbc.Card(
            [
                dbc.CardBody([
                    html.H4(id='card_title_vmin', children=['V Min'], className='CardTitle', style=CARD_TEXT_STYLE),
                    html.P(id='card_text_vmin', children={}, style=CARD_TEXT_STYLE)
                ]
                )
            ]
        )
    ),
    dbc.Row(
        dbc.Card(
            [
                dbc.CardBody([
                    html.H4(id='card_title_vmax', children=['V Max'], className='CardTitle', style=CARD_TEXT_STYLE),
                    html.P(id='card_text_vmax', children={}, style=CARD_TEXT_STYLE)
                ]
                )
            ]
        )
    ),
    dbc.Row(
        dbc.Card(
            [
                dbc.CardBody([
                    html.H4(id='card_title_granularity', children=['Granularity'], className='CardTitle', style=CARD_TEXT_STYLE),
                    html.P(id='card_text_granularity', children={}, style=CARD_TEXT_STYLE)
                ]
                )
            ]
        )
    )
])
# Content part, it contains only a title and the graph, in the CONTENT_STYLE

content = html.Div(
    [
        html.H2('Heatmap Vertical vs Horizontal Wells', style=TEXT_STYLE),
        html.Hr(),
        html.H4(children={}, style=TEXT_STYLE, id='graph_title'),
        dcc.Graph(
            id='graph'
        )
    ],
    style=CONTENT_STYLE
)

cards = html.Div([
    html.H2('Values', style=TEXT_STYLE),
    html.Hr(),
    content_second_row
],
style=VALUES_STYLE)

# Final layout is composed of sidebar and content
app.layout = html.Div([sidebar, content])


# First callback to get the
# @app.callback(
#     Output('text_area', 'value'),
#     Input('text_area_button', 'n_clicks'),
#     Input('text_area', 'value')
# )
# def update_text(n_clicks, value):
#     if n_clicks > 0:
#         return value



@app.callback(
    [Output('graph', 'figure'),
     Output('slider_vmin', 'disabled'),
     Output('slider_vmin', 'value'),
     Output('slider_vmax', 'min'),
     Output('slider_vmax', 'max'),
     Output('slider_vmax', 'marks'),
     Output('slider_vmax', 'step'),
     Output('slider_vmin', 'step'),
     Output('value_slider_vmin', 'children'),
     Output('value_slider_vmax', 'children'),
     Output('value_slider_granularity', 'children'),
     Output('graph_title', 'children')],
    [Input('dropdown', 'value'),
     Input('slider_vmin', 'value'),
     Input('slider_vmax', 'value'),
     Input('slider_granularity', 'value'),
     Input('text_area', 'value'),
     Input('text_area_button', 'n_clicks')]
)
def update_graph(value_selected, value_vmin, value_vmax, value_granularity, file_path, n_clicks):

    if n_clicks >0:

        Z_matrixes, unique_height, unique_perm, columns, Prod_matrixes = data_transformation(file_path)

        if value_selected in [9,10]:
            fig = go.Figure(data=
                go.Contour(
                    z=Z_matrixes[value_selected][0],
                    x=unique_height,
                    y=unique_perm,
                    colorscale=[[0, 'red'], [rescale(value_vmin, value_vmax), 'white'], [1, 'blue']],
                    dx=10,
                    contours=dict(
                        start=value_vmin,
                        end=value_vmax,
                        size=value_granularity
                    ),
                    colorbar=dict(
                        title='{}'.format(Z_matrixes[value_selected][1]),  # title here
                        titleside='top',
                        titlefont=dict(
                            size=14,
                            family='Arial, sans-serif')
                    )
                )
            )

            fig.update_layout(
                width=1000,
                height=675,
                yaxis=dict(
                    title_text='Permeability (mD)'
                ),
                xaxis=dict(
                    title_text='Gross Height (m)'
                )
            )

            fig.update_yaxes(type='log')

            return fig, False, value_vmin, 0, 10, marks_vmax, 0.1, 0.1, value_vmin, value_vmax, value_granularity, columns[value_selected-1]

        elif value_selected in [1,2,3,4]:

            marks_adapted = marks(Prod_matrixes[value_selected][0], Prod_matrixes[value_selected][1])

            fig = go.Figure(data=
            go.Contour(
                z=Z_matrixes[value_selected][0],
                x=unique_height,
                y=unique_perm,
                colorscale='Oranges',
                dx=10,
                contours=dict(
                    start=value_vmin,
                    end=value_vmax,
                    size=value_granularity
                ),
                colorbar=dict(
                    title='{}'.format(Z_matrixes[value_selected][1]),  # title here
                    titleside='top',
                    titlefont=dict(
                        size=14,
                        family='Arial, sans-serif')
                )
            )
            )

            fig.update_layout(
                width=1000,
                height=675,
                yaxis=dict(
                    title_text='Permeability (mD)'
                ),
                xaxis=dict(
                    title_text='Gross Height (m)'
                )
            )

            fig.update_yaxes(type='log')

            return fig, True, 0, Prod_matrixes[value_selected][0], Prod_matrixes[value_selected][1], marks_adapted, 1, 1,'None', value_vmax, value_granularity, columns[value_selected-1]

        else:
            fig = go.Figure(data=
            go.Contour(
                z=Z_matrixes[value_selected][0],
                x=unique_height,
                y=unique_perm,
                colorscale='Oranges',
                dx=10,
                contours=dict(
                    start=value_vmin,
                    end=value_vmax,
                    size=value_granularity
                ),
                colorbar=dict(
                    title='{}'.format(Z_matrixes[value_selected][1]),  # title here
                    titleside='top',
                    titlefont=dict(
                        size=14,
                        family='Arial, sans-serif')
                )
            )
            )

            fig.update_layout(
                width=1000,
                height=675,
                yaxis=dict(
                    title_text='Permeability (mD)'
                ),
                xaxis=dict(
                    title_text='Gross Height (m)'
                )
            )

            fig.update_yaxes(type='log')

            return fig, True, 0, 0, 10, marks_vmax, 0.1, 1,'None', value_vmax, value_granularity, columns[value_selected-1]

    else :
        return(no_update,no_update,no_update,no_update,no_update,no_update,no_update,no_update,no_update,no_update,no_update,no_update )





if __name__ == '__main__':
    app.run_server(debug=True)