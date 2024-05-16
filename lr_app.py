'''
This script corresponds to LATAM's comprehensive Lost Revenue calculation tool. This tool processes raw data from multiple Dataiku DSS data sources, cleans, process it, performs statistical prompts,
generates editable data tables, and stores manually-entered inputs into Snowflake databases. It includes modules for data entrance, visualizations, change logs, and data validation.
'''

# Importing necessary packages for interaction with dataiku
import dataiku
from dataiku import recipe

# Importing necessary packages for web app generation
import dash
from dash import Input, Output, State, dash_table, dcc, html
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import dash_ag_grid as dag

# Importing necessary packages for data manipulation
import datetime as dt
import pandas as pd
import numpy as np

# Importing necessary packages for user identification
import logging
from flask import request

# Importing CSS styling frameworks
app.config.external_stylesheets = [dbc.themes.MATERIA, dbc.icons.BOOTSTRAP]
app.config.suppress_callback_exceptions = True


def initial_data():
    '''
    This function generates the initial data state, by pulling the dataiku server data on start-up callback.
    After the initial callback, the data is converted to a dataframe, to ease further processing.
    '''
    dataset = dataiku.Dataset("LR_FINAL_DATASET")
    df = dataset.get_dataframe()
    return df

# Propagating source database to master script
df = initial_data()

#Puling other required databases
modified_dataset = dataiku.Dataset("LR_APP_LATEST_EDITS")
edits = dataiku.Dataset("LR_APP_EDITS")
siq_dataset = dataiku.Dataset("SIQ_APP_DATA")
df_siq = siq_dataset.get_dataframe()
edits_df = edits.get_dataframe()
ds_output = dataiku.Dataset("LR_APP_EDITS")

# Forcing numeric format into required fields, to enable aggregation
df['TOTAL_DOLLARS_LR_YTD'] = pd.to_numeric(df['TOTAL_DOLLARS_LR_YTD'], errors='coerce')
df['PROJ_LR_USD_YTG'] = pd.to_numeric(df['PROJ_LR_USD_YTG'], errors='coerce')
df['LR_USD_FY'] = pd.to_numeric(df['LR_USD_FY'], errors='coerce')

# Calculating boolean fields to validate if values exist in database
df['HAS_YTD'] = df['TOTAL_DOLLARS_LR_YTD'].apply(lambda x: x>0)
df['HAS_YTG'] = df['PROJ_LR_USD_YTG'].apply(lambda x: x>0)
df['HAS_FY'] = df['LR_USD_FY'].apply(lambda x: x>0)

#Predefining fields required for filtering
filter_fields = [
    'MATERIAL_KEY','MARKET_DESC','SUB_BUSINESS_UNIT_DESC',
    'PRODUCT_DESC','REGIONAL_SUPPLY_LEADER','SOURCE_DESC',
    'DEMAND_ANALYST_DESC','DEMAND_PLANNER_ID','ABOVE_MARKET_PLANNER_DESC',
    'SEGMENT', 'CLUSTER_DESC','SUB_REGION_DESC','REPORTING_MATERIAL_DESC',
    'LR_USD_FY', 'HAS_YTD','HAS_YTG','HAS_FY'
]

#Generating subset of data for filtering purposes
df_items = df[df.columns.intersection(filter_fields)]
#Sorting database for display purposes
df_items = df_items.sort_values(by='LR_USD_FY',ascending=False).reset_index(drop=True)

#Redefining Supply IQ database column names, to enable proper date sorting
new_column_names = {}
for col in df_siq.columns:
    if col.endswith('_value_sum') or col.endswith('_first'):
        new_name = col.split('_')[0]
        new_column_names[col] = new_name
    else:
        new_column_names[col] = col
df_siq.rename(columns=new_column_names, inplace=True)
column_order = ['Market','Location','Material','MaterialDescription','AmpPlanner','Version','KF'] + [col for col in df_siq.columns if col not in ['Market','Location','Material','MaterialDescription','AmpPlanner','Version','KF']]
df_siq = df_siq[column_order]
          
#Converting dataframes to dictionaries, to enable AG Grids ingestion
data = df.to_dict('records')
data_list = df_items.to_dict('records')
data_siq = df_siq.to_dict('records')

#Identifying unique values on each column on df_items, for dropdown propagation purposes
s = df_items.stack(dropna=True).groupby(level=[1]).unique()
s.to_frame('vals')

#Manually defining number formats required during AG Grid constructions
percentage = {'locale':{},'nully':'','prefix':None,'specifier':',.1%'}
units = {'locale': {'symbol': ['', ' EA']}, 'nully': '', 'prefix': None, 'specifier': '$,.0~f'}
days = {'locale': {'symbol': ['', ' days']}, 'nully': '', 'prefix': None, 'specifier': '$.0~f'}
dollars = {'locale': {'symbol': ['$ ', '']}, 'nully': '', 'prefix': None, 'specifier': '$,.0~f'}

#Predefining fields required for data changes registration
ed_cols = [
    'REGION_DESC','CLUSTER_DESC','SUB_REGION_DESC','MARKET_DESC','BUSINESS_UNIT_DESC','SUB_BUSINESS_UNIT_DESC',
    'SUB_BRAND_DESC','MATERIAL_KEY','REPORTING_MATERIAL_DESC','REGIONAL_SUPPLY_LEADER','SOURCE_DESC','DEMAND_ANALYST_DESC',
    'ABOVE_MARKET_PLANNER_DESC','DEMAND_PLANNER_ID','SEGMENT','Q4_VISIBILITY','Q3_VISIBILITY','Q2_VISIBILITY','Q1_VISIBILITY',
    'NPL','PRODUCT_DESC','TIMESTAMP','USERNAME','ASP','PROJ_SH_DAYS_Q3','OP_UNITS_YTG_Q3','PROJ_SO_DAYS_Q3','manual_in_stock_q3',
    'manual_lr_q3','manual_lrusd_q3','PROJ_SH_DAYS_Q2','OP_UNITS_YTG_Q2','PROJ_SO_DAYS_Q2','manual_in_stock_q2','PROJ_SH_DAYS_Q1',
    'PROJ_SO_DAYS_Q1','manual_in_stock_q1','OP_UNITS_YTG_Q1','manual_lr_q1','manual_lrusd_q1','PROJ_SH_DAYS_Q4','OP_UNITS_YTG_Q4',
    'manual_ytg_op','TOTAL_OP_UNITS_YTD','manual_fy_op','PROJ_SO_DAYS_Q4','manual_in_stock_q4','manual_lr_q4','manual_lrusd_q4',
    'manual_lr_q2','manual_lrusd_q2','manual_lr_ytg','manual_lrusd_ytg','TOTAL_SH_DAYS_YTD','TOTAL_SALES_YTD','manual_urusd_fy',
    'TOTAL_EXCESS_SALES_YTD','TOTAL_DOLLARS_UR_YTD','TOTAL_SO_DAYS_YTD','TOTAL_LR_YTD','manual_lr_fy','manual_lrusd_fy',
    'TOTAL_DOLLARS_LR_YTD','EDITED_ROOT_CAUSE', 'EDITED_RSL_COMMENTS'
]

#Identifying data visibility requirements, by switching dataframe boolean values 
q1_toggle = not df['Q1_VISIBILITY'].mode().values[0]
q2_toggle = not df['Q2_VISIBILITY'].mode().values[0]
q3_toggle = not df['Q3_VISIBILITY'].mode().values[0]
q4_toggle = not df['Q4_VISIBILITY'].mode().values[0]

#Username propagation
client = dataiku.api_client()

@app.callback(
    #Dash callback function: this function will be automatically triggered on start, to identify logged in user
    Output('user-output','children'), #Logged username will be stored in the navbar
    Input('dummy','children') #dummy div to act as auto-trigger
)

def get_logged_user(_):
    #Identifies username
    request_headers=dict(request.headers)
    auth_info_browser=client.get_auth_info_from_browser_headers(request_headers)
    logging.info((auth_info_browser))
    return auth_info_browser["authIdentifier"]


@app.callback(
    #Dash callback function: this function will be automatically triggered on start, to identify logged in user
    Output('string-output','data'),
    Input('dummy','children')
)

def get_logged_user(_):
    request_headers=dict(request.headers)
    auth_info_browser=client.get_auth_info_from_browser_headers(request_headers)
    logging.info((auth_info_browser))
    return auth_info_browser["authIdentifier"]

#Propagating username to master script
username = client.get_auth_info()['authIdentifier']

#Defining Dataiku structure for scenario triggering
project = client.get_project("AMRA_LOSTREVENUECALCULATION")
scenario = project.get_scenario("UPDATE_DATA_FOR_APP")

#Generating modal for changes confirmation


'''
Generating main page's layout
'''

def serve_layout():
    '''
    This function will generate the app's initial layout. 
    The layout is generated through callbacks to guarantee the app makes a server call to retrieve the latest available data.
    '''
    
    #Defining navigation bar layout
    '''
    This navbar will look like the table below:
    
    | Logo | Spacer | Header | Feedback | Documentation | Username |
    
    '''
    navbar = dbc.Container(
        dbc.Row(
            [
                dbc.Col(
                    html.Img(
                        src='https://brandid.pfizer.com/sites/default/files/images/logo-hierarchy-left.png',
                        style={'display': 'block','max-width':'230px','max-height':'45px','width': 'auto','height': 'auto'}
                    ),
                    id='navbar_logo',
                    width=1,
                    align='center'
                ),
                dbc.Col(
                    html.H1('LATAM Lost Revenue Application'),
                    id='navbar_spacer',
                    width=8
                ),
                dbc.Col(
                    dbc.Button(
                       'Feedback',
                       href='https://jira.pfizer.com/servicedesk/customer/portal/189/create/3684',
                       target='_blank',
                       style={
                           'color':'#0000C9',
                           'background-color':'white',
                           'border-color':'#0000C9',
                           'border-style':'solid',
                           'border-width':'1px',
                           'font-family':'Roboto',
                           'border-radius':'45px',
                           'width': '100%',
                           'height':'50px',
                           'justify':'center',
                           'align':'center',
                           'text-transform':'capitalize',
                           'box-shadow':'None'
                       }
                   ),
                    id='navbar_feedback',
                    width=1
                ),
                dbc.Col(
                    dbc.Button(
                       'Documentation',
                       href='https://dss-amer-dev.pfizer.com/projects/AMRA_LOSTREVENUECALCULATION/wiki/1/LATAM%20Lost%20Revenue%20Wiki',
                       target='_blank',
                       style={
                           'color':'#0000C9',
                           'background-color':'white',
                           'border-color':'#0000C9',
                           'border-style':'solid',
                           'border-width':'1px',
                           'font-family':'Roboto',
                           'border-radius':'45px',
                           'width': '100%',
                           'height':'50px',
                           'justify':'center',
                           'align':'center',
                           'text-transform':'capitalize',
                           'box-shadow':'None'
                       }
                   ),
                    id='navbar_doc',
                    width=1
                ),
                dbc.Col(
                    dbc.Button(
                       username,
                       href='',
                       target='_blank',
                       style={
                           'color':'#0000C9',
                           'background-color':'white',
                           'border-color':'#0000C9',
                           'border-style':'solid',
                           'border-width':'1px',
                           'font-family':'Roboto',
                           'border-radius':'45px',
                           'width': '100%',
                           'height':'50px',
                           'justify':'center',
                           'align':'center',
                           'text-transform':'capitalize',
                           'box-shadow':'None'
                       },
                   ),
                    id='navbar_user',
                    width=1
                ),
            ],
            style = {
                'height': '70px',
                'background-color':'white',
                'width': '95vw',
                'padding':'10px',
                'margin':'0px'
            }
        ),
        style={
          'box-shadow':'5px 0px #f2f2f8'  
        },
        fluid=True
    )
    
    #Defining main page's content layout
    content = dbc.Container([
        html.Div(dcc.ConfirmDialog(id='confirm-save',message='Changes have been properly saved!')),
        dbc.Row([
            #Leftmost column will include two main sections: dropdown selection filters, and item selection table
            dbc.Col([
                dbc.Row([html.Hr(),html.P('Item Overview')]),
                dbc.Row([
                    #Defining dropdown filters, each filter will include an APPLY button to propagate values, and recalculate relevant values on all other dropdowns
                    dbc.Col([
                        dbc.Row(html.Label('Sub Region')),
                        dbc.Row([
                            dbc.Col(dcc.Dropdown(id='f_srg_dd',options=s['SUB_REGION_DESC'], multi=True),width=8),
                            dbc.Col(dbc.Button('Apply',id='apply-srg',n_clicks=0, className="btn btn-secondary btn-sm"),width=1)
                        ])
                    ]),
                    
                    dbc.Col([
                        dbc.Row(html.Label('Cluster')),
                        dbc.Row([
                            dbc.Col(dcc.Dropdown(id='f_clu_dd',options=s['CLUSTER_DESC'], multi=True),width=8),
                            dbc.Col(dbc.Button('Apply',id='apply-clu',n_clicks=0, className="btn btn-secondary btn-sm"),width=1)
                        ])
                    ]),
                ], style={'font-size':'11px'}),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Row(html.Label('Market ')),
                        dbc.Row([
                            dbc.Col(dcc.Dropdown(id='f_mkt_dd',options=s['MARKET_DESC'], multi=True),width=8),
                            dbc.Col(dbc.Button('Apply',id='apply-mkt',n_clicks=0, className="btn btn-secondary btn-sm"),width=1)
                        ])
                    ]),
                    
                    dbc.Col([
                        dbc.Row(html.Label('Sub BU')),
                        dbc.Row([
                            dbc.Col(dcc.Dropdown(id='f_sbu_dd',options=s['SUB_BUSINESS_UNIT_DESC'], multi=True),width=8),
                            dbc.Col(dbc.Button('Apply',id='apply-sbu',n_clicks=0, className="btn btn-secondary btn-sm"),width=1)
                        ])
                    ]),
                ], style={'font-size':'11px'}),
                
                
                dbc.Row([
                    dbc.Col([
                        dbc.Row(html.Label('Product')),
                        dbc.Row([
                            dbc.Col(dcc.Dropdown(id='f_bnd_dd',options=s['PRODUCT_DESC'], multi=True),width=8),
                            dbc.Col(dbc.Button('Apply',id='apply-bnd',n_clicks=0, className="btn btn-secondary btn-sm"),width=1)
                        ])
                    ]),
                    
                    dbc.Col([
                        dbc.Row(html.Label('RSL')),
                        dbc.Row([
                            dbc.Col(dcc.Dropdown(id='f_rsl_dd',options=s['REGIONAL_SUPPLY_LEADER'], multi=True),width=8),
                            dbc.Col(dbc.Button('Apply',id='apply-rsl',n_clicks=0, className="btn btn-secondary btn-sm"),width=1)
                        ])
                    ]),
                ], style={'font-size':'11px'}),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Row(html.Label('Source')),
                        dbc.Row([
                            dbc.Col(dcc.Dropdown(id='f_src_dd',options=s['SOURCE_DESC'], multi=True),width=8),
                            dbc.Col(dbc.Button('Apply',id='apply-src',n_clicks=0, className="btn btn-secondary btn-sm"),width=1)
                        ])
                    ]),
                    
                    dbc.Col([
                        dbc.Row(html.Label('Segment')),
                        dbc.Row([
                            dbc.Col(dcc.Dropdown(id='f_seg_dd',options=s['SEGMENT'], multi=True),width=8),
                            dbc.Col(dbc.Button('Apply',id='apply-seg',n_clicks=0, className="btn btn-secondary btn-sm"),width=1)
                        ])
                    ]),
                ], style={'font-size':'11px'}),
               
                dbc.Row([
                    dbc.Col([
                        dbc.Row(html.Label('AMP Planner')),
                        dbc.Row([
                            dbc.Col(dcc.Dropdown(id='f_amp_dd',options=s['ABOVE_MARKET_PLANNER_DESC'], multi=True),width=8),
                            dbc.Col(dbc.Button('Apply',id='apply-amp',n_clicks=0, className="btn btn-secondary btn-sm"),width=1)
                        ])
                    ]),
                    
                    dbc.Col([
                        dbc.Row(html.Label('Demand Planner')),
                        dbc.Row([
                            dbc.Col(dcc.Dropdown(id='f_dem_dd',options=s['DEMAND_PLANNER_ID'], multi=True),width=8),
                            dbc.Col(dbc.Button('Apply',id='apply-dem',n_clicks=0, className="btn btn-secondary btn-sm"),width=1)
                        ])
                    ]),
                ], style={'font-size':'11px'}),

                dbc.Row([
                    dbc.Col([
                        dbc.Row(html.Label('Demand Analyst')),
                        dbc.Row([
                            dbc.Col(dcc.Dropdown(id='f_dan_dd',options=s['DEMAND_ANALYST_DESC'], multi=True),width=8),
                            dbc.Col(dbc.Button('Apply',id='apply-dan',n_clicks=0, className="btn btn-secondary btn-sm"),width=1)
                        ])
                    ]),
                    
                    dbc.Col([
                        dbc.Row(html.Label('Material')),
                        dbc.Row([
                            dbc.Col(dcc.Dropdown(id='f_mat_dd',options=s['MATERIAL_KEY'], multi=True),width=8),
                            dbc.Col(dbc.Button('Apply',id='apply-mat',n_clicks=0, className="btn btn-secondary btn-sm"),width=1)
                        ])
                    ]),
                ], style={'font-size':'11px'}),
               

                dbc.Row([
                    dbc.Col([html.Label('LR YTD? '),dcc.Dropdown(['True'], id='f_lr_ytd', multi=False)]),
                    dbc.Col([html.Label('LR YTG? '),dcc.Dropdown(['True'], id='f_lr_ytg', multi=False)]),
                    dbc.Col([html.Label('LR FY? '),dcc.Dropdown(['True'], id='f_lr_fy', multi=False)])
                ], style={'font-size':'11px'}),

                html.Hr(),
                dbc.Row([html.Div(id='ag-grid-container')])
            ], width = 3),

            dbc.Col([
                html.Hr(),
                html.P('LATAM Lost Revenue'),
                html.Div(id='output-details-ag-grid'),
                html.Hr(),
                dbc.Row([
                    dbc.Col([dbc.Button('Submit Changes',id='save-data-btn', n_clicks=0, className="me-1")], width=3),
                    dbc.Col([html.Div(id='output-data-div')], width=3),
                    dbc.Col([dbc.Button('Reload Data',id='reload-data-btn', n_clicks=0, style={'display':'none'})], width=3),
                    dbc.Col([html.Div(id='output-data-refresh')], width=3)
                ],
               ),
                html.Div(id='output-siq-div')
            ]),
        ])],
        fluid=True,
        className="dbc dbc-row-selectable")

    layout_page_1 = html.Div([
        navbar,
        content
        ]
    )
    
    return layout_page_1

# index layout
app.layout = serve_layout

@app.callback(
    Output('ag-grid-container','children'),
    [
        Input('f_srg_dd','value'),
        Input('f_clu_dd','value'),
        Input('f_mkt_dd','value'),
        Input('f_sbu_dd','value'),
        Input('f_bnd_dd','value'),
        Input('f_rsl_dd','value'),
        Input('f_src_dd','value'),
        Input('f_seg_dd','value'),
        Input('f_amp_dd','value'),
        Input('f_dem_dd','value'),
        Input('f_dan_dd','value'),
        Input('f_mat_dd','value'),
        Input('f_lr_ytd','value'),
        Input('f_lr_ytg','value'),
        Input('f_lr_fy','value')
    ]
)

def update_ag_grid(selected_value_srg, selected_value_clu, selected_value_mkt,
                   selected_value_sbu, selected_value_bnd, selected_value_rsl,
                   selected_value_src, selected_value_seg, selected_value_amp,
                   selected_value_dem, selected_value_dan, selected_value_mat,
                   selected_value_ytd, selected_value_ytg, selected_value_fy
                  ):
    
    new_dataset = dataiku.Dataset("LR_FINAL_DATASET")
    new_df = new_dataset.get_dataframe()
    new_df['TOTAL_DOLLARS_LR_YTD'] = pd.to_numeric(new_df['TOTAL_DOLLARS_LR_YTD'], errors='coerce')
    new_df['PROJ_LR_USD_YTG'] = pd.to_numeric(new_df['PROJ_LR_USD_YTG'], errors='coerce')
    new_df['LR_USD_FY'] = pd.to_numeric(new_df['LR_USD_FY'], errors='coerce')

    new_df['HAS_YTD'] = new_df['TOTAL_DOLLARS_LR_YTD'].apply(lambda x: x>0)
    new_df['HAS_YTG'] = new_df['PROJ_LR_USD_YTG'].apply(lambda x: x>0)
    new_df['HAS_FY'] = new_df['LR_USD_FY'].apply(lambda x: x>0)

    #Creating Items List for Material Selection
    new_filter_fields = [
        'MATERIAL_KEY','MARKET_DESC','SUB_BUSINESS_UNIT_DESC',
        'PRODUCT_DESC','REGIONAL_SUPPLY_LEADER','SOURCE_DESC',
        'DEMAND_ANALYST_DESC','DEMAND_PLANNER_ID','ABOVE_MARKET_PLANNER_DESC',
        'SEGMENT', 'CLUSTER_DESC','SUB_REGION_DESC','REPORTING_MATERIAL_DESC',
        'LR_USD_FY', 'HAS_YTD','HAS_YTG','HAS_FY'
    ]
    new_df_items = new_df[new_df.columns.intersection(new_filter_fields)]
    new_df_items = new_df_items.sort_values(by='LR_USD_FY',ascending=False).reset_index(drop=True)
    
    filtered_df = new_df_items.copy()
    
    if selected_value_srg:
        filtered_df = filtered_df[filtered_df['SUB_REGION_DESC'].isin(selected_value_srg)]
    if selected_value_clu:
        filtered_df = filtered_df[filtered_df['CLUSTER_DESC'].isin(selected_value_clu)]
    if selected_value_mkt:
        filtered_df = filtered_df[filtered_df['MARKET_DESC'].isin(selected_value_mkt)]
    if selected_value_sbu:
        filtered_df = filtered_df[filtered_df['SUB_BUSINESS_UNIT_DESC'].isin(selected_value_sbu)]
    if selected_value_bnd:
        filtered_df = filtered_df[filtered_df['PRODUCT_DESC'].isin(selected_value_bnd)]
    if selected_value_rsl:
        filtered_df = filtered_df[filtered_df['REGIONAL_SUPPLY_LEADER'].isin(selected_value_rsl)]
    if selected_value_src:
        filtered_df = filtered_df[filtered_df['SOURCE_DESC'].isin(selected_value_src)]
    if selected_value_seg:
        filtered_df = filtered_df[filtered_df['SEGMENT'].isin(selected_value_seg)]
    if selected_value_amp:
        filtered_df = filtered_df[filtered_df['ABOVE_MARKET_PLANNER_DESC'].isin(selected_value_amp)]
    if selected_value_dem:
        filtered_df = filtered_df[filtered_df['DEMAND_PLANNER_ID'].isin(selected_value_dem)]
    if selected_value_dan:
        filtered_df = filtered_df[filtered_df['DEMAND_ANALYST_DESC'].isin(selected_value_dan)]
    if selected_value_mat:
        filtered_df = filtered_df[filtered_df['MATERIAL_KEY'].isin(selected_value_mat)]
    if selected_value_ytd:
        filtered_df = filtered_df[filtered_df['HAS_YTD'] == True]
    if selected_value_ytg:
        filtered_df = filtered_df[filtered_df['HAS_YTG'] == True]
    if selected_value_fy:
        filtered_df = filtered_df[filtered_df['HAS_FY'] == True]
    
    ag_grid = dag.AgGrid(
        id='ag-grid',
        columnDefs=[
            {'headerName': 'Market', 'field':'MARKET_DESC'},
            {'headerName': 'Material', 'field':'MATERIAL_KEY'},
            {'headerName': 'YTD', 'field':'HAS_YTD'},
            {'headerName': 'YTG', 'field':'HAS_YTG'},
            {'headerName': 'FY', 'field':'HAS_FY'},
        ],
        rowData=filtered_df.to_dict('records'),
        style={'height':'600px','font-size':'11px'},
        rowStyle={'font-size': '11px'},
        columnSize = 'responsiveSizeToFit',
        dashGridOptions ={
            "animateRows":False,
            'rowSelection':'multiple',
        },
        defaultColDef={
            "filter": True,
            "checkboxSelection": {
                "function": 'params.column == params.columnApi.getAllDisplayedColumns()[0]'
            },
            "headerCheckboxSelection": {
                "function": 'params.column == params.columnApi.getAllDisplayedColumns()[0]'
            }
        },
    )
    
    return ag_grid

@app.callback(
    Output('output-details-ag-grid','children'),
    Input('ag-grid','selectedRows')
)

def update_details(selected_rows):
    if not selected_rows:
        return None
    
    selected_values_mkt = {row['MARKET_DESC'] for row in selected_rows}
    selected_values_mat = {row['MATERIAL_KEY'] for row in selected_rows}
    new_dataset = dataiku.Dataset("LR_FINAL_DATASET")
    new_df = new_dataset.get_dataframe()
    df_details = new_df[(new_df['MARKET_DESC'].isin(selected_values_mkt)) & (new_df['MATERIAL_KEY'].isin(selected_values_mat))]
    df_details = df_details.sort_values(by='LR_USD_FY',ascending=False).reset_index(drop=True)
    
    details_ag_grid = dag.AgGrid(
        id = 'details-ag-grid',
        columnDefs=[
            {'headerName': 'Regional Hierarchy',
             'children': [
                 {'headerName': 'Region', 'field': 'REGION_DESC', "columnGroupShow": "open"},
                 {'headerName': 'Sub-Region', 'field': 'SUB_REGION_DESC', "columnGroupShow": "open"},
                 {'headerName': 'Cluster', 'field': 'CLUSTER_DESC', "columnGroupShow": "open"},
                 {'headerName': 'Market', 'field': 'MARKET_DESC'},
             ]
            },
            
            {'headerName': 'Commercial Hierarchy',
             'children': [
                 {'headerName': 'BU', 'field': 'BUSINESS_UNIT_DESC', "columnGroupShow": "open"},
                {'headerName': 'Sub BU', 'field': 'SUB_BUSINESS_UNIT_DESC'},
                {'headerName': 'Product', 'field': 'PRODUCT_DESC'},     
             ]
            },
            
            {'headerName': 'Master Data',
             'children': [
                {'headerName': 'Material', 'field': 'MATERIAL_KEY'},
                {'headerName': 'Description', 'field': 'REPORTING_MATERIAL_DESC'},
                {'headerName': 'Source', 'field': 'SOURCE_DESC', "columnGroupShow": "open"},
                {'headerName': 'RSL', 'field': 'REGIONAL_SUPPLY_LEADER', "columnGroupShow": "open"},
                {'headerName': 'AMP', 'field': 'ABOVE_MARKET_PLANNER_DESC', "columnGroupShow": "open"},
                {'headerName': 'Demand Planner', 'field': 'DEMAND_PLANNER_ID', "columnGroupShow": "open"},
                {'headerName': 'Demand Analyst', 'field': 'DEMAND_ANALYST_DESC', "columnGroupShow": "open"},
                {'headerName': 'Segment', 'field': 'SEGMENT', "columnGroupShow": "open"},
                {'headerName': 'NPL', 'field': 'NPL', "columnGroupShow": "open"},
                {'headerName': 'ASP', 'field': 'ASP', 'hide': True},
                {'headerName': 'Q1_VISIBILITY', 'field': 'Q1_VISIBILITY', 'hide': True},
                {'headerName': 'Q2_VISIBILITY', 'field': 'Q2_VISIBILITY', 'hide': True},
                {'headerName': 'Q3_VISIBILITY', 'field': 'Q3_VISIBILITY', 'hide': True},
                {'headerName': 'Q4_VISIBILITY', 'field': 'Q4_VISIBILITY', 'hide': True},
             ]
            },
            
            {'headerName': 'Year-to-Date',
             'children': [
                 {'headerName': 'OP','field': 'TOTAL_OP_UNITS_YTD',"valueFormatter": {"function": "d3.format(',.0f')(params.value)"}, "columnGroupShow": "open"},
                 {'headerName': 'Sales','field': 'TOTAL_SALES_YTD',"valueFormatter": {"function": "d3.format(',.0f')(params.value)"}, "columnGroupShow": "open"},
                 {'headerName': 'Excess Sales','field': 'TOTAL_EXCESS_SALES_YTD',"valueFormatter": {"function": "d3.format(',.0f')(params.value)"}, "columnGroupShow": "open"},
                 {'headerName': 'SO Days','field': 'TOTAL_SO_DAYS_YTD',"valueFormatter": {"function": "d3.format(',.0f')(params.value)"},'cellDataType':'number','editable': True, 'cellStyle': {'background-color': '#FDF0B0'},'valueParser': {'function': 'Number(params.newValue)',"columnGroupShow": "open"}},
                 {'headerName': 'SH Days','field': 'TOTAL_SH_DAYS_YTD','type':'numericColumn', "valueFormatter": {"function": "d3.format(',.0f')(params.value)"},'cellDataType':'number','valueParser': {'function': 'Number(params.newValue)'},"columnGroupShow": "open"},
                 {
                     'headerName': 'In Stock',
                     'field':'manual_in_stock_ytd',
                     'valueGetter':{"function": "Number(params.data.TOTAL_SH_DAYS_YTD) === 0 ? 1 : 1-(Number(params.data.TOTAL_SO_DAYS_YTD) / Number(params.data.TOTAL_SH_DAYS_YTD))"},
                     'valueFormatter':{"function": "d3.format(',.1%')(params.value)"},
                     "columnGroupShow": "open",
                     'cellStyle': {"function": "params.value < 1 ? {'background-color': '#ff8f8f'} : {'background-color': '#ffffff'}"}
                 },
                 
                 {
                     'headerName': 'LR C',
                     'field': 'manual_lrc_ytd',
                     'valueGetter':{"function": "params.getValue('manual_in_stock_ytd') < 1 && params.data.TOTAL_SALES_YTD < params.data.TOTAL_OP_UNITS_YTD ? Number(params.data.TOTAL_OP_UNITS_YTD) * (1 - params.getValue('manual_in_stock_ytd')) : 0"},
                     "valueFormatter": {"function": "d3.format(',.0f')(params.value)"},
                     'cellStyle': {"function": "params.value > 0 ? {'background-color': '#ff8f8f'} : null"},
                     "columnGroupShow": "open",
                     'hide':True
                 },
                 {
                     'headerName': 'LR M',
                     'field': 'manual_lrm_ytd',
                     'valueGetter':{"function": "params.getValue('manual_in_stock_ytd') < 1 && params.data.TOTAL_SALES_YTD < params.data.TOTAL_OP_UNITS_YTD ? Number(params.data.TOTAL_OP_UNITS_YTD) - params.data.TOTAL_SALES_YTD : 0"},
                     "valueFormatter": {"function": "d3.format(',.0f')(params.value)"},
                     'cellStyle': {"function": "params.value > 0 ? {'background-color': '#ff8f8f'} : null"},
                     "columnGroupShow": "open",
                     'hide':True
                     
                 },
                 
                 {
                     'headerName': 'LR',
                     'field': 'manual_lr_ytd',
                     'valueGetter':{"function": "(params.getValue('TOTAL_SALES_YTD') + params.getValue('manual_lrc_ytd')) > Number(params.data.TOTAL_OP_UNITS_YTD) ? params.getValue('manual_lrm_ytd') : params.getValue('manual_lrc_ytd')"},
                     "valueFormatter": {"function": "d3.format(',.0f')(params.value)"},
                     'cellStyle': {"function": "params.value > 0 ? {'background-color': '#ff8f8f'} : null"},
                     "columnGroupShow": "open",
                     'cellStyle': {"function": "params.value > 0 ? {'background-color': '#ff8f8f'} : null"}
                 },
                 {
                     'headerName': 'LR$',
                     'field': 'manual_lrusd_ytd',
                     'valueGetter':{"function": "params.getValue('manual_lr_ytd') * Number(params.data.ASP)"},
                     "valueFormatter": {"function": "d3.format('$,.0f')(params.value)"},
                     'cellStyle': {"function": "params.value > 0 ? {'background-color': '#ff8f8f'} : null"}
                 },
                 {'headerName': 'UR$','field': 'TOTAL_DOLLARS_UR_YTD',"valueFormatter": {"function": "d3.format('$,.0f')(params.value)"},'cellStyle': {"function": "params.value > 0 ? {'background-color': '#92d050'} : null"}}
             ]
            },
            
            {'headerName': 'Quarter 1',
             'children': [
                 {'headerName': 'OP','field': 'OP_UNITS_YTG_Q1',"valueFormatter": {"function": "d3.format(',.0f')(params.value)"}, 'hide':q1_toggle},
                 {'headerName': 'SO Days','field': 'PROJ_SO_DAYS_Q1',"valueFormatter": {"function": "d3.format(',.0f')(params.value)"},'cellDataType':'number','editable': True, 'cellStyle': {'background-color': '#FDF0B0'},'valueParser': {'function': 'Number(params.newValue)',"columnGroupShow": "open"},'hide':q1_toggle},
                 {'headerName': 'SH Days','field': 'PROJ_SH_DAYS_Q1','type':'numericColumn', "valueFormatter": {"function": "d3.format(',.0f')(params.value)"},'cellDataType':'number','valueParser': {'function': 'Number(params.newValue)'},"columnGroupShow": "open", 'hide':q1_toggle},
                 {
                     'headerName': 'In Stock',
                     'field':'manual_in_stock_q1',
                     'valueGetter':{"function": "Number(params.data.PROJ_SH_DAYS_Q1) === 0 ? 1 : 1-(Number(params.data.PROJ_SO_DAYS_Q1) / Number(params.data.PROJ_SH_DAYS_Q1))"},
                     'valueFormatter':{"function": "d3.format(',.1%')(params.value)"},
                     "columnGroupShow": "open",
                     'hide':q1_toggle,
                     'cellStyle': {"function": "params.value < 1 ? {'background-color': '#ff8f8f'} : null"}
                 },
                 
                 {
                     'headerName': 'sales_lag_q1',
                     'field': 'sales_lag_q1',
                     'valueGetter': {"function":"params.data.Q1_VISIBILITY ? params.getValue('TOTAL_EXCESS_SALES_YTD') : 0"},
                     'hide':True
                     
                 },
                 
                 {
                     'headerName': 'Sales',
                     'field': 'manual_proj_sales_q1',
                     'valueGetter': {"function":"Number(params.data.PROJ_SH_DAYS_Q1) === 0 ? Number(params.data.OP_UNITS_YTG_Q1) : Number(params.data.OP_UNITS_YTG_Q1) * (1-(Number(params.data.PROJ_SO_DAYS_Q1) / Number(params.data.PROJ_SH_DAYS_Q1)))"},
                     "valueFormatter": {"function": "d3.format(',.0f')(params.value)"},
                     'valueParser': {'function': 'Number(params.value)'},
                     "columnGroupShow": "open",
                     'hide':True
                 },
                 
                 {
                     'headerName': 'Total Sales',
                     'field': 'total_sales_q1',
                     'valueGetter': {"function":"params.getValue('sales_lag_q1') + params.getValue('manual_proj_sales_q1')"},
                     "valueFormatter": {"function": "d3.format(',.0f')(params.value)"},
                     'hide':q1_toggle
                 },
                 
                 
                 {
                     'headerName': 'Excess',
                     'field': 'excess_sales_q1',
                     "valueGetter": {"function": "params.getValue('total_sales_q1') <= Number(params.data.OP_UNITS_YTG_Q1) ? 0 : params.getValue('total_sales_q1') - Number(params.data.OP_UNITS_YTG_Q1);"},
                     "valueFormatter": {"function": "d3.format(',.0f')(params.value)"},
                     "columnGroupShow": "open",
                     'hide':q1_toggle
                 },
                 {
                     'headerName': 'Sales Perf.',
                     'field': 'manual_sales_per_q1',
                     'valueGetter':{"function": "Number(params.data.OP_UNITS_YTG_Q1) === 0 ? 1  : (params.getValue('total_sales_q1')/ Number(params.data.OP_UNITS_YTG_Q1))"},
                     'valueFormatter':{"function": "d3.format(',.1%')(params.value)"},
                     "columnGroupShow": "open",
                     'hide':q1_toggle,
                     'cellStyle': {"function": "params.value < 1 ? {'background-color': '#ff8f8f'} : null"}
                 },
                 
                 
                 {
                     'headerName': 'LR C',
                     'field': 'manual_lrc_q1',
                     'valueGetter':{"function": "params.getValue('manual_in_stock_q1') < 1 && params.getValue('manual_sales_per_q1') < 1 ? Number(params.data.OP_UNITS_YTG_Q1) * (1 - params.getValue('manual_in_stock_q1')) : 0"},
                     "valueFormatter": {"function": "d3.format(',.0f')(params.value)"},
                     'cellStyle': {"function": "params.value > 0 ? {'background-color': '#ff8f8f'} : null"},
                     "columnGroupShow": "open",
                     'hide':q1_toggle
                 },
                 {
                     'headerName': 'LR M',
                     'field': 'manual_lrm_q1',
                     'valueGetter':{"function": "params.getValue('manual_in_stock_q1') < 1 && params.getValue('manual_sales_per_q1') < 1 ? Number(params.data.OP_UNITS_YTG_Q1) - params.getValue('total_sales_q1') : 0"},
                     "valueFormatter": {"function": "d3.format(',.0f')(params.value)"},
                     'cellStyle': {"function": "params.value > 0 ? {'background-color': '#ff8f8f'} : null"},
                     "columnGroupShow": "open",
                     'hide':q1_toggle
                     
                 },
                 
                 {
                     'headerName': 'LR',
                     'field': 'manual_lr_q1',
                     'valueGetter':{"function": "(params.getValue('total_sales_q1') + params.getValue('manual_lrc_q1')) > Number(params.data.OP_UNITS_YTG_Q1) ? params.getValue('manual_lrm_q1') : params.getValue('manual_lrc_q1')"},
                     "valueFormatter": {"function": "d3.format(',.0f')(params.value)"},
                     'cellStyle': {"function": "params.value > 0 ? {'background-color': '#ff8f8f'} : null"},
                     "columnGroupShow": "open",
                     'hide':q1_toggle,
                     'cellStyle': {"function": "params.value > 0 ? {'background-color': '#ff8f8f'} : null"}
                 },
                 {
                     'headerName': 'LR$',
                     'field': 'manual_lrusd_q1',
                     'valueGetter':{"function": "params.getValue('manual_lr_q1') * Number(params.data.ASP)"},
                     "valueFormatter": {"function": "d3.format('$,.0f')(params.value)"},
                     'cellStyle': {"function": "params.value > 0 ? {'background-color': '#ff8f8f'} : null"},
                     'hide':q1_toggle
                 }
             ],
            },
            
            {'headerName': 'Quarter 2',
             'children': [
                 {'headerName': 'OP','field': 'OP_UNITS_YTG_Q2',"valueFormatter": {"function": "d3.format(',.0f')(params.value)"},'hide':q2_toggle},
                 {'headerName': 'SO Days','field': 'PROJ_SO_DAYS_Q2',"valueFormatter": {"function": "d3.format(',.0f')(params.value)"},'cellDataType':'number','editable': True, 'cellStyle': {'background-color': '#FDF0B0'},'valueParser': {'function': 'Number(params.newValue)',"columnGroupShow": "open"},'hide':q2_toggle},
                 {'headerName': 'SH Days','field': 'PROJ_SH_DAYS_Q2','type':'numericColumn', "valueFormatter": {"function": "d3.format(',.0f')(params.value)"},'cellDataType':'number','valueParser': {'function': 'Number(params.newValue)'},"columnGroupShow": "open",'hide':q2_toggle},
                 {
                     'headerName': 'In Stock',
                     'field':'manual_in_stock_q2',
                     'valueGetter':{"function": "Number(params.data.PROJ_SH_DAYS_Q2) === 0 ? 1 : 1-(Number(params.data.PROJ_SO_DAYS_Q2) / Number(params.data.PROJ_SH_DAYS_Q2))"},
                     'valueFormatter':{"function": "d3.format(',.1%')(params.value)"},
                     "columnGroupShow": "open",
                     'hide':q2_toggle,
                     'cellStyle': {"function": "params.value < 1 ? {'background-color': '#ff8f8f'} : null"}
                 },
                 {
                     'headerName': 'skip_q1',
                     'field': 'skip_q1',
                     'valueGetter': {"function": "params.data.Q1_VISIBILITY ? params.getValue('excess_sales_q1') : params.getValue('TOTAL_EXCESS_SALES_YTD')"},
                     'hide':True
                 },
                 
                 {
                     'headerName': 'sales_lag_q2',
                     'field': 'sales_lag_q2',
                     'valueGetter': {"function":"params.data.Q2_VISIBILITY ? params.getValue('skip_q1') : 0"},
                     'hide':True
                 },
                 
                 {
                     'headerName': 'Sales',
                     'field': 'manual_proj_sales_q2',
                     'valueGetter': {"function":"Number(params.data.PROJ_SH_DAYS_Q2) === 0 ? Number(params.data.OP_UNITS_YTG_Q2) : Number(params.data.OP_UNITS_YTG_Q2) * (1-(Number(params.data.PROJ_SO_DAYS_Q2) / Number(params.data.PROJ_SH_DAYS_Q2)))"},
                     "valueFormatter": {"function": "d3.format(',.0f')(params.value)"},
                     'valueParser': {'function': 'Number(params.value)'},
                     "columnGroupShow": "open",
                     'hide':True
                 },
                 
                 {
                     'headerName': 'Total Sales',
                     'field': 'total_sales_q2',
                     'valueGetter': {"function":"params.getValue('sales_lag_q2') + params.getValue('manual_proj_sales_q2')"},
                     "valueFormatter": {"function": "d3.format(',.0f')(params.value)"},
                     'hide':q2_toggle
                 },
                 {
                     'headerName': 'Excess',
                     'field': 'excess_sales_q2',
                     "valueGetter": {"function": "params.getValue('total_sales_q2') <= Number(params.data.OP_UNITS_YTG_Q2) ? 0 : params.getValue('total_sales_q2') - Number(params.data.OP_UNITS_YTG_Q2);"},
                     "valueFormatter": {"function": "d3.format(',.0f')(params.value)"},
                     "columnGroupShow": "open",
                     'hide':q2_toggle
                 },
                 {
                     'headerName': 'Sales Perf.',
                     'field': 'manual_sales_per_q2',
                     'valueGetter':{"function": "Number(params.data.OP_UNITS_YTG_Q2) === 0 ? 1  : (params.getValue('total_sales_q2')/ Number(params.data.OP_UNITS_YTG_Q2))"},
                     'valueFormatter':{"function": "d3.format(',.1%')(params.value)"},
                     "columnGroupShow": "open",
                     'hide':q2_toggle,
                     'cellStyle': {"function": "params.value < 1 ? {'background-color': '#ff8f8f'} : null"}
                 },
                 
                 
                 {
                     'headerName': 'LR C',
                     'field': 'manual_lrc_q2',
                     'valueGetter':{"function": "params.getValue('manual_in_stock_q2') < 1 && params.getValue('manual_sales_per_q2') < 1 ? Number(params.data.OP_UNITS_YTG_Q2) * (1 - params.getValue('manual_in_stock_q2')) : 0"},
                     "valueFormatter": {"function": "d3.format(',.0f')(params.value)"},
                     'cellStyle': {"function": "params.value > 0 ? {'background-color': '#ff8f8f'} : null"},
                     "columnGroupShow": "open",
                     'hide':q2_toggle
                 },
                 {
                     'headerName': 'LR M',
                     'field': 'manual_lrm_q2',
                     'valueGetter':{"function": "params.getValue('manual_in_stock_q2') < 1 && params.getValue('manual_sales_per_q2') < 1 ? Number(params.data.OP_UNITS_YTG_Q2) - params.getValue('total_sales_q2') : 0"},
                     "valueFormatter": {"function": "d3.format(',.0f')(params.value)"},
                     'cellStyle': {"function": "params.value > 0 ? {'background-color': '#ff8f8f'} : null"},
                     "columnGroupShow": "open",
                     'hide':q2_toggle
                 },
                 
                 {
                     'headerName': 'LR',
                     'field': 'manual_lr_q2',
                     'valueGetter':{"function": "(params.getValue('total_sales_q2') + params.getValue('manual_lrc_q2')) > Number(params.data.OP_UNITS_YTG_Q2) ? params.getValue('manual_lrm_q2') : params.getValue('manual_lrc_q2')"},
                     "valueFormatter": {"function": "d3.format(',.0f')(params.value)"},
                     'cellStyle': {"function": "params.value > 0 ? {'background-color': '#ff8f8f'} : null"},
                     "columnGroupShow": "open",
                     'hide':q2_toggle,
                     'cellStyle': {"function": "params.value > 0 ? {'background-color': '#ff8f8f'} : null"}
                 },
                 {
                     'headerName': 'LR$',
                     'field': 'manual_lrusd_q2',
                     'valueGetter':{"function": "params.getValue('manual_lr_q2') * Number(params.data.ASP)"},
                     "valueFormatter": {"function": "d3.format('$,.0f')(params.value)"},
                     'cellStyle': {"function": "params.value > 0 ? {'background-color': '#ff8f8f'} : null"},
                     'hide':q2_toggle
                 }
             ]
            },
            
            {'headerName': 'Quarter 3',
             'children': [
                 {'headerName': 'OP','field': 'OP_UNITS_YTG_Q3',"valueFormatter": {"function": "d3.format(',.0f')(params.value)"},'hide':q3_toggle},
                 {'headerName': 'SO Days','field': 'PROJ_SO_DAYS_Q3',"valueFormatter": {"function": "d3.format(',.0f')(params.value)"},'cellDataType':'number','editable': True, 'cellStyle': {'background-color': '#FDF0B0'},'valueParser': {'function': 'Number(params.newValue)',"columnGroupShow": "open"},'hide':q3_toggle},
                 {'headerName': 'SH Days','field': 'PROJ_SH_DAYS_Q3','type':'numericColumn', "valueFormatter": {"function": "d3.format(',.0f')(params.value)"},'cellDataType':'number','valueParser': {'function': 'Number(params.newValue)'},"columnGroupShow": "open",'hide':q3_toggle},
                 {
                     'headerName': 'In Stock',
                     'field':'manual_in_stock_q3',
                     'valueGetter':{"function": "Number(params.data.PROJ_SH_DAYS_Q3) === 0 ? 1 : 1-(Number(params.data.PROJ_SO_DAYS_Q3) / Number(params.data.PROJ_SH_DAYS_Q3))"},
                     'valueFormatter':{"function": "d3.format(',.1%')(params.value)"},
                     "columnGroupShow": "open",
                     'hide':q3_toggle,
                     'cellStyle': {"function": "params.value < 1 ? {'background-color': '#ff8f8f'} : null"}
                 },
                 
                 {
                     'headerName': 'skip_q2',
                     'field': 'skip_q2',
                     'valueGetter': {"function": "params.data.Q2_VISIBILITY ? params.getValue('excess_sales_q2') : params.getValue('TOTAL_EXCESS_SALES_YTD')"},
                     'hide':True
                 },
                 
                 {
                     'headerName': 'sales_lag_q3',
                     'field': 'sales_lag_q3',
                     'valueGetter': {"function":"params.data.Q3_VISIBILITY ? params.getValue('skip_q2') : 0"},
                     'hide':True
                 },
                 
                 {
                     'headerName': 'Sales',
                     'field': 'manual_proj_sales_q3',
                     'valueGetter': {"function":"Number(params.data.PROJ_SH_DAYS_Q3) === 0 ? Number(params.data.OP_UNITS_YTG_Q3) : Number(params.data.OP_UNITS_YTG_Q3) * (1-(Number(params.data.PROJ_SO_DAYS_Q3) / Number(params.data.PROJ_SH_DAYS_Q3)))"},
                     "valueFormatter": {"function": "d3.format(',.0f')(params.value)"},
                     'valueParser': {'function': 'Number(params.value)'},
                     "columnGroupShow": "open",
                     'hide':True
                 },
                 
                 {
                     'headerName': 'Total Sales',
                     'field': 'total_sales_q3',
                     'valueGetter': {"function":"params.getValue('sales_lag_q3') + params.getValue('manual_proj_sales_q3')"},
                     "valueFormatter": {"function": "d3.format(',.0f')(params.value)"},
                     'hide':q3_toggle
                 },
                 {
                     'headerName': 'Excess',
                     'field': 'excess_sales_q3',
                     "valueGetter": {"function": "params.getValue('total_sales_q3') <= Number(params.data.OP_UNITS_YTG_Q3) ? 0 : params.getValue('total_sales_q3') - Number(params.data.OP_UNITS_YTG_Q3);"},
                     "valueFormatter": {"function": "d3.format(',.0f')(params.value)"},
                     "columnGroupShow": "open",
                     'hide':q3_toggle
                 },
                 {
                     'headerName': 'Sales Perf.',
                     'field': 'manual_sales_per_q3',
                     'valueGetter':{"function": "Number(params.data.OP_UNITS_YTG_Q3) === 0 ? 1  : (params.getValue('total_sales_q3')/ Number(params.data.OP_UNITS_YTG_Q3))"},
                     'valueFormatter':{"function": "d3.format(',.1%')(params.value)"},
                     "columnGroupShow": "open",
                     'hide':q3_toggle,
                     'cellStyle': {"function": "params.value < 1 ? {'background-color': '#ff8f8f'} : null"}
                 },
                 
                 
                 {
                     'headerName': 'LR C',
                     'field': 'manual_lrc_q3',
                     'valueGetter':{"function": "params.getValue('manual_in_stock_q3') < 1 && params.getValue('manual_sales_per_q3') < 1 ? Number(params.data.OP_UNITS_YTG_Q3) * (1 - params.getValue('manual_in_stock_q3')) : 0"},
                     "valueFormatter": {"function": "d3.format(',.0f')(params.value)"},
                     'cellStyle': {"function": "params.value > 0 ? {'background-color': '#ff8f8f'} : null"},
                     "columnGroupShow": "open",
                     'hide':q3_toggle
                 },
                 {
                     'headerName': 'LR M',
                     'field': 'manual_lrm_q3',
                     'valueGetter':{"function": "params.getValue('manual_in_stock_q3') < 1 && params.getValue('manual_sales_per_q3') < 1 ? Number(params.data.OP_UNITS_YTG_Q3) - params.getValue('total_sales_q3') : 0"},
                     "valueFormatter": {"function": "d3.format(',.0f')(params.value)"},
                     'cellStyle': {"function": "params.value > 0 ? {'background-color': '#ff8f8f'} : null"},
                     "columnGroupShow": "open",
                     'hide':q3_toggle
                 },
                 
                 {
                     'headerName': 'LR',
                     'field': 'manual_lr_q3',
                     'valueGetter':{"function": "(params.getValue('total_sales_q3') + params.getValue('manual_lrc_q3')) > Number(params.data.OP_UNITS_YTG_Q3) ? params.getValue('manual_lrm_q3') : params.getValue('manual_lrc_q3')"},
                     "valueFormatter": {"function": "d3.format(',.0f')(params.value)"},
                     'cellStyle': {"function": "params.value > 0 ? {'background-color': '#ff8f8f'} : null"},
                     "columnGroupShow": "open",
                     'hide':q3_toggle,
                     'cellStyle': {"function": "params.value > 0 ? {'background-color': '#ff8f8f'} : null"}
                 },
                 {
                     'headerName': 'LR$',
                     'field': 'manual_lrusd_q3',
                     'valueGetter':{"function": "params.getValue('manual_lr_q3') * Number(params.data.ASP)"},
                     "valueFormatter": {"function": "d3.format('$,.0f')(params.value)"},
                     'cellStyle': {"function": "params.value > 0 ? {'background-color': '#ff8f8f'} : null"},
                     'hide':q3_toggle
                 }
             ]
            },
            
            {'headerName': 'Quarter 4',
             'children': [
                 {'headerName': 'OP','field': 'OP_UNITS_YTG_Q4',"valueFormatter": {"function": "d3.format(',.0f')(params.value)"},'hide':q4_toggle},
                 {'headerName': 'SO Days','field': 'PROJ_SO_DAYS_Q4',"valueFormatter": {"function": "d3.format(',.0f')(params.value)"},'cellDataType':'number','editable': True, 'cellStyle': {'background-color': '#FDF0B0'},'valueParser': {'function': 'Number(params.newValue)',"columnGroupShow": "open"},'hide':q4_toggle},
                 {'headerName': 'SH Days','field': 'PROJ_SH_DAYS_Q4','type':'numericColumn', "valueFormatter": {"function": "d3.format(',.0f')(params.value)"},'cellDataType':'number','valueParser': {'function': 'Number(params.newValue)'},"columnGroupShow": "open",'hide':q4_toggle},
                 {
                     'headerName': 'In Stock',
                     'field':'manual_in_stock_q4',
                     'valueGetter':{"function": "Number(params.data.PROJ_SH_DAYS_Q4) === 0 ? 1 : 1-(Number(params.data.PROJ_SO_DAYS_Q4) / Number(params.data.PROJ_SH_DAYS_Q4))"},
                     'valueFormatter':{"function": "d3.format(',.1%')(params.value)"},
                     "columnGroupShow": "open",
                     'hide':q4_toggle,
                     'cellStyle': {"function": "params.value < 1 ? {'background-color': '#ff8f8f'} : null"}
                 },
                 
                 {
                     'headerName': 'skip_q3',
                     'field': 'skip_q3',
                     'valueGetter': {"function": "params.data.Q3_VISIBILITY ? params.getValue('excess_sales_q3') : params.getValue('TOTAL_EXCESS_SALES_YTD')"},
                     'hide':True
                 },
                 
                 {
                     'headerName': 'sales_lag_q4',
                     'field': 'sales_lag_q4',
                     'valueGetter': {"function":"params.data.Q4_VISIBILITY ? params.getValue('skip_q3') : 0"},
                     'hide':True
                 },
                 
                 {
                     'headerName': 'Sales',
                     'field': 'manual_proj_sales_q4',
                     'valueGetter': {"function":"Number(params.data.PROJ_SH_DAYS_Q4) === 0 ? Number(params.data.OP_UNITS_YTG_Q4) : Number(params.data.OP_UNITS_YTG_Q4) * (1-(Number(params.data.PROJ_SO_DAYS_Q4) / Number(params.data.PROJ_SH_DAYS_Q4)))"},
                     "valueFormatter": {"function": "d3.format(',.0f')(params.value)"},
                     'valueParser': {'function': 'Number(params.value)'},
                     "columnGroupShow": "open",
                     'hide':True
                 },
                 
                 {
                     'headerName': 'Total Sales',
                     'field': 'total_sales_q4',
                     'valueGetter': {"function":"params.getValue('sales_lag_q4') + params.getValue('manual_proj_sales_q4')"},
                     "valueFormatter": {"function": "d3.format(',.0f')(params.value)"},
                     'hide':q4_toggle
                 },
                 {
                     'headerName': 'Excess',
                     'field': 'excess_sales_q4',
                     "valueGetter": {"function": "params.getValue('total_sales_q4') <= Number(params.data.OP_UNITS_YTG_Q4) ? 0 : params.getValue('total_sales_q4') - Number(params.data.OP_UNITS_YTG_Q4);"},
                     "valueFormatter": {"function": "d3.format(',.0f')(params.value)"},
                     "columnGroupShow": "open",
                     'hide':q4_toggle
                 },
                 {
                     'headerName': 'Sales Perf.',
                     'field': 'manual_sales_per_q4',
                     'valueGetter':{"function": "Number(params.data.OP_UNITS_YTG_Q4) === 0 ? 1  : (params.getValue('total_sales_q4')/ Number(params.data.OP_UNITS_YTG_Q4))"},
                     'valueFormatter':{"function": "d3.format(',.1%')(params.value)"},
                     "columnGroupShow": "open",
                     'hide':q4_toggle,
                     'cellStyle': {"function": "params.value < 1 ? {'background-color': '#ff8f8f'} : null"}
                 },
                 
                 
                 {
                     'headerName': 'LR C',
                     'field': 'manual_lrc_q4',
                     'valueGetter':{"function": "params.getValue('manual_in_stock_q4') < 1 && params.getValue('manual_sales_per_q4') < 1 ? Number(params.data.OP_UNITS_YTG_Q4) * (1 - params.getValue('manual_in_stock_q4')) : 0"},
                     "valueFormatter": {"function": "d3.format(',.0f')(params.value)"},
                     'cellStyle': {"function": "params.value > 0 ? {'background-color': '#ff8f8f'} : null"},
                     "columnGroupShow": "open",
                     'hide':q4_toggle
                 },
                 {
                     'headerName': 'LR M',
                     'field': 'manual_lrm_q4',
                     'valueGetter':{"function": "params.getValue('manual_in_stock_q4') < 1 && params.getValue('manual_sales_per_q4') < 1 ? Number(params.data.OP_UNITS_YTG_Q4) - params.getValue('total_sales_q4') : 0"},
                     "valueFormatter": {"function": "d3.format(',.0f')(params.value)"},
                     'cellStyle': {"function": "params.value > 0 ? {'background-color': '#ff8f8f'} : null"},
                     "columnGroupShow": "open",
                     'hide':q4_toggle
                 },
                 
                 {
                     'headerName': 'LR',
                     'field': 'manual_lr_q4',
                     'valueGetter':{"function": "(params.getValue('total_sales_q4') + params.getValue('manual_lrc_q4')) > Number(params.data.OP_UNITS_YTG_Q4) ? params.getValue('manual_lrm_q4') : params.getValue('manual_lrc_q4')"},
                     "valueFormatter": {"function": "d3.format(',.0f')(params.value)"},
                     'cellStyle': {"function": "params.value > 0 ? {'background-color': '#ff8f8f'} : null"},
                     "columnGroupShow": "open",
                     'hide':q4_toggle,
                     'cellStyle': {"function": "params.value > 0 ? {'background-color': '#ff8f8f'} : null"}
                 },
                 {
                     'headerName': 'LR$',
                     'field': 'manual_lrusd_q4',
                     'valueGetter':{"function": "params.getValue('manual_lr_q4') * Number(params.data.ASP)"},
                     "valueFormatter": {"function": "d3.format('$,.0f')(params.value)"},
                     'cellStyle': {"function": "params.value > 0 ? {'background-color': '#ff8f8f'} : null"},
                     'hide':q4_toggle
                 }
             ]
            },
            
             
            {'headerName': 'Year-to-Go',
             'children': [
                 {'headerName': 'OP','field': 'manual_ytg_op',"valueFormatter": {"function": "d3.format(',.0f')(params.value)"}, "columnGroupShow": "open", 'valueGetter':{"function": "Number(params.data.OP_UNITS_YTG_Q1) + Number(params.data.OP_UNITS_YTG_Q2) + Number(params.data.OP_UNITS_YTG_Q3) + Number(params.data.OP_UNITS_YTG_Q4)"}},
                 {'headerName': 'Sales','field': 'manual_ytg_sales',"valueFormatter": {"function": "d3.format(',.0f')(params.value)"}, "columnGroupShow": "open", 'valueGetter':{"function": "params.getValue('manual_proj_sales_q1') + params.getValue('manual_proj_sales_q2') + params.getValue('manual_proj_sales_q3') + params.getValue('manual_proj_sales_q4')"}},
                 {'headerName': 'Sales Perf.','field': 'manual_ytg_perf',"valueFormatter": {"function": "d3.format('.0%')(params.value)"},'cellStyle': {"function": "params.value < 1 ? {'background-color': '#ff8f8f'} : null"}, "columnGroupShow": "open", 'valueGetter':{"function": "params.getValue('manual_ytg_op') === 0 ? 0 : params.getValue('manual_ytg_sales') / params.getValue('manual_ytg_op')"}},
                 {
                     'headerName': 'LR C',
                     'field': 'manual_lrc_ytg',
                     'valueGetter':{"function": "params.getValue('manual_lrc_q1') +params.getValue('manual_lrc_q2') +params.getValue('manual_lrc_q3')+params.getValue('manual_lrc_q4')"},
                     "valueFormatter": {"function": "d3.format(',.0f')(params.value)"},
                     'cellStyle': {"function": "params.value > 0 ? {'background-color': '#ff8f8f'} : null"},
                     "columnGroupShow": "open",
                 },
                 {
                     'headerName': 'LR M',
                     'field': 'manual_lrm_ytg',
                     'valueGetter':{"function": "params.getValue('manual_lrm_q1') +params.getValue('manual_lrm_q2') +params.getValue('manual_lrm_q3')+params.getValue('manual_lrm_q4')"},
                     "valueFormatter": {"function": "d3.format(',.0f')(params.value)"},
                     'cellStyle': {"function": "params.value > 0 ? {'background-color': '#ff8f8f'} : null"},
                     "columnGroupShow": "open",
                 },
                 
                 {
                     'headerName': 'LR',
                     'field': 'manual_lr_ytg',
                     'valueGetter':{"function": "params.getValue('manual_lr_q1') +params.getValue('manual_lr_q2') +params.getValue('manual_lr_q3')+params.getValue('manual_lr_q4')"},
                     "valueFormatter": {"function": "d3.format(',.0f')(params.value)"},
                     'cellStyle': {"function": "params.value > 0 ? {'background-color': '#ff8f8f'} : null"},
                     "columnGroupShow": "open",
                     'cellStyle': {"function": "params.value > 0 ? {'background-color': '#ff8f8f'} : null"}
                 },
                 {
                     'headerName': 'LR$',
                     'field': 'manual_lrusd_ytg',
                     'valueGetter':{"function": "params.getValue('manual_lr_ytg') * Number(params.data.ASP)"},
                     "valueFormatter": {"function": "d3.format('$,.0f')(params.value)"},
                     'cellStyle': {"function": "params.value > 0 ? {'background-color': '#ff8f8f'} : null"},
                 }
             ]
            },
            
           {'headerName': 'Full Year',
             'children': [
                 {'headerName': 'OP','field': 'manual_fy_op',"valueFormatter": {"function": "d3.format(',.0f')(params.value)"}, "columnGroupShow": "open", 'valueGetter':{"function": "Number(params.data.TOTAL_OP_UNITS_YTD) + Number(params.data.OP_UNITS_YTG_Q1) + Number(params.data.OP_UNITS_YTG_Q2) + Number(params.data.OP_UNITS_YTG_Q3) + Number(params.data.OP_UNITS_YTG_Q4)"}},
                 {
                     'headerName': 'Sales',
                     'field': 'manual_fy_sales',
                     "valueFormatter": {"function": "d3.format(',.0f')(params.value)"}, "columnGroupShow": "open", 'valueGetter':{"function": "params.data.TOTAL_SALES_YTD + params.getValue('manual_proj_sales_q1') + params.getValue('manual_proj_sales_q2') + params.getValue('manual_proj_sales_q3') + params.getValue('manual_proj_sales_q4')"}},
                 {'headerName': 'Excess','field': 'manual_fy_excess',"valueFormatter": {"function": "d3.format('.0%')(params.value)"},'cellStyle': {"function": "params.value < 1 ? {'background-color': '#ff8f8f'} : null"}, "columnGroupShow": "open", 'valueGetter':{"function": "params.getValue('manual_fy_sales') > params.getValue('manual_fy_op') ? params.getValue('manual_fy_sales') - params.getValue('manual_fy_op') : 0"}},
                 {'headerName': 'SO Days','field': 'manual_fy_so',"valueFormatter": {"function": "d3.format(',.0f')(params.value)"},'cellDataType':'number', 'cellStyle': {'background-color': '#FDF0B0'},'valueParser': {'function': 'Number(params.newValue)'},"columnGroupShow": "open", 'valueGetter':{"function": "params.data.TOTAL_SO_DAYS_YTD + Number(params.data.PROJ_SO_DAYS_Q1) + Number(params.data.PROJ_SO_DAYS_Q2) + Number(params.data.PROJ_SO_DAYS_Q3) + Number(params.data.PROJ_SO_DAYS_Q4)"}},
                 {'headerName': 'SH Days','field': 'manual_fy_sh','type':'numericColumn', "valueFormatter": {"function": "d3.format(',.0f')(params.value)"},'cellDataType':'number','valueParser': {'function': 'Number(params.newValue)'},"columnGroupShow": "open", 'valueGetter':{"function": "params.data.TOTAL_SH_DAYS_YTD + Number(params.data.PROJ_SH_DAYS_Q1) + Number(params.data.PROJ_SH_DAYS_Q2) + Number(params.data.PROJ_SH_DAYS_Q3) + Number(params.data.PROJ_SH_DAYS_Q4)"}},
                 {'headerName': 'In Stock','field':'manual_fy_is','valueFormatter':{"function": "d3.format(',.1%')(params.value)"},"columnGroupShow": "open", 'valueGetter':{"function": "params.getValue('manual_fy_sh') === 0 ? 1 : 1-(Number(params.getValue('manual_fy_so')) / Number(params.getValue('manual_fy_sh')))"}},
                 
                 {
                     'headerName': 'LR',
                     'field': 'manual_lr_fy',
                     'valueGetter':{"function": "params.getValue('manual_lr_ytg') + Number(params.data.TOTAL_LR_YTD)"},
                     "valueFormatter": {"function": "d3.format(',.0f')(params.value)"},
                     'cellStyle': {"function": "params.value > 0 ? {'background-color': '#ff8f8f'} : null"},
                     "columnGroupShow": "open",
                     'cellStyle': {"function": "params.value > 0 ? {'background-color': '#ff8f8f'} : null"}
                 },
                 {
                     'headerName': 'LR$',
                     'field': 'manual_lrusd_fy',
                     'valueGetter':{"function": "params.getValue('manual_lr_fy') * Number(params.data.ASP)"},
                     "valueFormatter": {"function": "d3.format('$,.0f')(params.value)"},
                     'cellStyle': {"function": "params.value > 0 ? {'background-color': '#ff8f8f'} : null"},
                 },
                 
                 {
                     'headerName': 'UR$',
                     'field': 'manual_urusd_fy',
                     'valueGetter':{"function": "params.getValue('manual_fy_excess') * Number(params.data.ASP)"},
                     "valueFormatter": {"function": "d3.format('$,.0f')(params.value)"},
                     'cellStyle': {"function": "params.value > 0 ? {'background-color': '#92d050'} : null"},
                 }
             ]
            },
            
            {
                'headerName': 'Root Cause',
                'field': 'EDITED_ROOT_CAUSE',
                'editable': True,
                
                'cellEditor': 'agSelectCellEditor',
                'cellEditorParams': {
                    'values': ['Quality Investigation','Global Allocation','Capacity Constraints','Site Transfer','Regulatory Issue','Oversales','Supply Delays','Import Process','Natural Disaster','Unforeseen Tender','Competitor SO']
                }
            },
            
            {
                'headerName': 'RSL Comments',
                'field': 'EDITED_RSL_COMMENTS',
                'editable': True
            }
        ],
        rowData = df_details.to_dict('records'),
        dashGridOptions = {
            "skipHeaderOnAutoSize": False,
            "suppressColumnVirtualisation": True,
            "animateRows":False,
            'rowSelection':'single',
        },
        columnSize = 'autoSize',
        defaultColDef={
            'minWidth': 6,
            'wrapHeaderText': True,
        },
        style={'height':'500px','font-size':'11px'},
        rowStyle={'font-size': '11px'},
        csvExportParams={
                "fileName": "lost_revenue_extract.csv",
            },
        
    )
    
    lr_table = html.Div([
        html.Button("Download CSV", id="main-extract-button", n_clicks=0),
        details_ag_grid
    ])
    
    return lr_table

@app.callback(
    Output('output-siq-div','children'),
    Input('details-ag-grid','selectedRows')
)

def update_siq_details(selected_rows):
    if not selected_rows:
        return None
    selected_values_mkt = {row['MARKET_DESC'] for row in selected_rows}
    selected_values_mat = {row['MATERIAL_KEY'] for row in selected_rows}
    
    df_siq_view = df_siq
    df_siq_view = df_siq_view[(df_siq_view['Market'].isin(selected_values_mkt)) & (df_siq_view['Material'].isin(selected_values_mat))]
    df_siq_view = df_siq_view.sort_values('KF').reset_index(drop=True)
    df_siq_view = df_siq_view.sort_values('Location').reset_index(drop=True)
    
    stcondsiq = []
    stcondsiq.append(
        {
            "condition": f"params.value < 0",
            "style": {"backgroundColor": '#ff8f8f', "color": '#ffffff'},
        }
    )
    
    siq_ag_grid = dag.AgGrid(
            id="siq_ag_grid",
            columnDefs=[{'field':i} for i in df_siq_view],
            defaultColDef={"cellStyle": {"styleConditions": stcondsiq}},
            rowData=df_siq_view.to_dict("records"),
            style={'height':'300px','font-size':'11px'},
            rowStyle={'font-size': '11px'},
            columnSize = 'autoSize',
            dashGridOptions = {"suppressColumnVirtualisation": True}
    )
    
    return siq_ag_grid

@app.callback(
    Output("confirm-save", "displayed"),
    [
        Input('save-data-btn','n_clicks'),
        Input('string-output','data'),
    ],
    State('details-ag-grid','rowData'),
)
    
def save_data_to_dataframe(n_clicks, string_data, rowData):
    if n_clicks > 0:
        if not rowData:
            return 'No data to save!!! :('
        start_time = datetime.datetime.now() 

        
        user_log = string_data
        
        output_df = pd.DataFrame(rowData)
        new_edits = dataiku.Dataset("LR_APP_EDITS")
        new_edits_df = new_edits.get_dataframe()
        #output_df = output_df[output_df.columns.intersection(ed_cols)]
        print(output_df)
        
        output_df['TIMESTAMP'] = dt.datetime.now()
        output_df['USERNAME'] = user_log
        
        #Recalculating LR based on confirmed changes
        
        #Recalculating YTD figures
        output_df['PROJ_IS_YTD'] = output_df.apply(lambda row: 1- (row['TOTAL_SO_DAYS_YTD'] / row['TOTAL_SH_DAYS_YTD']) if row['TOTAL_SH_DAYS_YTD'] != 0 else 1, axis=1)
        
        output_df['LRC_YTD'] = output_df.apply(lambda row: (row['TOTAL_OP_UNITS_YTD'] * (1 - row['PROJ_IS_YTD'])) if row['TOTAL_SALES_YTD'] < row['TOTAL_OP_UNITS_YTD'] and row['PROJ_IS_YTD'] < 1 else 0, axis=1)
        output_df['LRM_YTD'] = output_df.apply(lambda row: (row['TOTAL_OP_UNITS_YTD'] - row['TOTAL_SALES_YTD']) if row['TOTAL_SALES_YTD'] < row['TOTAL_OP_UNITS_YTD'] and row['PROJ_IS_YTD'] < 1 else 0, axis=1)
        output_df['LR_EVAL_YTD'] = output_df.apply(lambda row: (row['TOTAL_SALES_YTD'] + row['LRC_YTD']), axis=1)
        output_df['PROJ_LR_YTD'] = output_df.apply(lambda row: (row['LRM_YTD']) if row['LR_EVAL_YTD'] > row['TOTAL_OP_UNITS_YTD'] else row['LRC_YTD'], axis=1)
        output_df['PROJ_LR_USD_YTD'] = output_df.apply(lambda row: (row['PROJ_LR_YTD'] * row['ASP']), axis=1)
        
        #RECALCULATING Q1 FIGURES
        output_df['PROJ_IS_Q1'] = output_df.apply(lambda row: 1- (row['PROJ_SO_DAYS_Q1'] / row['PROJ_SH_DAYS_Q1']) if row['PROJ_SH_DAYS_Q1'] != 0 else 1, axis=1)
        output_df['SALES_LAG_Q1'] = output_df.apply(lambda row: (row['TOTAL_EXCESS_SALES_YTD']) if row['Q1_VISIBILITY'] else 0, axis=1)
        output_df['PROJ_SALES_Q1'] = output_df.apply(lambda row: (row['OP_UNITS_YTG_Q1'] * row['PROJ_IS_Q1']) if row['PROJ_SH_DAYS_Q1'] != 0 else row['OP_UNITS_YTG_Q1'], axis=1)
        output_df['TOTAL_SALES_Q1'] = output_df.apply(lambda row: row['SALES_LAG_Q1'] + row['PROJ_SALES_Q1'], axis=1)
        output_df['EXCESS_Q1'] = output_df.apply(lambda row: (row['TOTAL_SALES_Q1'] - row['OP_UNITS_YTG_Q1']) if row['TOTAL_SALES_Q1'] > row['OP_UNITS_YTG_Q1'] else 0, axis=1)
        output_df['PERF_Q1'] = output_df.apply(lambda row: (row['TOTAL_SALES_Q1'] / row['OP_UNITS_YTG_Q1']) if row['OP_UNITS_YTG_Q1'] > 0 else 1, axis=1)
        output_df['LRC_Q1'] = output_df.apply(lambda row: (row['OP_UNITS_YTG_Q1'] * (1 - row['PROJ_IS_Q1'])) if row['PERF_Q1'] < 1 and row['PROJ_IS_Q1'] < 1 else 0, axis=1)
        output_df['LRM_Q1'] = output_df.apply(lambda row: (row['OP_UNITS_YTG_Q1'] - row['TOTAL_SALES_Q1']) if row['PERF_Q1'] < 1 and row['PROJ_IS_Q1'] < 1 else 0, axis=1)
        output_df['LR_EVAL_Q1'] = output_df.apply(lambda row: (row['TOTAL_SALES_Q1'] + row['LRC_Q1']), axis=1)
        output_df['PROJ_LR_Q1'] = output_df.apply(lambda row: (row['LRM_Q1']) if row['LR_EVAL_Q1'] > row['OP_UNITS_YTG_Q1'] else row['LRC_Q1'], axis=1)
        output_df['PROJ_LR_USD_Q1'] = output_df.apply(lambda row: (row['PROJ_LR_Q1'] * row['ASP']), axis=1)
        
        #RECALCULATING Q2 FIGURES
        output_df['SKIP_Q1'] = output_df.apply(lambda row: (row['EXCESS_Q1']) if row['Q1_VISIBILITY'] else row['TOTAL_EXCESS_SALES_YTD'], axis=1)
        output_df['SALES_LAG_Q2'] = output_df.apply(lambda row: (row['SKIP_Q1']) if row['Q2_VISIBILITY'] else 0, axis=1)
        output_df['PROJ_IS_Q2'] = output_df.apply(lambda row: 1- (row['PROJ_SO_DAYS_Q2'] / row['PROJ_SH_DAYS_Q2']) if row['PROJ_SH_DAYS_Q2'] != 0 else 1, axis=1)
        output_df['PROJ_SALES_Q2'] = output_df.apply(lambda row: (row['OP_UNITS_YTG_Q2'] * row['PROJ_IS_Q2']) if row['PROJ_SH_DAYS_Q2'] != 0 else row['OP_UNITS_YTG_Q2'], axis=1)
        output_df['TOTAL_SALES_Q2'] = output_df.apply(lambda row: row['SALES_LAG_Q2'] + row['PROJ_SALES_Q2'], axis=1)
        output_df['EXCESS_Q2'] = output_df.apply(lambda row: (row['TOTAL_SALES_Q2'] - row['OP_UNITS_YTG_Q2']) if row['TOTAL_SALES_Q2'] > row['OP_UNITS_YTG_Q2'] else 0, axis=1)
        output_df['PERF_Q2'] = output_df.apply(lambda row: (row['TOTAL_SALES_Q2'] / row['OP_UNITS_YTG_Q2']) if row['OP_UNITS_YTG_Q2'] > 0 else 1, axis=1)
        output_df['LRC_Q2'] = output_df.apply(lambda row: (row['OP_UNITS_YTG_Q2'] * (1 - row['PROJ_IS_Q2'])) if row['PERF_Q2'] < 1 and row['PROJ_IS_Q2'] < 1 else 0, axis=1)
        output_df['LRM_Q2'] = output_df.apply(lambda row: (row['OP_UNITS_YTG_Q2'] - row['TOTAL_SALES_Q2']) if row['PERF_Q2'] < 1 and row['PROJ_IS_Q2'] < 1 else 0, axis=1)
        output_df['PROJ_LR_Q2'] = output_df.apply(lambda row: row['LRM_Q2'] if (row['TOTAL_SALES_Q2'] + row['LRC_Q2']) > row['OP_UNITS_YTG_Q2'] else row['LRC_Q2'], axis=1)
        output_df['PROJ_LR_USD_Q2'] = output_df.apply(lambda row: (row['PROJ_LR_Q2'] * row['ASP']), axis=1)
        
        #RECALCULATING Q3 FIGURES
        output_df['SKIP_Q2'] = output_df.apply(lambda row: (row['EXCESS_Q2']) if row['Q2_VISIBILITY'] else row['TOTAL_EXCESS_SALES_YTD'], axis=1)
        output_df['SALES_LAG_Q3'] = output_df.apply(lambda row: (row['SKIP_Q2']) if row['Q3_VISIBILITY'] else 0, axis=1)
        output_df['PROJ_IS_Q3'] = output_df.apply(lambda row: 1- (row['PROJ_SO_DAYS_Q3'] / row['PROJ_SH_DAYS_Q3']) if row['PROJ_SH_DAYS_Q3'] != 0 else 1, axis=1)
        output_df['PROJ_SALES_Q3'] = output_df.apply(lambda row: (row['OP_UNITS_YTG_Q3'] * row['PROJ_IS_Q3']) if row['PROJ_SH_DAYS_Q3'] != 0 else row['OP_UNITS_YTG_Q3'], axis=1)
        output_df['TOTAL_SALES_Q3'] = output_df.apply(lambda row: row['SALES_LAG_Q3'] + row['PROJ_SALES_Q3'], axis=1)
        output_df['EXCESS_Q3'] = output_df.apply(lambda row: (row['TOTAL_SALES_Q3'] - row['OP_UNITS_YTG_Q3']) if row['TOTAL_SALES_Q3'] > row['OP_UNITS_YTG_Q3'] else 0, axis=1)
        output_df['PERF_Q3'] = output_df.apply(lambda row: (row['TOTAL_SALES_Q3'] / row['OP_UNITS_YTG_Q3']) if row['OP_UNITS_YTG_Q3'] > 0 else 1, axis=1)
        output_df['LRC_Q3'] = output_df.apply(lambda row: (row['OP_UNITS_YTG_Q3'] * (1 - row['PROJ_IS_Q3'])) if row['PERF_Q3'] < 1 and row['PROJ_IS_Q3'] < 1 else 0, axis=1)
        output_df['LRM_Q3'] = output_df.apply(lambda row: (row['OP_UNITS_YTG_Q3'] - row['TOTAL_SALES_Q3']) if row['PERF_Q3'] < 1 and row['PROJ_IS_Q3'] < 1 else 0, axis=1)
        output_df['PROJ_LR_Q3'] = output_df.apply(lambda row: row['LRM_Q3'] if (row['TOTAL_SALES_Q3'] + row['LRC_Q3']) > row['OP_UNITS_YTG_Q3'] else row['LRC_Q3'], axis=1)
        output_df['PROJ_LR_USD_Q3'] = output_df.apply(lambda row: (row['PROJ_LR_Q3'] * row['ASP']), axis=1)
        
        #RECALCULATING Q4 FIGURES
        output_df['SKIP_Q3'] = output_df.apply(lambda row: (row['EXCESS_Q3']) if row['Q3_VISIBILITY'] else row['TOTAL_EXCESS_SALES_YTD'], axis=1)
        output_df['SALES_LAG_Q4'] = output_df.apply(lambda row: (row['SKIP_Q3']) if row['Q4_VISIBILITY'] else 0, axis=1)
        output_df['PROJ_IS_Q4'] = output_df.apply(lambda row: 1- (row['PROJ_SO_DAYS_Q4'] / row['PROJ_SH_DAYS_Q4']) if row['PROJ_SH_DAYS_Q4'] != 0 else 1, axis=1)
        output_df['PROJ_SALES_Q4'] = output_df.apply(lambda row: (row['OP_UNITS_YTG_Q4'] * row['PROJ_IS_Q4']) if row['PROJ_SH_DAYS_Q4'] != 0 else row['OP_UNITS_YTG_Q4'], axis=1)
        output_df['TOTAL_SALES_Q4'] = output_df.apply(lambda row: row['SALES_LAG_Q4'] + row['PROJ_SALES_Q4'], axis=1)
        output_df['EXCESS_Q4'] = output_df.apply(lambda row: (row['TOTAL_SALES_Q4'] - row['OP_UNITS_YTG_Q4']) if row['TOTAL_SALES_Q4'] > row['OP_UNITS_YTG_Q4'] else 0, axis=1)
        output_df['PERF_Q4'] = output_df.apply(lambda row: (row['TOTAL_SALES_Q4'] / row['OP_UNITS_YTG_Q4']) if row['OP_UNITS_YTG_Q4'] > 0 else 1, axis=1)
        output_df['LRC_Q4'] = output_df.apply(lambda row: (row['OP_UNITS_YTG_Q4'] * (1 - row['PROJ_IS_Q4'])) if row['PERF_Q4'] < 1 and row['PROJ_IS_Q4'] < 1 else 0, axis=1)
        output_df['LRM_Q4'] = output_df.apply(lambda row: (row['OP_UNITS_YTG_Q4'] - row['TOTAL_SALES_Q4']) if row['PERF_Q4'] < 1 and row['PROJ_IS_Q4'] < 1 else 0, axis=1)
        output_df['PROJ_LR_Q4'] = output_df.apply(lambda row: row['LRM_Q4'] if (row['TOTAL_SALES_Q4'] + row['LRC_Q4']) > row['OP_UNITS_YTG_Q4'] else row['LRC_Q4'], axis=1)
        output_df['PROJ_LR_USD_Q4'] = output_df.apply(lambda row: (row['PROJ_LR_Q4'] * row['ASP']), axis=1)
        
        output_df = new_edits_df.append(output_df)
        
        
        output_string = output_df.to_string()
        ds_output.write_with_schema(output_df,drop_and_create=False)
        scenario.run_and_wait()
        
        current_time = datetime.datetime.now() 
        duration = current_time - start_time
        
        
        return True
    
@app.callback(
    [
        Output('f_srg_dd','options'),
        Output('f_clu_dd','options'),
        Output('f_mkt_dd','options'),
        Output('f_sbu_dd','options'),
        Output('f_bnd_dd','options'),
        Output('f_rsl_dd','options'),
        Output('f_src_dd','options'),
        Output('f_seg_dd','options'),
        Output('f_amp_dd','options'),
        Output('f_dem_dd','options'),
        Output('f_dan_dd','options'),
        Output('f_mat_dd','options')
    ],
    [
      Input('apply-srg','n_clicks'),
      Input('apply-clu','n_clicks'),
      Input('apply-mkt','n_clicks'),
      Input('apply-sbu','n_clicks'),
      Input('apply-bnd','n_clicks'),
      Input('apply-rsl','n_clicks'),
      Input('apply-src','n_clicks'),
      Input('apply-seg','n_clicks'),
      Input('apply-amp','n_clicks'),
      Input('apply-dem','n_clicks'),
      Input('apply-dan','n_clicks'),
      Input('apply-mat','n_clicks')
    ],
    [
        State('f_srg_dd','value'),
        State('f_clu_dd','value'),
        State('f_mkt_dd','value'),
        State('f_sbu_dd','value'),
        State('f_bnd_dd','value'),
        State('f_rsl_dd','value'),
        State('f_src_dd','value'),
        State('f_seg_dd','value'),
        State('f_amp_dd','value'),
        State('f_dem_dd','value'),
        State('f_dan_dd','value'),
        State('f_mat_dd','value'),
        State('f_lr_ytd','value'),
        State('f_lr_ytg','value'),
        State('f_lr_fy','value')
    ]
)

def update_filters(n_srg, n_clu, n_mkt, n_sbu, n_bnd, n_rsl, n_src, n_seg, n_amp, n_dem, n_dan, n_mat, sel_srg, sel_clu, sel_mkt, sel_sbu, sel_bnd, sel_rsl, sel_src, sel_seg, sel_amp, sel_dem, sel_dan, sel_mat, sel_ytd, sel_ytg, sel_fy):
    
    fil_dd_df = df_items.copy()
    
    if n_srg > 0:
        if sel_srg:
            fil_dd_df = fil_dd_df[fil_dd_df['SUB_REGION_DESC'].isin(sel_srg)]

    if n_clu > 0:
        if sel_clu:
            fil_dd_df = fil_dd_df[fil_dd_df['CLUSTER_DESC'].isin(sel_clu)]
    if n_mkt > 0:
        if sel_mkt:
            fil_dd_df = fil_dd_df[fil_dd_df['MARKET_DESC'].isin(sel_mkt)]
    if n_sbu > 0:
        if sel_sbu:
            fil_dd_df = fil_dd_df[fil_dd_df['SUB_BUSINESS_UNIT_DESC'].isin(sel_sbu)]
    if n_bnd > 0:
        if sel_bnd:
            fil_dd_df = fil_dd_df[fil_dd_df['PRODUCT_DESC'].isin(sel_bnd)]
    if n_rsl > 0:
        if sel_rsl:
            fil_dd_df = fil_dd_df[fil_dd_df['REGIONAL_SUPPLY_LEADER'].isin(sel_rsl)]
    if n_src > 0:
        if sel_src:
            fil_dd_df = fil_dd_df[fil_dd_df['SOURCE_DESC'].isin(sel_src)]
    if n_seg > 0:
        if sel_seg:
            fil_dd_df = fil_dd_df[fil_dd_df['SEGMENT'].isin(sel_seg)]
    if n_amp > 0:
        if sel_amp:
            fil_dd_df = fil_dd_df[fil_dd_df['ABOVE_MARKET_PLANNER_DESC'].isin(sel_amp)]
    if n_dem > 0:
        if sel_dem:
            fil_dd_df = fil_dd_df[fil_dd_df['DEMAND_PLANNER_ID'].isin(sel_dem)]
    if n_dan > 0:
        if sel_dan:
            fil_dd_df = fil_dd_df[fil_dd_df['DEMAND_ANALYST_DESC'].isin(sel_dan)]
    if n_mat > 0:
        if sel_mat:
            fil_dd_df = fil_dd_df[fil_dd_df['MATERIAL_KEY'].isin(sel_mat)]
    if sel_ytd:
        fil_dd_df = fil_dd_df[fil_dd_df['HAS_YTD'] == True]
    if sel_ytg:
        fil_dd_df = fil_dd_df[fil_dd_df['HAS_YTG'] == True]
    if sel_fy:
        fil_dd_df = fil_dd_df[fil_dd_df['HAS_FY'] == True]
    f_srg = fil_dd_df['SUB_REGION_DESC'].drop_duplicates().sort_values()
    f_clu = fil_dd_df['CLUSTER_DESC'].drop_duplicates().sort_values()
    f_mkt = fil_dd_df['MARKET_DESC'].drop_duplicates().sort_values()
    f_sbu = fil_dd_df['SUB_BUSINESS_UNIT_DESC'].drop_duplicates().sort_values()
    f_bnd = fil_dd_df['PRODUCT_DESC'].drop_duplicates().sort_values()
    f_rsl = fil_dd_df['REGIONAL_SUPPLY_LEADER'].drop_duplicates().sort_values()
    f_src = fil_dd_df['SOURCE_DESC'].drop_duplicates().sort_values()
    f_seg = fil_dd_df['SEGMENT'].drop_duplicates().sort_values()
    f_amp = fil_dd_df['ABOVE_MARKET_PLANNER_DESC'].drop_duplicates().sort_values()
    f_dem = fil_dd_df['DEMAND_PLANNER_ID'].drop_duplicates().sort_values()
    f_dan = fil_dd_df['DEMAND_ANALYST_DESC'].drop_duplicates().sort_values()
    f_mat = fil_dd_df['MATERIAL_KEY'].drop_duplicates().sort_values()
    
    return f_srg, f_clu, f_mkt, f_sbu, f_bnd, f_rsl, f_src, f_seg, f_amp, f_dem, f_dan, f_mat



@app.callback(
    Output("details-ag-grid", "exportDataAsCsv"),
    Input("main-extract-button", "n_clicks"),
)
def export_data_as_csv(n_clicks):
    if n_clicks:
        return True
    return False
