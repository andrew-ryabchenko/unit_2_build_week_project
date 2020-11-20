#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input,Output,State
import plotly.express as px
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import rf_model


# In[2]:


data = pd.read_csv('cars_v2(ModelReady).csv', index_col='Unnamed: 0')
data['state']=data['state'].apply(lambda x: x.upper())
file = open('./aval_models.json','r')
aval_models = json.loads(file.read())
file.close()


# In[3]:


brands = [{'label': brand.capitalize(), 'value':brand} for brand in data['manufacturer'].value_counts().index]
year_options = [{'label': year, 'value':year} for year in range(2020,1980,-1)]
map_data = pd.crosstab(data['state'], 'count')
map_data['state'] = map_data.index


# In[ ]:



# Price Histogram figure
figure_price = px.histogram(data,x='price', nbins=20, title='Price Distribution')
figure_price.update_xaxes(title_text='Price USD')
figure_price.update_yaxes(title_text='Entries Count')
figure_price.update_traces(marker_color='rgb(252,78,42)', marker_line_color='rgb(227,26,28)',
                  marker_line_width=1.5, opacity=0.6)

# Brands Bar Chart figure
figure_brands = px.bar(x=data['manufacturer'].value_counts(), y=data['manufacturer'].value_counts().index,
                      orientation='h', title='Manufacturers')
figure_brands.update_layout( yaxis = dict(tickmode = 'linear',tick0 = 0,dtick = 1))
figure_brands.update_yaxes(title_text='')
figure_brands.update_xaxes(title_text='Entries Count')
figure_brands.update_traces(marker_color='rgb(252,78,42)', marker_line_color='rgb(227,26,28)',
                  marker_line_width=1.5, opacity=0.6)

# Chloropleth figure
figure_map = px.choropleth(map_data, locations='state', color='count',
                           locationmode='USA-states',
                           scope='usa',
                           labels={'state': 'State',
                                   'count': 'Number of Entries'},
                           color_continuous_scale=px.colors.sequential.YlOrRd,
                           title='Entries by State'
                          )

# Age distribution figure
figure_age = px.histogram(data, x='age', nbins = 20, title='Age Distribution')
figure_age.update_xaxes(title_text='Age')
figure_age.update_yaxes(title_text='Entries Count')
figure_age.update_traces(marker_color='rgb(252,78,42)', marker_line_color='rgb(227,26,28)',
                  marker_line_width=1.5, opacity=0.6)

app = dash.Dash()

app.layout = html.Div([
    
html.Div(['______Training Data Overview______'], id='body_title', className = 'containers'),
    
html.Div(id='output', className = 'containers'),
html.Div(id='mae', className = 'containers'),
     

html.Div([dcc.Dropdown(
    id="brand_selector", 
    options = brands,
    placeholder = 'Make', 
    value=None)], id='brand_selector_container', className = 'containers'),

html.Div([dcc.Dropdown(
    id="model_selector",
    placeholder='Model')], id='model_selector_container', className = 'containers'),
    
html.Div([html.Button(children='Submit', id='submit_button',n_clicks=0, className='containers')], id='submit_button_container', className='containers'),

html.Div([dcc.Dropdown(
    id='year_selector',
    options=year_options,
    placeholder='Year')],id='year_selector_container', className = 'containers'),

html.Div([dcc.Dropdown(
    id='condition_selector',
    options = [
    {'label':'New', 'value':'new'},
    {'label':'Excellent', 'value':'excellent'},
    {'label':'Good', 'value':'good'},
    {'label':'Fair', 'value':'fair'}
    ], placeholder='Condition')], id='condition_selector_container', className = 'containers'),
    
html.Div([dcc.Input(
            id='odometer_selector',
            className = 'containers',
            type='number',
            placeholder='Mileage',
            min=0, max = 300000, size='0'
        )], id='odometer_selector_container', className = 'containers'),
    
html.Div([dcc.Graph(
    id='price_distribution_plot',
    figure=figure_price,
    responsive = True,
    config={'displayModeBar':False}
    )], id='price_distribution_plot_container', className = 'containers'),
    
html.Div([dcc.Graph(
    id='brand_distribution',
    figure=figure_brands,
    responsive=True,
    config={'displayModeBar':False}
    )], id='brand_distribution_plot_container', className = 'containers'),

html.Div([dcc.Graph(
    id='map_plot',
    figure=figure_map,
    responsive=True,
    config={'displayModeBar':False}
    )], id='map_plot_container', className = 'containers'),
    
html.Div([dcc.Graph(
    id='age_distribution_plot',
    figure=figure_age,
    responsive=True,
    config={'displayModeBar':False}
    )], id='age_distribution_plot_container', className = 'containers')


], id='main_container')
                          
#______________________________________________________________________________                         

# Assign options to Model Selector

@app.callback(
    Output("model_selector", "options"),
    [Input("brand_selector", "value")])

def model_options(value):
    if (value):
        return [{'label': model.capitalize(), 'value':model} for model in aval_models[value]]
    return []

# Gather case data
@app.callback(
    [Output("output", "children"),Output("mae", "children"),Output("output", "style")],
    [Input("submit_button", "n_clicks")],
    [State("condition_selector","value"),
     State("odometer_selector", "value"),
     State("brand_selector", "value"),
     State("model_selector", "value"),
     State("year_selector", "value")])

def gather_case_data(n_clicks, cond, odo, brand, model, year):
    if (n_clicks>0):
        case = {'condition': cond, 'odometer': odo,  
                'manufacturer': brand, 'model': model, 'year': year}

        for i in case.keys():
            if (case[i] == None):
                return ('Incomplete entry', None, {'color':'red'})
        
        case['age'] = 2020 - case['year']
        
        if (case['age']==0):
            case['miles_per_year']=int(odo)
        
        if (odo==0):
            case['miles_per_year']=0
        
        if (case['age']>0 and odo>0):
            case['miles_per_year'] = int(case['odometer']/case['age'])
        
        sample = pd.DataFrame({'age': [case['age']], 'odometer': [odo], 'miles_per_year':[case['miles_per_year']],
                                                                                             'condition': [cond]})
        
        obj = rf_model.SelectModel(case['manufacturer'], case['model'], data)
        prediction = obj.fit_predict(sample)
        
        fig1 = px.histogram(x=obj.y, nbins=30, title=f"{case['manufacturer']} {case['model']} Price Distribution")
        fig1.update_xaxes(title_text='Price USD')
        fig1.update_yaxes(title_text='Entries Count')
        fig1.update_traces(marker_color='rgb(252,78,42)', marker_line_color='rgb(227,26,28)',
                          marker_line_width=1.5, opacity=0.6)
        
        
        fig2 = px.histogram(x=2020 - obj.X['age'], nbins = 30, title=f"{case['manufacturer']} {case['model']} Age Distribution")
        fig2.update_xaxes(title_text='Age')
        fig2.update_yaxes(title_text='Entries Count')
        fig2.update_traces(marker_color='rgb(252,78,42)', marker_line_color='rgb(227,26,28)',
                          marker_line_width=1.5, opacity=0.6)
        
        
        return (f'Predicted price: {prediction}$', 
                f"Mean Absolute Error for {case['manufacturer']} {case['model']} population: {obj.mae}$", 
                {'color':'white'})
                
    return (None, None, None)


if __name__ == '__main__':
    app.run_server(debug=False, port=8055)

