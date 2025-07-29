# app.py (Flask Backend)
from flask import Flask, jsonify, render_template, request, abort
import pandas as pd
import json
import os

app = Flask(__name__)

# Load real data from CSV
DATA_PATH = os.path.join(os.path.dirname(__file__), 'Data', 'medical_dahboard.csv')
df_real = pd.read_csv(DATA_PATH)

@app.route('/')
def index():
    """Renders the main dashboard HTML page."""
    return render_template('index.html')

def get_filtered_df(args):
    df = df_real.copy()
    # Apply global filters
    if args.get('classification') and args['classification'] != 'All':
        df = df[df['Medical Classification'] == args['classification']]
    if args.get('definition') and args['definition'] != 'All':
        df = df[df['Medical Definition'] == args['definition']]
    if args.get('drg') and args['drg'] != 'All':
        df = df[df['DRG Code'] == args['drg']]
    if args.get('state') and args['state'] != 'All' and args['state'] != 'All States':
        df = df[df['Provider State'] == args['state']]
    if args.get('region') and args['region'] != 'All':
        df = df[df['Hospital Referral Region'] == args['region']]
    # Range filters
    if args.get('reim_min') is not None:
        df = df[df['Reimbursement Rate'] >= float(args['reim_min'])]
    if args.get('reim_max') is not None:
        df = df[df['Reimbursement Rate'] <= float(args['reim_max'])]
    if args.get('charge_min') is not None:
        df = df[df['Average Covered Charges'] >= float(args['charge_min'])]
    if args.get('charge_max') is not None:
        df = df[df['Average Covered Charges'] <= float(args['charge_max'])]
    if args.get('discharge_min') is not None:
        df = df[df['Total Discharges'] >= int(args['discharge_min'])]
    if args.get('discharge_max') is not None:
        df = df[df['Total Discharges'] <= int(args['discharge_max'])]
    return df

@app.route('/api/filters')
def get_filters():
    df = df_real
    filters = {
        'classifications': sorted(df['Medical Classification'].dropna().unique()),
        'definitions': sorted(df['Medical Definition'].dropna().unique()),
        'drg_codes': sorted(df['DRG Code'].dropna().unique()),
        'states': sorted(df['Provider State'].dropna().unique()),
        'regions': sorted(df['Hospital Referral Region'].dropna().unique()),
        'reim_min': float(df['Reimbursement Rate'].min()),
        'reim_max': float(df['Reimbursement Rate'].max()),
        'charge_min': float(df['Average Covered Charges'].min()),
        'charge_max': float(df['Average Covered Charges'].max()),
        'discharge_min': int(df['Total Discharges'].min()),
        'discharge_max': int(df['Total Discharges'].max()),
    }
    return jsonify(filters)

@app.route('/api/treemap_data')
def get_treemap_data():
    args = request.args.to_dict()
    df = get_filtered_df(args)
    classification_filter = args.get('classification', 'All')
    group_by = args.get('group_by', 'Definition')
    # Apply local classification filter if present
    if classification_filter != 'All':
        df = df[df['Medical Classification'] == classification_filter]
    # Group by the selected dimension
    if group_by == 'DRG':
        grouped_data = df.groupby('DRG Code')['Total Discharges'].sum().reset_index()
        grouped_data.rename(columns={'DRG Code': 'Label'}, inplace=True)
    else: # Default to Definition
        grouped_data = df.groupby('Medical Definition')['Total Discharges'].sum().reset_index()
        grouped_data.rename(columns={'Medical Definition': 'Label'}, inplace=True)
    treemap_data = grouped_data.to_dict(orient='records')
    return jsonify(treemap_data)

@app.route('/api/bar_chart_data')
def get_bar_chart_data():
    args = request.args.to_dict()
    df = get_filtered_df(args)
    definition = args.get('definition', 'All')
    sort = args.get('sort', 'highest')
    # Filter by definition if not All
    if definition != 'All' and definition != 'All Definitions':
        df = df[df['Medical Definition'] == definition]
    # Group by Medical Definition
    grouped = df.groupby('Medical Definition').agg({
        'Reimbursement Rate': 'mean',
        'Average Covered Charges': 'mean',
        'Average Total Payments': 'mean'
    }).reset_index()
    grouped['Gap'] = grouped['Average Covered Charges'] - grouped['Average Total Payments']
    # Sorting
    if sort == 'highest':
        grouped = grouped.sort_values('Reimbursement Rate', ascending=False)
        y = grouped['Reimbursement Rate']
    elif sort == 'alpha':
        grouped = grouped.sort_values('Medical Definition')
        y = grouped['Reimbursement Rate']
    elif sort == 'charges':
        grouped = grouped.sort_values('Average Covered Charges', ascending=False)
        y = grouped['Average Covered Charges']
    elif sort == 'payments':
        grouped = grouped.sort_values('Average Total Payments', ascending=False)
        y = grouped['Average Total Payments']
    elif sort == 'gap':
        grouped = grouped.sort_values('Gap', ascending=False)
        y = grouped['Gap']
    else:
        y = grouped['Reimbursement Rate']
    data = [
        {'Label': row['Medical Definition'], 'Value': val}
        for row, val in zip(grouped.to_dict('records'), y)
    ]
    return jsonify(data)

@app.route('/api/choropleth_data')
def get_choropleth_data():
    args = request.args.to_dict()
    df = get_filtered_df(args)
    classification = args.get('classification', 'All')
    definition = args.get('definition', 'All')
    if classification != 'All' and classification != 'All Classifications':
        df = df[df['Medical Classification'] == classification]
    if definition != 'All' and definition != 'All Definitions':
        df = df[df['Medical Definition'] == definition]
    grouped = df.groupby('Provider State').agg({'Average Covered Charges': 'mean'}).reset_index()
    grouped.rename(columns={'Provider State': 'State'}, inplace=True)
    data = grouped.to_dict(orient='records')
    return jsonify(data)

@app.route('/api/scatter_data')
def get_scatter_data():
    args = request.args.to_dict()
    df = get_filtered_df(args)
    colorBy = args.get('colorBy', '').split(',') if args.get('colorBy') else []
    regression = args.get('regression', 'false') == 'true'
    traces = []
    color_map = {
        'Circulatory System': '#4fd1c5',
        'Blood Diseases': '#1e90ff',
        'Alcohol and Drug Use': '#ffb347',
        'Respiratory System': '#b388ff',
        'Endocrine, Nutritional & Metabolic': '#ff6b6b',
        'Musculoskeletal System': '#43e97b',
    }
    if colorBy:
        for cls in colorBy:
            sub = df[df['Medical Classification'] == cls]
            traces.append({
                'x': sub['Average Covered Charges'].tolist(),
                'y': sub['Average Total Payments'].tolist(),
                'name': cls,
                'color': color_map.get(cls, '#4fd1c5'),
                'text': sub['Provider Name'].tolist()
            })
    else:
        traces.append({
            'x': df['Average Covered Charges'].tolist(),
            'y': df['Average Total Payments'].tolist(),
            'name': 'All',
            'color': '#4fd1c5',
            'text': df['Provider Name'].tolist()
        })
    # Regression line (simple linear fit)
    regression_line = None
    if regression and len(df) > 1:
        import numpy as np
        x = df['Average Covered Charges'].values
        y = df['Average Total Payments'].values
        coeffs = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = coeffs[0] * x_line + coeffs[1]
        regression_line = {'x': x_line.tolist(), 'y': y_line.tolist()}
    return jsonify({'traces': traces, 'regression': regression_line})

@app.route('/api/provider_treemap_data')
def get_provider_treemap_data():
    args = request.args.to_dict()
    df = get_filtered_df(args)
    state = args.get('state', 'All')
    topN = int(args.get('topN', 10))
    if state != 'All' and state != 'All States':
        df = df[df['Provider State'] == state]
    grouped = df.groupby('Provider Name')['Total Discharges'].sum().reset_index()
    grouped = grouped.sort_values('Total Discharges', ascending=False).head(topN)
    grouped.rename(columns={'Provider Name': 'Label'}, inplace=True)
    data = grouped.to_dict(orient='records')
    return jsonify(data)

@app.route('/api/histogram_data')
def get_histogram_data():
    args = request.args.to_dict()
    df = get_filtered_df(args)
    binWidth = float(args.get('binWidth', 0.1))
    classification = args.get('classification', 'All')
    if classification != 'All':
        df = df[df['Medical Classification'] == classification]
    x = df['Reimbursement Rate'].tolist()
    return jsonify({'x': x})

if __name__ == '__main__':
    app.run(debug=True)
