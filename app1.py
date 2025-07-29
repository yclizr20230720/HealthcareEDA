# app.py (Flask Backend)
from flask import Flask, jsonify, render_template, request
import pandas as pd
import numpy as np
import json
from flask_cors import CORS # Import CORS
from sklearn.linear_model import LinearRegression # For scatter plot regression line

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# --- Mock Data Generation ---
# This mock data simulates the structure needed for the dashboard's treemap.
# In a real application, you would load your correctly formatted CSV here.
def generate_mock_data(num_records=2000):
    np.random.seed(42) # for reproducibility

    drg_codes = [
        '039 - EXTRACRANIAL PROCEDURES W CC',
        '001 - HEART TRANSPLANT OR IMPLANT OF HEART ASSIST SYSTEM W MCC',
        '291 - HEART FAILURE & SHOCK W MCC',
        '292 - HEART FAILURE & SHOCK W CC',
        '897 - ALCOHOL/DRUG ABUSE OR DEPENDENCE W CC',
        '898 - ALCOHOL/DRUG ABUSE OR DEPENDENCE W/O CC/MCC',
        '302 - KIDNEY TRANSPLANT',
        '286 - ACUTE MYOCARDIAL INFARCTION, DISCHARGED ALIVE W MCC',
        '287 - ACUTE MYOCARDIAL INFARCTION, DISCHARGED ALIVE W CC',
        '880 - ACUTE LEUKEMIA WITHOUT MCC',
        '881 - CHRONIC LEUKEMIA WITHOUT MCC',
        '917 - POISONING & TOXIC EFFECTS OF DRUGS W MCC',
        '918 - POISONING & TOXIC EFFECTS OF DRUGS W CC',
        '190 - CHRONIC OBSTRUCTIVE PULMONARY DISEASE W MCC',
        '191 - CHRONIC OBSTRUCTIVE PULMONARY DISEASE W CC',
        '640 - MISCELLANEOUS DISORDERS OF NUTRITION, METABOLISM, FLUIDS & ELECTROLYTES W MCC',
        '641 - MISCELLANEOUS DISORDERS OF NUTRITION, METABOLISM, FLUIDS & ELECTROLYTES W CC',
        '470 - MAJOR JOINT REPLACEMENT OR REATTACHMENT OF LOWER EXTREMITY W MCC',
        '471 - MAJOR JOINT REPLACEMENT OR REATTACHMENT OF LOWER EXTREMITY W CC',
    ]
    medical_definitions = [
        'EXTRACRANIAL PROCEDURES', 'HEART TRANSPLANT', 'HEART FAILURE', 'ALCOHOL/DRUG ABUSE',
        'KIDNEY TRANSPLANT', 'ACUTE MYOCARDIAL INFARCTION', 'ACUTE LEUKEMIA',
        'POISONING & TOXIC EFFECTS OF DRUGS', 'COPD', 'METABOLIC DISORDERS',
        'JOINT REPLACEMENT'
    ]
    medical_classification_map = {
        'EXTRACRANIAL PROCEDURES': 'Circulatory System',
        'HEART TRANSPLANT': 'Circulatory System',
        'HEART FAILURE': 'Circulatory System',
        'ALCOHOL/DRUG ABUSE': 'Alcohol and Drug Use',
        'KIDNEY TRANSPLANT': 'Blood Diseases',
        'ACUTE MYOCARDIAL INFARCTION': 'Circulatory System',
        'ACUTE LEUKEMIA': 'Blood Diseases',
        'POISONING & TOXIC EFFECTS OF DRUGS': 'Alcohol and Drug Use',
        'COPD': 'Respiratory System',
        'METABOLIC DISORDERS': 'Endocrine, Nutritional & Metabolic',
        'JOINT REPLACEMENT': 'Musculoskeletal System'
    }

    provider_names = [f'Hospital {chr(65 + i)}' for i in range(50)]
    provider_states = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY']
    hospital_referral_regions = [
        'New York City', 'Los Angeles', 'Houston', 'Miami', 'Atlanta', 'Chicago',
        'Seattle', 'Portland', 'Phoenix', 'Denver', 'Boston', 'Philadelphia',
        'Dallas', 'San Francisco', 'Washington D.C.', 'Detroit', 'Minneapolis',
        'St. Louis', 'Cleveland', 'Baltimore', 'Pittsburgh', 'San Diego',
        'Orlando', 'Tampa', 'Charlotte', 'Nashville', 'Kansas City', 'Indianapolis'
    ]

    data = {
        'DRG Code': np.random.choice(drg_codes, num_records),
        'Medical Definition': np.random.choice(medical_definitions, num_records),
        'Total Discharges': np.random.randint(11, 1571, num_records),
        'Provider Name': np.random.choice(provider_names, num_records),
        'Provider State': np.random.choice(provider_states, num_records),
        'Hospital Referral Region': np.random.choice(hospital_referral_regions, num_records),
    }

    df = pd.DataFrame(data)

    # Map Medical Definition to Medical Classification
    df['Medical Classification'] = df['Medical Definition'].map(medical_classification_map)

    # Generate Average Covered Charges and Average Total Payments with some correlation
    df['Average Covered Charges'] = np.random.uniform(2537, 351798, num_records)
    df['Average Total Payments'] = df['Average Covered Charges'] * np.random.uniform(0.4, 0.9, num_records) # Payments are generally less than charges

    # Calculate Reimbursement Rate
    df['Reimbursement Rate'] = df['Average Total Payments'] / df['Average Covered Charges']
    df['Reimbursement Rate'] = df['Reimbursement Rate'].clip(0.05, 1.87) # Clip to specified range

    # Ensure some definitions/DRGs have higher discharges for treemap
    for _ in range(num_records // 10):
        idx = np.random.randint(0, num_records)
        df.loc[idx, 'Total Discharges'] = np.random.randint(500, 2000)

    return df

df_mock = generate_mock_data(num_records=5000) # Generate more data for better visualization

@app.route('/')
def index():
    """Renders the main dashboard HTML page."""
    # Get unique values for dropdowns
    classifications = ['All'] + sorted(df_mock['Medical Classification'].dropna().unique().tolist())
    definitions = ['All'] + sorted(df_mock['Medical Definition'].dropna().unique().tolist())
    drg_codes = ['All'] + sorted(df_mock['DRG Code'].dropna().unique().tolist())
    provider_states = ['All States'] + sorted(df_mock['Provider State'].dropna().unique().tolist())
    hospital_regions = ['All'] + sorted(df_mock['Hospital Referral Region'].dropna().unique().tolist())

    # Get min/max for range sliders
    min_reimbursement = float(df_mock['Reimbursement Rate'].min())
    max_reimbursement = float(df_mock['Reimbursement Rate'].max())
    min_charges = float(df_mock['Average Covered Charges'].min())
    max_charges = float(df_mock['Average Covered Charges'].max())
    min_discharges = float(df_mock['Total Discharges'].min())
    max_discharges = float(df_mock['Total Discharges'].max())

    return render_template('index.html',
                           classifications=classifications,
                           definitions=definitions,
                           drg_codes=drg_codes,
                           provider_states=provider_states,
                           hospital_regions=hospital_regions,
                           min_reimbursement=min_reimbursement,
                           max_reimbursement=max_reimbursement,
                           min_charges=min_charges,
                           max_charges=max_charges,
                           min_discharges=min_discharges,
                           max_discharges=max_discharges)

def apply_global_filters(df, filters):
    """Applies global filters to the DataFrame."""
    filtered_df = df.copy()

    if filters.get('classification') and filters['classification'] != 'All':
        filtered_df = filtered_df[filtered_df['Medical Classification'] == filters['classification']]
    if filters.get('definition') and filters['definition'] != 'All':
        filtered_df = filtered_df[filtered_df['Medical Definition'] == filters['definition']]
    if filters.get('drg_code') and filters['drg_code'] != 'All':
        filtered_df = filtered_df[filtered_df['DRG Code'] == filters['drg_code']]
    if filters.get('provider_state') and filters['provider_state'] != 'All States':
        filtered_df = filtered_df[filtered_df['Provider State'] == filters['provider_state']]
    if filters.get('hospital_region') and filters['hospital_region'] != 'All':
        filtered_df = filtered_df[filtered_df['Hospital Referral Region'] == filters['hospital_region']]

    # Numeric filters
    if filters.get('reimbursement_rate_min') is not None:
        filtered_df = filtered_df[filtered_df['Reimbursement Rate'] >= float(filters['reimbursement_rate_min'])]
    if filters.get('reimbursement_rate_max') is not None:
        filtered_df = filtered_df[filtered_df['Reimbursement Rate'] <= float(filters['reimbursement_rate_max'])]
    if filters.get('charges_min') is not None:
        filtered_df = filtered_df[filtered_df['Average Covered Charges'] >= float(filters['charges_min'])]
    if filters.get('charges_max') is not None:
        filtered_df = filtered_df[filtered_df['Average Covered Charges'] <= float(filters['charges_max'])]
    if filters.get('discharges_min') is not None:
        filtered_df = filtered_df[filtered_df['Total Discharges'] >= int(filters['discharges_min'])]
    if filters.get('discharges_max') is not None:
        filtered_df = filtered_df[filtered_df['Total Discharges'] <= int(filters['discharges_max'])]

    return filtered_df

@app.route('/api/treemap_data')
def get_treemap_data():
    """API for 'Total discharges by medical definition' treemap."""
    filters = request.args.to_dict()
    filtered_df = apply_global_filters(df_mock, filters)

    group_by = filters.get('group_by_treemap', 'Definition') # Renamed to avoid conflict

    if group_by == 'DRG':
        grouped_data = filtered_df.groupby('DRG Code')['Total Discharges'].sum().reset_index()
        grouped_data.rename(columns={'DRG Code': 'Label'}, inplace=True)
    else: # Default to Definition
        grouped_data = filtered_df.groupby('Medical Definition')['Total Discharges'].sum().reset_index()
        grouped_data.rename(columns={'Medical Definition': 'Label'}, inplace=True)

    treemap_data = grouped_data.to_dict(orient='records')
    return jsonify(treemap_data)

@app.route('/api/reimbursement_bar_data')
def get_reimbursement_bar_data():
    """API for 'Reimbursement rate by medical classification' bar chart."""
    filters = request.args.to_dict()
    filtered_df = apply_global_filters(df_mock, filters)

    selected_definition = filters.get('bar_chart_definition', 'All')
    sort_option = filters.get('bar_chart_sort', 'highest_to_lowest_reimbursement')

    if selected_definition != 'All':
        filtered_df = filtered_df[filtered_df['Medical Definition'] == selected_definition]

    # Group by Medical Classification and Definition to get average reimbursement
    grouped_data = filtered_df.groupby(['Medical Classification', 'Medical Definition']).agg(
        AverageReimbursement=('Reimbursement Rate', 'mean'),
        AverageCoveredCharges=('Average Covered Charges', 'mean'),
        AverageTotalPayments=('Average Total Payments', 'mean')
    ).reset_index()

    # Sort based on option
    if sort_option == 'highest_to_lowest_reimbursement':
        grouped_data = grouped_data.sort_values(by='AverageReimbursement', ascending=False)
    elif sort_option == 'alphabetical':
        grouped_data = grouped_data.sort_values(by='Medical Definition', ascending=True)
    elif sort_option == 'highest_to_lowest_charges':
        grouped_data = grouped_data.sort_values(by='AverageCoveredCharges', ascending=False)
    elif sort_option == 'highest_to_lowest_payments':
        grouped_data = grouped_data.sort_values(by='AverageTotalPayments', ascending=False)
    elif sort_option == 'largest_gap':
        grouped_data['Gap'] = grouped_data['AverageCoveredCharges'] - grouped_data['AverageTotalPayments']
        grouped_data = grouped_data.sort_values(by='Gap', ascending=False)


    return jsonify(grouped_data.to_dict(orient='records'))

@app.route('/api/choropleth_data')
def get_choropleth_data():
    """API for 'Geographic variation in average covered charges' choropleth map."""
    filters = request.args.to_dict()
    filtered_df = apply_global_filters(df_mock, filters)

    # Group by state and calculate average covered charges
    state_charges = filtered_df.groupby('Provider State')['Average Covered Charges'].mean().reset_index()
    state_charges.rename(columns={'Provider State': 'State', 'Average Covered Charges': 'AvgCharges'}, inplace=True)

    return jsonify(state_charges.to_dict(orient='records'))

@app.route('/api/scatter_data')
def get_scatter_data():
    """API for 'Average total payments vs. average covered charges' scatter plot."""
    filters = request.args.to_dict()
    filtered_df = apply_global_filters(df_mock, filters)

    color_by_classification = filters.get('color_by_classification', 'All') # Default to All for coloring
    show_regression = filters.get('show_regression', 'false') == 'true'

    plot_data = []

    if color_by_classification != 'All':
        # Filter for specific classification if selected for coloring
        filtered_df = filtered_df[filtered_df['Medical Classification'] == color_by_classification]
        # If a specific classification is selected, we only plot that one
        classifications_to_plot = [color_by_classification]
    else:
        # Otherwise, plot all relevant classifications
        classifications_to_plot = ['Circulatory System', 'Blood Diseases', 'Alcohol and Drug Use'] # As per prompt

    for classification in classifications_to_plot:
        class_df = filtered_df[filtered_df['Medical Classification'] == classification]
        if not class_df.empty:
            plot_data.append({
                'classification': classification,
                'charges': class_df['Average Covered Charges'].tolist(),
                'payments': class_df['Average Total Payments'].tolist()
            })

            # Calculate regression line for this classification
            if show_regression and len(class_df) > 1:
                X = class_df['Average Covered Charges'].values.reshape(-1, 1)
                y = class_df['Average Total Payments'].values
                model = LinearRegression()
                model.fit(X, y)
                x_range = np.array([X.min(), X.max()]).reshape(-1, 1)
                y_pred = model.predict(x_range)
                plot_data.append({
                    'classification': classification + ' (Regression)',
                    'regression_x': x_range.flatten().tolist(),
                    'regression_y': y_pred.flatten().tolist()
                })

    return jsonify(plot_data)


@app.route('/api/provider_treemap_data')
def get_provider_treemap_data():
    """API for 'Total discharges by provider' treemap."""
    filters = request.args.to_dict()
    filtered_df = apply_global_filters(df_mock, filters)

    state_filter = filters.get('provider_treemap_state', 'All States')
    top_n = int(filters.get('top_n_providers', 10))

    if state_filter != 'All States':
        filtered_df = filtered_df[filtered_df['Provider State'] == state_filter]

    # Group by provider name and sum discharges
    grouped_data = filtered_df.groupby('Provider Name')['Total Discharges'].sum().reset_index()
    grouped_data = grouped_data.sort_values(by='Total Discharges', ascending=False)

    # Limit to top N providers
    grouped_data = grouped_data.head(top_n)

    grouped_data.rename(columns={'Provider Name': 'Label'}, inplace=True)
    return jsonify(grouped_data.to_dict(orient='records'))

@app.route('/api/histogram_data')
def get_histogram_data():
    """API for 'Distribution of reimbursement rates' histogram."""
    filters = request.args.to_dict()
    filtered_df = apply_global_filters(df_mock, filters)

    bin_width = float(filters.get('bin_width', 0.1))
    segment_by_classification = filters.get('segment_by_classification', 'All')

    # If segment_by_classification is 'All', we'll return a single histogram
    # Otherwise, we'll return data for each specified classification
    if segment_by_classification == 'All':
        data_to_plot = {
            'All': filtered_df['Reimbursement Rate'].tolist()
        }
    else:
        # Only segment by the specified classifications from the prompt
        target_classifications = ['Circulatory System', 'Blood Diseases', 'Alcohol and Drug Use']
        data_to_plot = {}
        for cls in target_classifications:
            class_df = filtered_df[filtered_df['Medical Classification'] == cls]
            if not class_df.empty:
                data_to_plot[cls] = class_df['Reimbursement Rate'].tolist()

    return jsonify({
        'data': data_to_plot,
        'bin_width': bin_width
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0') # Changed host to '0.0.0.0'
