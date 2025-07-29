# app.py (Flask Backend)
from flask import Flask, jsonify, render_template, request, redirect, url_for, flash
import pandas as pd
import numpy as np
import json
from flask_cors import CORS
from sklearn.linear_model import LinearRegression
import io # For file handling in memory

app = Flask(__name__)
app.secret_key = 'supersecretkey' # Needed for flash messages
CORS(app)

# Global variable to store uploaded DataFrame
# In a real-world multi-user application, this would be stored per-user (e.g., in a session, database, or cloud storage)
uploaded_df = None

# --- Mock Data Generation ---
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

mock_df = generate_mock_data(num_records=5000)

# Define required columns for validation
REQUIRED_COLUMNS = [
    'DRG Code', 'Medical Definition', 'Medical Classification', 'Total Discharges',
    'Average Covered Charges', 'Average Total Payments', 'Reimbursement Rate',
    'Provider Name', 'Provider State', 'Hospital Referral Region'
]

def get_current_data():
    """Returns the uploaded DataFrame if available, otherwise the mock DataFrame."""
    return uploaded_df if uploaded_df is not None else mock_df

@app.route('/')
def index():
    """Renders the main dashboard HTML page."""
    current_data = get_current_data()

    # Get unique values for dropdowns
    classifications = ['All'] + sorted(current_data['Medical Classification'].dropna().unique().tolist())
    definitions = ['All'] + sorted(current_data['Medical Definition'].dropna().unique().tolist())
    drg_codes = ['All'] + sorted(current_data['DRG Code'].dropna().unique().tolist())
    provider_states = ['All States'] + sorted(current_data['Provider State'].dropna().unique().tolist())
    hospital_regions = ['All'] + sorted(current_data['Hospital Referral Region'].dropna().unique().tolist())

    # Get min/max for range sliders
    min_reimbursement = float(current_data['Reimbursement Rate'].min())
    max_reimbursement = float(current_data['Reimbursement Rate'].max())
    min_charges = float(current_data['Average Covered Charges'].min())
    max_charges = float(current_data['Average Covered Charges'].max())
    min_discharges = float(current_data['Total Discharges'].min())
    max_discharges = float(current_data['Total Discharges'].max())

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
                           max_discharges=max_discharges,
                           uploaded_data_exists=(uploaded_df is not None),
                           current_page='analysis_report') # Pass current page name

@app.route('/data_access')
def data_access():
    """Renders the data access page."""
    return render_template('data_access.html', required_columns=REQUIRED_COLUMNS, current_page='data_access') # Pass current page name

@app.route('/data_engineering')
def data_engineering():
    """Placeholder for Data Engineering page."""
    flash('Data Engineering functionality is coming soon!', 'info')
    return render_template('coming_soon.html', current_page='data_engineering')

@app.route('/machine_learning')
def machine_learning():
    """Placeholder for Machine Learning page."""
    flash('Machine Learning functionality is coming soon!', 'info')
    return render_template('coming_soon.html', current_page='machine_learning')

@app.route('/job_status_review')
def job_status_review():
    """Placeholder for Job Status Review page."""
    flash('Job Status Review functionality is coming soon!', 'info')
    return render_template('coming_soon.html', current_page='job_status_review')


def read_uploaded_file(file, filename):
    """Reads an uploaded file into a pandas DataFrame based on its extension."""
    df = None
    file_extension = filename.rsplit('.', 1)[1].lower()
    try:
        if file_extension == 'csv':
            df = pd.read_csv(io.StringIO(file.read().decode('utf-8')))
        elif file_extension in ['xls', 'xlsx']:
            df = pd.read_excel(io.BytesIO(file.read()))
        elif file_extension == 'txt':
            # Assuming tab-separated or comma-separated for TXT, try comma first
            try:
                df = pd.read_csv(io.StringIO(file.read().decode('utf-8')), sep=',')
            except Exception:
                df = pd.read_csv(io.StringIO(file.read().decode('utf-8')), sep='\t')
        else:
            raise ValueError("Unsupported file type.")
    except Exception as e:
        raise ValueError(f"Error reading file: {e}")
    return df

@app.route('/upload_file', methods=['POST'])
def upload_file():
    """Handles file uploads from the data access page."""
    global uploaded_df

    if 'file' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('data_access'))

    file = request.files['file']
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('data_access'))

    if file:
        try:
            df = read_uploaded_file(file, file.filename)

            # Validate columns
            missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
            if missing_columns:
                flash(f"Uploaded file is missing required columns: {', '.join(missing_columns)}. Please ensure your file has the correct format.", 'error')
                return redirect(url_for('data_access'))

            # Ensure numeric columns are correct type (coercing errors to NaN)
            numeric_cols = ['Total Discharges', 'Average Covered Charges', 'Average Total Payments', 'Reimbursement Rate']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    if df[col].isnull().any():
                        flash(f"Warning: Column '{col}' contains non-numeric values that were converted to NaN. Please check your data.", 'warning')

            # Recalculate Reimbursement Rate if needed or if it might be incorrect
            if 'Reimbursement Rate' not in df.columns or df['Reimbursement Rate'].isnull().any():
                 df['Reimbursement Rate'] = df['Average Total Payments'] / df['Average Covered Charges']
                 flash("Reimbursement Rate was calculated from 'Average Total Payments' and 'Average Covered Charges'.", 'info')

            # Map Medical Definition to Medical Classification if Classification is missing
            if 'Medical Classification' not in df.columns:
                flash("Warning: 'Medical Classification' column missing. Attempting to infer from 'Medical Definition' (may not be accurate).", 'warning')
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
                df['Medical Classification'] = df['Medical Definition'].map(medical_classification_map).fillna('Other')


            uploaded_df = df
            flash('File uploaded and data loaded successfully! Dashboard updated.', 'success')
            return redirect(url_for('index'))

        except ValueError as e:
            flash(f'File upload failed: {e}', 'error')
            return redirect(url_for('data_access'))
        except Exception as e:
            flash(f'An unexpected error occurred during file processing: {e}', 'error')
            return redirect(url_for('data_access'))

    flash('No file selected.', 'error')
    return redirect(url_for('data_access'))

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
    current_data = get_current_data()
    filters = request.args.to_dict()
    filtered_df = apply_global_filters(current_data, filters)

    group_by = filters.get('group_by_treemap', 'Definition')

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
    current_data = get_current_data()
    filters = request.args.to_dict()
    filtered_df = apply_global_filters(current_data, filters)

    selected_definition = filters.get('bar_chart_definition', 'All')
    sort_option = filters.get('bar_chart_sort', 'highest_to_lowest_reimbursement')

    if selected_definition != 'All':
        filtered_df = filtered_df[filtered_df['Medical Definition'] == selected_definition]

    grouped_data = filtered_df.groupby(['Medical Classification', 'Medical Definition']).agg(
        AverageReimbursement=('Reimbursement Rate', 'mean'),
        AverageCoveredCharges=('Average Covered Charges', 'mean'),
        AverageTotalPayments=('Average Total Payments', 'mean')
    ).reset_index()

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
    current_data = get_current_data()
    filters = request.args.to_dict()
    filtered_df = apply_global_filters(current_data, filters)

    state_charges = filtered_df.groupby('Provider State')['Average Covered Charges'].mean().reset_index()
    state_charges.rename(columns={'Provider State': 'State', 'Average Covered Charges': 'AvgCharges'}, inplace=True)

    return jsonify(state_charges.to_dict(orient='records'))

@app.route('/api/scatter_data')
def get_scatter_data():
    """API for 'Average total payments vs. average covered charges' scatter plot."""
    current_data = get_current_data()
    filters = request.args.to_dict()
    filtered_df = apply_global_filters(current_data, filters)

    color_by_classification = filters.get('color_by_classification', 'All')
    show_regression = filters.get('show_regression', 'false') == 'true'

    plot_data = []

    if color_by_classification != 'All':
        filtered_df = filtered_df[filtered_df['Medical Classification'] == color_by_classification]
        classifications_to_plot = [color_by_classification]
    else:
        classifications_to_plot = ['Circulatory System', 'Blood Diseases', 'Alcohol and Drug Use']

    for classification in classifications_to_plot:
        class_df = filtered_df[filtered_df['Medical Classification'] == classification]
        if not class_df.empty:
            plot_data.append({
                'classification': classification,
                'charges': class_df['Average Covered Charges'].tolist(),
                'payments': class_df['Average Total Payments'].tolist()
            })

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
    current_data = get_current_data()
    filters = request.args.to_dict()
    filtered_df = apply_global_filters(current_data, filters)

    state_filter = filters.get('provider_treemap_state', 'All States')
    top_n = int(filters.get('top_n_providers', 10))

    if state_filter != 'All States':
        filtered_df = filtered_df[filtered_df['Provider State'] == state_filter]

    grouped_data = filtered_df.groupby('Provider Name')['Total Discharges'].sum().reset_index()
    grouped_data = grouped_data.sort_values(by='Total Discharges', ascending=False)
    grouped_data = grouped_data.head(top_n)

    grouped_data.rename(columns={'Provider Name': 'Label'}, inplace=True)
    return jsonify(grouped_data.to_dict(orient='records'))

@app.route('/api/histogram_data')
def get_histogram_data():
    """API for 'Distribution of reimbursement rates' histogram."""
    current_data = get_current_data()
    filters = request.args.to_dict()
    filtered_df = apply_global_filters(current_data, filters)

    bin_width = float(filters.get('bin_width', 0.1))
    segment_by_classification = filters.get('segment_by_classification', 'All')

    if segment_by_classification == 'All':
        data_to_plot = {
            'All': filtered_df['Reimbursement Rate'].tolist()
        }
    else:
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
    app.run(debug=True, host='0.0.0.0')
