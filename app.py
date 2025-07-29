# app.py (Flask Backend)
from flask import Flask, jsonify, render_template, request, redirect, url_for, flash, send_file
import pandas as pd
import numpy as np
import json
from flask_cors import CORS
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, classification_report, confusion_matrix
from scipy.stats import f_oneway # For ANOVA
import io # For file handling in memory
import warnings
import os # For file operations
import subprocess # For launching Jupyter Notebook
import sys # To check operating system

# H2O.ai imports
import h2o
from h2o.automl import H2OAutoML

# Suppress specific scikit-learn warnings for cleaner output in a dashboard context
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

app = Flask(__name__)
app.secret_key = 'supersecretkey' # Needed for flash messages
CORS(app)

# Global variable to store uploaded DataFrame
uploaded_df = None

# Global H2O.ai context
h2o_cluster_status = {'is_running': False, 'flow_url': None, 'version': None, 'status_message': 'H2O cluster not initialized.'}
h2o_automl_jobs = {} # To store AutoML job instances

# --- Mock Data Generation (kept for initial demo if no file is uploaded) ---
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
    """Renders the main dashboard HTML page (Analysis Report)."""
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
    """Renders the data engineering page."""
    # This page doesn't directly use current_data for its initial render,
    # but the JS will fetch data as needed.
    return render_template('data_engineering.html', current_page='data_engineering')

@app.route('/machine_learning')
def machine_learning():
    """Renders the Machine Learning page (Scikit-learn models)."""
    current_data = get_current_data()
    classifications = sorted(current_data['Medical Classification'].dropna().unique().tolist())
    definitions = sorted(current_data['Medical Definition'].dropna().unique().tolist())
    provider_states = sorted(current_data['Provider State'].dropna().unique().tolist())
    hospital_regions = sorted(current_data['Hospital Referral Region'].dropna().unique().tolist())

    return render_template('machine_learning.html',
                           classifications=classifications,
                           definitions=definitions,
                           provider_states=provider_states,
                           hospital_regions=hospital_regions,
                           current_page='machine_learning')

@app.route('/h2o_ai_integration')
def h2o_ai_integration():
    """Renders the H2O.ai integration page."""
    # Pass H2O cluster status to the template
    return render_template('h2o_ai_integration.html',
                           h2o_cluster_status=h2o_cluster_status,
                           current_page='h2o_ai_integration') # Updated page name

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

# --- ML Data Preprocessing ---
def preprocess_ml_data(df):
    """
    Preprocesses the DataFrame for ML models:
    - Fills missing numerical values with median.
    - Encodes categorical features using one-hot encoding.
    - Creates 'Cost Per Discharge' feature.
    - Scales numerical features.
    """
    df_processed = df.copy()

    # Fill missing numerical values with median for robustness against outliers.
    for col in ['Total Discharges', 'Average Covered Charges', 'Average Total Payments', 'Reimbursement Rate']:
        if col in df_processed.columns and df_processed[col].isnull().any():
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())

    # Create 'Cost Per Discharge'
    # Handle division by zero if Total Discharges can be 0. Replace with NaN then fill.
    df_processed['Cost Per Discharge'] = df_processed.apply(
        lambda row: row['Average Covered Charges'] / row['Total Discharges'] if row['Total Discharges'] != 0 else np.nan,
        axis=1
    )
    df_processed['Cost Per Discharge'] = df_processed['Cost Per Discharge'].replace([np.inf, -np.inf], np.nan).fillna(df_processed['Cost Per Discharge'].median())


    # Categorical features for encoding
    categorical_features = [
        'DRG Code', 'Medical Definition', 'Medical Classification',
        'Provider State', 'Hospital Referral Region', 'Provider Name'
    ]
    # Filter to only include columns that exist in the DataFrame
    categorical_features = [col for col in categorical_features if col in df_processed.columns]

    # One-hot encode categorical features
    # Handle potential new categories in test set by using handle_unknown='ignore' (for prediction)
    # For training, ensure all categories are present or handle appropriately
    df_processed = pd.get_dummies(df_processed, columns=categorical_features, drop_first=True)

    # Identify numerical features for scaling (exclude one-hot encoded and original categorical)
    numerical_features = [
        'Total Discharges', 'Average Covered Charges', 'Average Total Payments',
        'Reimbursement Rate', 'Cost Per Discharge'
    ]
    numerical_features = [col for col in numerical_features if col in df_processed.columns]

    scaler = StandardScaler()
    if numerical_features:
        df_processed[numerical_features] = scaler.fit_transform(df_processed[numerical_features])

    return df_processed, scaler # Return scaler if needed for inverse transform or new data

# --- Scikit-learn ML Model Endpoints ---

@app.route('/api/ml/predictive_pricing', methods=['POST'])
def run_predictive_pricing():
    data = request.get_json()
    target_variable = data.get('target_variable', 'Average Total Payments')
    n_estimators = int(data.get('n_estimators', 100))
    max_depth = int(data.get('max_depth', 10))

    current_data = get_current_data()
    df_ml, _ = preprocess_ml_data(current_data)

    # Define features (excluding target and highly correlated ones if target is charges/payments)
    features = [col for col in df_ml.columns if col not in [
        target_variable, 'Average Covered Charges', 'Average Total Payments', 'Reimbursement Rate'
    ]]
    # Add back relevant features based on target
    if target_variable == 'Reimbursement Rate':
        features.extend(['Average Covered Charges', 'Average Total Payments'])
    elif target_variable == 'Average Total Payments':
        features.append('Average Covered Charges')


    # Filter features that actually exist in the DataFrame
    X = df_ml[[f for f in features if f in df_ml.columns]]
    y = df_ml[target_variable]

    if X.empty or len(X) < 2:
        return jsonify({'error': 'Insufficient data for training. Please upload more data or adjust filters.'}), 400

    # Handle cases where target variable might have NaN after preprocessing
    combined_df = pd.concat([X, y], axis=1).dropna()
    X = combined_df[X.columns]
    y = combined_df[y.name]

    if X.empty or len(X) < 2:
        return jsonify({'error': 'Insufficient valid data points after handling missing values for prediction.'}), 400


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)

    return jsonify({
        'r2_score': r2,
        'mae': mae,
        'feature_importances': feature_importances.to_dict(),
        'target_variable': target_variable
    })

@app.route('/api/ml/hospital_clustering', methods=['POST'])
def run_hospital_clustering():
    data = request.get_json()
    n_clusters = int(data.get('n_clusters', 3))
    clustering_features = data.get('clustering_features', ['Average Covered Charges', 'Reimbursement Rate', 'Total Discharges'])

    current_data = get_current_data()
    df_ml, _ = preprocess_ml_data(current_data)

    # Ensure selected features exist and handle potential NaNs after preprocessing
    features_for_clustering = [f for f in clustering_features if f in df_ml.columns]
    if not features_for_clustering:
         return jsonify({'error': 'No valid features selected for clustering or features are missing.'}), 400

    X = df_ml[features_for_clustering].dropna() # Drop rows with NaNs in selected features

    if X.empty or len(X) < n_clusters:
        return jsonify({'error': f'Not enough valid data points ({len(X)}) for {n_clusters} clusters after filtering missing values. Reduce cluster count or upload more data.'}), 400

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # n_init for robustness
    df_ml.loc[X.index, 'cluster'] = kmeans.fit_predict(X) # Assign clusters back to original (preprocessed) index

    # Get original (unscaled) data for cluster centroids for better interpretability
    # Ensure 'cluster' column exists before trying to group
    original_data = current_data.copy()
    original_data = original_data.loc[X.index].copy() # Align with filtered X
    original_data['cluster'] = df_ml.loc[X.index, 'cluster'].astype(int) # Ensure integer type for cluster

    cluster_centroids = original_data.groupby('cluster')[clustering_features].mean().to_dict(orient='index')

    # Prepare data for scatter plot
    # Use the original, unscaled values for better visualization interpretation
    # Ensure at least two features are selected for scatter plot
    if len(clustering_features) < 2:
        return jsonify({'error': 'Please select at least two features for clustering visualization.'}), 400

    scatter_data = original_data[[clustering_features[0], clustering_features[1], 'cluster']].to_dict(orient='records')

    return jsonify({
        'centroids': cluster_centroids,
        'scatter_data': scatter_data,
        'x_axis_feature': clustering_features[0],
        'y_axis_feature': clustering_features[1],
        'n_clusters': n_clusters,
        'clustering_features': clustering_features # Pass back for table rendering
    })

@app.route('/api/ml/geographic_analysis', methods=['POST'])
def run_geographic_analysis():
    data = request.get_json()
    group_by_geo = data.get('group_by', 'Provider State')
    target_metric = data.get('metric', 'Average Covered Charges')
    n_geo_clusters = int(data.get('n_clusters', 3))

    current_data = get_current_data()

    if group_by_geo not in current_data.columns:
        return jsonify({'error': f'Grouping feature "{group_by_geo}" not found in data.'}), 400
    if target_metric not in current_data.columns:
        return jsonify({'error': f'Target metric "{target_metric}" not found in data.'}), 400

    # Filter out NaNs in target metric and grouping feature
    df_filtered = current_data.dropna(subset=[group_by_geo, target_metric])

    if df_filtered.empty:
        return jsonify({'error': 'No valid data points for geographic analysis after filtering NaNs.'}), 400

    # ANOVA Test
    unique_groups = df_filtered[group_by_geo].unique()
    if len(unique_groups) < 2:
        return jsonify({'error': f'Not enough unique groups ({group_by_geo}) for ANOVA test.'}), 400

    groups_data = [df_filtered[df_filtered[group_by_geo] == g][target_metric].values for g in unique_groups]
    # Filter out empty arrays from groups_data, as f_oneway requires at least one observation per group
    groups_data = [arr for arr in groups_data if len(arr) > 0]

    if len(groups_data) < 2:
        return jsonify({'error': 'Not enough groups with data for ANOVA test after filtering.'}), 400

    f_statistic, p_value = f_oneway(*groups_data)

    anova_results = {
        'f_statistic': f_statistic,
        'p_value': p_value,
        'conclusion': 'Significant differences exist between groups (p < 0.05).' if p_value < 0.05 else 'No significant differences found between groups (p >= 0.05).',
        'group_by': group_by_geo,
        'target_metric': target_metric
    }

    # Geographic Clustering (K-Means on aggregated data)
    aggregated_geo_data = df_filtered.groupby(group_by_geo).agg(
        AvgCharges=('Average Covered Charges', 'mean'),
        AvgPayments=('Average Total Payments', 'mean'),
        AvgDischarges=('Total Discharges', 'mean'),
        AvgReimbursement=('Reimbursement Rate', 'mean')
    ).reset_index()

    # Features for clustering states/regions
    geo_clustering_features = ['AvgCharges', 'AvgPayments', 'AvgDischarges', 'AvgReimbursement']
    geo_clustering_features = [f for f in geo_clustering_features if f in aggregated_geo_data.columns]

    # Map the original target_metric name to its aggregated version for chart data preparation
    chart_target_metric_agg_name = {
        'Average Covered Charges': 'AvgCharges',
        'Average Total Payments': 'AvgPayments',
        'Reimbursement Rate': 'AvgReimbursement',
        'Total Discharges': 'AvgDischarges'
    }.get(target_metric, target_metric)


    if not geo_clustering_features or len(aggregated_geo_data) < n_geo_clusters:
        return jsonify({
            'anova_results': anova_results,
            'geo_clustering_warning': 'Not enough data points or features for geographic clustering. Skipping clustering visualization.',
            'geo_clusters': {},
            'geo_chart_data': [],
            'group_by_geo': group_by_geo,
            'target_metric': target_metric,
            'chart_target_metric_agg_name': chart_target_metric_agg_name
        })

    # Scale data for clustering
    scaler_geo = StandardScaler()
    scaled_geo_data = scaler_geo.fit_transform(aggregated_geo_data[geo_clustering_features])

    kmeans_geo = KMeans(n_clusters=n_geo_clusters, random_state=42, n_init=10)
    aggregated_geo_data['geo_cluster'] = kmeans_geo.fit_predict(scaled_geo_data)

    geo_cluster_summary = aggregated_geo_data.groupby('geo_cluster')[geo_clustering_features].mean().to_dict(orient='index')

    # Prepare data for map/bar chart visualization
    geo_chart_data = aggregated_geo_data[[group_by_geo, chart_target_metric_agg_name, 'geo_cluster']].to_dict(orient='records')


    return jsonify({
        'anova_results': anova_results,
        'geo_clusters': geo_cluster_summary,
        'geo_chart_data': geo_chart_data,
        'group_by_geo': group_by_geo,
        'target_metric': target_metric,
        'chart_target_metric_agg_name': chart_target_metric_agg_name
    })


@app.route('/api/ml/classification_optimization', methods=['POST'])
def run_classification_optimization():
    data = request.get_json()
    target_threshold = int(data.get('target_threshold', 100))
    model_type = data.get('model_type', 'Decision Tree')

    current_data = get_current_data()

    if 'Medical Definition' not in current_data.columns or 'Total Discharges' not in current_data.columns:
        return jsonify({'error': 'Required columns "Medical Definition" or "Total Discharges" missing for classification target.'}), 400

    df_with_target = current_data.copy()
    df_with_target['Medical Definition'] = df_with_target['Medical Definition'].astype(str)

    df_with_target['is_high_volume'] = df_with_target.groupby('Medical Definition')['Total Discharges'].transform(lambda x: (x.sum() > target_threshold).astype(int))

    df_ml_final, _ = preprocess_ml_data(df_with_target)

    if 'is_high_volume' not in df_ml_final.columns:
        return jsonify({'error': 'Failed to create "is_high_volume" target column after preprocessing.'}), 500

    df_ml_final.dropna(subset=['is_high_volume'], inplace=True)

    if df_ml_final.empty:
        return jsonify({'error': 'No data available after defining high/low volume target and filtering NaNs. Adjust threshold or upload more data.'}), 400

    features = [col for col in df_ml_final.columns if col not in [
        'is_high_volume', 'Total Discharges', 'Average Covered Charges', 'Average Total Payments', 'Reimbursement Rate'
    ] and not col.startswith(('DRG Code_', 'Medical Definition_', 'Provider Name_'))]

    X = df_ml_final[[f for f in features if f in df_ml_final.columns]]
    y = df_ml_final['is_high_volume']

    if len(y.unique()) < 2:
        return jsonify({'error': 'Only one class (high/low volume) found in the target variable. Adjust threshold or provide more diverse data.'}), 400

    if X.empty or len(X) < 2:
        return jsonify({'error': 'Insufficient data for classification. Please upload more data or adjust filters.'}), 400

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = None
    feature_importances = {}
    coefficients = {}

    if model_type == 'Decision Tree':
        model = DecisionTreeClassifier(max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10).to_dict()
    elif model_type == 'Logistic Regression':
        model = LogisticRegression(random_state=42, solver='liblinear', max_iter=1000)
        model.fit(X_train, y_train)
        if hasattr(model, 'coef_') and len(model.coef_[0]) == len(X.columns): # Check if coef_ exists and matches feature count
            coefficients = pd.Series(model.coef_[0], index=X.columns).sort_values(key=abs, ascending=False).head(10).to_dict()
    elif model_type == 'SVM':
        # Using LinearSVC for interpretability of coefficients, suitable for binary classification
        # For non-linear SVMs (SVC with rbf, poly kernels), coefficients are not directly interpretable
        # Added max_iter for convergence
        model = LinearSVC(random_state=42, dual=False, max_iter=2000) # dual=False for n_samples > n_features
        try:
            model.fit(X_train, y_train)
            if hasattr(model, 'coef_') and len(model.coef_[0]) == len(X.columns):
                coefficients = pd.Series(model.coef_[0], index=X.columns).sort_values(key=abs, ascending=False).head(10).to_dict()
            else:
                coefficients = {"Warning": "Coefficients not directly available for this SVM configuration or data."}
        except Exception as e:
            return jsonify({'error': f'SVM model training failed: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Invalid model type selected.'}), 400

    y_pred = model.predict(X_test)

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1]) # Ensure order of labels for consistency
    tn, fp, fn, tp = cm.ravel() # Unpack confusion matrix values

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    for key, value in report.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (np.float32, np.float64)):
                    report[key][sub_key] = float(sub_value)
        elif isinstance(value, (np.float32, np.float64)):
            report[key] = float(value)

    return jsonify({
        'classification_report': report,
        'feature_importances': feature_importances,
        'coefficients': coefficients,
        'target_threshold': target_threshold,
        'model_type': model_type,
        'confusion_matrix_data': {'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)} # Return CM data
    })

@app.route('/api/ml/anomaly_detection', methods=['POST'])
def run_anomaly_detection():
    data = request.get_json()
    contamination = float(data.get('contamination', 0.05))

    current_data = get_current_data()
    df_ml_temp, _ = preprocess_ml_data(current_data)

    features = ['Average Covered Charges', 'Average Total Payments', 'Reimbursement Rate', 'Total Discharges', 'Cost Per Discharge']
    features = [f for f in features if f in df_ml_temp.columns]

    if not features:
        return jsonify({'error': 'No valid features selected for anomaly detection.'}), 400

    X = df_ml_temp[features].dropna()

    if X.empty or len(X) < 2:
        return jsonify({'error': 'Insufficient data for anomaly detection after filtering missing values. Adjust filters or upload more data.'}), 400

    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(X)

    original_indices = X.index
    df_ml_temp.loc[original_indices, 'anomaly_score'] = model.decision_function(X)
    df_ml_temp.loc[original_indices, 'is_anomaly'] = model.predict(X)

    original_data_for_anom = current_data.loc[original_indices].copy()
    original_data_for_anom['is_anomaly'] = df_ml_temp.loc[original_indices, 'is_anomaly']
    original_data_for_anom['anomaly_score'] = df_ml_temp.loc[original_indices, 'anomaly_score']

    anomalies = original_data_for_anom[original_data_for_anom['is_anomaly'] == -1].to_dict(orient='records')
    inliers = original_data_for_anom[original_data_for_anom['is_anomaly'] == 1].to_dict(orient='records')

    return jsonify({
        'anomalies': anomalies,
        'inliers': inliers,
        'contamination': contamination,
        'features_used': features
    })


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
        # Only plot specific classifications if 'All' is selected, to avoid too many traces
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
                # Ensure X is 2D for sklearn
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

# --- Data Engineering Endpoints ---

@app.route('/api/data_characteristics/missing_values')
def get_missing_values():
    """Returns missing value counts and percentages per column."""
    df = get_current_data()
    missing_data = df.isnull().sum()
    total_rows = df.shape[0]
    missing_info = []
    for col, count in missing_data.items():
        percentage = (count / total_rows) * 100 if total_rows > 0 else 0
        missing_info.append({
            'column': col,
            'missing_count': int(count),
            'missing_percentage': round(percentage, 2)
        })
    return jsonify(missing_info)

@app.route('/api/data_characteristics/outlier_boxplots')
def get_outlier_boxplots():
    """Returns data for box plots of numerical columns."""
    df = get_current_data()
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    plot_data = []
    for col in numerical_cols:
        if not df[col].empty:
            plot_data.append({
                'name': col,
                'y': df[col].dropna().tolist() # Send raw data for Plotly to render box plot
            })
    return jsonify(plot_data)

@app.route('/api/data_characteristics/distribution_histograms')
def get_distribution_histograms():
    """Returns data for histograms of key numerical distributions."""
    df = get_current_data()
    target_cols = ['Average Covered Charges', 'Average Total Payments', 'Total Discharges', 'Reimbursement Rate']
    plot_data = []
    for col in target_cols:
        if col in df.columns and not df[col].empty:
            plot_data.append({
                'name': col,
                'x': df[col].dropna().tolist() # Send raw data for Plotly to bin
            })
    return jsonify(plot_data)

@app.route('/api/data_characteristics/correlation_matrix')
def get_correlation_matrix():
    """Returns the correlation matrix for numerical features."""
    df = get_current_data()
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numerical_cols:
        return jsonify({'error': 'No numerical columns found for correlation matrix.'}), 400
    
    # Filter out columns with all NaNs or constant values which can cause issues with corr()
    valid_cols = [col for col in numerical_cols if df[col].nunique() > 1 and not df[col].isnull().all()]
    
    if not valid_cols or len(valid_cols) < 2:
        return jsonify({'error': 'Not enough varying numerical data for correlation matrix.'}), 400

    corr_matrix = df[valid_cols].corr().fillna(0) # Fill NaN from correlation (e.g., if a column was constant)
    
    return jsonify({
        'z': corr_matrix.values.tolist(),
        'x': corr_matrix.columns.tolist(),
        'y': corr_matrix.index.tolist()
    })

@app.route('/api/data_characteristics/geographic_maps')
def get_geographic_maps():
    """Returns data for geographic distribution maps."""
    df = get_current_data()

    # Provider Distribution (Choropleth)
    if 'Provider State' in df.columns:
        provider_counts = df.groupby('Provider State').size().reset_index(name='count')
        provider_choropleth_data = provider_counts.to_dict(orient='records')
    else:
        provider_choropleth_data = []

    # Average Costs by Region (Choropleth/Heatmap)
    if 'Hospital Referral Region' in df.columns and 'Average Covered Charges' in df.columns:
        region_avg_costs = df.groupby('Hospital Referral Region')['Average Covered Charges'].mean().reset_index()
        region_avg_costs_data = region_avg_costs.to_dict(orient='records')
    else:
        region_avg_costs_data = []

    return jsonify({
        'provider_choropleth': provider_choropleth_data,
        'region_avg_costs': region_avg_costs_data
    })

@app.route('/api/data_characteristics/medical_classification_analysis')
def get_medical_classification_analysis():
    """Returns data for medical classification analysis charts."""
    df = get_current_data()

    # Procedure Count by Medical Classification (Bar Chart)
    if 'Medical Classification' in df.columns and 'Total Discharges' in df.columns:
        class_discharges = df.groupby('Medical Classification')['Total Discharges'].sum().reset_index()
        class_discharges_data = class_discharges.to_dict(orient='records')
    else:
        class_discharges_data = []

    # Cost Distribution by Medical Classification (Box Plot)
    if 'Medical Classification' in df.columns and 'Average Covered Charges' in df.columns:
        class_cost_data = df[['Medical Classification', 'Average Covered Charges']].dropna().to_dict(orient='records')
    else:
        class_cost_data = []

    return jsonify({
        'class_discharges': class_discharges_data,
        'class_cost_distribution': class_cost_data
    })

@app.route('/api/data_characteristics/cost_structure_analysis')
def get_cost_structure_analysis():
    """Returns data for cost structure analysis charts."""
    df = get_current_data()

    # Scatter Plot: Charges vs. Payments
    if 'Average Covered Charges' in df.columns and 'Average Total Payments' in df.columns:
        charges_payments_scatter = df[['Average Covered Charges', 'Average Total Payments']].dropna().to_dict(orient='records')
    else:
        charges_payments_scatter = []

    # Violin Plot: Cost distribution by state
    if 'Provider State' in df.columns and 'Average Covered Charges' in df.columns:
        state_cost_violin = df[['Provider State', 'Average Covered Charges']].dropna().to_dict(orient='records')
    else:
        state_cost_violin = []

    # Bubble Chart: Volume vs. Average Cost (bubble size = reimbursement rate)
    if all(col in df.columns for col in ['Medical Definition', 'Total Discharges', 'Average Covered Charges', 'Reimbursement Rate']):
        bubble_data_agg = df.groupby('Medical Definition').agg(
            AvgCharges=('Average Covered Charges', 'mean'),
            TotalDischarges=('Total Discharges', 'sum'),
            AvgReimbursement=('Reimbursement Rate', 'mean')
        ).reset_index().dropna()
        bubble_chart_data = bubble_data_agg.to_dict(orient='records')
    else:
        bubble_chart_data = []

    return jsonify({
        'charges_payments_scatter': charges_payments_scatter,
        'state_cost_violin': state_cost_violin,
        'bubble_chart': bubble_chart_data
    })

@app.route('/api/data_characteristics/feature_distribution_analysis')
def get_feature_distribution_analysis():
    """Returns data for feature distribution analysis (log-histograms, density plots)."""
    df = get_current_data()
    numerical_cols = ['Average Covered Charges', 'Average Total Payments', 'Total Discharges', 'Reimbursement Rate']
    plot_data = []

    for col in numerical_cols:
        if col in df.columns and not df[col].empty:
            # Log-scale histogram
            log_values = np.log1p(df[col].dropna()).tolist() # log1p handles zero/negative values
            plot_data.append({
                'feature': col,
                'type': 'log_histogram',
                'x': log_values
            })

            # Density plot (using histogram and KDE approximation for simplicity)
            # For a true KDE, a more complex library or server-side calculation would be needed.
            # Here, we approximate by sending raw data for Plotly to render a histogram,
            # and potentially overlay a smoothed line in JS if needed.
            plot_data.append({
                'feature': col,
                'type': 'density_plot',
                'x': df[col].dropna().tolist()
            })
    return jsonify(plot_data)

@app.route('/api/data_characteristics/categorical_feature_analysis')
def get_categorical_feature_analysis():
    """Returns data for categorical feature analysis."""
    df = get_current_data()
    categorical_cols = ['Medical Classification', 'Provider State', 'Hospital Referral Region']
    plot_data = []

    # Count plots
    for col in categorical_cols:
        if col in df.columns and not df[col].empty:
            counts = df[col].value_counts().reset_index()
            counts.columns = ['category', 'count']
            plot_data.append({
                'feature': col,
                'type': 'count_plot',
                'data': counts.to_dict(orient='records')
            })

    # Cross-tabulation heatmaps (example: Classification vs. State)
    if 'Medical Classification' in df.columns and 'Provider State' in df.columns:
        crosstab = pd.crosstab(df['Medical Classification'], df['Provider State'])
        plot_data.append({
            'feature_x': 'Medical Classification',
            'feature_y': 'Provider State',
            'type': 'crosstab_heatmap',
            'z': crosstab.values.tolist(),
            'x': crosstab.columns.tolist(),
            'y': crosstab.index.tolist()
        })
    
    # Proportion stacked bar charts (example: Medical Classification by State)
    if 'Medical Classification' in df.columns and 'Provider State' in df.columns:
        # Calculate proportions of each classification within each state
        state_class_counts = df.groupby(['Provider State', 'Medical Classification']).size().unstack(fill_value=0)
        state_class_proportions = state_class_counts.div(state_class_counts.sum(axis=1), axis=0).fillna(0)

        plot_data.append({
            'type': 'stacked_bar_proportions',
            'x_axis': 'Provider State',
            'y_axis': 'Proportion',
            'data': state_class_proportions.reset_index().to_dict(orient='records'),
            'columns': state_class_proportions.columns.tolist() # The medical classifications
        })

    return jsonify(plot_data)

# --- H2O.ai Endpoints ---

# Comprehensive metrics extraction function
def get_model_metrics(model):
    metrics = {}
    try:
        # Get default model performance object
        perf = model.model_performance()

        # Common metrics for most models
        metric_methods = ['auc', 'rmse', 'mae', 'mean_residual_deviance', 'logloss', 'gini', 'r2']

        for metric_name in metric_methods:
            try:
                # Check if the performance object has the method
                if hasattr(perf, metric_name):
                    value = getattr(perf, metric_name)()
                    # Handle different return types (e.g., auc() returns a float, gini() might return a list)
                    if isinstance(value, list) and len(value) > 0:
                        # For cases like Gini, which might return [value, threshold]
                        metrics[metric_name] = value[0][0] if isinstance(value[0], list) else value[0]
                    else:
                        metrics[metric_name] = value
            except Exception as e:
                # print(f"Warning: Could not get metric '{metric_name}' for model {model.model_id}: {e}")
                continue

        # Get additional model info
        try:
            metrics['model_type'] = model._model_json['algo']
            metrics['model_category'] = model._model_json.get('model_category', 'Unknown')
        except Exception as e:
            # print(f"Warning: Could not get model_type/category for model {model.model_id}: {e}")
            metrics['model_type'] = 'Unknown'
            metrics['model_category'] = 'Unknown'

        # Specific metrics for classification models
        if model.model_category == 'Binomial' or model.model_category == 'Multinomial':
            try:
                cm = perf.confusion_matrix().as_data_frame().to_dict(orient='records')
                metrics['confusion_matrix'] = cm
            except Exception as e:
                # print(f"Warning: Could not get confusion matrix for model {model.model_id}: {e}")
                pass

            try:
                # For ROC curve, extract FPR and TPR points
                roc_curve_df = perf.roc_curve().as_data_frame()
                metrics['roc_curve'] = {
                    'fpr': roc_curve_df['fpr'].tolist(),
                    'tpr': roc_curve_df['tpr'].tolist(),
                    'thresholds': roc_curve_df['thresholds'].tolist()
                }
            except Exception as e:
                # print(f"Warning: Could not get ROC curve for model {model.model_id}: {e}")
                pass

    except Exception as e:
        metrics['error'] = f'Could not retrieve metrics for model {model.model_id}: {str(e)}'
        return metrics
    
    return metrics


@app.route('/api/h2o/init', methods=['POST'])
def h2o_init_cluster():
    global h2o_cluster_status
    if h2o_cluster_status['is_running']:
        return jsonify({'status': 'already_running', 'message': 'H2O cluster is already running.', **h2o_cluster_status})

    try:
        # Attempt to connect to an existing cluster first, or start a new one
        h2o.init(strict_version_check=False, port=54321) # Use a fixed port for consistency

        # Get cluster info and construct URL
        cluster_info = h2o.cluster().show_status()
        # Extract IP and port from cluster info or use default
        flow_url = "http://localhost:54321"  # Default H2O flow URL

        h2o_cluster_status['is_running'] = True
        h2o_cluster_status['flow_url'] = flow_url
        # Get version more safely
        try:
            h2o_cluster_status['version'] = h2o.cluster().version
        except:
            h2o_cluster_status['version'] = "Unknown"
        h2o_cluster_status['status_message'] = 'H2O cluster initialized successfully.'
        return jsonify({'status': 'success', 'message': 'H2O cluster initialized successfully.', **h2o_cluster_status})
    except Exception as e:
        h2o_cluster_status['is_running'] = False
        h2o_cluster_status['flow_url'] = None
        h2o_cluster_status['version'] = None
        h2o_cluster_status['status_message'] = f'Failed to initialize H2O cluster: {str(e)}'
        return jsonify({'status': 'error', 'message': f'Failed to initialize H2O cluster: {str(e)}'}), 500

@app.route('/api/h2o/status', methods=['GET'])
def h2o_get_status():
    global h2o_cluster_status
    # Check if H2O is actually connected
    try:
        if h2o_cluster_status['is_running']:
            # Ping the cluster to ensure it's still alive
            cloud_status = h2o.cluster().get_cloud_status()
            h2o_cluster_status['flow_url'] = cloud_status.get('cloud_url') # Update flow_url on status check
            h2o_cluster_status['status_message'] = 'H2O cluster is running.'
        else:
            h2o_cluster_status['status_message'] = 'H2O cluster not initialized.'
    except Exception:
        h2o_cluster_status['is_running'] = False
        h2o_cluster_status['flow_url'] = None
        h2o_cluster_status['version'] = None
        h2o_cluster_status['status_message'] = 'H2O cluster connection lost or not running.'
    return jsonify(h2o_cluster_status)

@app.route('/api/h2o/upload_data', methods=['POST'])
def h2o_upload_data():
    if not h2o_cluster_status['is_running']:
        return jsonify({'error': 'H2O cluster is not running. Please initialize it first.'}), 400

    df = get_current_data()
    if df.empty:
        return jsonify({'error': 'No data loaded in the dashboard to upload to H2O. Please upload a file via Data Access first.'}), 400

    try:
        # Generate a unique frame ID
        frame_id = f"uploaded_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        # Corrected: Use destination_frame instead of frame_id for H2OFrame constructor
        h2o_frame = h2o.H2OFrame(df, destination_frame=frame_id)
        return jsonify({'status': 'success', 'message': f'Data uploaded to H2O as frame: {frame_id}', 'frame_id': frame_id})
    except Exception as e:
        return jsonify({'error': f'Failed to upload data to H2O: {str(e)}'}), 500

@app.route('/api/h2o/list_frames', methods=['GET'])
def h2o_list_frames():
    if not h2o_cluster_status['is_running']:
        return jsonify({'error': 'H2O cluster is not running.'}), 400
    try:
        frames = h2o.ls()
        frame_list = []
        for f in frames['key']:
            try:
                # Attempt to get frame details, skip if it's not a valid frame
                frame_obj = h2o.get_frame(f)
                frame_list.append({
                    'frame_id': f,
                    'rows': frame_obj.nrows,
                    'cols': frame_obj.ncols,
                    'columns': frame_obj.col_names # Include column names for UI selection
                })
            except Exception:
                # Skip keys that are not valid H2OFrame IDs (e.g., models, internal keys)
                continue
        return jsonify({'status': 'success', 'frames': frame_list})
    except Exception as e:
        return jsonify({'error': f'Failed to list H2O frames: {str(e)}'}), 500

@app.route('/api/h2o/run_automl', methods=['POST'])
def h2o_run_automl():
    if not h2o_cluster_status['is_running']:
        return jsonify({'error': 'H2O cluster is not running. Please initialize it first.'}), 400

    data = request.get_json()
    frame_id = data.get('frame_id')
    target_column = data.get('target_column')
    ignored_columns = data.get('ignored_columns', [])
    max_runtime_secs = int(data.get('max_runtime_secs', 300))
    max_models = int(data.get('max_models', 10))
    sort_metric = data.get('sort_metric', 'AUTO')
    is_classification = data.get('is_classification', True) # Assume classification by default

    if not frame_id or not target_column:
        return jsonify({'error': 'Missing H2O frame ID or target column.'}), 400

    try:
        hf = h2o.get_frame(frame_id)
        x = [col for col in hf.col_names if col != target_column and col not in ignored_columns]
        y = target_column

        # Ensure target column is categorical for classification
        if is_classification:
            if hf[y].isfactor()[0] is False: # Check if it's not already a factor
                hf[y] = hf[y].asfactor()
        else: # Regression
            if hf[y].isfactor()[0] is True:
                hf[y] = hf[y].asnumeric()

        # Split data for training and validation (optional, AutoML can do internal validation)
        # For simplicity, let AutoML handle internal validation for now.
        # train, valid = hf.split_frame(ratios=[0.8], seed=42)

        aml = H2OAutoML(
            max_runtime_secs=max_runtime_secs,
            max_models=max_models,
            seed=42,
            nfolds=5, # Use cross-validation
            sort_metric=sort_metric,
            exclude_algos=["DeepLearning"] # Exclude DeepLearning for faster runs in demo
        )

        # Generate a unique job ID for tracking
        job_id = f"automl_job_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        h2o_automl_jobs[job_id] = {'aml_instance': aml, 'status': 'running', 'start_time': pd.Timestamp.now(), 'leaderboard': None}

        # Run AutoML in a separate thread/process if possible for non-blocking UI
        # For Flask's built-in server, this is tricky without external libraries like Celery.
        # For now, we'll just kick it off and rely on frontend polling.
        aml.train(x=x, y=y, training_frame=hf) # Blocking call for now

        # Once train completes (or if it's non-blocking and we check later)
        h2o_automl_jobs[job_id]['status'] = 'completed'
        h2o_automl_jobs[job_id]['leaderboard'] = aml.leaderboard.as_data_frame().to_dict(orient='records')

        return jsonify({'status': 'started', 'message': 'AutoML job started.', 'job_id': job_id})
    except Exception as e:
        return jsonify({'error': f'Failed to start AutoML job: {str(e)}'}), 500

@app.route('/api/h2o/get_automl_status/<job_id>', methods=['GET'])
def h2o_get_automl_status(job_id):
    job_info = h2o_automl_jobs.get(job_id)
    if not job_info:
        return jsonify({'error': 'AutoML job not found.'}), 404

    aml = job_info['aml_instance']
    status = job_info['status']
    leaderboard = job_info['leaderboard']

    if status == 'running':
        # H2O's AutoML instance doesn't expose a direct progress percentage easily
        # We can infer progress by checking the number of models trained vs max_models
        # or by checking if leader exists.
        progress_message = "Training in progress..."
        if aml.leader:
            progress_message = f"Training in progress. Current leader: {aml.leader.model_id}"
        
        return jsonify({
            'status': status,
            'message': progress_message,
            'leader_model_id': aml.leader.model_id if aml.leader else None,
            'leaderboard_available': False # Leaderboard is only available after completion for simplicity
        })
    elif status == 'completed':
        return jsonify({
            'status': status,
            'message': 'AutoML job completed.',
            'leader_model_id': aml.leader.model_id if aml.leader else None,
            'leaderboard_available': True,
            'leaderboard': leaderboard
        })
    else:
        return jsonify({'status': status, 'message': 'Unknown job status.'})

@app.route('/api/h2o/get_model_details/<model_id>', methods=['GET'])
def h2o_get_model_details(model_id):
    if not h2o_cluster_status['is_running']:
        return jsonify({'error': 'H2O cluster is not running.'}), 400
    try:
        model = h2o.get_model(model_id)
        
        # Use the comprehensive metrics extraction function
        metrics_data = get_model_metrics(model)

        var_importances = None
        if hasattr(model, 'varimp') and model.varimp:
            var_importances = model.varimp.as_data_frame().to_dict(orient='records')
        
        return jsonify({
            'status': 'success',
            'model_id': model_id,
            'algo': metrics_data.get('model_type', model.algo), # Use extracted type, fallback to model.algo
            'model_category': metrics_data.get('model_category', model.model_category), # Use extracted category, fallback
            'metrics': metrics_data, # Send all extracted metrics
            'var_importances': var_importances,
            # roc_curve_data is now part of 'metrics' if available
        })
    except Exception as e:
        return jsonify({'error': f'Failed to get model details for {model_id}: {str(e)}'}), 500

@app.route('/api/h2o/download_mojo/<model_id>', methods=['GET'])
def h2o_download_mojo(model_id):
    if not h2o_cluster_status['is_running']:
        return jsonify({'error': 'H2O cluster is not running.'}), 400
    try:
        model = h2o.get_model(model_id)
        # Create a temporary directory for MOJO files
        mojo_dir = 'h2o_mojos'
        os.makedirs(mojo_dir, exist_ok=True)
        mojo_path = h2o.download_mojo(model, path=mojo_dir)
        
        return send_file(mojo_path, as_attachment=True, download_name=f"{model_id}.zip")
    except Exception as e:
        return jsonify({'error': f'Failed to download MOJO for {model_id}: {str(e)}'}), 500

@app.route('/api/h2o/predict', methods=['POST'])
def h2o_predict():
    if not h2o_cluster_status['is_running']:
        return jsonify({'error': 'H2o cluster is not running.'}), 400
    
    data = request.get_json()
    model_id = data.get('model_id')
    input_data = data.get('input_data') # Expecting a list of dictionaries for new data

    if not model_id or not input_data:
        return jsonify({'error': 'Missing model ID or input data for prediction.'}), 400

    try:
        model = h2o.get_model(model_id)
        # Convert input_data (list of dicts) to pandas DataFrame, then to H2OFrame
        input_df = pd.DataFrame(input_data)
        input_hf = h2o.H2OFrame(input_df)

        predictions = model.predict(input_hf).as_data_frame().to_dict(orient='records')
        return jsonify({'status': 'success', 'predictions': predictions})
    except Exception as e:
        return jsonify({'error': f'Failed to make prediction with model {model_id}: {str(e)}'}), 500

@app.route('/api/launch_jupyter', methods=['POST'])
def launch_jupyter():
    """
    Launches a Jupyter Notebook server.
    This assumes Jupyter is installed and accessible in the environment.
    """
    try:
        command = ['jupyter', 'notebook', '--no-browser', '--port=8888', '--allow-root']
        
        if sys.platform == "win32":
            # For Windows, use creationflags to detach the process
            # and avoid the console window popping up.
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            subprocess.Popen(command,
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL,
                             stdin=subprocess.DEVNULL,
                             close_fds=True, # Close file descriptors on child process exit
                             creationflags=subprocess.DETACHED_PROCESS,
                             startupinfo=startupinfo)
        else:
            # For Unix-like systems, use preexec_fn=os.setsid to detach
            subprocess.Popen(command,
                             stdout=subprocess.DEVNULL, # Suppress stdout
                             stderr=subprocess.DEVNULL, # Suppress stderr
                             preexec_fn=os.setsid) # Detach from parent process on Unix-like systems

        return jsonify({'status': 'success', 'message': 'Jupyter Notebook launched. Access it at http://localhost:8888', 'jupyter_url': 'http://localhost:8888'}), 200
    except FileNotFoundError:
        return jsonify({'error': 'Jupyter Notebook command not found. Please ensure Jupyter is installed and in your system PATH.'}), 500
    except Exception as e:
        return jsonify({'error': f'Failed to launch Jupyter Notebook: {str(e)}'}), 500


# --- Global Error Handlers for API routes ---
@app.errorhandler(404)
def not_found(error):
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Not Found', 'message': 'The requested API endpoint does not exist.'}), 404
    # Fallback for non-API routes or if no specific template exists
    return render_template('404.html'), 404 

@app.errorhandler(500)
def internal_server_error(error):
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Internal Server Error', 'message': str(error)}), 500
    # Fallback for non-API routes or if no specific template exists
    return render_template('500.html'), 500 


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

