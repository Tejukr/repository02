# app.py (updated)
from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file, g
from functools import wraps
import pickle
import pandas as pd
import os
import sqlite3
import json
import datetime
import traceback
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -------------------- CONFIG & CONSTANTS --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_FOLDER = os.path.join(BASE_DIR, 'static')
TEMPLATE_FOLDER = os.path.join(BASE_DIR, 'templates')

app = Flask(__name__, static_folder=STATIC_FOLDER, template_folder=TEMPLATE_FOLDER)
app.secret_key = os.environ.get('FLASK_SECRET', 'dev-secret-key')

MODEL_FILE = os.path.join(BASE_DIR, 'models', 'model.pkl')
# Try multiple possible feature file names for compatibility with older runs
FEATURES_FILE_CANDIDATES = [
    os.path.join(BASE_DIR, 'models', 'model_features.pkl'),
    os.path.join(BASE_DIR, 'models', 'model_features'),
    os.path.join(BASE_DIR, 'models', 'feature_columns.pkl'),
    os.path.join(BASE_DIR, 'models', 'feature_columns')
]
DATA_CSV = os.path.join(BASE_DIR, 'data', 'house_data.csv')
DB_FILE = os.path.join(BASE_DIR, 'data', 'app.db')
EXPORT_DIR = os.path.join(BASE_DIR, 'data', 'exports')
os.makedirs(EXPORT_DIR, exist_ok=True)

# -------------------- GLOBALS FOR MODEL --------------------
model_artifacts = None
feature_columns = None

# -------------------- CONTEXT PROCESSOR (makes links available in templates) --------------------
@app.context_processor
def inject_helpers():
    """
    Make a few helpful items available in all templates:
     - links: dict of commonly used endpoints (or None when endpoint missing)
     - now: function returning utcnow()
     - current_year: year integer
    """
    def safe_url(endpoint_name, **values):
        try:
            return url_for(endpoint_name, **values)
        except Exception:
            return None

    links = {
        'home': safe_url('home'),
        'dashboard': safe_url('dashboard'),
        'predict': safe_url('predict'),
        'visuals': safe_url('visuals'),
        'history': safe_url('history'),
        'model_performance': safe_url('model_performance'),
        'admin': safe_url('admin'),
        'login': safe_url('login'),
        'signup': safe_url('signup'),
    }

    return {
        'links': links,
        'now': datetime.datetime.utcnow,
        'current_year': datetime.datetime.utcnow().year
    }

# -------------------- MODEL LOADER --------------------
def get_model():
    """
    Lazy-load model artifacts (model, imputer, scaler, log_target flag) and feature columns.
    Returns tuple: (model, feature_columns, imputer, scaler, log_target_bool)
    """
    global model_artifacts, feature_columns
    if model_artifacts is not None and feature_columns is not None:
        return (
            model_artifacts.get('model'),
            feature_columns,
            model_artifacts.get('imputer'),
            model_artifacts.get('scaler'),
            bool(model_artifacts.get('log_target'))
        )

    # load model
    if os.path.exists(MODEL_FILE):
        try:
            with open(MODEL_FILE, 'rb') as f:
                loaded = pickle.load(f)
                # allow both saving approaches: either raw model or dict with model+imputer+scaler
                if isinstance(loaded, dict) and 'model' in loaded:
                    model_artifacts = {
                        'model': loaded.get('model'),
                        'imputer': loaded.get('imputer'),
                        'scaler': loaded.get('scaler'),
                        'log_target': loaded.get('log_target', False)
                    }
                else:
                    model_artifacts = {
                        'model': loaded,
                        'imputer': None,
                        'scaler': None,
                        'log_target': False
                    }
            print(f"Loaded model from {MODEL_FILE}")
        except Exception as e:
            print("Error loading model:", e)
            traceback.print_exc()
            model_artifacts = None
    else:
        print("Model file not found at:", MODEL_FILE)
        model_artifacts = None

    # load features - try several candidate filenames
    feature_columns = []
    for candidate in FEATURES_FILE_CANDIDATES:
        if os.path.exists(candidate):
            try:
                with open(candidate, 'rb') as f:
                    feature_columns = pickle.load(f)
                    # if saved as dict or other container, try to extract list
                    if isinstance(feature_columns, dict) and 'features' in feature_columns:
                        feature_columns = feature_columns['features']
                print("Loaded feature columns from:", candidate)
                break
            except Exception as e:
                print("Error loading feature file", candidate, e)
                traceback.print_exc()
                feature_columns = []

    if not feature_columns:
        # fallback: if model has attribute `.feature_names_in_`, use it
        try:
            if model_artifacts and model_artifacts.get('model') is not None and hasattr(model_artifacts.get('model'), 'feature_names_in_'):
                feature_columns = list(model_artifacts.get('model').feature_names_in_)
                print("Using model.feature_names_in_ for features")
        except Exception:
            pass

    if feature_columns is None:
        feature_columns = []

    if model_artifacts is None:
        return None, feature_columns, None, None, False

    return (
        model_artifacts.get('model'),
        feature_columns,
        model_artifacts.get('imputer'),
        model_artifacts.get('scaler'),
        bool(model_artifacts.get('log_target'))
    )

# -------------------- FEATURE ENGINEERING HELPER --------------------
def engineer_features(df, current_year=None):
    """
    Apply the same feature engineering as in train_model.py
    """
    if current_year is None:
        current_year = datetime.datetime.now().year
    
    df = df.copy()
    
    # Calculate house age
    if "Built Year" in df.columns:
        df["house_age"] = current_year - df["Built Year"].fillna(current_year)
        df["house_age"] = df["house_age"].clip(lower=0, upper=200)
    
    # Renovation status
    if "Renovation Year" in df.columns:
        df["is_renovated"] = (df["Renovation Year"] > 0).astype(int)
        df["years_since_renovation"] = current_year - df["Renovation Year"].fillna(0)
        df["years_since_renovation"] = df["years_since_renovation"].clip(lower=0, upper=200)
    else:
        df["is_renovated"] = 0
        df["years_since_renovation"] = 0
    
    # Area ratios and derived features
    if "living area" in df.columns and "lot area" in df.columns:
        df["living_to_lot_ratio"] = df["living area"] / (df["lot area"] + 1)
        df["total_area"] = df["living area"] + df.get("Area of the basement", 0).fillna(0)
    
    if "living area" in df.columns and "number of bedrooms" in df.columns:
        df["area_per_bedroom"] = df["living area"] / (df["number of bedrooms"] + 1)
    
    if "living area" in df.columns and "number of bathrooms" in df.columns:
        df["area_per_bathroom"] = df["living area"] / (df["number of bathrooms"] + 1)
    
    # Basement ratio
    if "Area of the basement" in df.columns and "living area" in df.columns:
        df["basement_ratio"] = df["Area of the basement"] / (df["living area"] + df["Area of the basement"] + 1)
    
    return df

# -------------------- DATABASE HELPERS --------------------
def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def ensure_users_schema(conn):
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()

    # read existing columns
    c.execute("PRAGMA table_info(users)")
    cols = [r['name'] for r in c.fetchall()]

    if 'role' not in cols:
        try:
            c.execute("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'user'")
            conn.commit()
        except Exception as e:
            print('Could not add role column:', e)

    if 'last_login' not in cols:
        try:
            c.execute("ALTER TABLE users ADD COLUMN last_login DATETIME")
            conn.commit()
        except Exception as e:
            print('Could not add last_login column:', e)

    # ensure default admin exists
    try:
        c.execute("SELECT id FROM users WHERE username = ?", ('admin',))
        if c.fetchone() is None:
            c.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", ('admin', 'admin', 'admin'))
            conn.commit()
            print('Inserted default admin user (admin/admin)')
    except Exception as e:
        print('Error ensuring default admin:', e)

def ensure_predictions_schema(conn):
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            input_data TEXT,
            predicted_price REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()

    # ensure columns exist (PRAGMA shows them)
    c.execute("PRAGMA table_info(predictions)")
    cols = [r['name'] for r in c.fetchall()]
    needed = ['user_id', 'input_data', 'predicted_price', 'timestamp']
    for col in needed:
        if col not in cols:
            try:
                if col == 'user_id':
                    c.execute("ALTER TABLE predictions ADD COLUMN user_id INTEGER")
                elif col == 'input_data':
                    c.execute("ALTER TABLE predictions ADD COLUMN input_data TEXT")
                elif col == 'predicted_price':
                    c.execute("ALTER TABLE predictions ADD COLUMN predicted_price REAL")
                elif col == 'timestamp':
                    c.execute("ALTER TABLE predictions ADD COLUMN timestamp DATETIME")
                conn.commit()
            except Exception as e:
                print(f'Could not add column {col} to predictions:', e)

def ensure_activity_schema(conn):
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS activity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            username TEXT,
            action TEXT,
            details TEXT,
            ip TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()

def current_user_id(conn=None):
    if 'user_id' in session:
        return session.get('user_id')
    if 'username' in session:
        close_conn = False
        if conn is None:
            conn = get_db()
            close_conn = True
        try:
            c = conn.cursor()
            c.execute('SELECT id FROM users WHERE username = ?', (session['username'],))
            row = c.fetchone()
            if row:
                uid = row['id']
                session['user_id'] = uid
                return uid
        except Exception as e:
            print('Error getting current_user_id:', e)
        finally:
            if close_conn:
                conn.close()
    return None

# -------------------- ACTIVITY LOGGING --------------------
def log_activity(action, details=None, conn=None):
    created_conn = False
    try:
        if conn is None:
            conn = get_db()
            created_conn = True
        ensure_activity_schema(conn)
        c = conn.cursor()
        ip = None
        try:
            ip = request.headers.get('X-Forwarded-For', request.remote_addr)
        except Exception:
            ip = None
        user_id = session.get('user_id')
        username = session.get('username')
        try:
            details_json = json.dumps(details) if details is not None else None
        except Exception:
            details_json = str(details)
        c.execute(
            'INSERT INTO activity (user_id, username, action, details, ip) VALUES (?, ?, ?, ?, ?)',
            (user_id, username, action, details_json, ip)
        )
        conn.commit()
    except Exception as e:
        print('Failed to log activity:', e)
        traceback.print_exc()
    finally:
        if created_conn and conn:
            conn.close()

def format_timestamp(ts, to_ist=True):
    """Format a timestamp string from database to readable format (default IST)."""
    if ts is None:
        return '—'
    try:
        # Try parsing as datetime string
        if isinstance(ts, str):
            # Handle various SQLite datetime formats
            for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M:%S.%f']:
                try:
                    dt = datetime.datetime.strptime(ts, fmt)
                    # Convert UTC to IST (UTC+5:30)
                    if to_ist:
                        dt = dt + datetime.timedelta(hours=5, minutes=30)
                    return dt.strftime('%Y-%m-%d %H:%M:%S IST')
                except ValueError:
                    continue
        # If it's already a datetime object
        if isinstance(ts, datetime.datetime):
            # Convert UTC to IST (UTC+5:30)
            if to_ist:
                ts = ts + datetime.timedelta(hours=5, minutes=30)
            return ts.strftime('%Y-%m-%d %H:%M:%S IST')
        return str(ts)
    except Exception:
        return str(ts) if ts else '—'

def preview_payload(raw, max_items=3, max_chars=140):
    """Return a short, human-readable preview from a JSON string or object."""
    if raw is None or raw == '':
        return '—'
    data = raw
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except Exception:
            data = raw
    if isinstance(data, dict):
        items = list(data.items())
        preview = ', '.join(f"{k}: {v}" for k, v in items[:max_items])
        if len(items) > max_items:
            preview += ' ...'
        return preview or '—'
    if isinstance(data, (list, tuple)):
        items = list(data)[:max_items]
        preview = ', '.join(str(v) for v in items)
        if len(data) > max_items:
            preview += ' ...'
        return preview or '—'
    text = str(data)
    if len(text) > max_chars:
        text = text[: max_chars - 3] + '...'
    return text or '—'

# -------------------- STARTUP: ENSURE TABLES --------------------
with get_db() as conn:
    try:
        ensure_users_schema(conn)
        ensure_predictions_schema(conn)
        ensure_activity_schema(conn)
    except Exception as e:
        print('Error ensuring DB schema on startup:', e)
        traceback.print_exc()

# -------------------- AUTH DECORATORS --------------------
def login_required(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        if 'username' not in session:
            flash('Please log in to access this page', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return wrapped

def admin_required(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        if 'username' not in session:
            flash('Please log in to access this page', 'warning')
            return redirect(url_for('login'))
        if session.get('role') != 'admin':
            flash('Admin required', 'danger')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return wrapped

# -------------------- ROUTES --------------------
@app.route('/')
def home():
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'GET':
        return render_template('signup.html')
    username = request.form.get('username', '').strip()
    password = request.form.get('password', '').strip()
    if not username or not password:
        flash('Username and password required', 'danger')
        return render_template('signup.html')
    try:
        with get_db() as conn:
            c = conn.cursor()
            c.execute('INSERT INTO users (username, password, role) VALUES (?, ?, ?)', (username, password, 'user'))
            conn.commit()
            log_activity('signup', {'created_username': username}, conn=conn)
            flash('Account created. Please log in.', 'success')
            return redirect(url_for('login'))
    except sqlite3.IntegrityError:
        flash('Username already exists', 'danger')
        return render_template('signup.html')
    except Exception as e:
        print('Signup error:', e)
        flash('Error creating account', 'danger')
        return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    username = request.form.get('username', '').strip()
    password = request.form.get('password', '').strip()
    if not username or not password:
        flash('Username and password required', 'danger')
        return render_template('login.html')
    try:
        with get_db() as conn:
            c = conn.cursor()
            c.execute('SELECT id, username, role FROM users WHERE username = ? AND password = ?', (username, password))
            row = c.fetchone()
            if row:
                session.permanent = True
                session['username'] = row['username']
                session['user_id'] = row['id']
                session['role'] = row['role'] if row['role'] else 'user'
                c.execute('UPDATE users SET last_login = ? WHERE id = ?', (datetime.datetime.utcnow(), row['id']))
                conn.commit()
                log_activity('login', {'username': username}, conn=conn)
                flash('Logged in successfully', 'success')
                return redirect(url_for('dashboard'))
            else:
                flash('Invalid credentials', 'danger')
                return render_template('login.html')
    except Exception as e:
        print('Login error:', e)
        flash('Error during login', 'danger')
        return render_template('login.html')

@app.route('/logout')
def logout():
    try:
        log_activity('logout', {'username': session.get('username')})
    except Exception:
        pass
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    if session.get('role') == 'admin':
        return redirect(url_for('admin_dashboard'))
    recent_predictions = []
    user_pred_total = 0
    try:
        with get_db() as conn:
            c = conn.cursor()
            uid = current_user_id(conn)
            if uid is None:
                flash('User not found', 'warning')
                return redirect(url_for('logout'))
            c.execute('SELECT COUNT(*) AS cnt FROM predictions WHERE user_id = ?', (uid,))
            r = c.fetchone()
            user_pred_total = int(r['cnt']) if r else 0
            c.execute('''
                SELECT p.id, p.user_id, p.input_data, p.predicted_price, p.timestamp, u.username
                FROM predictions p LEFT JOIN users u ON p.user_id = u.id
                WHERE p.user_id = ?
                ORDER BY p.timestamp DESC LIMIT 5
            ''', (uid,))
            fetched = [dict(row) for row in c.fetchall()]
            # Assign sequential IDs starting from 1 for user's own predictions
            for idx, row in enumerate(fetched, start=1):
                row['input_preview'] = preview_payload(row.get('input_data'))
                row['formatted_timestamp'] = format_timestamp(row.get('timestamp'))
                row['display_id'] = idx  # Sequential ID for user view
                row['actual_id'] = row['id']  # Keep actual DB ID for reference
            recent_predictions = fetched
    except Exception as e:
        print('Dashboard error:', e)
        traceback.print_exc()
        flash('Error loading dashboard data', 'warning')

    return render_template('dashboard.html',
                           recent_predictions=recent_predictions,
                           user_pred_total=user_pred_total)

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'GET':
        return render_template('predict.html')

    model_obj, feat_cols, imputer, scaler, log_target = get_model()
    if model_obj is None or not feat_cols:
        flash('Prediction model is not available', 'danger')
        return render_template('predict.html')

    keys = [
        'living area', 'number of bedrooms', 'number of bathrooms', 'number of floors',
        'condition of the house', 'grade of the house', 'Area of the house(excluding basement)',
        'Area of the basement', 'Built Year'
    ]
    # Optional features that might be needed for feature engineering
    optional_keys = ['lot area', 'waterfront present', 'number of views', 
                     'Renovation Year', 'Number of schools nearby', 'Distance from the airport']
    
    input_data = {}
    try:
        for k in keys:
            v = request.form.get(k)
            if v is None:
                raise ValueError(f'Missing input: {k}')
            try:
                val = float(v)
            except Exception:
                val = v
            input_data[k] = val
        
        # Add optional features if provided (default to 0)
        for k in optional_keys:
            v = request.form.get(k)
            if v is not None:
                try:
                    input_data[k] = float(v)
                except:
                    input_data[k] = 0
            elif k in feat_cols:  # If feature is expected but not provided, set default
                input_data[k] = 0

        # Create DataFrame and apply feature engineering
        df_input = pd.DataFrame([input_data])
        df_input = engineer_features(df_input)
        
        # Ensure all required features are present
        for col in feat_cols:
            if col not in df_input.columns:
                df_input[col] = 0
        
        df_input = df_input[feat_cols]
        
        # Apply imputation
        if imputer is not None:
            df_input = pd.DataFrame(imputer.transform(df_input), columns=feat_cols)
        
        # Apply scaling
        if scaler is not None:
            df_input = pd.DataFrame(scaler.transform(df_input), columns=feat_cols)
        
        predicted = float(model_obj.predict(df_input)[0])
        predicted_price = float(np.expm1(predicted) if log_target else predicted)

        with get_db() as conn:
            ensure_predictions_schema(conn)
            c = conn.cursor()
            now = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
            c.execute('INSERT INTO predictions (user_id, input_data, predicted_price, timestamp) VALUES (?, ?, ?, ?)',
                      (current_user_id(conn), json.dumps(input_data), predicted_price, now))
            conn.commit()
            log_activity('predict', {'input': input_data, 'predicted_price': predicted_price}, conn=conn)

        return render_template('predict.html', predicted_price=predicted_price, input_preview=input_data)
    except Exception as e:
        print('Prediction error:', e)
        traceback.print_exc()
        flash('Error during prediction: ' + str(e), 'danger')
        return render_template('predict.html')

@app.route('/history')
@login_required
def history():
    rows = []
    try:
        with get_db() as conn:
            ensure_predictions_schema(conn)
            uid = current_user_id(conn)
            c = conn.cursor()
            c.execute('SELECT * FROM predictions WHERE user_id = ? ORDER BY timestamp DESC', (uid,))
            fetched = [dict(r) for r in c.fetchall()]
            # Assign sequential IDs starting from 1 for user's own predictions
            for idx, r in enumerate(fetched, start=1):
                r['input_preview'] = preview_payload(r.get('input_data'))
                r['formatted_timestamp'] = format_timestamp(r.get('timestamp'))
                r['display_id'] = idx  # Sequential ID for user view
                r['actual_id'] = r['id']  # Keep actual DB ID for reference
            rows = fetched
            log_activity('view_history', {'returned_count': len(rows)}, conn=conn)
    except Exception as e:
        print('History error:', e)
        traceback.print_exc()
        flash('Error loading history', 'warning')
    return render_template('history.html', predictions=rows)

@app.route('/model-performance')
@login_required
def model_performance():
    if not os.path.exists(DATA_CSV):
        flash('Data CSV is missing', 'warning')
        return redirect(url_for('dashboard'))
    model_obj, feat_cols, imputer, scaler, log_target = get_model()
    if model_obj is None or not feat_cols:
        flash('Model is not available', 'warning')
        return redirect(url_for('dashboard'))
    try:
        df = pd.read_csv(DATA_CSV)
        if 'Price' not in df.columns:
            flash('Data CSV does not contain Price column', 'warning')
            return redirect(url_for('dashboard'))
        
        # Apply feature engineering
        df = engineer_features(df)
        
        # Ensure all required features are present
        for col in feat_cols:
            if col not in df.columns:
                df[col] = 0
        
        X = df[feat_cols].copy()
        
        # Apply imputation
        if imputer is not None:
            X = pd.DataFrame(imputer.transform(X), columns=feat_cols)
        else:
            X = X.fillna(0)
        
        # Apply scaling
        if scaler is not None:
            X = pd.DataFrame(scaler.transform(X), columns=feat_cols)
        
        y = df['Price']
        y_pred = model_obj.predict(X)
        if log_target:
            y_pred = np.expm1(y_pred)
        mse = float(((y - y_pred) ** 2).mean())
        rmse = float(np.sqrt(mse))
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        plt.figure(figsize=(6, 4))
        actual_vals = y.values if hasattr(y, 'values') else np.array(y)
        predicted_vals = np.array(y_pred)
        plt.scatter(actual_vals, predicted_vals, c='#58d5c9', alpha=0.7, edgecolors='#0b1a2b', linewidths=0.3, label='Predictions')
        min_val = min(actual_vals.min(), predicted_vals.min())
        max_val = max(actual_vals.max(), predicted_vals.max())
        plt.plot([min_val, max_val], [min_val, max_val], '--', color='#ff5e7e', linewidth=1.3, label='Perfect fit')
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title('Actual vs Predicted Price')
        plt.legend()
        plot_path = os.path.join(app.static_folder, 'model_performance.png')
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        return render_template('model_performance.html', mse=mse, rmse=rmse, r2=r2, plot_url=url_for('static', filename='model_performance.png'))
    except Exception as e:
        print('Model performance error:', e)
        traceback.print_exc()
        flash('Error computing model performance', 'danger')
        return redirect(url_for('dashboard'))

@app.route('/visuals')
@login_required
def visuals():
    if not os.path.exists(DATA_CSV):
        flash('Data CSV is missing', 'warning')
        return redirect(url_for('dashboard'))
    try:
        df = pd.read_csv(DATA_CSV)
        price_list = df['Price'].dropna().head(10).tolist() if 'Price' in df.columns else []
        bedrooms = df['number of bedrooms'].dropna().unique().tolist() if 'number of bedrooms' in df.columns else []
        avg_price_by_bed = {}
        if 'number of bedrooms' in df.columns and 'Price' in df.columns:
            grp = df.groupby('number of bedrooms')['Price'].mean().reset_index()
            avg_price_by_bed = {int(r['number of bedrooms']): float(r['Price']) for _, r in grp.iterrows()}
        area_pairs = []
        if 'living area' in df.columns and 'Price' in df.columns:
            area_pairs = list(zip(df['living area'].fillna(0).head(20).tolist(), df['Price'].fillna(0).head(20).tolist()))
        return render_template('visuals.html', price_list=price_list, bedrooms=bedrooms, avg_price_by_bed=avg_price_by_bed, area_pairs=area_pairs)
    except Exception as e:
        print('Visuals error:', e)
        traceback.print_exc()
        flash('Error generating visuals', 'warning')
        return redirect(url_for('dashboard'))

@app.route('/api/price-trend')
@login_required
def api_price_trend():
    if not os.path.exists(DATA_CSV):
        return json.dumps({'labels': [], 'values': []}), 200, {'Content-Type': 'application/json'}
    try:
        df = pd.read_csv(DATA_CSV)
        if 'number of bedrooms' not in df.columns or 'Price' not in df.columns:
            return json.dumps({'labels': [], 'values': []}), 200, {'Content-Type': 'application/json'}
        grp = df.groupby('number of bedrooms')['Price'].mean().reset_index()
        labels = grp['number of bedrooms'].astype(str).tolist()
        values = grp['Price'].round(2).tolist()
        return json.dumps({'labels': labels, 'values': values}), 200, {'Content-Type': 'application/json'}
    except Exception as e:
        print('Price trend API error:', e)
        traceback.print_exc()
        return json.dumps({'labels': [], 'values': []}), 200, {'Content-Type': 'application/json'}

# -------------------- ADMIN ROUTES --------------------
@app.route('/admin')
@admin_required
def admin_dashboard():
    user_count = admin_count = pred_count = 0
    recent_predictions = []
    recent_activity = []
    try:
        with get_db() as conn:
            c = conn.cursor()
            c.execute('SELECT COUNT(*) AS cnt FROM users')
            r = c.fetchone()
            user_count = int(r['cnt']) if r else 0
            c.execute("SELECT COUNT(*) as cnt FROM users WHERE role = 'admin'")
            r = c.fetchone()
            admin_count = int(r['cnt']) if r else 0
            c.execute('SELECT COUNT(*) AS cnt FROM predictions')
            r = c.fetchone()
            pred_count = int(r['cnt']) if r else 0
            c.execute('''
                SELECT p.id, p.user_id, p.input_data, p.predicted_price, p.timestamp, u.username
                FROM predictions p LEFT JOIN users u ON p.user_id = u.id
                ORDER BY p.timestamp DESC LIMIT 10
            ''')
            fetched_preds = [dict(r) for r in c.fetchall()]
            for row in fetched_preds:
                row['input_preview'] = preview_payload(row.get('input_data'))
                row['formatted_timestamp'] = format_timestamp(row.get('timestamp'))
            recent_predictions = fetched_preds

            c.execute('''
                SELECT id, username, action, details, ip, timestamp, user_id
                FROM activity
                ORDER BY timestamp DESC LIMIT 8
            ''')
            fetched_act = [dict(r) for r in c.fetchall()]
            for row in fetched_act:
                row['details_preview'] = preview_payload(row.get('details'))
                row['formatted_timestamp'] = format_timestamp(row.get('timestamp'))
            recent_activity = fetched_act
    except Exception as e:
        print('Admin dashboard error:', e)
        traceback.print_exc()
        flash('Error loading admin dashboard', 'warning')
    return render_template('admin_dashboard.html',
                           user_count=user_count,
                           admin_count=admin_count,
                           pred_count=pred_count,
                           recent_predictions=recent_predictions,
                           recent_activity=recent_activity)

@app.route('/admin/users')
@admin_required
def admin_users():
    try:
        with get_db() as conn:
            c = conn.cursor()
            c.execute('SELECT id, username, role, last_login FROM users')
            users = [dict(r) for r in c.fetchall()]
    except Exception as e:
        print('Admin users error:', e)
        traceback.print_exc()
        flash('Error loading users', 'warning')
        users = []
    return render_template('admin_users.html', users=users)

@app.route('/admin/users/create', methods=['GET', 'POST'])
@admin_required
def admin_create_user():
    if request.method == 'GET':
        return render_template('admin_create_user.html')
    username = request.form.get('username', '').strip()
    password = request.form.get('password', '').strip()
    role = request.form.get('role', 'user')
    if not username or not password:
        flash('Username and password required', 'danger')
        return render_template('admin_create_user.html')
    try:
        with get_db() as conn:
            c = conn.cursor()
            c.execute('INSERT INTO users (username, password, role) VALUES (?, ?, ?)', (username, password, role))
            conn.commit()
            log_activity('admin_create_user', {'created_username': username, 'role': role}, conn=conn)
            flash('User created', 'success')
            return redirect(url_for('admin_users'))
    except sqlite3.IntegrityError:
        flash('Username already exists', 'danger')
        return render_template('admin_create_user.html')
    except Exception as e:
        print('Admin create user error:', e)
        traceback.print_exc()
        flash('Error creating user', 'danger')
        return render_template('admin_create_user.html')

@app.route('/admin/users/<int:user_id>/delete', methods=['POST'])
@admin_required
def admin_delete_user(user_id):
    if session.get('user_id') == user_id:
        flash('Cannot delete the currently logged-in admin', 'warning')
        return redirect(url_for('admin_users'))
    try:
        with get_db() as conn:
            c = conn.cursor()
            c.execute('DELETE FROM users WHERE id = ?', (user_id,))
            conn.commit()
            log_activity('admin_delete_user', {'deleted_user_id': user_id}, conn=conn)
            flash('User deleted', 'success')
    except Exception as e:
        print('Admin delete user error:', e)
        traceback.print_exc()
        flash('Error deleting user', 'danger')
    return redirect(url_for('admin_users'))

@app.route('/admin/users/<int:user_id>/role', methods=['POST'])
@admin_required
def admin_change_role(user_id):
    new_role = request.form.get('role')
    if not new_role:
        flash('Role required', 'danger')
        return redirect(url_for('admin_users'))
    try:
        with get_db() as conn:
            c = conn.cursor()
            c.execute('UPDATE users SET role = ? WHERE id = ?', (new_role, user_id))
            conn.commit()
            log_activity('admin_change_role', {'user_id': user_id, 'new_role': new_role}, conn=conn)
            flash('Role updated', 'success')
    except Exception as e:
        print('Admin change role error:', e)
        traceback.print_exc()
        flash('Error updating role', 'danger')
    return redirect(url_for('admin_users'))

@app.route('/admin/predictions')
@admin_required
def admin_predictions():
    rows = []
    try:
        with get_db() as conn:
            c = conn.cursor()
            c.execute('''
                SELECT p.id, p.user_id, p.input_data, p.predicted_price, p.timestamp, u.username
                FROM predictions p LEFT JOIN users u ON p.user_id = u.id
                ORDER BY p.timestamp DESC
            ''')
            fetched = [dict(r) for r in c.fetchall()]
            for row in fetched:
                row['input_preview'] = preview_payload(row.get('input_data'))
                row['formatted_timestamp'] = format_timestamp(row.get('timestamp'))
            rows = fetched
    except Exception as e:
        print('Admin predictions error:', e)
        traceback.print_exc()
        flash('Error loading predictions', 'warning')
        rows = []
    return render_template('admin_predictions.html', predictions=rows)

@app.route('/admin/predictions/<int:pred_id>/delete', methods=['POST'])
@admin_required
def admin_delete_prediction(pred_id):
    try:
        with get_db() as conn:
            c = conn.cursor()
            c.execute('DELETE FROM predictions WHERE id = ?', (pred_id,))
            conn.commit()
            log_activity('admin_delete_prediction', {'deleted_prediction_id': pred_id}, conn=conn)
            flash('Prediction deleted', 'success')
    except Exception as e:
        print('Admin delete prediction error:', e)
        traceback.print_exc()
        flash('Error deleting prediction', 'danger')
    return redirect(url_for('admin_predictions'))

@app.route('/admin/export-predictions')
@admin_required
def admin_export_predictions():
    try:
        with get_db() as conn:
            c = conn.cursor()
            c.execute('''
                SELECT p.id, p.user_id, p.input_data, p.predicted_price, p.timestamp, u.username
                FROM predictions p LEFT JOIN users u ON p.user_id = u.id
                ORDER BY p.timestamp DESC
            ''')
            rows = [dict(r) for r in c.fetchall()]
        for r in rows:
            try:
                r_input = json.loads(r['input_data']) if r.get('input_data') else {}
            except Exception:
                r_input = {'raw_input': r.get('input_data')}
            r.update({f'input_{k}': v for k, v in (r_input.items())})
            r.pop('input_data', None)
        df = pd.DataFrame(rows)
        ts = datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(EXPORT_DIR, f'predictions_export_{ts}.csv')
        df.to_csv(path, index=False)
        log_activity('admin_export_predictions', {'export_path': path})
        return send_file(path, as_attachment=True)
    except Exception as e:
        print('Admin export predictions error:', e)
        traceback.print_exc()
        flash('Error exporting predictions', 'danger')
        return redirect(url_for('admin_predictions'))

@app.route('/admin/activity')
@admin_required
def admin_activity():
    activities = []
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        offset = (page - 1) * per_page
        with get_db() as conn:
            ensure_activity_schema(conn)
            c = conn.cursor()
            c.execute('SELECT COUNT(*) as cnt FROM activity')
            total = int(c.fetchone()['cnt'] or 0)
            c.execute('SELECT * FROM activity ORDER BY timestamp DESC LIMIT ? OFFSET ?', (per_page, offset))
            fetched = [dict(r) for r in c.fetchall()]
            for row in fetched:
                row['details_preview'] = preview_payload(row.get('details'), max_chars=200)
                row['formatted_timestamp'] = format_timestamp(row.get('timestamp'))
            activities = fetched
        return render_template('admin_activity.html', activities=activities, page=page, per_page=per_page, total=total)
    except Exception as e:
        print('Admin activity error:', e)
        traceback.print_exc()
        flash('Error loading activity', 'warning')
        return redirect(url_for('admin_dashboard'))

@app.route('/admin/activity/export')
@admin_required
def admin_activity_export():
    try:
        with get_db() as conn:
            ensure_activity_schema(conn)
            df = pd.read_sql_query('SELECT * FROM activity ORDER BY timestamp DESC', conn)
        ts = datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(EXPORT_DIR, f'activity_export_{ts}.csv')
        df.to_csv(path, index=False)
        log_activity('admin_export_activity', {'export_path': path})
        return send_file(path, as_attachment=True)
    except Exception as e:
        print('Admin export activity error:', e)
        traceback.print_exc()
        flash('Error exporting activity', 'danger')
        return redirect(url_for('admin_dashboard'))

@app.route('/admin/activity/<int:activity_id>/delete', methods=['POST'])
@admin_required
def admin_delete_activity(activity_id):
    try:
        with get_db() as conn:
            ensure_activity_schema(conn)
            c = conn.cursor()
            c.execute('DELETE FROM activity WHERE id = ?', (activity_id,))
            conn.commit()
            log_activity('admin_delete_activity', {'deleted_activity_id': activity_id}, conn=conn)
            flash('Activity entry deleted', 'success')
    except Exception as e:
        print('Admin delete activity error:', e)
        traceback.print_exc()
        flash('Error deleting activity entry', 'danger')
    return redirect(url_for('admin_activity', page=request.args.get('page', 1)))

@app.route('/admin/activity/clear', methods=['POST'])
@admin_required
def admin_clear_activity():
    try:
        with get_db() as conn:
            ensure_activity_schema(conn)
            c = conn.cursor()
            c.execute('DELETE FROM activity')
            conn.commit()
        flash('All activity logs cleared', 'success')
    except Exception as e:
        print('Admin clear activity error:', e)
        traceback.print_exc()
        flash('Error clearing activity log', 'danger')
    return redirect(url_for('admin_activity'))

# -------------------- ERROR HANDLING --------------------
@app.errorhandler(500)
def handle_500(e):
    print('Internal server error:', e)
    traceback.print_exc()
    flash('Internal server error', 'danger')
    return redirect(url_for('dashboard'))

# -------------------- ENTRYPOINT --------------------
if __name__ == '__main__':
    app.run(debug=True)
