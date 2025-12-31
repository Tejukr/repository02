import pickle
import pandas as pd
import os

print("=" * 50)
print("MODEL DEBUG SCRIPT")
print("=" * 50)

# Check if files exist
print("\n1. Checking files...")
model_exists = os.path.exists('linear_regression_model.pkl')
features_exists = os.path.exists('model_features.pkl')
data_exists = os.path.exists('data/house_data.csv')

print(f"   Model file exists: {model_exists}")
print(f"   Features file exists: {features_exists}")
print(f"   Data file exists: {data_exists}")

# Load and check model
print("\n2. Loading model...")
try:
    with open('linear_regression_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print(f"   ✓ Model loaded successfully")
    print(f"   Model type: {type(model)}")
    print(f"   Model coef shape: {model.coef_}")
    print(f"   Model intercept: {model.intercept_}")
except Exception as e:
    print(f"   ✗ Error loading model: {e}")
    model = None

# Load features
print("\n3. Loading features...")
try:
    with open('model_features.pkl', 'rb') as f:
        features = pickle.load(f)
    print(f"   ✓ Features loaded successfully")
    print(f"   Features: {features}")
except Exception as e:
    print(f"   ✗ Error loading features: {e}")
    features = None

# Test prediction
print("\n4. Testing prediction...")
if model and features:
    try:
        test_data = {
            'living area': 1500,
            'number of bedrooms': 3,
            'number of bathrooms': 2,
            'number of floors': 2,
            'condition of the house': 3,
            'grade of the house': 8,
            'Area of the house(excluding basement)': 1200,
            'Area of the basement': 500,
            'Built Year': 2000
        }
        
        df_test = pd.DataFrame([test_data])
        print(f"   Test data: {test_data}")
        
        # Make prediction
        prediction = model.predict(df_test)[0]
        print(f"   ✓ Prediction: ₹ {prediction:,.2f}")
        
        # Test with different values
        print("\n5. Testing with different values...")
        test_data2 = {
            'living area': 3000,
            'number of bedrooms': 5,
            'number of bathrooms': 3,
            'number of floors': 2,
            'condition of the house': 5,
            'grade of the house': 10,
            'Area of the house(excluding basement)': 2500,
            'Area of the basement': 800,
            'Built Year': 2010
        }
        
        df_test2 = pd.DataFrame([test_data2])
        prediction2 = model.predict(df_test2)[0]
        print(f"   Test data 2: {test_data2}")
        print(f"   ✓ Prediction 2: ₹ {prediction2:,.2f}")
        
        if prediction != prediction2:
            print("\n   ✓ Model is working correctly (predictions differ)")
        else:
            print("\n   ✗ WARNING: Predictions are the same!")
            
    except Exception as e:
        print(f"   ✗ Error during prediction: {e}")
else:
    print("   ✗ Cannot test - model or features not loaded")

print("\n" + "=" * 50)