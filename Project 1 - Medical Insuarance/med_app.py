import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Page configure.
st.set_page_config(page_title="Medical Insurance Cost Predictor", layout="wide")
st.title("Medical Insurance Cost Predictor")
st.markdown("""
This Dashboard Helps an Insurance company estimate annual medical charges based on patient characteristics.
Use the sidebar to input a patient's detail and get an instant prediction.
""")

# 1. Cached Data loading & model training
@st.cache_data
def load_data():
    df = pd.read_csv('insurance.csv')
    return df

@st.cache_resource
def train_model(input_df):
    # FIX: Create a copy of the dataframe. This ensures we don't modify the 
    # original data, allowing your EDA charts below to still use the string 'smoker' column.
    df = input_df.copy()
    
    # Feature engineering
    df['smoker_num'] = (df['smoker'] == 'yes').astype(int)
    
    # FIX: Assign the result back to 'df' (or use inplace=True). 
    # This successfully removes the string column from the training data.
    df = df.drop('smoker', axis=1)
    
    # FIX: Added dtype=int. This forces the dummy variables to be 0s and 1s 
    # instead of booleans, which plays much nicer with scikit-learn.
    df = pd.get_dummies(df, columns=['sex', 'region'], drop_first=True, dtype=int)
    
    X = df.drop('charges', axis=1)
    y = df['charges']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Metrics
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    coef = pd.Series(model.coef_, index=X.columns)
    return model, X_train, X_test, y_test, mae, r2, coef

df = load_data()
model, X_train, X_test, y_test, mae, r2, coef = train_model(df)

# 2. Sidebar - User Input
st.sidebar.header("Patient Information")

age = st.sidebar.slider("Age", 18, 64, 30)
sex = st.sidebar.selectbox("Sex", ["male", 'female'])
bmi = st.sidebar.slider("BMI", 15.0, 55.0, 30.0, step=0.1)
children = st.sidebar.selectbox("Number of Children", [0, 1, 2, 3, 4, 5])
smoker = st.sidebar.selectbox("Smoker", ["no", "yes"])
region = st.sidebar.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

# 3. Preprocess input and predict.
# Create a data frame with the same columns as X_train
input_dict = {
    "age": age,
    "bmi": bmi,
    "children": children,
    "smoker_num": 1 if smoker == "yes" else 0,
    "sex_male": 1 if sex == "male" else 0,
    "region_northwest": 1 if region == "northwest" else 0,
    "region_southeast": 1 if region == "southeast" else 0,
    "region_southwest": 1 if region == "southwest" else 0
    # note: region_northeast is the reference category, so not included.
}

# Ensure all expected columns exist
for col in X_train.columns:
    if col not in input_dict:
        input_dict[col] = 0
        
input_df = pd.DataFrame([input_dict])
# reorder columns to match X_train.
input_df = input_df[X_train.columns]

# Predict
predicted_charge = model.predict(input_df)[0]

# 4. main area - results & Insights
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Model MAE", f"${mae:,.0f}")
with col2:
    st.metric("Model R2", f"{r2:.3f}")
with col3:
    st.metric("Predicted Annual charge", f"${predicted_charge:,.2f}")

# 5. Explanatory plots.
st.subheader("Model insights & Feature importance.")
fig_coef, ax = plt.subplots(figsize=(8, 5))
coef.sort_values().plot(kind="barh", ax=ax, color="steelblue")
ax.set_title("Linear Regression Coefficients (Impact on charges)")
ax.set_xlabel("Change in charge (USD) per unit increase in feature.")
st.pyplot(fig_coef)

# 6. Exploratory data analysis.
st.subheader("Exploratory Data Analysis")
tab1, tab2, tab3, tab4 = st.tabs(["Charge Distribution", "Age vs Charge", "BMI vs Charge", "Smoker Effect"])

with tab1:
    fig, ax = plt.subplots()
    sns.histplot(data=df, x='charges', kde=True, ax=ax)
    ax.set_title("Distribution of Annual charges")
    ax.set_xlabel("Charges ($)")
    st.pyplot(fig)

with tab2:
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='age', y='charges', hue='smoker', alpha=0.6, ax=ax)
    ax.set_title("Age vs Charges (colored by smoking status)")
    st.pyplot(fig)

with tab3:
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='bmi', y='charges', hue='smoker', alpha=0.6, ax=ax)
    ax.set_title("BMI vs Charges (colored by smoking status)")
    st.pyplot(fig)

with tab4:
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x='smoker', y='charges', ax=ax)
    ax.set_title("Charges by smoker Status")
    st.pyplot(fig)

# 7. Business Recommendations
st.subheader("Business Recommendations")
st.markdown("""
Based on the model insights, here are actionable suggestions for the insurance company:

1. **Smoking is the largest risk factor** - coefficient for the smoker is large and positive.
            implement higher premiums for smokers or offer incentives for quitting.
2. **BMI and age also significantly increase charges** - individuals with high bmi or older age cost more.
            consider wellness programs for weight management and regular health check-ups.
3. **Regional differences are small** - The model show limited impact of region, so pricing can be uniform across geographies.            
4. **Number of Children has a positive but modest effect** - family plans may be priced accordingly, but the effect is not as strong as smoking.            
5. **Use the interactive tool** - Agents can quickly estimate charges for prospective clients, improving transparency and customer trust.

These recommendations help the company set fair, risk-based premiums while encouraging healthy behaviours.            
""")