import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load and preprocess the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('./Diabetes Classification.csv')
    df = df.drop('Unnamed: 0', axis=1)
    df['Gender'] = df['Gender'].str.capitalize()
    return df

st.set_page_config(layout="centered", page_title="Sugar Sage", page_icon="🧬",
                   menu_items={
                       "Get Help": "https://www.kaggle.com/datasets/simaanjali/diabetes-classification-dataset",
                       "Report a bug": "https://monika15verma.wixsite.com/monikavdev/contact",
                       "About": "This app helps track and predict diabetes-related risks."
                       })
st.title('🧬 Sugar Sage')
st.info('A wise guide for tracking and predicting diabetes.')

def display_correlation_heatmap(correlation_matrix):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        cbar=False,
        square=True,
        annot_kws={"size": 5},
        ax=ax
    )
    ax.tick_params(axis='x', labelsize=5)
    ax.tick_params(axis='y', labelsize=5)
    ax.set_title("Correlation Heatmap", fontsize=5)
    st.pyplot(fig)
    
def show_prediction_progress(prediction_prob):
    st.write('##### Diabetic Risk Prediction')
    st.write("This section shows the likelihood of the input data belonging to the diabetic class. The progress bar below represents the predicted risk percentage.")

    progress = int(prediction_prob * 100)

    if prediction_prob >= 0.5:
        label = f"High Risk: {progress}%"
        color = 'red'
    else:
        label = f"Low Risk: {progress}%"
        color = 'green'

    st.markdown(
        f"""
        <div style="border: 1px solid #ddd; border-radius: 5px; width: 100%; background-color: #f3f3f3; margin-top: 10px;">
            <div style="height: 24px; width: {progress}%; background-color: {color}; text-align: center; line-height: 24px; color: white; border-radius: 5px;">
                {label}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("**Note:** A risk percentage of 50% or higher indicates a higher likelihood of diabetes.")
    
def render_metric_pie_chart(metric_name, value, color):
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.pie(
        [value, 1 - value],
        colors=[color, "#E0E0E0"],
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.3},
    )
    ax.set_title(f"{metric_name}: {value:.4f}", fontsize=10)
    st.pyplot(fig)

def display_metrics_as_pie_charts(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    st.write("##### Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    blue_shades = ["#03045E", "#023E8A", "#0077B6", "#0096C7"]
    with col1:
        render_metric_pie_chart("Accuracy", accuracy, blue_shades[0])
    with col2:
        render_metric_pie_chart("Precision", precision, blue_shades[1])
    with col3:
        render_metric_pie_chart("Recall", recall, blue_shades[2])
    with col4:
        render_metric_pie_chart("F1 Score", f1, blue_shades[3])

# Function to display feature importance
def display_feature_importance(model, feature_names):
    st.write("##### Feature Importance")
    
    # Check if the model supports feature importance
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_[0])
    else:
        st.warning("Feature importance is not available for the selected model.")
        return

    # Create a DataFrame for feature importance
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    }).sort_values(by="Importance", ascending=True)

    st.bar_chart(importance_df.set_index("Feature"))

# Add hyperparameter selection for each model
def get_model_with_hyperparameters(model_name):
    with st.sidebar.header('Hyperparameters'):
        st.write(f"### Adjust the hyperparameters for {model_name}:")
        if model_name == 'Logistic Regression':
            c_value = st.sidebar.slider('C (Inverse of Regularization Strength)', 0.01, 10.0, 1.0)
            model = LogisticRegression(C=c_value, max_iter=1000)
        elif model_name == 'Decision Tree':
            max_depth = st.sidebar.slider('Max Depth', 1, 20, 5)
            min_samples_split = st.sidebar.slider('Min Samples Split', 2, 10, 2)
            model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
        elif model_name == 'Random Forest':
            n_estimators = st.sidebar.slider('Number of Estimators', 10, 200, 100)
            max_depth = st.sidebar.slider('Max Depth', 1, 20, 10)
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        elif model_name == 'Support Vector Machine':
            c_value = st.sidebar.slider('C (Regularization Parameter)', 0.01, 10.0, 1.0)
            kernel = st.sidebar.selectbox('Kernel', ['linear', 'poly', 'rbf', 'sigmoid'], index=2)
            model = SVC(C=c_value, kernel=kernel, probability=True)
        elif model_name == 'K-Nearest Neighbor':
            n_neighbors = st.sidebar.slider('Number of Neighbors', 1, 20, 5)
            weights = st.sidebar.selectbox('Weights', ['uniform', 'distance'], index=0)
            model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
        else:
            st.error("Invalid model selected.")
            model = None
        return model

# Main app
def main():
    # Load dataset
    df = load_data()
    sample_count, feature_count = df.shape
    dataset = df.copy()

    gender_map = {'M': 0, 'F': 1}
    dataset['Gender'] = dataset['Gender'].str.capitalize().map(gender_map)

    X = dataset.drop('Diagnosis', axis=1)
    y = dataset['Diagnosis'] 

    if y.dtypes != 'int64' or y.isnull().any():
        st.error("The target variable 'Diagnosis' contains invalid or missing values.")
        
    trained_models = {}
    
    # Normalize the data before training the models
    scaler = StandardScaler()

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    with st.sidebar.header('Input Features'):
        dataset_description = dataset.describe()
        
        gender = st.sidebar.selectbox('Gender', options=['Male', 'Female'], index=0)
        age = st.sidebar.slider(
            'Age',
            min_value=int(dataset_description['Age']['min']),
            max_value=int(dataset_description['Age']['max']),
            value=int(dataset_description['Age']['mean'])
        )
        bmi = st.sidebar.slider(
            'BMI',
            min_value=float(dataset_description['BMI']['min']),
            max_value=float(dataset_description['BMI']['max']),
            value=float(dataset_description['BMI']['mean'])
        )
        chol = st.sidebar.slider(
            'Cholesterol (Chol)',
            min_value=float(dataset_description['Chol']['min']),
            max_value=float(dataset_description['Chol']['max']),
            value=float(dataset_description['Chol']['mean'])
        )
        tg = st.sidebar.slider(
            'Triglycerides (TG)',
            min_value=float(dataset_description['TG']['min']),
            max_value=float(dataset_description['TG']['max']),
            value=float(dataset_description['TG']['mean'])
        )
        hdl = st.sidebar.slider(
            'High-Density Lipoprotein (HDL)',
            min_value=float(dataset_description['HDL']['min']),
            max_value=float(dataset_description['HDL']['max']),
            value=float(dataset_description['HDL']['mean'])
        )
        ldl = st.sidebar.slider(
            'Low-Density Lipoprotein (LDL)',
            min_value=float(dataset_description['LDL']['min']),
            max_value=float(dataset_description['LDL']['max']),
            value=float(dataset_description['LDL']['mean'])
        )
        cr = st.sidebar.slider(
            'Creatinine (Cr)',
            min_value=float(dataset_description['Cr']['min']),
            max_value=float(dataset_description['Cr']['max']),
            value=float(dataset_description['Cr']['mean'])
        )
        bun = st.sidebar.slider(
            'Blood Urea Nitrogen (BUN)',
            min_value=float(dataset_description['BUN']['min']),
            max_value=float(dataset_description['BUN']['max']),
            value=float(dataset_description['BUN']['mean'])
        )
        
        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [0 if gender == 'Male' else 1],
            'BMI': [bmi],
            'Chol': [chol],
            'TG': [tg],
            'HDL': [hdl],
            'LDL': [ldl],
            'Cr': [cr],
            'BUN': [bun]
        }, columns=X.columns)  

    
    with st.expander('About', icon='✏️'):
        st.subheader('Introduction')
        st.write('This app is designed to help users track their diabetes-related data and provide insights into their condition.')
        st.write('The data used in this app is from the [Kaggle](https://www.kaggle.com/datasets/simaanjali/diabetes-classification-dataset). This dataset was created to support research and development of risk prediction models for heart disease, diabetes and impaired kidney function.')
        
        st.subheader('About Dataset')
        st.write(
            """
            This dataset contains clinical data from a number of patients that have been analyzed to examine cardiovascular health and kidney function. 
            This data is important for evaluating the risk of heart disease and diabetes, as well as the impaired kidney function often associated with these conditions.
            
            This dataset was created to support research and development of risk prediction models for heart disease, diabetes, and impaired kidney function. 
            With relevant features and clear diagnosis labels, this dataset can be used to build and test accurate prediction models.
            """
        )

        st.subheader('Feature Descriptions')
        st.write('Below are the descriptions of the features included in the dataset:')
        st.write('1. **Age**: Represents the age of the patient in years. Age can be a risk factor for diabetes, as the risk of diabetes increases with age.')
        st.write('2. **Gender**: Indicates the gender of the patient, which can be a factor in the prediction of diabetes. Some studies suggest that women may have a different risk than men in developing diabetes.')
        st.write('3. **Body Mass Index (BMI)**: BMI is a measure that uses a person\'s height and weight to determine whether they are in the normal weight, overweight, or obese category. A high BMI is associated with a higher risk of diabetes.')
        st.write('4. **Chol (Total Cholesterol)**: Total cholesterol level in the blood. High cholesterol can be a risk factor for heart disease and diabetes.')
        st.write('5. **TG (Triglycerides)**: Represents the level of triglycerides in the blood. High levels can increase the risk of heart disease and diabetes.')
        st.write('6. **HDL (High-Density Lipoprotein)**: The "good" cholesterol that helps transport excess cholesterol from body tissues back to the liver for further processing or excretion. High levels of HDL are usually considered good for heart health.')
        st.write('7. **LDL (Low-Density Lipoprotein)**: The "bad" cholesterol that can cause plaque buildup in the arteries, increasing the risk of heart disease and stroke. High LDL levels can be a risk factor for diabetes.')
        st.write('8. **Cr (Creatinine)**: A waste product of muscle metabolism that is excreted from the body through the kidneys. Creatinine levels in the blood can provide information about kidney function. Kidney disease may be linked to the risk of diabetes.')
        st.write('9. **BUN (Blood Urea Nitrogen)**: A measure used to evaluate kidney and liver function. High levels of BUN may indicate kidney or liver disorders that can be related to diabetes.')
        st.write('10. **Diagnosis**: An indicator that someone has diabetes.')
        
    with st.expander('Data', icon='📊'):
        st.write('**Diabetes Dataset**')
        st.write(f'This dataset contains **{sample_count} samples** and **{feature_count} features**. The features include:')
        st.write(df.head())
        
        st.write('**Statistical Summary**') 
        st.write(dataset.describe())
        st.write('**Missing Values**')
        st.write(dataset.isnull().sum().to_dict())
        st.write('**Unique Gender Values:**')
        st.write(df['Gender'].unique())
        
    with st.expander('Data Preprocessing', icon='✂️'):
        st.write('**Data Preprocessing**')
        st.write('The dataset has been preprocessed to remove any unnecessary columns and handle missing values.')
        st.write('The following preprocessing steps were applied:')
        st.write('1. Removed the **Unnamed: 0** column.')
        st.write('2. Handled corrupt data values.')
        st.write('3. Converted categorical variables into numerical format.')
        st.write('4. Normalized the data.')
        st.write('5. Split the data into features and target variable.')
            
    with st.expander('Data Visualizations', icon='📈'):
        st.write('**Explore the dataset with Streamlit charts**')

        st.write('**1. Age Distribution**')
        st.bar_chart(dataset['Age'].value_counts().sort_index())

        st.write('**2. Gender Count**')
        gender_count = df['Gender'].value_counts()
        st.bar_chart(gender_count)

        st.write('**3. BMI vs Cholesterol**')
        st.write('Scatter plot of BMI vs Cholesterol:')
        st.dataframe(dataset[['BMI', 'Chol']])  
        st.vega_lite_chart(dataset, {
            'mark': 'circle',
            'encoding': {
                'x': {'field': 'BMI', 'type': 'quantitative'},
                'y': {'field': 'Chol', 'type': 'quantitative'},
                'color': {'field': 'Diagnosis', 'type': 'nominal'}
            }
        })

        st.write('**4. Correlation Heatmap**')
        correlation_matrix = dataset.corr()
        display_correlation_heatmap(correlation_matrix)
        
        st.write('**5. Diagnosis Count**')
        diagnosis_count = dataset['Diagnosis'].value_counts()
        st.bar_chart(diagnosis_count)       

    with st.expander('Model Evaluation', icon='🤖'):
        st.write('##### Build and Evaluate Models')
        model_name = st.selectbox(
            'Select a model to build and evaluate:',
            ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Support Vector Machine', 'K-Nearest Neighbor']
        )
        st.dataframe(input_data)

        # Normalize the input data for prediction
        input_data_normalized = scaler.transform(input_data)
        
        if model_name:
            if model_name not in trained_models:
                with st.spinner(f'Building {model_name} model...'):
                    model = get_model_with_hyperparameters(model_name)

                    # Train the model using normalized data
                    model.fit(X_train, y_train)
                    trained_models[model_name] = model
            else:
                model = trained_models[model_name]

            # Evaluate the model
            y_pred = model.predict(X_test)
            y_pred_prob = model.predict_proba(input_data_normalized)[:, 1][0] if hasattr(model, 'predict_proba') else model.decision_function(input_data_normalized)[0]
            st.divider()
            show_prediction_progress(y_pred_prob)
            st.divider()
            display_metrics_as_pie_charts(y_test, y_pred)
            st.divider()
            display_feature_importance(model, X.columns)



# Run the app
if __name__ == '__main__':
    main()