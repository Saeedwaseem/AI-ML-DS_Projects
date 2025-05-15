import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans

# Step 2: Load the dataset
file_path = "C:/Users/Dell/Documents/PYTHON TRAINING/PRODUCT SALES.csv"  
data = pd.read_csv(file_path)

# Step 3: Explore the dataset (EDA)
print(data.info())
print(data.describe())
print(data.head())

# Visualize the distribution of age groups and products
sns.countplot(data['Age_Group'])
plt.title('Distribution of Age Groups')
plt.show()

sns.countplot(data['Product_Category'])
plt.title('Product Category Distribution')
plt.show()

# Step 4: Data Cleaning
# Drop unnecessary columns (if any)
columns_to_drop = ['how amny']  # Example column to drop
data = data.drop(columns=columns_to_drop, errors='ignore')

# Check for missing values
print(data.isnull().sum())
data = data.dropna()  # Drop rows with missing values (alternative: fillna())

# Step 5: Data Transformation
# Encode categorical columns
label_encoders = {}
categorical_columns = ['Age_Group', 'Customer_Gender', 'Country', 'Product_Category']

for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Convert Date to datetime and extract useful features
data['Date_of_sale'] = pd.to_datetime(data['Date_of_sale'])
data['Year'] = data['Date_of_sale'].dt.year
data['Month'] = data['Date_of_sale'].dt.month
data['Day'] = data['Date_of_sale'].dt.day

# Step 6: Feature Engineering
# Create new features like total_sales or product_popularity
data['Age_Product_Interaction'] = data['Age_Group'] * data['Product_Category']

# Step 7: Splitting for Supervised Learning
X = data[['Age_Group', 'Customer_Gender', 'Country', 'Product_Category', 'Age_Product_Interaction']]
y = data['Customer_Gender']  # Example target variable for segmentation (can be updated)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 8: Supervised Learning - Segmentation using Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Predict on test data
y_pred = rf_model.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 9: Unsupervised Segmentation using KMeans
# Cluster customers into segments
kmeans = KMeans(n_clusters=3, random_state=42)
data['Segment'] = kmeans.fit_predict(X_scaled)

# Visualize the segments
sns.scatterplot(data=data, x='Age_Group', y='Product_Category', hue='Segment', palette='viridis')
plt.title('Customer Segments')
plt.show()

# Step 10: Save the model and prepare for deployment
import joblib

# Save models
joblib.dump(rf_model, 'segmentation_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(kmeans, 'kmeans_model.pkl')

print("Models saved for deployment!")

# Step 11: Deployment Example
# Load model and predict on new data
loaded_model = joblib.load('segmentation_model.pkl')
loaded_scaler = joblib.load('scaler.pkl')

new_data = pd.DataFrame({
    'Age_Group': [2],
    'Customer_Gender': [1],
    'Country': [3],
    'Product_Category': [1],
    'Age_Product_Interaction': [2]
})
new_data_scaled = loaded_scaler.transform(new_data)
segment = loaded_model.predict(new_data_scaled)

print(f"Predicted segment: {segment[0]}")