"""
Data preprocessing module for diabetes prediction project.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

class DataPreprocessor:
    """
    Class for preprocessing diabetes dataset.
    """
    
    def __init__(self, data_path=None):
        """
        Initialize the preprocessor.
        
        Args:
            data_path (str): Path to the dataset file
        """
        self.data_path = data_path
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.feature_names = None
        
    def load_data(self, data_path=None):
        """
        Load the diabetes dataset.
        
        Args:
            data_path (str): Path to the dataset file
        """
        if data_path:
            self.data_path = data_path
        
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"Data loaded successfully. Shape: {self.data.shape}")
            return self.data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def explore_data(self):
        """
        Perform basic data exploration.
        """
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
        
        print("Dataset Overview:")
        print("=" * 50)
        print(f"Shape: {self.data.shape}")
        print(f"\nColumn names: {list(self.data.columns)}")
        print(f"\nData types:\n{self.data.dtypes}")
        print(f"\nMissing values:\n{self.data.isnull().sum()}")
        print(f"\nBasic statistics:\n{self.data.describe()}")
        
        # Check for zero values (which might indicate missing data)
        print("\nZero values in each column:")
        zero_counts = (self.data == 0).sum()
        for col, count in zero_counts.items():
            if count > 0:
                print(f"  {col}: {count} ({count/len(self.data)*100:.1f}%)")
        
        # Target variable distribution
        print(f"\nTarget variable distribution:")
        print(self.data['Outcome'].value_counts())
        print(f"Percentage of diabetes cases: {self.data['Outcome'].mean()*100:.1f}%")
    
    def handle_missing_values(self, method='median'):
        """
        Handle missing values in the dataset.
        Zero values in certain columns likely represent missing data.
        
        Args:
            method (str): Method to handle missing values ('median', 'mean', 'mode')
        """
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
        
        # Columns where 0 is not a valid value
        zero_not_valid = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        
        print("Handling missing values (zeros)...")
        for col in zero_not_valid:
            if col in self.data.columns:
                # Replace 0 with NaN
                self.data[col] = self.data[col].replace(0, np.nan)
                
                # Handle missing values based on method
                if method == 'median':
                    fill_value = self.data[col].median()
                elif method == 'mean':
                    fill_value = self.data[col].mean()
                else:  # mode
                    fill_value = self.data[col].mode()[0]
                
                # Fill missing values
                self.data[col] = self.data[col].fillna(fill_value)
                print(f"  {col}: Filled with {method} value ({fill_value:.2f})")
        
        print("Missing values handled successfully.")
    
    def detect_outliers(self, method='iqr'):
        """
        Detect outliers in the dataset.
        
        Args:
            method (str): Method to detect outliers ('iqr', 'zscore')
        """
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
        
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col != 'Outcome']
        
        outliers_info = {}
        
        for col in numeric_columns:
            if method == 'iqr':
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = self.data[(self.data[col] < lower_bound) | 
                                   (self.data[col] > upper_bound)]
            else:  # zscore
                z_scores = np.abs((self.data[col] - self.data[col].mean()) / 
                                self.data[col].std())
                outliers = self.data[z_scores > 3]
            
            outliers_info[col] = len(outliers)
            print(f"{col}: {len(outliers)} outliers detected")
        
        return outliers_info
    
    def remove_outliers(self, method='iqr', columns=None):
        """
        Remove outliers from the dataset.
        
        Args:
            method (str): Method to detect outliers ('iqr', 'zscore')
            columns (list): Columns to remove outliers from
        """
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
        
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns
            columns = [col for col in columns if col != 'Outcome']
        
        initial_shape = self.data.shape
        
        for col in columns:
            if method == 'iqr':
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                self.data = self.data[(self.data[col] >= lower_bound) & 
                                    (self.data[col] <= upper_bound)]
            else:  # zscore
                z_scores = np.abs((self.data[col] - self.data[col].mean()) / 
                                self.data[col].std())
                self.data = self.data[z_scores <= 3]
        
        final_shape = self.data.shape
        removed_rows = initial_shape[0] - final_shape[0]
        print(f"Removed {removed_rows} rows with outliers.")
        print(f"Dataset shape: {initial_shape} -> {final_shape}")
    
    def create_features(self):
        """
        Create additional features from existing ones.
        """
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
        
        # Create BMI categories
        self.data['BMI_Category'] = pd.cut(self.data['BMI'], 
                                          bins=[0, 18.5, 25, 30, float('inf')], 
                                          labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
        
        # Create age groups
        self.data['Age_Group'] = pd.cut(self.data['Age'], 
                                       bins=[0, 25, 35, 45, float('inf')], 
                                       labels=['Young', 'Adult', 'Middle-aged', 'Senior'])
        
        # Create glucose categories
        self.data['Glucose_Category'] = pd.cut(self.data['Glucose'], 
                                              bins=[0, 100, 126, float('inf')], 
                                              labels=['Normal', 'Pre-diabetic', 'Diabetic'])
        
        # Blood pressure categories
        self.data['BP_Category'] = pd.cut(self.data['BloodPressure'], 
                                         bins=[0, 80, 90, float('inf')], 
                                         labels=['Normal', 'High-Normal', 'High'])
        
        print("Additional features created successfully.")
        print(f"New dataset shape: {self.data.shape}")
    
    def prepare_features(self, include_categorical=False):
        """
        Prepare features for machine learning.
        
        Args:
            include_categorical (bool): Whether to include categorical features
        """
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
        
        # Select features
        if include_categorical:
            # Include both numerical and categorical features
            categorical_cols = self.data.select_dtypes(include=['category', 'object']).columns
            numerical_cols = [col for col in self.data.select_dtypes(include=[np.number]).columns 
                            if col != 'Outcome']
            
            # One-hot encode categorical features
            if len(categorical_cols) > 0:
                categorical_df = pd.get_dummies(self.data[categorical_cols], prefix=categorical_cols)
                numerical_df = self.data[numerical_cols]
                self.X = pd.concat([numerical_df, categorical_df], axis=1)
            else:
                self.X = self.data[numerical_cols]
        else:
            # Only numerical features
            self.X = self.data.select_dtypes(include=[np.number]).drop('Outcome', axis=1)
        
        self.y = self.data['Outcome']
        self.feature_names = list(self.X.columns)
        
        print(f"Features prepared. Shape: {self.X.shape}")
        print(f"Feature names: {self.feature_names}")
    
    def scale_features(self, scaler_type='standard'):
        """
        Scale features using StandardScaler or MinMaxScaler.
        
        Args:
            scaler_type (str): Type of scaler ('standard' or 'minmax')
        """
        if self.X is None:
            print("No features prepared. Please prepare features first.")
            return
        
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()
        
        self.X = pd.DataFrame(
            self.scaler.fit_transform(self.X),
            columns=self.X.columns,
            index=self.X.index
        )
        
        print(f"Features scaled using {scaler_type} scaler.")
    
    def split_data(self, test_size=0.2, random_state=42, stratify=True):
        """
        Split data into training and testing sets.
        
        Args:
            test_size (float): Proportion of test set
            random_state (int): Random state for reproducibility
            stratify (bool): Whether to stratify split based on target
        """
        if self.X is None or self.y is None:
            print("No data prepared. Please prepare features first.")
            return
        
        stratify_param = self.y if stratify else None
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=stratify_param
        )
        
        print(f"Data split completed:")
        print(f"  Training set: {self.X_train.shape}")
        print(f"  Test set: {self.X_test.shape}")
        print(f"  Training target distribution: {self.y_train.value_counts().to_dict()}")
        print(f"  Test target distribution: {self.y_test.value_counts().to_dict()}")
    
    def get_processed_data(self):
        """
        Get the processed training and testing data.
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        if self.X_train is None:
            print("Data not split yet. Please split data first.")
            return None
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def save_processed_data(self, filepath_prefix='processed_data'):
        """
        Save processed data to CSV files.
        
        Args:
            filepath_prefix (str): Prefix for saved files
        """
        if self.X_train is None:
            print("No processed data to save.")
            return
        
        try:
            self.X_train.to_csv(f'{filepath_prefix}_X_train.csv', index=False)
            self.X_test.to_csv(f'{filepath_prefix}_X_test.csv', index=False)
            self.y_train.to_csv(f'{filepath_prefix}_y_train.csv', index=False)
            self.y_test.to_csv(f'{filepath_prefix}_y_test.csv', index=False)
            print(f"Processed data saved with prefix: {filepath_prefix}")
        except Exception as e:
            print(f"Error saving data: {e}")

def main():
    """
    Example usage of the DataPreprocessor class.
    """
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load data
    preprocessor.load_data('diabetes.csv')
    
    # Explore data
    preprocessor.explore_data()
    
    # Handle missing values
    preprocessor.handle_missing_values(method='median')
    
    # Detect outliers
    preprocessor.detect_outliers(method='iqr')
    
    # Prepare features
    preprocessor.prepare_features(include_categorical=False)
    
    # Scale features
    preprocessor.scale_features(scaler_type='standard')
    
    # Split data
    preprocessor.split_data(test_size=0.2, random_state=42)
    
    # Get processed data
    X_train, X_test, y_train, y_test = preprocessor.get_processed_data()
    
    print("Data preprocessing completed successfully!")

if __name__ == "__main__":
    main()
