import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib to use 'Agg' backend before importing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

def load_and_process_data(filepath='predictor/ml_model/data/student-mat.csv'):
    """
    Tải và tiền xử lý dữ liệu từ dataset UCI Student Performance
    
    Yêu cầu đề tài:
    - Thu thập/xây dựng bộ dữ liệu
    - Làm sạch dữ liệu: Kiểm tra dữ liệu thiếu và mã hóa dữ liệu phân loại
    - Chuẩn hóa giá trị số
    """
    # 1. Tải dữ liệu
    try:
        data = pd.read_csv(filepath, sep=';')
        print(f"Loaded dataset with {data.shape[0]} samples and {data.shape[1]} features")
    except FileNotFoundError:
        print(f"Dataset not found at {filepath}. Attempting to download...")
        # Tạo thư mục nếu chưa có
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Download từ UCI ML Repository
        import urllib.request
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
        zip_path = os.path.dirname(filepath) + "/student.zip"
        
        try:
            urllib.request.urlretrieve(url, zip_path)
            # Giải nén file
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(filepath))
            # Thử lại đọc file
            data = pd.read_csv(filepath, sep=';')
            print(f"Downloaded dataset with {data.shape[0]} samples and {data.shape[1]} features")
        except Exception as e:
            print(f"Failed to download dataset: {e}")
            return None, None

    # 2. Kiểm tra dữ liệu thiếu
    print("\n--- Missing Values Analysis ---")
    missing_values = data.isnull().sum()
    print(missing_values[missing_values > 0] if any(missing_values > 0) else "No missing values found")
    
    # Nếu có giá trị thiếu, thực hiện imputation
    if any(missing_values > 0):
        numeric_imputer = SimpleImputer(strategy='mean')
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        
        # Áp dụng imputation cho các cột phù hợp
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        
        data[numeric_cols] = numeric_imputer.fit_transform(data[numeric_cols])
        data[categorical_cols] = categorical_imputer.fit_transform(data[categorical_cols])
    
    # 3. Mã hóa dữ liệu phân loại
    print("\n--- Encoding Categorical Features ---")
    categorical_features = data.select_dtypes(include=['object']).columns
    print(f"Categorical features: {', '.join(categorical_features)}")
    
    # Binary encoding cho các biến nhị phân (yes/no)
    binary_cols = ['schoolsup', 'famsup', 'paid', 'activities', 
                   'nursery', 'higher', 'internet', 'romantic']
    
    for col in binary_cols:
        if col in data.columns:
            data[col] = data[col].map({'yes': 1, 'no': 0})
            print(f"Binary encoded: {col}")
    
    # One-hot encoding cho biến phân loại khác
    non_binary_cat = [col for col in categorical_features if col not in binary_cols]
    if non_binary_cat:
        data = pd.get_dummies(data, columns=non_binary_cat, drop_first=True)
        print(f"One-hot encoded: {', '.join(non_binary_cat)}")
    
    # 4. Chuẩn hóa giá trị số
    print("\n--- Normalizing Numerical Features ---")
    numeric_features = [col for col in data.select_dtypes(include=['int64', 'float64']).columns 
                        if col not in ['G1', 'G2', 'G3']]
    print(f"Numerical features to normalize: {', '.join(numeric_features)}")
    
    # Sử dụng MinMaxScaler để chuẩn hóa về khoảng [0, 1]
    scaler = MinMaxScaler()
    data[numeric_features] = scaler.fit_transform(data[numeric_features])
    
    # 5. Xác định đặc trưng và mục tiêu
    X = data.drop(columns=['G3'])  # G3 là điểm cuối kỳ (mục tiêu cần dự đoán)
    y = data['G3']
    
    # Lưu metadata cho quá trình dự đoán
    metadata = {
        'binary_cols': binary_cols,
        'non_binary_cat': non_binary_cat if len(non_binary_cat) > 0 else [],
        'numeric_features': numeric_features,
        'scaler': scaler,
        'feature_names': list(X.columns)
    }
    
    return X, y, metadata

def analyze_correlations(X, y, metadata, output_dir='predictor/ml_model/analysis'):
    """
    Phân tích tương quan giữa các yếu tố và điểm số
    
    Yêu cầu đề tài:
    - Phân tích tương quan giữa các yếu tố và điểm số
    - Phát hiện mối quan hệ như thời gian học và điểm số, ảnh hưởng của hỗ trợ gia đình
    """
    # Tạo thư mục output nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Kết hợp X và y để phân tích tương quan
    data_for_corr = X.copy()
    data_for_corr['G3'] = y  # Thêm điểm cuối kỳ vào dataframe
    
    print("\n--- Correlation Analysis ---")
    
    # 1. Tính ma trận tương quan
    correlation_matrix = data_for_corr.corr()
    
    # 2. Lấy tương quan với điểm số G3
    correlations_with_g3 = correlation_matrix['G3'].sort_values(ascending=False)
    print("Top correlations with final grade (G3):")
    print(correlations_with_g3.head(10))
    
    # 3. Trực quan hóa tương quan với heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix Heatmap')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_heatmap.png")
    
    # 4. Phân tích các yếu tố quan trọng theo yêu cầu đề tài
    key_factors = [
    # Đặc trưng học tập & kết quả trước đó
    'G1', 'G2', 'studytime', 'failures', 'absences', 
    
    # Thông tin cá nhân và gia đình
    'age', 'Medu', 'Fedu', 'famrel', 'famsup',
    
    # Thói quen và lối sống
    'internet', 'freetime', 'goout', 'activities', 'higher',
    'romantic', 'health', 'Dalc', 'Walc',
    
    # Yếu tố hỗ trợ giáo dục
    'schoolsup', 'paid', 'traveltime'
    ]

    if 'famsup' in data_for_corr.columns:
        key_factors.append('famsup')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=key_factors, y=[correlations_with_g3[factor] for factor in key_factors])
    plt.title('Correlation with Final Grade for Key Factors')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/key_factors_correlation.png")
    
    # 5. Tạo scatterplot cho một số mối quan hệ quan trọng
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    if 'G1' in data_for_corr.columns:
        sns.scatterplot(x='G1', y='G3', data=data_for_corr, ax=axes[0, 0])
        axes[0, 0].set_title('First Period Grade vs Final Grade')
    
    if 'G2' in data_for_corr.columns:
        sns.scatterplot(x='G2', y='G3', data=data_for_corr, ax=axes[0, 1])
        axes[0, 1].set_title('Second Period Grade vs Final Grade')
    
    if 'studytime' in data_for_corr.columns:
        sns.boxplot(x='studytime', y='G3', data=data_for_corr, ax=axes[1, 0])
        axes[1, 0].set_title('Study Time vs Final Grade')
    
    if 'famsup' in data_for_corr.columns:
        sns.boxplot(x='famsup', y='G3', data=data_for_corr, ax=axes[1, 1])
        axes[1, 1].set_title('Family Support vs Final Grade')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/key_relationships.png")
    
    # 6. Lưu phân tích tương quan vào file CSV để tham khảo sau
    correlations_with_g3.to_csv(f"{output_dir}/correlations_with_final_grade.csv")
    
    # Trả về top features dựa trên tương quan
    top_features = correlations_with_g3.drop('G3')[:10].index.tolist()
    return top_features, correlations_with_g3

def train_regression_model(X, y, metadata, top_features):
    """
    Áp dụng mô hình hồi quy để xây dựng ứng dụng dự đoán điểm số
    
    Yêu cầu đề tài:
    - Áp dụng mô hình hồi quy để dự đoán điểm số
    """
    print("\n--- Training Regression Model ---")
    
    # 1. Chia dữ liệu thành tập huấn luyện và kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # 2. Huấn luyện mô hình RandomForest (mô hình hồi quy mạnh)
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    
    # 3. Đánh giá mô hình RandomForest trên tập test
    rf_predictions = rf_model.predict(X_test)
    rf_mse = mean_squared_error(y_test, rf_predictions)
    rf_r2 = r2_score(y_test, rf_predictions)
    
    print(f"RandomForest Mean Squared Error: {rf_mse:.4f}")
    print(f"RandomForest R² Score: {rf_r2:.4f}")
    
    # 4. Cross-validation để đánh giá độ tin cậy của mô hình
    cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='r2')
    print(f"Cross-validation R² scores: {cv_scores}")
    print(f"Mean CV R² score: {cv_scores.mean():.4f}")
    
    # 5. Huấn luyện mô hình Linear Regression để so sánh
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    # 6. Đánh giá mô hình Linear Regression
    lr_predictions = lr_model.predict(X_test)
    lr_mse = mean_squared_error(y_test, lr_predictions)
    lr_r2 = r2_score(y_test, lr_predictions)
    
    print(f"Linear Regression Mean Squared Error: {lr_mse:.4f}")
    print(f"Linear Regression R² Score: {lr_r2:.4f}")
    
    # 7. So sánh hai mô hình và chọn mô hình tốt nhất
    best_model = rf_model if rf_r2 > lr_r2 else lr_model
    print(f"\nBest model: {'RandomForest' if rf_r2 > lr_r2 else 'Linear Regression'}")
    
    # 8. Phân tích tầm quan trọng của đặc trưng (chỉ áp dụng cho RandomForest)
    if rf_r2 > lr_r2:
        feature_importances = rf_model.feature_importances_
        features_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': feature_importances
        }).sort_values('Importance', ascending=False)
        
        print("\nFeature Importances:")
        print(features_df.head(10))
        
        # Trực quan hóa tầm quan trọng của đặc trưng
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=features_df.head(10))
        plt.title('Top 10 Most Important Features')
        plt.tight_layout()
        plt.savefig("predictor/ml_model/analysis/feature_importance.png")
    
    # 9. Trực quan hóa dự đoán vs thực tế
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, rf_predictions, alpha=0.5)
    plt.plot([0, 20], [0, 20], '--r')
    plt.xlabel('Actual Scores')
    plt.ylabel('Predicted Scores')
    plt.title('RandomForest: Prediction vs Actual')
    plt.tight_layout()
    plt.savefig("predictor/ml_model/analysis/rf_prediction_vs_actual.png")
    
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, lr_predictions, alpha=0.5)
    plt.plot([0, 20], [0, 20], '--r')
    plt.xlabel('Actual Scores')
    plt.ylabel('Predicted Scores')
    plt.title('Linear Regression: Prediction vs Actual')
    plt.tight_layout()
    plt.savefig("predictor/ml_model/analysis/lr_prediction_vs_actual.png")
    
    return best_model

def save_model_and_metadata(model, metadata, filepath='predictor/ml_model/student_score_model.pkl'):
    """
    Lưu mô hình và metadata cho sử dụng sau này
    """
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Đóng gói mô hình và metadata
    model_package = {
        'model': model,
        'metadata': metadata,
        'creation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Lưu mô hình
    joblib.dump(model_package, filepath)
    print(f"\nModel and metadata saved to {filepath}")

def generate_synthetic_data(n_samples=1000):
    """
    Tạo dữ liệu tổng hợp khi không có dữ liệu thực hoặc để bổ sung dữ liệu
    """
    np.random.seed(42)
    
    # Tạo đặc trưng với phạm vi tương tự như dữ liệu UCI
    age = np.clip(np.random.normal(16.5, 1.2, n_samples), 15, 22)
    studytime = np.random.randint(1, 5, n_samples)  # 1-4
    failures = np.clip(np.random.poisson(0.5, n_samples), 0, 3)  # 0-3
    absences = np.clip(np.random.poisson(3.5, n_samples), 0, 30)
    schoolsup = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    famsup = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
    paid = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    activities = np.random.choice([0, 1], n_samples)
    higher = np.random.choice([0, 1], n_samples, p=[0.1, 0.9])
    internet = np.random.choice([0, 1], n_samples, p=[0.2, 0.8])
    romantic = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    freetime = np.random.randint(1, 6, n_samples)  # 1-5
    goout = np.random.randint(1, 6, n_samples)  # 1-5
    health = np.random.randint(1, 6, n_samples)  # 1-5
    
    # Tạo điểm giai đoạn 1
    g1 = np.clip(
        10 + 
        (studytime - 1) * 1.2 - 
        failures * 2 -
        absences * 0.1 +
        np.random.normal(0, 2, n_samples),
        0, 20
    )
    
    # Tạo điểm giai đoạn 2
    g2 = np.clip(
        g1 * 0.8 + 
        (studytime - 1) * 1 -
        failures * 1 -
        absences * 0.1 +
        np.random.normal(0, 1.5, n_samples),
        0, 20
    )
    
    # Tạo dataframe với tất cả các đặc trưng
    data = pd.DataFrame({
        'age': age,
        'studytime': studytime,
        'failures': failures,
        'absences': absences,
        'schoolsup': schoolsup,
        'famsup': famsup,
        'paid': paid,
        'activities': activities,
        'higher': higher,
        'internet': internet,
        'romantic': romantic,
        'freetime': freetime, 
        'goout': goout,
        'health': health,
        'G1': g1,
        'G2': g2
    })
    
    # Tạo điểm cuối kỳ (G3)
    g3 = np.clip(
        g1 * 0.3 + g2 * 0.6 +
        (studytime - 1) * 0.8 -
        failures * 1.2 -
        absences * 0.05 +
        famsup * 0.4 +
        np.random.normal(0, 1.2, n_samples),
        0, 20
    )
    
    # Thêm G3 vào dataframe
    data['G3'] = g3
    
    # Chuẩn hóa các đặc trưng số
    numeric_features = ['age', 'studytime', 'failures', 'absences', 
                       'freetime', 'goout', 'health']
    scaler = MinMaxScaler()
    data[numeric_features] = scaler.fit_transform(data[numeric_features])
    
    # Tách đặc trưng và nhãn
    X = data.drop(columns=['G3'])
    y = data['G3']
    
    # Tạo metadata
    metadata = {
        'binary_cols': ['schoolsup', 'famsup', 'paid', 'activities', 
                        'higher', 'internet', 'romantic'],
        'non_binary_cat': [],
        'numeric_features': numeric_features,
        'scaler': scaler,
        'feature_names': list(X.columns)
    }
    
    print(f"Generated synthetic dataset with {n_samples} samples")
    return X, y, metadata

def generate_synthetic_data_nonlinear(n_samples=1000):
    """
    Tạo dữ liệu tổng hợp với mối quan hệ phi tuyến tính phức tạp
    """
    np.random.seed(42)
    
    # Tạo các biến đặc trưng giống như trước
    age = np.clip(np.random.normal(16.5, 1.2, n_samples), 15, 22)
    studytime = np.random.randint(1, 5, n_samples)  # 1-4
    failures = np.clip(np.random.poisson(0.5, n_samples), 0, 3)  # 0-3
    absences = np.clip(np.random.poisson(3.5, n_samples), 0, 30)
    schoolsup = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    famsup = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
    paid = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    activities = np.random.choice([0, 1], n_samples)
    higher = np.random.choice([0, 1], n_samples, p=[0.1, 0.9])
    internet = np.random.choice([0, 1], n_samples, p=[0.2, 0.8])
    romantic = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    freetime = np.random.randint(1, 6, n_samples)  # 1-5
    goout = np.random.randint(1, 6, n_samples)  # 1-5
    health = np.random.randint(1, 6, n_samples)  # 1-5
    dalc = np.random.randint(1, 6, n_samples)  # 1-5
    walc = np.random.randint(1, 6, n_samples)  # 1-5
    
    # Tạo biến tác động (influence factors) ngẫu nhiên cho mỗi yếu tố
    # Mỗi lần chạy sẽ tạo ra một bộ trọng số khác nhau mà mô hình phải học
    random_weights = np.random.uniform(-1, 1, 15)
    random_weights[0:5] = np.abs(random_weights[0:5])  # Đảm bảo một số yếu tố luôn tích cực
    
    # Tạo G1 với mối quan hệ phi tuyến và ngẫu nhiên
    g1_base = 10
    g1_components = [
        studytime * random_weights[0],
        -failures * np.abs(random_weights[1]),
        -absences * np.abs(random_weights[2]) * 0.1,
        np.sin(studytime) * random_weights[3],
        famsup * random_weights[4],
        internet * random_weights[5],
        np.exp(higher - 0.5) * random_weights[6],
        np.log1p(health) * random_weights[7]
    ]
    
    g1 = np.clip(g1_base + sum(g1_components) + np.random.normal(0, 2, n_samples), 0, 20)
    
    # Tạo G2 với mối quan hệ phi tuyến và phụ thuộc G1
    g2_base = g1 * 0.5 + 5
    g2_components = [
        studytime * random_weights[8] * 0.8,
        -failures * np.abs(random_weights[9]),
        -absences * np.abs(random_weights[10]) * 0.1,
        famsup * random_weights[11] * 0.7,
        internet * random_weights[12],
        np.sqrt(health) * random_weights[13],
        np.square(g1/20) * random_weights[14] * 5
    ]
    
    g2 = np.clip(g2_base + sum(g2_components) + np.random.normal(0, 1.5, n_samples), 0, 20)
    
    # Tạo G3 với mối quan hệ phức tạp với G1, G2 và các yếu tố khác
    # Tạo bộ trọng số mới để G3 có mối quan hệ khác với G1, G2
    random_weights_g3 = np.random.uniform(-1, 1, 20)
    random_weights_g3[0:2] = np.abs(random_weights_g3[0:2])  # G1, G2 luôn tích cực
    
    g3_components = [
        g1 * np.abs(random_weights_g3[0]) * 0.3,
        g2 * np.abs(random_weights_g3[1]) * 0.5,
        (studytime - 1) * random_weights_g3[2],
        -failures * np.abs(random_weights_g3[3]) * 1.1,
        -absences * np.abs(random_weights_g3[4]) * 0.05,
        famsup * random_weights_g3[5],
        paid * random_weights_g3[6],
        activities * random_weights_g3[7],
        higher * random_weights_g3[8],
        internet * random_weights_g3[9],
        -goout * np.abs(random_weights_g3[10]) * 0.1 if random_weights_g3[10] < 0 else 0,
        health * random_weights_g3[11] * 0.1,
        -np.mean([dalc, walc], axis=0) * np.abs(random_weights_g3[12]) * 0.2,
        # Các tương tác phi tuyến
        internet * studytime * random_weights_g3[13] * 0.15,
        famsup * absences * random_weights_g3[14] * (-0.01),
        np.square(g2/20) * random_weights_g3[15] * 3,
        np.sin(g1 * np.pi / 20) * random_weights_g3[16] * 2,
        np.log1p(absences) * random_weights_g3[17] * (-0.3),
        np.exp(higher - 0.5) * random_weights_g3[18] * 0.5,
        np.tanh(internet) * random_weights_g3[19] * 1.5
    ]
    
    g3 = np.clip(10 + sum(g3_components) + np.random.normal(0, 1.2, n_samples), 0, 20)
    
    # Tạo dataframe với tất cả các đặc trưng
    data = pd.DataFrame({
        'age': age,
        'studytime': studytime,
        'failures': failures,
        'absences': absences,
        'schoolsup': schoolsup,
        'famsup': famsup,
        'paid': paid,
        'activities': activities,
        'higher': higher,
        'internet': internet,
        'romantic': romantic,
        'freetime': freetime, 
        'goout': goout,
        'health': health,
        'dalc': dalc,
        'walc': walc,
        'G1': g1,
        'G2': g2,
        'G3': g3
    })
    
    # Chuẩn hóa các đặc trưng số
    numeric_features = ['age', 'studytime', 'failures', 'absences', 
                       'freetime', 'goout', 'health', 'dalc', 'walc']
    scaler = MinMaxScaler()
    data[numeric_features] = scaler.fit_transform(data[numeric_features])
    
    # Tách đặc trưng và nhãn
    X = data.drop(columns=['G3'])
    y = data['G3']
    
    # Hiển thị các trọng số được tạo ngẫu nhiên để tham khảo
    print("\nRandom weights used for G3:")
    features = ['g1', 'g2', 'studytime', 'failures', 'absences', 'famsup', 'paid', 
                'activities', 'higher', 'internet', 'goout', 'health', 'alcohol',
                'internet*study', 'famsup*absence', 'g2^2', 'sin(g1)', 'log(absence)', 
                'exp(higher)', 'tanh(internet)']
    for i, (feat, weight) in enumerate(zip(features, random_weights_g3)):
        print(f"{feat}: {weight:.4f}")
        
    # Phân tích tương quan để kiểm tra dữ liệu
    corr = data.corr()['G3'].sort_values(ascending=False)
    print("\nCorrelations with G3 in synthetic data:")
    print(corr.head(10))
    
    # Tạo metadata
    metadata = {
        'binary_cols': ['schoolsup', 'famsup', 'paid', 'activities', 
                        'higher', 'internet', 'romantic'],
        'non_binary_cat': [],
        'numeric_features': numeric_features,
        'scaler': scaler,
        'feature_names': list(X.columns),
        'note': 'Generated with complex nonlinear relationships - no predefined weights'
    }
    
    return X, y, metadata

def main(use_real_data=True):
    """
    Quy trình chính để xử lý dữ liệu, phân tích và huấn luyện mô hình
    """
    # Bước 1: Tải và xử lý dữ liệu
    if use_real_data:
        X, y, metadata = load_and_process_data()
        
        if X is None:
            print("Falling back to synthetic data generation")
            #X, y, metadata = generate_synthetic_data(2000)
            X, y, metadata = generate_synthetic_data_nonlinear(2000)
    else:
        print("Using synthetic data as per configuration")
        #X, y, metadata = generate_synthetic_data(2000)
        X, y, metadata = generate_synthetic_data_nonlinear(2000)
    
    # Bước 2: Phân tích tương quan
    top_features, correlations = analyze_correlations(X, y, metadata)
    
    # Bước 3: Huấn luyện mô hình hồi quy
    model = train_regression_model(X, y, metadata, top_features)
    
    # Bước 4: Lưu mô hình và metadata
    save_model_and_metadata(model, metadata)
    
    print("\nModel training and analysis complete!")

if __name__ == "__main__":
    main(use_real_data=True)  # Set to False to use synthetic data