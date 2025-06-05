from predictor.ml_utils import CustomRandomForestRegressor
import pandas as pd
import numpy as np

# Tạo mô hình test
model = CustomRandomForestRegressor(n_estimators=10, random_state=42)

# Tạo dữ liệu giả
X_train = pd.DataFrame({
    'G1': [10, 12, 15], 
    'G2': [11, 13, 16],
    'studytime': [2, 3, 4],
    'traveltime': [1, 2, 4],
    'failures': [0, 1, 0],
    'absences': [2, 5, 10],
    'Dalc': [1, 2, 3],
    'Walc': [2, 3, 4],
    'health': [3, 4, 2]
})
y_train = np.array([12, 14, 17])

# Huấn luyện
model.fit(X_train, y_train)

# Tạo vài bản ghi test với traveltime tăng dần
X_test = pd.DataFrame({
    'G1': [12, 12, 12, 12],
    'G2': [14, 14, 14, 14],
    'studytime': [3, 3, 3, 3],
    'traveltime': [1, 2, 3, 4],  # Tăng dần traveltime
    'failures': [0, 0, 0, 0],
    'absences': [5, 5, 5, 5],
    'Dalc': [1, 1, 1, 1],
    'Walc': [2, 2, 2, 2],
    'health': [4, 4, 4, 4]
})

# Dự đoán và in kết quả
predictions = model.predict(X_test)
for i, travel in enumerate([1, 2, 3, 4]):
    print(f"traveltime={travel}, prediction={predictions[i]}")