import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

class CustomRandomForestRegressor(RandomForestRegressor):
    def __init__(self, feature_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.feature_weights = {
            'G2': 2.0,        # Điểm G2 rất quan trọng (tích cực)
            'G1': 1.0,        # Điểm G1 quan trọng (tích cực)
            'studytime': 0.5, # Thời gian học ảnh hưởng tích cực
            'traveltime': -0.3,  # Thời gian đi lại ảnh hưởng tiêu cực
            'failures': -0.5, # Số lần thất bại ảnh hưởng tiêu cực
            'absences': -0.2, # Vắng mặt ảnh hưởng tiêu cực
            'Dalc': -0.3,     # Rượu ngày thường ảnh hưởng tiêu cực
            'Walc': -0.2,     # Rượu cuối tuần ảnh hưởng tiêu cực
            'health': 0.2     # Sức khỏe tốt ảnh hưởng tích cực
        }

    def fit(self, X, y, sample_weight=None):
        if self.feature_weights:
            # Apply feature weights during training
            sample_weight = np.ones(X.shape[0])
            for feature, weight in self.feature_weights.items():
                if feature in X.columns:
                    sample_weight *= (1 + weight * abs(X[feature]))
        
        # Apply sample weights to give more importance to mid-range predictions
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])
        
        # Give more weight to samples where G2 is not extremely different from G1
        if 'G1' in X.columns and 'G2' in X.columns:
            g1_g2_diff = abs(X['G1'] - X['G2'])
            sample_weight *= np.exp(-g1_g2_diff * 0.1)  # Reduce weight for large differences
            
        return super().fit(X, y, sample_weight=sample_weight)
    
    def predict(self, X):
        """Dự đoán với hiệu chỉnh thông minh"""
        # Dự đoán ban đầu
        predictions = super().predict(X)
        
        # Điều chỉnh dự đoán dựa trên kiến thức lĩnh vực
        for i in range(len(predictions)):
            # --- Hiệu chỉnh dựa trên điểm G1, G2 ---
            if 'G1' in X.columns and 'G2' in X.columns:
                g1 = X['G1'].iloc[i]
                g2 = X['G2'].iloc[i]
                
                # 1. Nếu cả G1 và G2 đều cao (≥ 16)
                if g1 >= 16 and g2 >= 16:
                    # Điểm cuối cùng nên ít nhất bằng điểm cao nhất
                    predictions[i] = max(predictions[i], max(g1, g2))
                
                # 2. Nếu G2 > G1 (xu hướng tăng) và G2 cao (≥ 14)
                if g2 > g1 and g2 >= 14:
                    # Điểm cuối cùng nên ít nhất bằng G2
                    predictions[i] = max(predictions[i], g2)
                
                # 3. Nếu G2 < G1 (xu hướng giảm) nhưng G1 và G2 đều trên trung bình (≥ 10)
                if g2 < g1 and g1 >= 10 and g2 >= 10:
                    # Điểm cuối cùng có thể giảm nhưng không nên thấp hơn G2 quá nhiều
                    min_g3 = g2 - (g1 - g2) * 0.5  # Giảm không quá 50% của chênh lệch
                    predictions[i] = max(predictions[i], min_g3)
            
            # --- Hiệu chỉnh dựa trên tiêu thụ rượu (Dalc, Walc) ---
            if 'Dalc' in X.columns and 'Walc' in X.columns:
                dalc = X['Dalc'].iloc[i]
                walc = X['Walc'].iloc[i]
                
                # Tính điểm tiêu thụ rượu tổng hợp (trọng số Dalc cao hơn vì ảnh hưởng tới ngày học)
                alcohol_score = (dalc * 0.6 + walc * 0.4)
                
                # Áp dụng hiệu chỉnh theo thang điểm:
                # - Mức tiêu thụ rượu thấp (1-2): không ảnh hưởng hoặc ảnh hưởng ít
                # - Mức trung bình (3): bắt đầu ảnh hưởng tiêu cực
                # - Mức cao (4-5): ảnh hưởng tiêu cực rõ rệt
                
                if alcohol_score > 2:
                    # Giảm điểm dự đoán theo mức độ sử dụng rượu
                    # Công thức: giảm 3-8% điểm tùy theo mức độ rượu
                    alcohol_penalty = 0.03 * (alcohol_score - 2) * predictions[i]
                    predictions[i] = max(0, predictions[i] - alcohol_penalty)
            
            # --- Hiệu chỉnh dựa trên sức khỏe ---
            if 'health' in X.columns:
                health = X['health'].iloc[i]
                
                # Sức khỏe là yếu tố tích cực đối với học tập
                # - Sức khỏe kém (1-2): ảnh hưởng tiêu cực
                # - Sức khỏe trung bình (3): trung tính
                # - Sức khỏe tốt (4-5): ảnh hưởng tích cực
                
                if health > 3:
                    # Tăng điểm nếu sức khỏe tốt (2% mỗi điểm trên 3)
                    health_bonus = 0.02 * (health - 3) * predictions[i]
                    predictions[i] = min(20, predictions[i] + health_bonus)
                elif health < 3:
                    # Giảm điểm nếu sức khỏe kém (2% mỗi điểm dưới 3)
                    health_penalty = 0.02 * (3 - health) * predictions[i]
                    predictions[i] = max(0, predictions[i] - health_penalty)
            
            # --- Hiệu chỉnh dựa trên thời gian học ---
            if 'studytime' in X.columns:
                studytime = X['studytime'].iloc[i]
                
                # Thời gian học nhiều hơn thường dẫn đến kết quả tốt hơn
                # - Thời gian học ít (1): có thể ảnh hưởng tiêu cực
                # - Thời gian học trung bình (2-3): trung tính đến tích cực
                # - Thời gian học nhiều (4): ảnh hưởng tích cực
                
                if studytime >= 3:
                    # Tăng điểm cho học sinh chăm chỉ (3% mỗi cấp độ trên 2)
                    study_bonus = 0.03 * (studytime - 2) * predictions[i]
                    predictions[i] = min(20, predictions[i] + study_bonus)
                elif studytime == 1:
                    # Giảm điểm cho học sinh học quá ít
                    study_penalty = 0.03 * predictions[i]
                    predictions[i] = max(0, predictions[i] - study_penalty)
            
            # --- Hiệu chỉnh dựa trên số lần thất bại trước đó ---
            if 'failures' in X.columns:
                failures = X['failures'].iloc[i]
                
                # Số lần thất bại trước đó có tương quan tiêu cực mạnh với kết quả
                if failures > 0:
                    # Giảm điểm dựa trên số lần thất bại (5% mỗi lần)
                    failures_penalty = 0.05 * failures * predictions[i]
                    predictions[i] = max(0, predictions[i] - failures_penalty)
            
            # --- Hiệu chỉnh dựa trên sự vắng mặt ---
            if 'absences' in X.columns:
                absences = X['absences'].iloc[i]
                
                # Vắng mặt quá nhiều ảnh hưởng tiêu cực
                # Ngưỡng hợp lý có thể là khoảng 10 buổi
                if absences > 10:
                    # Giảm điểm 0.5% cho mỗi buổi vắng vượt quá 10
                    absences_penalty = 0.005 * (absences - 10) * predictions[i]
                    predictions[i] = max(0, predictions[i] - absences_penalty)
        
            # --- Hiệu chỉnh dựa trên thời gian di chuyển đến trường ---
            if 'traveltime' in X.columns:
                traveltime = X['traveltime'].iloc[i]
                
                # Traveltime cao (3-4: >30 phút) ảnh hưởng tiêu cực
                if traveltime >= 3:
                    # Giảm 2% điểm cho mỗi cấp độ traveltime trên 2
                    traveltime_penalty = 0.02 * (traveltime - 2) * predictions[i]
                    predictions[i] = max(0, predictions[i] - traveltime_penalty)
        
        # Đảm bảo kết quả nằm trong khoảng hợp lệ [0, 20]
        predictions = np.clip(predictions, 0, 20)
        
        return predictions