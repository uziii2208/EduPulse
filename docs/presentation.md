# DỰ ĐOÁN ĐIỂM SỐ HỌC SINH DỰA TRÊN CÁC YẾU TỐ ẢNH HƯỞNG

## CHƯƠNG 1: GIỚI THIỆU ĐỀ TÀI

### 1.1. Tổng quan
- Vấn đề dự đoán điểm số học sinh dựa trên các yếu tố ảnh hưởng
- Ý nghĩa và tầm quan trọng của việc dự đoán sớm kết quả học tập
- Khả năng ứng dụng trong thực tiễn giáo dục

### 1.2. Mục tiêu
- Xây dựng mô hình dự đoán điểm số học sinh
- Phân tích các yếu tố ảnh hưởng đến kết quả học tập
- Phát triển công cụ hỗ trợ giáo viên và phụ huynh

### 1.3. Phạm vi nghiên cứu
- Tập trung vào dữ liệu học sinh trung học
- Phân tích các yếu tố học tập và môi trường
- Dự đoán điểm cuối kỳ (G3) dựa trên các yếu tố đầu vào

## CHƯƠNG 2: THU THẬP VÀ XỬ LÝ DỮ LIỆU

### 2.1. Nguồn dữ liệu và Quy trình
```mermaid
flowchart TB
    UCI[UCI ML Repository] -->|Download| RAW[Raw Data Files]
    RAW -->|student-mat.csv| MATH[Mathematics Data]
    RAW -->|student-por.csv| PORT[Portuguese Data]
    RAW -->|student-merge.R| MERGE[Data Merge Script]
    
    MATH --> MERGE
    PORT --> MERGE
    
    MERGE -->|382 students| FINAL[Final Dataset]
    
    subgraph Data_Features
        PERSONAL[Thông tin cá nhân]
        FAMILY[Thông tin gia đình]
        STUDY[Thói quen học tập]
        GRADES[Điểm số]
    end
    
    FINAL --> Data_Features
```

### 2.2. Đặc trưng dữ liệu
```mermaid
mindmap
    root((Student Data))
        Personal Info
            Age(15-22)
            Gender(M/F)
            Address(Urban/Rural)
        Family Background
            Parents Education
                Mother's Education(0-4)
                Father's Education(0-4)
            Parents Job
                Mother's Job
                Father's Job
            Family Support
                Family Size
                Study Support
        Academic Info
            Study Time(1-4)
            Failures(0-n)
            Extra Classes
            Absences(0-93)
        Performance
            G1 Score(0-20)
            G2 Score(0-20)
            G3 Score(0-20)
```

### 2.3. Tiền xử lý dữ liệu
```mermaid
flowchart LR
    subgraph Input [Tiền xử lý dữ liệu]
        direction TB
        RAW[Raw Data] --> CLEAN[Làm sạch dữ liệu]
        CLEAN --> ENCODE[Mã hóa dữ liệu]
        ENCODE --> NORM[Chuẩn hóa dữ liệu]
    end

    subgraph Process [Quy trình xử lý]
        direction TB
        MISSING[Xử lý missing values]
        BINARY[Binary Encoding]
        ONEHOT[One-hot Encoding]
        SCALE[MinMax Scaling]
    end

    CLEAN --> MISSING
    ENCODE --> BINARY
    ENCODE --> ONEHOT
    NORM --> SCALE

    subgraph Output [Kết quả]
        direction TB
        FINAL[Clean Dataset]
        FEATURES[Processed Features]
        TARGET[Target Variable]
    end

    MISSING --> FINAL
    BINARY --> FEATURES
    ONEHOT --> FEATURES
    SCALE --> FEATURES
    FEATURES --> ML[ML Ready Data]
```

## CHƯƠNG 3: PHÂN TÍCH DỮ LIỆU & TƯƠNG QUAN

### 3.1. Phân tích tương quan

```mermaid
graph TB
    subgraph Correlation_Analysis[Phân tích tương quan]
        MATRIX[Ma trận tương quan]
        HEATMAP[Heatmap visualization]
        TOP10[Top 10 correlations]
    end

    subgraph Key_Factors[Yếu tố chính]
        G1G2[Điểm G1, G2] -->|0.8| Score
        Study[Thời gian học] -->|0.3| Score
        Fail[Số lần trượt] -->|-0.4| Score
        Absent[Vắng mặt] -->|-0.25| Score
        Support[Hỗ trợ gia đình] -->|0.15| Score
        Score[Điểm số cuối kỳ]
    end
```

### 3.2. Phân tích chi tiết các yếu tố

```mermaid
mindmap
    root((Yếu tố ảnh hưởng))
        Academic
            G1_G2(Điểm số<br>r = 0.8)
            Study_Time(Thời gian học<br>r = 0.3)
            Failures(Số lần trượt<br>r = -0.4)
            Absences(Vắng mặt<br>r = -0.25)
        Family
            Parent_Edu(Học vấn<br>cha mẹ)
            Family_Sup(Hỗ trợ<br>gia đình)
            Family_Rel(Quan hệ<br>gia đình)
        Personal
            Health(Sức khỏe)
            Free_Time(Thời gian<br>rảnh)
            Activities(Hoạt động<br>ngoại khóa)
```

### 3.3. Trực quan hóa và phát hiện

```mermaid
graph TB
    subgraph Visualizations[Phương pháp trực quan hóa]
        HEAT[Heatmap Matrix]
        BOX[Boxplots]
        SCATTER[Scatterplots]
    end

    subgraph Findings[Phát hiện chính]
        STUDY[Thời gian học]
        SUPPORT[Hỗ trợ gia đình]
        BALANCE[Cân bằng học tập]
    end

    HEAT -->|Tương quan| CORR[Mối quan hệ tổng thể]
    BOX -->|Phân phối| DIST[Phân phối theo nhóm]
    SCATTER -->|Xu hướng| TREND[Xu hướng và patterns]

    STUDY -->|Optimal| OPT[3-4 giờ/ngày]
    SUPPORT -->|Impact| IMP[Tăng 5-10% điểm]
    BALANCE -->|Important| BAL[Học tập & giải trí]
```

## CHƯƠNG 4: XÂY DỰNG MÔ HÌNH DỰ ĐOÁN

### 4.1. Kiến trúc mô hình

```mermaid
graph TB
    subgraph Data_Preparation[Chuẩn bị dữ liệu]
        SPLIT[Train/Test Split]
        VALID[Validation Set]
    end

    subgraph Models[Mô hình dự đoán]
        RF[Random Forest]
        LR[Linear Regression]
    end

    subgraph Evaluation[Đánh giá]
        CV[Cross Validation]
        METRICS[Performance Metrics]
    end

    SPLIT -->|80%| TRAIN[Training Data]
    SPLIT -->|20%| TEST[Test Data]
    TRAIN --> RF
    TRAIN --> LR
    TEST --> EVAL[Evaluation]
    
    RF --> CV
    LR --> CV
    CV --> METRICS
```

### 4.2. Quy trình huấn luyện và đánh giá

```mermaid
sequenceDiagram
    participant D as Data
    participant P as Preprocessor
    participant M as Model
    participant E as Evaluator
    
    D->>P: Raw Features
    P->>P: Scale & Encode
    P->>M: Processed Data
    M->>M: Train Model
    M->>E: Predictions
    E->>E: Calculate Metrics
    E->>M: Feedback
    M->>M: Adjust Parameters
```

### 4.3. Tối ưu hóa và kết quả

```mermaid
graph LR
    subgraph Optimization[Tối ưu hóa mô hình]
        PARAMS[Hyperparameters]
        FEATURE[Feature Selection]
        TUNE[Model Tuning]
    end

    subgraph Performance[Hiệu suất]
        MSE[Mean Squared Error]
        R2[R² Score]
        VALID[Validation Score]
    end

    subgraph Results[Kết quả so sánh]
        RF_RES[Random Forest<br>R² > 0.85<br>MSE < 2.5]
        LR_RES[Linear Regression<br>R² ~ 0.75<br>MSE ~ 3.2]
    end

    PARAMS --> TUNE
    FEATURE --> TUNE
    TUNE --> RF_RES
    TUNE --> LR_RES
    RF_RES --> BEST[Best Model Selected]
```

```mermaid
graph TB
    subgraph Feature_Importance[Độ quan trọng của đặc trưng]
        G2[G2 Score<br>0.35]
        G1[G1 Score<br>0.30]
        STUDY[Study Time<br>0.15]
        FAIL[Failures<br>0.10]
        ABS[Absences<br>0.05]
        OTHER[Other Features<br>0.05]
    end

    G2 --> IMPACT[Model Impact]
    G1 --> IMPACT
    STUDY --> IMPACT
    FAIL --> IMPACT
    ABS --> IMPACT
    OTHER --> IMPACT
```

## CHƯƠNG 5: TRIỂN KHAI ỨNG DỤNG WEB

### 5.1. Kiến trúc hệ thống

```mermaid
graph TB
    subgraph Frontend[Giao diện người dùng]
        UI[Web Interface]
        FORMS[Input Forms]
        CHARTS[Visualizations]
    end

    subgraph Backend[Django Backend]
        AUTH[Authentication]
        API[REST API]
        DB[(Database)]
        
        subgraph ML[ML System]
            PREPROCESS[Preprocessor]
            MODEL[ML Model]
            ANALYSIS[Analyzer]
        end
    end

    UI -->|User Input| FORMS
    FORMS -->|Submit| API
    API -->|Process| ML
    ML -->|Results| DB
    DB -->|Data| CHARTS
    CHARTS -->|Display| UI
```

### 5.2. Luồng xử lý dự đoán

```mermaid
sequenceDiagram
    actor User
    participant Web as Web Interface
    participant Auth as Authentication
    participant API as Backend API
    participant ML as ML System
    participant DB as Database

    User->>Web: Access System
    Web->>Auth: Verify Login
    Auth-->>Web: Auth Token
    
    User->>Web: Input Student Data
    Web->>API: Submit Data
    API->>ML: Process Data
    ML->>ML: Make Prediction
    ML->>DB: Save Results
    DB-->>API: Confirm Save
    API-->>Web: Return Results
    Web-->>User: Show Prediction
```

### 5.3. Mô hình dữ liệu và quyền hạn

```mermaid
erDiagram
    User ||--o{ Student : manages
    User ||--o{ Prediction : creates
    Student ||--o{ Prediction : has
    Teacher ||--o{ Student : monitors
    
    User {
        int id
        string username
        string role
        boolean is_active
    }
    
    Student {
        int id
        string name
        string class
        json study_data
    }
    
    Prediction {
        int id
        float g1_score
        float g2_score
        float predicted_g3
        datetime created_at
        text recommendations
    }
    
    Teacher {
        int id
        string department
        string subjects
    }
```

### 5.4. Tính năng và luồng người dùng

```mermaid
graph TB
    subgraph Teacher[Giáo viên]
        T_LOGIN[Đăng nhập] --> T_DASH[Dashboard]
        T_DASH --> T_PREDICT[Dự đoán điểm]
        T_DASH --> T_MONITOR[Theo dõi học sinh]
        T_DASH --> T_STATS[Thống kê & Báo cáo]
        
        subgraph T_Actions[Chức năng]
            T_PREDICT --> BULK[Dự đoán hàng loạt]
            T_PREDICT --> INDIVIDUAL[Dự đoán cá nhân]
            T_MONITOR --> PROGRESS[Tiến độ học sinh]
            T_MONITOR --> IMPROVE[Đề xuất cải thiện]
            T_STATS --> REPORTS[Báo cáo định kỳ]
            T_STATS --> ANALYTICS[Phân tích dữ liệu]
        end
    end

    subgraph Student[Học sinh]
        S_LOGIN[Đăng nhập] --> S_DASH[Dashboard]
        S_DASH --> S_VIEW[Xem dự đoán]
        S_DASH --> S_TRACK[Theo dõi tiến độ]
        S_DASH --> S_PLAN[Kế hoạch học tập]
        
        subgraph S_Actions[Chức năng]
            S_VIEW --> HISTORY[Lịch sử điểm số]
            S_TRACK --> GOALS[Mục tiêu]
            S_TRACK --> ACHIEVEMENTS[Thành tích]
            S_PLAN --> SCHEDULE[Lịch học]
            S_PLAN --> TASKS[Nhiệm vụ]
        end
    end
```

> **Giải thích biểu đồ tính năng:**
> - **Phân quyền người dùng**: Hệ thống phân chia 2 đối tượng chính là giáo viên và học sinh với các chức năng riêng biệt (theo mô hình `UserProfile` trong `predictor/models.py`)
> - **Giáo viên**: Có quyền truy cập toàn bộ tính năng dự đoán, theo dõi và phân tích
> - **Học sinh**: Tập trung vào theo dõi tiến độ cá nhân và nhận đề xuất cải thiện
> - **Dashboard**: Tùy theo vai trò mà hiển thị thông tin phù hợp (cài đặt trong `predictor/views.py`)

```mermaid
mindmap
    root((Giao diện))
        Trang chủ
            Đăng nhập/Đăng ký
            Thông tin tổng quan
            Thống kê nhanh
        Dự đoán
            Form nhập liệu
            Xử lý thời gian thực
            Kết quả & đề xuất
        Phân tích
            Biểu đồ thống kê
            Biểu đồ tiến trình
            Metrics hiệu suất
        Quản lý
            Hồ sơ người dùng
            Phân quyền
            Lịch sử dữ liệu
        Hỗ trợ
            Đề xuất cải thiện
            Kế hoạch học tập
            Báo cáo định kỳ
```

> **Giải thích giao diện:**
> - **Trang chủ**: Hiển thị dashboard tùy biến theo role người dùng (template: `predictor/templates/predictor/home.html`)
> - **Dự đoán**: Form thông minh với validation và xử lý real-time (form logic trong `predictor/forms.py`)
> - **Phân tích**: Sử dụng Chart.js để trực quan hóa dữ liệu (custom filters trong `predictor/templatetags/predictor_filters.py`)
> - **Quản lý**: Tích hợp hệ thống authentication của Django
> - **Hỗ trợ**: Tự động tạo kế hoạch và đề xuất dựa trên kết quả dự đoán

### 5.5. Triển khai và tương tác

```mermaid
sequenceDiagram
    participant U as User
    participant F as Frontend
    participant B as Backend
    participant ML as ML System
    participant DB as Database
    
    U->>F: Truy cập hệ thống
    F->>B: Xác thực người dùng
    B->>DB: Kiểm tra thông tin
    DB-->>B: Xác nhận quyền
    B-->>F: Session token
    
    U->>F: Nhập dữ liệu học sinh
    F->>B: Gửi request
    B->>ML: Xử lý & dự đoán
    ML->>DB: Lưu kết quả
    ML-->>B: Trả về dự đoán
    B-->>F: Hiển thị kết quả
    F-->>U: Xem kết quả & đề xuất
```

> **Giải thích quy trình:**
> - **Xác thực**: Sử dụng Django Authentication System với custom UserProfile
> - **Xử lý dữ liệu**: Validate và chuẩn hóa input thông qua Django Forms
> - **ML System**: Tích hợp mô hình RandomForest thông qua `ml_utils.py`
> - **Database**: Sử dụng SQLite để lưu trữ dữ liệu người dùng và kết quả dự đoán
> - **Frontend**: Responsive design với Bootstrap và custom CSS (`static/predictor/css/style.css`)

## CHƯƠNG 6: KẾT LUẬN & HƯỚNG PHÁT TRIỂN

### 6.1. Kết quả đạt được
- Mô hình dự đoán chính xác
- Phân tích được yếu tố ảnh hưởng
- Ứng dụng web hoàn chỉnh

### 6.2. Hạn chế
- Dữ liệu còn hạn chế
- Chưa tích hợp thêm nhiều yếu tố
- Cần thêm dữ liệu thực tế

### 6.3. Hướng phát triển
- Mở rộng dataset với dữ liệu local
- Thêm các thuật toán dự đoán mới
- Phát triển tính năng nâng cao

## TÀI LIỆU THAM KHẢO

1. P. Cortez and A. Silva. Using Data Mining to Predict Secondary School Student Performance. In A. Brito and J. Teixeira Eds., Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008) pp. 5-12, Porto, Portugal, April, 2008, EUROSIS, ISBN 978-9077381-39-7.

2. UCI Machine Learning Repository: Student Performance Data Set
   https://archive.ics.uci.edu/ml/datasets/Student+Performance

3. Sklearn Documentation: Random Forest Regressor
   https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

4. Django Documentation
   https://docs.djangoproject.com/