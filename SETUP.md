# Hướng Dẫn Cài Đặt Chi Tiết

## Yêu Cầu Hệ Thống
- Python 3.10 trở lên
- Git
- pip (Python package installer)

## Các Bước Cài Đặt

### 1. Clone dự án
```bash
git clone ....
cd NEW_DOAN_KTDL
```

### 2. Tạo môi trường ảo

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Cài đặt các gói phụ thuộc
```bash
pip install -r requirements.txt
```

### 4. Cài đặt cơ sở dữ liệu
```bash
python manage.py makemigrations
python manage.py migrate
```

### 5. Khởi tạo mô hình Machine Learning
```bash
python manage.py init_model
```

### 6. Tạo tài khoản admin (nếu chưa có)
```bash
python manage.py createsuperuser
```
Hoặc sử dụng tài khoản có sẵn:
- Username: admin
- Password: 6h1ehcjb!

### 7. Chạy máy chủ phát triển
```bash
python manage.py runserver
```

Sau khi chạy lệnh này, bạn có thể truy cập ứng dụng tại http://127.0.0.1:8000/

## Cấu Hình Môi Trường (Tùy Chọn)

Bạn có thể cấu hình các biến môi trường sau:

- `SECRET_KEY`: Khóa bảo mật Django
- `DEBUG`: Đặt là 'True' để bật chế độ debug hoặc 'False' cho môi trường sản xuất
- `ALLOWED_HOSTS`: Danh sách các host được phép truy cập, phân tách bằng dấu phẩy
- `STATIC_URL`: URL cho tài nguyên tĩnh
- `MEDIA_URL`: URL cho tệp phương tiện

## Xử Lý Sự Cố Thường Gặp

### Lỗi khi cài đặt các gói Python
Nếu gặp lỗi khi cài đặt các gói phụ thuộc:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Lỗi khi khởi tạo mô hình ML
Kiểm tra xem bạn đã cài đặt các gói scikit-learn, pandas và numpy chưa:
```bash
pip install scikit-learn pandas numpy
```

### Lỗi kết nối cơ sở dữ liệu
Đảm bảo bạn đã chạy các lệnh migration:
```bash
python manage.py makemigrations predictor
python manage.py migrate
```

### Lỗi khi chạy máy chủ
Kiểm tra cổng 8000 có đang được sử dụng không. Nếu có, bạn có thể chỉ định cổng khác:
```bash
python manage.py runserver 8080
```

## Các Scripts Cài Đặt Tự Động

### Windows (setup.ps1)
```powershell
# Tạo và kích hoạt môi trường ảo
python -m venv venv
.\venv\Scripts\activate

# Cài đặt dependencies
pip install -r requirements.txt

# Cài đặt database
python manage.py makemigrations
python manage.py migrate

# Khởi tạo mô hình ML
python manage.py init_model

# Thông báo hoàn thành
Write-Host "Cài đặt hoàn tất! Để chạy server, sử dụng lệnh: python manage.py runserver"
```

### Linux/Mac (setup.sh)
```bash
#!/bin/bash

# Tạo và kích hoạt môi trường ảo
python3 -m venv venv
source venv/bin/activate

# Cài đặt dependencies
pip install -r requirements.txt

# Cài đặt database
python manage.py makemigrations
python manage.py migrate

# Khởi tạo mô hình ML
python manage.py init_model

# Thông báo hoàn thành
echo "Cài đặt hoàn tất! Để chạy server, sử dụng lệnh: python manage.py runserver"
```

## Hỗ Trợ Thêm
Nếu bạn gặp vấn đề khi cài đặt, vui lòng tạo issue trên GitHub hoặc liên hệ với nhóm phát triển để được hỗ trợ.