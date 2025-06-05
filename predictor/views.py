import django
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required, user_passes_test
from django.http import HttpResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from matplotlib import colors
from matplotlib.table import Table
import xlsxwriter
from reportlab.platypus import TableStyle, SimpleDocTemplate, Paragraph, Spacer  # Add this import
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import landscape, letter
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from predictor.forms import ActionCompletionForm, StudentPredictionForm, ActualScoreForm, CustomAuthenticationForm, UserRegistrationForm
from predictor.models import StudentPrediction, UserProfile, ImprovementAction
from .ml_utils import CustomRandomForestRegressor
import joblib
import numpy as np
import pandas as pd
import os
import io
from django.db.models import Avg, Count, Max, Min, StdDev
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Sử dụng backend không cần GUI
import matplotlib.pyplot as plt
import base64
from django.db.models import Count, Avg, FloatField
from django.db.models.functions import Cast
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'NEW_DOAN_KTDL.settings')
django.setup()
def load_model():
    try:
        model_dir = Path(__file__).parent / 'ml_model'
        model_path = model_dir / 'student_score_model.pkl'
        
        if not model_path.exists():
            model_path = model_dir / 'model.joblib'  # Fallback
            
        if model_path.exists():
            model_package = joblib.load(model_path)
            if isinstance(model_package, dict):
                model = model_package.get('model')
                metadata = model_package.get('metadata', {})
                return model, metadata
        
        return None, None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

@login_required
def home(request):
    # Khởi tạo các biến ngay từ đầu
    prediction = None
    recommendation = None
    result_category = None
    prediction_id = None
    score_trend = None
    print("check")
    if request.method == 'POST':
        form = StudentPredictionForm(request.POST)
        
        # Thêm debug để kiểm tra dữ liệu POST
        print("Form data:", request.POST)
        print("Form is valid:", form.is_valid())
        
        # Thêm logic xử lý form không valid
        if not form.is_valid():
            print("Form errors:", form.errors)
            # Hiển thị thông báo lỗi chi tiết cho từng trường
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f"Lỗi ở trường '{field}': {error}")
            # Vẫn tiếp tục render form với lỗi được hiển thị
            return render(request, 'predictor/home.html', {'form': form})
            
        # Tiếp tục xử lý nếu form hợp lệ
        if form.is_valid():
            print("Form is valid")
            model, metadata = load_model()
            if model is None:
                messages.error(request, 'Lỗi: Không thể tải mô hình dự đoán')
                return render(request, 'predictor/home.html', {'form': form})

            # Chuẩn bị dữ liệu với tất cả các trường
            try:
                input_data = pd.DataFrame({
                    'age': [form.cleaned_data['age']],
                    'Medu': [form.cleaned_data['Medu']],
                    'Fedu': [form.cleaned_data['Fedu']],
                    'traveltime': [form.cleaned_data['traveltime']],
                    'studytime': [form.cleaned_data['studytime']],
                    'failures': [form.cleaned_data['failures']],
                    'schoolsup': [0],  # Default value
                    'famsup': [int(form.cleaned_data['famsup'])],
                    'paid': [int(form.cleaned_data['paid'])],
                    'activities': [int(form.cleaned_data['activities'])],
                    'nursery': [int(form.cleaned_data['nursery'])],
                    'higher': [int(form.cleaned_data['higher'])],
                    'internet': [int(form.cleaned_data['internet'])],
                    'romantic': [int(form.cleaned_data['romantic'])],
                    'famrel': [form.cleaned_data['famrel']],
                    'freetime': [form.cleaned_data['freetime']],
                    'goout': [form.cleaned_data['goout']],
                    'Dalc': [form.cleaned_data['Dalc']],
                    'Walc': [form.cleaned_data['Walc']],
                    'health': [form.cleaned_data['health']],
                    'absences': [form.cleaned_data['absences']],
                    'G1': [form.cleaned_data['g1_score']],
                    'G2': [form.cleaned_data['g2_score']],
                })
                print(f"Input data: {input_data}")
            except KeyError as e:
                messages.error(request, f'Lỗi: Trường dữ liệu {str(e)} không tồn tại trong form')
                return render(request, 'predictor/home.html', {'form': form})
            except Exception as e:
                messages.error(request, f'Lỗi khi xử lý dữ liệu: {str(e)}')
                return render(request, 'predictor/home.html', {'form': form})

            try:
                # Thêm one-hot encoding cho Mjob và Fjob
                mjob_cols = pd.get_dummies(pd.Series([form.cleaned_data['Mjob']]), prefix='Mjob')
                fjob_cols = pd.get_dummies(pd.Series([form.cleaned_data['Fjob']]), prefix='Fjob')
                
                # Kết hợp tất cả các features
                input_data = pd.concat([input_data, mjob_cols, fjob_cols], axis=1)
                
                # Đảm bảo thứ tự cột giống với lúc training
                if 'feature_names' in metadata:
                    missing_cols = set(metadata['feature_names']) - set(input_data.columns)
                    for col in missing_cols:
                        # Sử dụng giá trị mặc định phù hợp với loại cột
                        if col in metadata.get('binary_cols', []):
                            input_data[col] = 0  # Giá trị mặc định cho biến nhị phân
                        elif any(col.startswith(prefix) for prefix in metadata.get('non_binary_cat', [])):
                            input_data[col] = 0  # Giá trị mặc định cho one-hot encoding
                        else:
                            input_data[col] = 0  # Giá trị mặc định cho các cột khác
                        
                    # Đảm bảo thứ tự cột đúng
                    input_data = input_data.reindex(columns=metadata['feature_names'], fill_value=0)
                
                # Chuẩn hóa các đặc trưng số nếu cần
                if 'numeric_features' in metadata and 'scaler' in metadata:
                    numeric_features = [f for f in metadata['numeric_features'] if f in input_data.columns]
                    if numeric_features:
                        input_data[numeric_features] = metadata['scaler'].transform(input_data[numeric_features])
            except Exception as e:
                import traceback
                print(f"Error in data preparation: {e}")
                print(traceback.format_exc())
                messages.error(request, f'Lỗi khi chuẩn bị dữ liệu: {str(e)}')
                return render(request, 'predictor/home.html', {'form': form})

            # Dự đoán
            try:
                prediction = float(model.predict(input_data)[0])
                # Đảm bảo kết quả nằm trong thang điểm hợp lý (0-20)
                prediction = max(0, min(20, prediction))
            except Exception as e:
                import traceback
                print(f"Error in prediction: {e}")
                print(traceback.format_exc())
                messages.error(request, f'Lỗi khi dự đoán: {str(e)}')
                prediction = None
                return render(request, 'predictor/home.html', {'form': form})
            
            try:
                # Tạo đề xuất dựa trên dữ liệu đầu vào
                recommendations = []
                
                # Học tập
                if form.cleaned_data['g2_score'] < 10:
                    recommendations.append('Cải thiện điểm số kỳ 2 thông qua việc học tập chăm chỉ hơn')
                if form.cleaned_data['failures'] > 0:
                    recommendations.append('Tập trung khắc phục các môn học đã thất bại trước đây')
                if form.cleaned_data['g1_score'] < 10:
                    recommendations.append('Cải thiện điểm số kỳ 1 để tạo nền tảng tốt')
                if form.cleaned_data['absences'] > 5:
                    recommendations.append('Giảm số lần vắng mặt để không bỏ lỡ bài học')
                if form.cleaned_data['studytime'] < 10:
                    recommendations.append('Tăng thời gian học tập lên ít nhất 10 giờ/tuần')

                # Hoạt động và sức khỏe
                if not form.cleaned_data['activities']:
                    recommendations.append('Tham gia các hoạt động ngoại khóa để phát triển toàn diện')
                if form.cleaned_data['freetime'] >= 4 and form.cleaned_data['goout'] >= 4:
                    recommendations.append('Cân bằng thời gian giữa học tập và giải trí')
                if form.cleaned_data['Dalc'] > 2 or form.cleaned_data['Walc'] > 2:
                    recommendations.append('Hạn chế sử dụng rượu bia để bảo vệ sức khỏe')
                if form.cleaned_data['health'] < 3:
                    recommendations.append('Cần chú ý cải thiện sức khỏe')

                # Hỗ trợ giáo dục
                # Kiểm tra nếu trường paid_classes tồn tại trong form, nếu không dùng trường paid
                has_paid_classes = 'paid_classes' in form.cleaned_data
                if (not form.cleaned_data.get('paid_classes' if has_paid_classes else 'paid')) and form.cleaned_data['g2_score'] < 12:
                    recommendations.append('Cân nhắc việc tham gia các lớp học thêm')
                if form.cleaned_data['traveltime'] >= 3:
                    recommendations.append('Cân nhắc tìm nơi ở gần trường hơn hoặc phương tiện di chuyển phù hợp')
                
                recommendation = ' | '.join(recommendations) if recommendations else 'Tiếp tục duy trì phương pháp học tập hiện tại'

                # Tính toán xu hướng điểm số
                score_trend = form.cleaned_data['g2_score'] - form.cleaned_data['g1_score']
                if score_trend > 0:
                    recommendation = f"Xu hướng tích cực (+{score_trend} điểm) | " + recommendation
                elif score_trend < 0:
                    recommendation = f"Xu hướng cần cải thiện ({score_trend} điểm) | " + recommendation

                # Phân loại kết quả
                if prediction is not None:
                    if prediction >= 14:
                        result_category = "Xuất sắc"
                    elif prediction >= 12:
                        result_category = "Khá"  
                    elif prediction >= 10:
                        result_category = "Trung bình"
                    else:
                        result_category = "Cần cải thiện"
                
                # Nếu dự đoán thành công, lưu kết quả vào cơ sở dữ liệu
                if prediction is not None:
                    try:
                        student_prediction = StudentPrediction(
                            student_name=form.cleaned_data.get('student_name', ''),
                            student_id=form.cleaned_data.get('student_id', ''),
                            g1_score=form.cleaned_data['g1_score'],
                            g2_score=form.cleaned_data['g2_score'],
                            studytime=form.cleaned_data['studytime'],
                            failures=form.cleaned_data['failures'],
                            absences=form.cleaned_data['absences'],
                            dalc=form.cleaned_data['Dalc'],
                            walc=form.cleaned_data['Walc'],
                            health=form.cleaned_data['health'],
                            predicted_score=prediction,
                            result_category=result_category,
                            recommendation=recommendation,
                            user=request.user  # Thêm dòng này
                        )
                        student_prediction.save()
                        prediction_id = student_prediction.id
                        messages.success(request, "Đã dự đoán điểm số thành công!")
                    except Exception as e:
                        import traceback
                        print(f"Error saving prediction: {e}")
                        print(traceback.format_exc())
                        messages.error(request, f'Lỗi khi lưu kết quả dự đoán: {str(e)}')
            except Exception as e:
                import traceback
                print(f"Error in recommendation generation: {e}")
                print(traceback.format_exc())
                messages.error(request, f'Lỗi khi tạo khuyến nghị: {str(e)}')
    else:
        print("GET request - rendering empty form")
        form = StudentPredictionForm()

    # Thêm prediction_id vào context để có thể dùng cho form nhập điểm thực tế
    return render(request, 'predictor/home.html', {
        'form': form,
        'prediction': round(prediction, 2) if prediction is not None else None,
        'recommendation': recommendation,
        'result_category': result_category,
        'prediction_id': prediction_id
    })

@login_required
def update_actual_score(request, prediction_id):
    prediction = get_object_or_404(StudentPrediction, id=prediction_id)
    
    if request.method == 'POST':
        form = ActualScoreForm(request.POST)
        if form.is_valid():
            prediction.actual_score = form.cleaned_data['actual_score']
            prediction.save()  # save() sẽ tự động tính prediction_diff
            messages.success(request, 'Điểm thực tế đã được cập nhật thành công!')
            return redirect('prediction_stats')
    else:
        form = ActualScoreForm()
    
    return render(request, 'predictor/update_score.html', {
        'form': form,
        'prediction': prediction
    })

@login_required
def prediction_stats(request):
    # Chỉ lấy dự đoán của người dùng hiện tại
    print(f"User: {request}")
    predictions_with_actual = StudentPrediction.objects.filter(
        actual_score__isnull=False,
        user=request.user
    )
    
    # Nếu không có dữ liệu, hiển thị thông báo
    if not predictions_with_actual.exists():
        messages.info(request, 'Chưa có dữ liệu điểm thực tế để thống kê.')
        return render(request, 'predictor/stats.html', {'no_data': True})
    
    # Tính các thống kê
    stats = {
        'total_predictions': predictions_with_actual.count(),
        'avg_predicted': predictions_with_actual.aggregate(Avg('predicted_score'))['predicted_score__avg'],
        'avg_actual': predictions_with_actual.aggregate(Avg('actual_score'))['actual_score__avg'],
        'avg_diff': predictions_with_actual.aggregate(Avg('prediction_diff'))['prediction_diff__avg'],
        'max_diff': predictions_with_actual.aggregate(Max('prediction_diff'))['prediction_diff__max'],
        'min_diff': predictions_with_actual.aggregate(Min('prediction_diff'))['prediction_diff__min'],
        'std_diff': predictions_with_actual.aggregate(StdDev('prediction_diff'))['prediction_diff__stddev'],
    }
    
    # Tính tỷ lệ dự đoán chính xác (sai số < 1.0)
    accurate_predictions = predictions_with_actual.filter(prediction_diff__lt=1.0).count()
    stats['accuracy_rate'] = (accurate_predictions / stats['total_predictions']) * 100 if stats['total_predictions'] > 0 else 0
    
    # Thống kê theo từng loại kết quả
    category_stats = []
    categories = predictions_with_actual.values('result_category').annotate(count=Count('id')).order_by('-count')
    
    for category in categories:
        cat_name = category['result_category']
        cat_predictions = predictions_with_actual.filter(result_category=cat_name)
        cat_avg_diff = cat_predictions.aggregate(Avg('prediction_diff'))['prediction_diff__avg']
        
        category_stats.append({
            'name': cat_name,
            'count': category['count'],
            'avg_diff': cat_avg_diff
        })
    
    # Lấy danh sách 10 dự đoán gần nhất có điểm thực tế
    recent_predictions = predictions_with_actual.order_by('-updated_at')[:10]
    
    return render(request, 'predictor/stats.html', {
        'stats': stats,
        'category_stats': category_stats,
        'recent_predictions': recent_predictions,
        'no_data': False
    })

@login_required
def prediction_list(request):
    # Admin có thể xem tất cả dự đoán
    if request.user.is_staff:
        predictions = StudentPrediction.objects.all().order_by('-created_at')
    else:
        # Người dùng thông thường chỉ xem dự đoán của mình
        predictions = StudentPrediction.objects.filter(user=request.user).order_by('-created_at')
    
    return render(request, 'predictor/prediction_list.html', {
        'predictions': predictions,
        'is_admin': request.user.is_staff
    })

@login_required
def generate_improvement_plan(request, prediction_id):
    """Tạo kế hoạch cải thiện chi tiết cho học sinh"""
    prediction = get_object_or_404(StudentPrediction, id=prediction_id)
    
    # Kiểm tra xem đã có kế hoạch cải thiện chưa
    existing_actions = prediction.improvement_actions.all()
    
    if existing_actions.exists():
        # Đã có kế hoạch, hiển thị và cho phép cập nhật
        action_forms = []
        for action in existing_actions:
            form = ActionCompletionForm(initial={
                'action_id': action.id,
                'completed': action.completed
            })
            action_forms.append({
                'action': action,
                'form': form
            })
        
        if request.method == 'POST':
            action_id = request.POST.get('action_id')
            if action_id:
                action = get_object_or_404(ImprovementAction, id=action_id)
                form = ActionCompletionForm(request.POST)
                if form.is_valid():
                    action.completed = form.cleaned_data['completed']
                    action.save()
                    messages.success(request, "Cập nhật trạng thái thành công!")
                    return redirect('improvement_plan', prediction_id=prediction_id)
        
        # Tính toán tỷ lệ hoàn thành
        total_actions = existing_actions.count()
        completed_actions = existing_actions.filter(completed=True).count()
        completion_rate = (completed_actions / total_actions * 100) if total_actions > 0 else 0
        
        return render(request, 'predictor/improvement_plan.html', {
            'prediction': prediction,
            'action_forms': action_forms,
            'completion_rate': completion_rate,
            'total_actions': total_actions,
            'completed_actions': completed_actions
        })
    else:
        # Chưa có kế hoạch, tạo mới
        # Xác định điểm yếu dựa trên các chỉ số
        weaknesses = []
        
        # Phân tích thời gian học
        if prediction.studytime < 10:
            weaknesses.append("Thời gian học tập")
            ImprovementAction.objects.create(
                prediction=prediction,
                area='Thời gian học tập',
                current_level=f"{prediction.studytime} giờ/tuần",
                target_level="Tối thiểu 10 giờ/tuần",
                action_description="1. Lập lịch học cố định mỗi ngày\n2. Sử dụng kỹ thuật Pomodoro (học 25 phút, nghỉ 5 phút)\n3. Đặt mục tiêu học tập hàng ngày"
            )
        
        # Phân tích số buổi vắng mặt
        if prediction.absences > 3:
            weaknesses.append("Chuyên cần")
            ImprovementAction.objects.create(
                prediction=prediction,
                area='Chuyên cần',
                current_level=f"{prediction.absences} buổi vắng",
                target_level="Tối đa 3 buổi vắng/học kỳ",
                action_description="1. Đặt báo thức sớm hơn 30 phút\n2. Chuẩn bị tài liệu từ tối hôm trước\n3. Liên hệ bạn học để nhắc nhở lẫn nhau"
            )
        
        # Phân tích tiêu thụ rượu
        if prediction.dalc > 2 or prediction.walc > 3:
            weaknesses.append("Lối sống")
            ImprovementAction.objects.create(
                prediction=prediction,
                area='Lối sống - Rượu/Bia',
                current_level=f"Ngày thường: {prediction.dalc}/5, Cuối tuần: {prediction.walc}/5",
                target_level="Giảm xuống mức 1-2/5",
                action_description="1. Tìm hoạt động thay thế tích cực (thể thao, đọc sách)\n2. Tham gia các câu lạc bộ học thuật\n3. Thiết lập mục tiêu học tập rõ ràng"
            )
        
        # Phân tích điểm số
        if prediction.g1_score < 10 or prediction.g2_score < 10:
            weaknesses.append("Kiến thức nền tảng")
            ImprovementAction.objects.create(
                prediction=prediction,
                area='Điểm số',
                current_level=f"G1: {prediction.g1_score}, G2: {prediction.g2_score}",
                target_level="Mỗi kỳ trên 12 điểm",
                action_description="1. Ôn tập lại kiến thức cơ bản\n2. Đăng ký học thêm hoặc tìm gia sư\n3. Thành lập nhóm học tập cùng bạn bè\n4. Đặt câu hỏi với giáo viên mỗi khi không hiểu bài"
            )
        
        # Nếu điểm dự đoán thấp hơn điểm G2, đề xuất cách cải thiện
        if prediction.predicted_score < prediction.g2_score:
            weaknesses.append("Xu hướng điểm số")
            ImprovementAction.objects.create(
                prediction=prediction,
                area='Xu hướng điểm số',
                current_level=f"Dự đoán ({prediction.predicted_score}) thấp hơn G2 ({prediction.g2_score})",
                target_level=f"Duy trì hoặc vượt điểm G2 ({prediction.g2_score})",
                action_description="1. Tăng cường thời gian ôn tập\n2. Tìm kiếm sự giúp đỡ từ giáo viên\n3. Xem lại các đề thi/kiểm tra trước đây\n4. Thực hành giải các bài tập khó"
            )
        
        # Tạo kế hoạch học tập hàng tuần
        weekly_plan = generate_weekly_study_plan(prediction)
        prediction.improvement_plan = weekly_plan
        prediction.save()
        
        # Tải lại trang để hiển thị kế hoạch mới tạo
        messages.success(request, "Đã tạo kế hoạch cải thiện thành công!")
        return redirect('improvement_plan', prediction_id=prediction_id)


def generate_weekly_study_plan(prediction):
    """Tạo lịch học tập theo tuần dựa trên đặc điểm học sinh"""
    # Tạo lịch học khởi đầu
    study_hours_needed = max(10, prediction.studytime + 2)  # Thêm ít nhất 2 giờ từ mức hiện tại
    
    # Phân bổ giờ học qua các ngày trong tuần
    days = ["Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7", "Chủ nhật"]
    
    # Phân bổ nhiều giờ hơn cho các ngày trong tuần, ít hơn vào cuối tuần
    weekday_hours = study_hours_needed * 0.7 / 5  # 70% thời gian cho 5 ngày trong tuần
    weekend_hours = study_hours_needed * 0.3 / 2  # 30% thời gian cho 2 ngày cuối tuần
    
    plan = "# KẾ HOẠCH HỌC TẬP HÀNG TUẦN\n\n"
    
    for i, day in enumerate(days):
        if i < 5:  # Ngày trong tuần
            hours = round(weekday_hours, 1)
            if prediction.g1_score < 10 or prediction.g2_score < 10:
                plan += f"### {day}\n- 18:00 - 19:30: Ôn tập kiến thức cơ bản (1.5 giờ)\n- 20:00 - {20 + (hours - 1.5):.1f}: Làm bài tập và chuẩn bị bài mới ({hours - 1.5:.1f} giờ)\n\n"
            else:
                plan += f"### {day}\n- 18:30 - {18.5 + hours:.1f}: Học tập và làm bài tập ({hours} giờ)\n\n"
        else:  # Cuối tuần
            hours = round(weekend_hours, 1)
            if i == 5:  # Thứ 7
                plan += f"### {day}\n- 09:00 - 11:00: Ôn tập và làm bài tập (2 giờ)\n- 15:00 - {15 + (hours - 2):.1f}: Học bài mới ({hours - 2:.1f} giờ)\n\n"
            else:  # Chủ nhật
                plan += f"### {day}\n- 15:00 - {15 + hours:.1f}: Chuẩn bị cho tuần mới ({hours} giờ)\n\n"
    
    # Thêm lời khuyên dựa trên điểm yếu
    plan += "\n## Lời khuyên cụ thể\n\n"
    
    if prediction.absences > 3:
        plan += "- **Chuyên cần**: Đặt báo thức sớm hơn và chuẩn bị bài từ tối hôm trước\n"
    
    if prediction.studytime < 10:
        plan += "- **Thời gian học**: Sử dụng kỹ thuật Pomodoro (25 phút học, 5 phút nghỉ)\n"
    
    if prediction.dalc > 2 or prediction.walc > 3:
        plan += "- **Lối sống**: Hạn chế rượu bia, thay thế bằng hoạt động thể thao hoặc sở thích lành mạnh\n"
    
    if prediction.g1_score < 10 or prediction.g2_score < 10:
        plan += "- **Kiến thức**: Tập trung vào các chủ đề cơ bản trước khi học các kiến thức nâng cao\n"
    
    return plan


@login_required
def student_progress_tracker(request, student_id=None):
    """Theo dõi tiến độ cải thiện của học sinh theo thời gian"""
    
    # Nếu có student_id, hiển thị tiến độ riêng của học sinh đó
    if student_id:
        predictions = StudentPrediction.objects.filter(student_id=student_id).order_by('created_at')
        student_name = predictions.first().student_name if predictions.exists() else "Học sinh"
        
        # Nếu không có dữ liệu, thông báo và chuyển hướng
        if not predictions.exists():
            messages.warning(request, f"Không tìm thấy dữ liệu cho học sinh với ID {student_id}")
            return redirect('student_list')
        
        # Chuẩn bị dữ liệu cho biểu đồ tiến độ
        dates = [p.created_at.strftime('%d/%m/%Y') for p in predictions]
        g1_scores = [p.g1_score for p in predictions]
        g2_scores = [p.g2_score for p in predictions]
        predicted_scores = [p.predicted_score for p in predictions]
        actual_scores = [p.actual_score if p.actual_score else None for p in predictions]
        
        # Tạo biểu đồ tiến độ
        plt.figure(figsize=(12, 6))
        plt.plot(dates, g1_scores, marker='o', label='Điểm G1')
        plt.plot(dates, g2_scores, marker='s', label='Điểm G2')
        plt.plot(dates, predicted_scores, marker='^', label='Điểm dự đoán')
        
        # Chỉ vẽ điểm thực tế cho những dự đoán đã có điểm
        valid_dates = []
        valid_scores = []
        for i, score in enumerate(actual_scores):
            if score is not None:
                valid_dates.append(dates[i])
                valid_scores.append(score)
                
        if valid_scores:
            plt.plot(valid_dates, valid_scores, marker='*', linestyle='--', label='Điểm thực tế')
        
        plt.title(f'Tiến độ học tập của {student_name}')
        plt.xlabel('Ngày dự đoán')
        plt.ylabel('Điểm số')
        plt.ylim(0, 20)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Chuyển biểu đồ thành base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        
        progress_chart = base64.b64encode(image_png).decode('utf-8')
        
        # Phân tích tiến độ học tập
        progress_stats = analyze_student_progress(predictions)
        
        # Tạo biểu đồ phân tích các yếu tố khác
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # Biểu đồ 1: Thời gian học
        study_times = [p.studytime for p in predictions]
        axs[0, 0].plot(dates, study_times, marker='o', color='green')
        axs[0, 0].set_title('Thời gian học tập')
        axs[0, 0].set_ylabel('Giờ/tuần')
        axs[0, 0].set_xticks([])
        axs[0, 0].grid(True, linestyle='--', alpha=0.7)
        
        # Biểu đồ 2: Vắng mặt
        absences = [p.absences for p in predictions]
        axs[0, 1].plot(dates, absences, marker='s', color='red')
        axs[0, 1].set_title('Số buổi vắng mặt')
        axs[0, 1].set_xticks([])
        axs[0, 1].grid(True, linestyle='--', alpha=0.7)
        
        # Biểu đồ 3: Tiêu thụ rượu
        dalc = [p.dalc for p in predictions]
        walc = [p.walc for p in predictions]
        axs[1, 0].plot(dates, dalc, marker='o', label='Ngày thường')
        axs[1, 0].plot(dates, walc, marker='s', label='Cuối tuần')
        axs[1, 0].set_title('Mức tiêu thụ rượu')
        axs[1, 0].set_xlabel('Ngày dự đoán')
        axs[1, 0].set_yticks([1, 2, 3, 4, 5])
        axs[1, 0].legend()
        axs[1, 0].grid(True, linestyle='--', alpha=0.7)
        
        # Biểu đồ 4: Sai số dự đoán (nếu có điểm thực tế)
        prediction_diffs = []
        valid_dates_diff = []
        for p in predictions:
            if p.prediction_diff is not None:
                prediction_diffs.append(p.prediction_diff)
                valid_dates_diff.append(p.created_at.strftime('%d/%m/%Y'))
        
        if prediction_diffs:
            axs[1, 1].bar(valid_dates_diff, prediction_diffs, color='purple')
            axs[1, 1].set_title('Sai số dự đoán')
            axs[1, 1].set_xlabel('Ngày dự đoán')
            axs[1, 1].set_ylabel('Sai số')
            axs[1, 1].grid(True, linestyle='--', alpha=0.7)
        else:
            axs[1, 1].text(0.5, 0.5, 'Chưa có dữ liệu điểm thực tế', 
                      horizontalalignment='center', verticalalignment='center')
            axs[1, 1].set_title('Sai số dự đoán')
        
        plt.tight_layout()
        
        # Chuyển biểu đồ phân tích thành base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        
        factors_chart = base64.b64encode(image_png).decode('utf-8')
        
        # Đánh giá kế hoạch cải thiện
        improvement_status = []
        for p in predictions:
            actions = p.improvement_actions.all()
            total = actions.count()
            completed = actions.filter(completed=True).count()
            rate = (completed / total * 100) if total > 0 else 0
            
            improvement_status.append({
                'date': p.created_at,
                'total_actions': total,
                'completed_actions': completed,
                'completion_rate': rate,
                'prediction_id': p.id
            })
        
        return render(request, 'predictor/student_progress.html', {
            'student_name': student_name,
            'student_id': student_id,
            'predictions': predictions,
            'progress_chart': progress_chart,
            'factors_chart': factors_chart,
            'progress_stats': progress_stats,
            'improvement_status': improvement_status
        })
    else:
        # Hiển thị danh sách các học sinh có dữ liệu
        students = StudentPrediction.objects.values('student_id', 'student_name')\
                    .filter(student_id__isnull=False)\
                    .distinct()\
                    .order_by('student_name')
        
        # Tính số lượng dự đoán cho mỗi học sinh
        for student in students:
            student['prediction_count'] = StudentPrediction.objects.filter(
                student_id=student['student_id']
            ).count()
            
            # Lấy dự đoán gần nhất
            latest_prediction = StudentPrediction.objects.filter(
                student_id=student['student_id']
            ).order_by('-created_at').first()
            
            if latest_prediction:
                student['latest_score'] = latest_prediction.predicted_score
                student['latest_category'] = latest_prediction.result_category
        
        return render(request, 'predictor/student_list.html', {
            'students': students
        })


def analyze_student_progress(predictions):
    """Phân tích tiến độ học tập của học sinh dựa trên các dự đoán"""
    if not predictions or len(predictions) < 2:
        return {
            'has_progress': False,
            'message': 'Cần ít nhất 2 dự đoán để phân tích tiến độ.'
        }
    
    # Sắp xếp dự đoán theo thời gian
    sorted_predictions = sorted(predictions, key=lambda p: p.created_at)
    
    first_prediction = sorted_predictions[0]
    last_prediction = sorted_predictions[-1]
    
    # So sánh điểm giữa lần đầu và lần cuối
    g1_change = last_prediction.g1_score - first_prediction.g1_score
    g2_change = last_prediction.g2_score - first_prediction.g2_score
    
    # Kiểm tra các yếu tố khác
    study_change = last_prediction.studytime - first_prediction.studytime
    absences_change = last_prediction.absences - first_prediction.absences
    dalc_change = last_prediction.dalc - first_prediction.dalc
    walc_change = last_prediction.walc - first_prediction.walc
    
    # Kiểm tra điểm thực tế nếu có
    actual_progress = None
    if hasattr(first_prediction, 'actual_score') and hasattr(last_prediction, 'actual_score'):
        if first_prediction.actual_score and last_prediction.actual_score:
            actual_change = last_prediction.actual_score - first_prediction.actual_score
            actual_progress = {
                'change': actual_change,
                'percentage': (actual_change / first_prediction.actual_score) * 100 if first_prediction.actual_score else 0
            }
    
    # Phân tích tiến độ
    is_improving = g1_change > 0 and g2_change > 0
    
    # Tính toán xu hướng dựa trên nhiều yếu tố
    factors_improving = 0
    factors_total = 4
    
    if g1_change > 0:
        factors_improving += 1
    if g2_change > 0:
        factors_improving += 1
    if study_change > 0:
        factors_improving += 1
    if absences_change < 0:
        factors_improving += 1
    
    improvement_score = (factors_improving / factors_total) * 10
    
    progress_stats = {
        'has_progress': True,
        'g1_change': g1_change,
        'g2_change': g2_change,
        'actual_progress': actual_progress,
        'study_change': study_change,
        'absences_change': absences_change,
        'dalc_change': dalc_change,
        'walc_change': walc_change,
        'is_improving': is_improving,
        'improvement_score': improvement_score,
        'period': {
            'start': first_prediction.created_at,
            'end': last_prediction.created_at,
            'days': (last_prediction.created_at - first_prediction.created_at).days
        }
    }
    
    # Thêm phân tích định tính
    if improvement_score >= 7.5:
        progress_stats['assessment'] = 'Cải thiện xuất sắc'
        progress_stats['recommendation'] = 'Tiếp tục phương pháp học tập hiện tại'
    elif improvement_score >= 5:
        progress_stats['assessment'] = 'Đang cải thiện tốt'
        progress_stats['recommendation'] = 'Tập trung vào các điểm yếu còn lại'
    elif improvement_score >= 2.5:
        progress_stats['assessment'] = 'Có cải thiện nhưng chưa đủ'
        progress_stats['recommendation'] = 'Cần nỗ lực nhiều hơn và thực hiện đầy đủ kế hoạch'
    else:
        progress_stats['assessment'] = 'Chưa có cải thiện đáng kể'
        progress_stats['recommendation'] = 'Cần xem lại toàn bộ phương pháp học tập và tìm hỗ trợ'
    
    return progress_stats

def login_view(request):
    if request.method == 'POST':
        form = CustomAuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('home')
    else:
        form = CustomAuthenticationForm()
    return render(request, 'predictor/login.html', {'form': form})

def register_view(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            # Tạo profile cho user mới
            UserProfile.objects.create(user=user)
            # Đăng nhập user sau khi đăng ký
            login(request, user)
            return redirect('home')
    else:
        form = UserRegistrationForm()
    return render(request, 'predictor/register.html', {'form': form})

def logout_view(request):
    logout(request)
    return redirect('login')

# Hàm kiểm tra quyền giáo viên
def is_teacher_or_staff(user):
    return user.is_staff or (hasattr(user, 'profile') and user.profile.is_teacher)

# @user_passes_test(is_teacher_or_staff)
@login_required
def export_report(request, format='excel'):
    """Xuất báo cáo dự đoán ra file Excel hoặc PDF"""
    
    # Lấy tất cả dự đoán
    predictions = StudentPrediction.objects.all().order_by('-created_at')
    
    if format == 'excel':
        # Tạo file Excel trong bộ nhớ
        output = io.BytesIO()
        workbook = xlsxwriter.Workbook(output, {'remove_timezone': True})
        worksheet = workbook.add_worksheet('Dự đoán điểm số')
        
        # Định dạng
        header_format = workbook.add_format({'bold': True, 'bg_color': '#4B82BC', 'color': 'white'})
        date_format = workbook.add_format({'num_format': 'dd/mm/yyyy hh:mm'})
        
        # Tiêu đề
        headers = [
            'ID', 'Tên học sinh', 'Mã học sinh', 'Điểm G1', 'Điểm G2', 
            'Thời gian học', 'Số lần thất bại', 'Số buổi vắng', 
            'Điểm dự đoán', 'Điểm thực tế', 'Sai số', 'Phân loại',
            'Ngày tạo'
        ]
        
        for col, header in enumerate(headers):
            worksheet.write(0, col, header, header_format)
        
        # Dữ liệu
        for row, prediction in enumerate(predictions, start=1):
            worksheet.write(row, 0, prediction.id)
            worksheet.write(row, 1, prediction.student_name or 'Không có tên')
            worksheet.write(row, 2, prediction.student_id or 'Không có mã')
            worksheet.write(row, 3, prediction.g1_score)
            worksheet.write(row, 4, prediction.g2_score)
            worksheet.write(row, 5, prediction.studytime)
            worksheet.write(row, 6, prediction.failures)
            worksheet.write(row, 7, prediction.absences)
            worksheet.write(row, 8, prediction.predicted_score)
            worksheet.write(row, 9, prediction.actual_score if prediction.actual_score else 'Chưa có')
            worksheet.write(row, 10, prediction.prediction_diff if prediction.prediction_diff else 'N/A')
            worksheet.write(row, 11, prediction.result_category)
            worksheet.write(row, 12, prediction.created_at, date_format)
        
        # Thiết lập chiều rộng cột
        for i, width in enumerate([5, 20, 15, 8, 8, 12, 15, 12, 12, 12, 8, 15, 20]):
            worksheet.set_column(i, i, width)
        
        workbook.close()
        
        # Tạo HTTP response với file Excel
        output.seek(0)
        response = HttpResponse(
            output, 
            content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        response['Content-Disposition'] = 'attachment; filename=du_doan_diem_so.xlsx'
        
        return response
    
    elif format == 'pdf':
        # Tạo file PDF trong bộ nhớ
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=landscape(letter))
        
        # Dữ liệu cho bảng
        data = [
            ['ID', 'Tên học sinh', 'Điểm G1', 'Điểm G2', 'Dự đoán', 'Thực tế', 'Phân loại']
        ]
        
        for p in predictions:
            data.append([
                str(p.id),
                p.student_name or 'Không tên',
                str(p.g1_score),
                str(p.g2_score),
                f"{p.predicted_score:.2f}",
                f"{p.actual_score:.2f}" if p.actual_score else 'Chưa có',
                p.result_category
            ])
        
        # Tạo bảng
        t = Table(data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.blue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))
        
        # Thêm tiêu đề
        styles = getSampleStyleSheet()
        title = Paragraph("Báo cáo dự đoán điểm số học sinh", styles['Heading1'])
        
        # Xây dựng PDF
        elements = [title, Spacer(1, 20), t]
        doc.build(elements)
        
        # Tạo HTTP response với file PDF
        buffer.seek(0)
        response = HttpResponse(buffer, content_type='application/pdf')
        response['Content-Disposition'] = 'attachment; filename=du_doan_diem_so.pdf'
        
        return response
    
    # Format không hợp lệ
    messages.error(request, f"Định dạng {format} không được hỗ trợ")
    return redirect('prediction_list')

@login_required
def visualization_dashboard(request):
    """Trang hiển thị các biểu đồ trực quan về dữ liệu"""
    
    # Lấy tất cả dự đoán có điểm thực tế
    predictions_with_actual = StudentPrediction.objects.filter(actual_score__isnull=False)
    
    if not predictions_with_actual.exists():
        messages.info(request, 'Chưa có đủ dữ liệu để hiển thị biểu đồ.')
        return render(request, 'predictor/visualization.html', {'no_data': True})
    
    # 1. Biểu đồ phân phối điểm theo loại kết quả
    score_distribution_chart = generate_score_distribution_chart(predictions_with_actual)
    
    # 2. Biểu đồ so sánh điểm dự đoán và điểm thực tế
    prediction_vs_actual_chart = generate_prediction_comparison_chart(predictions_with_actual)
    
    # 3. Biểu đồ mối quan hệ giữa thời gian học và điểm số
    studytime_score_chart = generate_studytime_score_chart(predictions_with_actual)
    
    # 4. Biểu đồ mối quan hệ giữa vắng mặt và điểm số
    absence_score_chart = generate_absence_score_chart(predictions_with_actual)
    
    # 5. Biểu đồ radar các yếu tố ảnh hưởng
    radar_chart = generate_factors_radar_chart()
    
    return render(request, 'predictor/visualization.html', {
        'score_distribution_chart': score_distribution_chart,
        'prediction_vs_actual_chart': prediction_vs_actual_chart,
        'studytime_score_chart': studytime_score_chart,
        'absence_score_chart': absence_score_chart,
        'radar_chart': radar_chart,
        'no_data': False
    })

def generate_score_distribution_chart(predictions):
    """Tạo biểu đồ phân phối điểm theo loại kết quả"""
    plt.figure(figsize=(10, 6))
    
    # Lấy dữ liệu phân phối điểm theo loại
    categories = predictions.values('result_category')\
                .annotate(count=Count('id'))\
                .order_by('result_category')
    
    # Biểu đồ hình tròn
    labels = [f"{cat['result_category']} ({cat['count']})" for cat in categories]
    sizes = [cat['count'] for cat in categories]
    
    # Màu sắc cho từng loại
    colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
    
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Phân phối học sinh theo loại kết quả')
    
    # Chuyển biểu đồ thành base64 để hiển thị trong HTML
    return get_chart_as_base64(plt)

def generate_prediction_comparison_chart(predictions):
    """Tạo biểu đồ so sánh điểm dự đoán và điểm thực tế"""
    plt.figure(figsize=(12, 6))
    
    # Lấy 20 dự đoán gần nhất có điểm thực tế
    recent_predictions = predictions.order_by('-created_at')[:20]
    
    # Danh sách ID hoặc tên học sinh
    labels = [f"#{p.id}" for p in recent_predictions]
    predicted = [p.predicted_score for p in recent_predictions]
    actual = [p.actual_score for p in recent_predictions]
    
    x = range(len(labels))
    width = 0.35
    
    # Vẽ biểu đồ cột
    plt.bar([i - width/2 for i in x], predicted, width, label='Dự đoán')
    plt.bar([i + width/2 for i in x], actual, width, label='Thực tế')
    
    plt.xlabel('Mã dự đoán')
    plt.ylabel('Điểm số')
    plt.title('So sánh điểm dự đoán và điểm thực tế')
    plt.xticks(x, labels, rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    return get_chart_as_base64(plt)

def generate_studytime_score_chart(predictions):
    """Tạo biểu đồ mối quan hệ giữa thời gian học và điểm số"""
    plt.figure(figsize=(10, 6))
    
    # Tính trung bình điểm thực tế theo thời gian học
    studytime_scores = predictions.values('studytime')\
                    .annotate(avg_score=Avg('actual_score'))\
                    .order_by('studytime')
    
    # Vẽ biểu đồ đường
    x = [item['studytime'] for item in studytime_scores]
    y = [item['avg_score'] for item in studytime_scores]
    
    plt.plot(x, y, marker='o', linestyle='-', linewidth=2, markersize=10)
    
    plt.xlabel('Mức thời gian học (1-4)')
    plt.ylabel('Điểm trung bình')
    plt.title('Mối quan hệ giữa thời gian học và điểm số')
    plt.xticks(x)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Thêm giá trị trên biểu đồ
    for i, j in zip(x, y):
        plt.annotate(f'{j:.2f}', xy=(i, j), xytext=(0, 5), 
                     textcoords='offset points', ha='center')
    
    return get_chart_as_base64(plt)

def generate_absence_score_chart(predictions):
    """Tạo biểu đồ mối quan hệ giữa vắng mặt và điểm số"""
    plt.figure(figsize=(10, 6))
    
    # Nhóm theo số buổi vắng
    absence_groups = {}
    for p in predictions:
        # Nhóm các giá trị vắng mặt: 0, 1-3, 4-7, 8+
        if p.absences == 0:
            group = '0'
        elif p.absences <= 3:
            group = '1-3'
        elif p.absences <= 7:
            group = '4-7'
        else:
            group = '8+'
            
        if group not in absence_groups:
            absence_groups[group] = []
        absence_groups[group].append(p.actual_score)
    
    # Tính trung bình và độ lệch chuẩn
    box_data = [absence_groups[group] for group in ['0', '1-3', '4-7', '8+'] if group in absence_groups]
    
    # Vẽ biểu đồ hộp
    plt.boxplot(box_data, patch_artist=True)
    plt.xticks(range(1, len(box_data) + 1), [group for group in ['0', '1-3', '4-7', '8+'] if group in absence_groups])
    plt.xlabel('Số buổi vắng mặt')
    plt.ylabel('Điểm số')
    plt.title('Ảnh hưởng của việc vắng mặt đến kết quả học tập')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    return get_chart_as_base64(plt)

def generate_factors_radar_chart():
    """Tạo biểu đồ radar các yếu tố ảnh hưởng đến kết quả"""
    plt.figure(figsize=(8, 8))
    
    # Các yếu tố và trọng số ảnh hưởng (lấy từ mô hình)
    # Đây là dữ liệu mẫu, thay thế bằng trọng số thực từ mô hình của bạn
    factors = ['G2', 'G1', 'Studytime', 'Absences', 'Failures', 'Health', 'Dalc/Walc']
    weights = [0.79, 0.62, 0.31, -0.28, -0.25, 0.18, -0.20]
    
    # Chuyển về giá trị dương để hiển thị
    abs_weights = [abs(w) for w in weights]
    
    # Số lượng yếu tố
    N = len(factors)
    
    # Góc cho mỗi yếu tố
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Đóng vòng tròn
    
    # Giá trị trọng số
    values = abs_weights + [abs_weights[0]]  # Đóng vòng tròn
    
    # Vẽ biểu đồ radar
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2, linestyle='solid')
    ax.fill(angles, values, alpha=0.25)
    
    # Thêm nhãn và giá trị
    plt.xticks(angles[:-1], factors)
    plt.yticks([0.2, 0.4, 0.6, 0.8], ['0.2', '0.4', '0.6', '0.8'], color='gray', size=8)
    
    # Thêm nhãn tác động (tích cực/tiêu cực)
    for i, (angle, weight) in enumerate(zip(angles[:-1], weights)):
        impact = "+" if weight > 0 else "-"
        plt.annotate(f"{impact}", xy=(angle, abs(weight) + 0.05),
                    ha='center', va='center', color='blue' if weight > 0 else 'red')
    
    plt.title('Ảnh hưởng của các yếu tố đến điểm số')
    
    return get_chart_as_base64(plt)

def get_chart_as_base64(plt):
    """Chuyển đổi biểu đồ matplotlib thành chuỗi base64 để hiển thị trong HTML"""
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()  # Đóng biểu đồ để giải phóng bộ nhớ
    
    return base64.b64encode(image_png).decode('utf-8')
