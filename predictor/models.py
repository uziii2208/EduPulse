from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User

# Create your models here.

class StudentPrediction(models.Model):
    # Thông tin học sinh
    student_name = models.CharField(max_length=100, blank=True)
    student_id = models.CharField(max_length=50, blank=True)
    
    # Đặc trưng đầu vào chính
    g1_score = models.FloatField()
    g2_score = models.FloatField()
    studytime = models.IntegerField()
    failures = models.IntegerField()
    absences = models.IntegerField()
    
    # Thông tin phụ
    dalc = models.IntegerField()  # Tiêu thụ rượu trong tuần
    walc = models.IntegerField()  # Tiêu thụ rượu cuối tuần
    health = models.IntegerField()  # Sức khỏe
    
    # Kết quả dự đoán
    predicted_score = models.FloatField()
    result_category = models.CharField(max_length=50)
    recommendation = models.TextField()
    
    # Điểm thực tế (sẽ nhập sau khi có kết quả)
    actual_score = models.FloatField(null=True, blank=True)
    
    # Đánh giá độ chính xác
    prediction_diff = models.FloatField(null=True, blank=True)
    
    # Thông tin cải thiện
    improvement_plan = models.TextField(null=True, blank=True)
    
    # Thời gian
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Thêm trường user để lưu người tạo dự đoán
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='predictions', null=True)
    
    def save(self, *args, **kwargs):
        # Tính toán sự khác biệt giữa điểm dự đoán và thực tế
        if self.predicted_score is not None and self.actual_score is not None:
            self.prediction_diff = abs(self.predicted_score - self.actual_score)
        super().save(*args, **kwargs)
    
    def __str__(self):
        return f"{self.student_name or 'Học sinh'} - Dự đoán: {self.predicted_score:.2f}, Thực tế: {self.actual_score or 'Chưa có'}"


class ImprovementAction(models.Model):
    """Lưu các hành động cải thiện khuyến nghị cho học sinh"""
    prediction = models.ForeignKey(StudentPrediction, on_delete=models.CASCADE, related_name='improvement_actions')
    area = models.CharField(max_length=100)  # Lĩnh vực cần cải thiện
    current_level = models.CharField(max_length=100)  # Mức hiện tại
    target_level = models.CharField(max_length=100)  # Mức mục tiêu
    action_description = models.TextField()  # Mô tả hành động
    completed = models.BooleanField(default=False)  # Đánh dấu hoàn thành
    
    def __str__(self):
        return f"Cải thiện {self.area} cho {self.prediction.student_name or 'Học sinh'}"


class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    is_teacher = models.BooleanField(default=False)
    department = models.CharField(max_length=100, blank=True, null=True)
    
    def __str__(self):
        return f"{self.user.username} - {'Giáo viên' if self.is_teacher else 'Học sinh'}"
