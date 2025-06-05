from django.contrib import admin
from .models import StudentPrediction, UserProfile, ImprovementAction

# Tạo lớp Admin cho StudentPrediction với các tùy chỉnh
class StudentPredictionAdmin(admin.ModelAdmin):
    list_display = ('student_name', 'student_id', 'g1_score', 'g2_score', 'predicted_score', 
                    'actual_score', 'result_category', 'created_at')
    list_filter = ('result_category', 'created_at')
    search_fields = ('student_name', 'student_id', 'result_category')
    date_hierarchy = 'created_at'
    ordering = ('-created_at',)

# Đăng ký model với Admin site
admin.site.register(StudentPrediction, StudentPredictionAdmin)

# Nếu bạn có các model khác, hãy đăng ký chúng ở đây
try:
    admin.site.register(UserProfile)
    admin.site.register(ImprovementAction)
except admin.sites.AlreadyRegistered:
    pass  # Bỏ qua nếu đã đăng ký
