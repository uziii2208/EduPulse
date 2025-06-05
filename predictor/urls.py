from django.urls import path
from . import views

urlpatterns = [
    # Auth URLs
    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('logout/', views.logout_view, name='logout'),
    
    # Các URLs hiện có
    path('', views.home, name='home'),
    path('update-score/<int:prediction_id>/', views.update_actual_score, name='update_score'),
    path('stats/', views.prediction_stats, name='prediction_stats'),
    path('predictions/', views.prediction_list, name='prediction_list'),
    
    # URLs mới
    path('improvement-plan/<int:prediction_id>/', views.generate_improvement_plan, name='improvement_plan'),
    path('progress/<str:student_id>/', views.student_progress_tracker, name='student_progress'),
    path('students/', views.student_progress_tracker, name='student_list'),

    # Thêm vào urlpatterns
    path('export-report/<str:format>/', views.export_report, name='export_report'),
    path('visualization/', views.visualization_dashboard, name='visualization'),
]