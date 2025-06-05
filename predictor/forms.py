import os
from django import forms
import django
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.models import User
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'NEW_DOAN_KTDL.settings')
django.setup()
EDUCATION_CHOICES = [
    (0, 'Không'),
    (1, 'Tiểu học'),
    (2, 'THCS'),
    (3, 'THPT'),
    (4, 'Đại học trở lên')
]

JOB_CHOICES = [
    ('teacher', 'Giáo viên'),
    ('health', 'Y tế'),
    ('services', 'Dịch vụ công'),
    ('at_home', 'Nội trợ'),
    ('other', 'Khác')
]

class StudentPredictionForm(forms.Form):
    # Thông tin cá nhân
    age = forms.IntegerField(
        label='Tuổi',
        min_value=15,
        max_value=22,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    address = forms.ChoiceField(
        label='Nơi ở',
        choices=[('U', 'Thành thị'), ('R', 'Nông thôn')],
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    famsize = forms.ChoiceField(
        label='Quy mô gia đình',
        choices=[('LE3', '≤ 3 người'), ('GT3', '> 3 người')],
        widget=forms.Select(attrs={'class': 'form-control'})
    )

    # Thông tin học vấn gia đình
    Medu = forms.ChoiceField(
        label='Trình độ học vấn của mẹ',
        choices=EDUCATION_CHOICES,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    Fedu = forms.ChoiceField(
        label='Trình độ học vấn của cha',
        choices=EDUCATION_CHOICES,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    Mjob = forms.ChoiceField(
        label='Nghề nghiệp của mẹ',
        choices=JOB_CHOICES,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    Fjob = forms.ChoiceField(
        label='Nghề nghiệp của cha',
        choices=JOB_CHOICES,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    
    # Mối quan hệ gia đình
    famrel = forms.IntegerField(
        label='Chất lượng mối quan hệ gia đình (1-5)',
        min_value=1,
        max_value=5,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    
    # Thời gian và học tập
    traveltime = forms.IntegerField(
        label='Thời gian di chuyển đến trường',
        min_value=1,
        max_value=4,
        help_text='1: <15 phút, 2: 15-30 phút, 3: 30-60 phút, 4: >60 phút',
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    studytime = forms.IntegerField(
        label='Thời gian học (giờ/tuần)',
        min_value=1,
        max_value=20,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    failures = forms.IntegerField(
        label='Số lần thất bại trước đây',
        min_value=0,
        max_value=3,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    absences = forms.IntegerField(
        label='Số lần vắng mặt',
        min_value=0,
        max_value=30,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )

    # Điểm số
    g1_score = forms.IntegerField(
        label='Điểm kỳ 1 (0-20)',
        min_value=0,
        max_value=20,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    g2_score = forms.IntegerField(
        label='Điểm kỳ 2 (0-20)',
        min_value=0,
        max_value=20,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )

    # Hỗ trợ và hoạt động
    famsup = forms.BooleanField(
        label='Hỗ trợ từ gia đình',
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    paid = forms.BooleanField(
        label='Học thêm',
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    activities = forms.BooleanField(
        label='Hoạt động ngoại khóa',
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    nursery = forms.BooleanField(
        label='Đã học mẫu giáo',
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    higher = forms.BooleanField(
        label='Dự định học đại học',
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    internet = forms.BooleanField(
        label='Có Internet tại nhà',
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    romantic = forms.BooleanField(
        label='Có mối quan hệ tình cảm',
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )

    # Thời gian rảnh và sức khỏe
    freetime = forms.IntegerField(
        label='Thời gian rảnh (1-5)',
        min_value=1,
        max_value=5,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    goout = forms.IntegerField(
        label='Thời gian đi chơi (1-5)',
        min_value=1,
        max_value=5,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    Dalc = forms.IntegerField(
        label='Uống rượu trong ngày (1-5)',
        min_value=1,
        max_value=5,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    Walc = forms.IntegerField(
        label='Uống rượu cuối tuần (1-5)',
        min_value=1,
        max_value=5,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    health = forms.IntegerField(
        label='Tình trạng sức khỏe (1-5)',
        min_value=1,
        max_value=5,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )

    # Thêm vào form hiện tại
    student_name = forms.CharField(
        label='Tên học sinh',
        max_length=100,
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    student_id = forms.CharField(
        label='Mã học sinh',
        max_length=50,
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )

class ActualScoreForm(forms.Form):
    actual_score = forms.FloatField(
        label='Điểm thực tế',
        min_value=0,
        max_value=20,
        widget=forms.NumberInput(attrs={'step': '0.1'})
    )

# Bổ sung vào forms.py hiện có

class ActionCompletionForm(forms.Form):
    """Form đánh dấu hoàn thành hành động cải thiện"""
    action_id = forms.IntegerField(widget=forms.HiddenInput())
    completed = forms.BooleanField(required=False, label="Đã hoàn thành")
    notes = forms.CharField(
        widget=forms.Textarea(attrs={'rows': 3}), 
        required=False,
        label="Ghi chú"
    )

class CustomAuthenticationForm(AuthenticationForm):
    class Meta:
        model = User
        fields = ['username', 'password']
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['username'].widget.attrs.update({'class': 'form-control', 'placeholder': 'Tên đăng nhập'})
        self.fields['password'].widget.attrs.update({'class': 'form-control', 'placeholder': 'Mật khẩu'})

class UserRegistrationForm(UserCreationForm):
    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2']
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['username'].widget.attrs.update({'class': 'form-control', 'placeholder': 'Tên đăng nhập'})
        self.fields['email'].widget.attrs.update({'class': 'form-control', 'placeholder': 'Email'})
        self.fields['password1'].widget.attrs.update({'class': 'form-control', 'placeholder': 'Mật khẩu'})
        self.fields['password2'].widget.attrs.update({'class': 'form-control', 'placeholder': 'Nhập lại mật khẩu'})