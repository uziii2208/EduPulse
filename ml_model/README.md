# Machine Learning Model Setup

## Overview
The machine learning model used in this project is a Random Forest Regressor trained to predict student scores based on various factors including:
- Study time (hours/week)
- Completed assignments
- Previous test scores
- Family support
- Attendance rate
- Grade level

## Initial Setup

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Initialize the model with sample data:
```bash
python manage.py init_model
```

This will:
- Generate synthetic training data
- Train a Random Forest model
- Save the model to `predictor/ml_model/model.pkl`
- Create a feature importance plot in `static/predictor/images/`

## Model Details

### Features
- `study_time`: Hours spent studying per week (1-10)
- `completed_assignments`: Number of completed assignments (0-20)
- `previous_scores`: Previous test scores (0-10)
- `family_support`: Whether student has family support (0/1)
- `attendance_rate`: Class attendance percentage (0-100)
- `grade_level`: Student's grade level (10-12)

### Model Parameters
- Algorithm: Random Forest Regressor
- Number of estimators: 100
- Max depth: 10
- Random state: 42

### Feature Importance
A visualization of feature importance is generated during initialization and can be found at `static/predictor/images/feature_importance.png`.

## Customization

To modify the model or use your own data:

1. Create a new dataset following the feature format above
2. Modify the `init_model.py` command to use your data
3. Re-run the initialization command

## Production Use
For production deployment:
- Use real student data instead of synthetic data
- Adjust model parameters based on your needs
- Consider implementing model retraining based on actual results
- Add validation metrics and monitoring

## Update Model
To update the model with new data:
```bash
python manage.py init_model --retrain