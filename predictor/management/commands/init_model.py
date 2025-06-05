from django.core.management.base import BaseCommand
from django.conf import settings
import os
import sys
from pathlib import Path

class Command(BaseCommand):
    help = 'Initializes the ML model for student score prediction'

    def handle(self, *args, **kwargs):
        try:
            # Add ml_model directory to Python path
            project_root = Path(__file__).resolve().parent.parent.parent.parent
            ml_model_path = project_root / 'ml_model'
            sys.path.append(str(ml_model_path))

            # Import main training function
            from train_model import main
            
            self.stdout.write('Training model...')
            
            # Run training
            main()
            
            self.stdout.write(self.style.SUCCESS('Model successfully trained and saved'))

        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error initializing model: {str(e)}'))
            raise