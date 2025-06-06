# Generated by Django 5.2.1 on 2025-05-21 12:57

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('predictor', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='studentprediction',
            name='improvement_plan',
            field=models.TextField(blank=True, null=True),
        ),
        migrations.CreateModel(
            name='ImprovementAction',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('area', models.CharField(max_length=100)),
                ('current_level', models.CharField(max_length=100)),
                ('target_level', models.CharField(max_length=100)),
                ('action_description', models.TextField()),
                ('completed', models.BooleanField(default=False)),
                ('prediction', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='improvement_actions', to='predictor.studentprediction')),
            ],
        ),
    ]
