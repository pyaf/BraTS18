# Generated by Django 2.1.3 on 2018-12-03 14:55

import base.models
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('base', '0007_scan_scan_id'),
    ]

    operations = [
        migrations.AddField(
            model_name='scan',
            name='seg_file',
            field=models.FileField(null=True, upload_to=base.models.get_seg_path),
        ),
    ]
