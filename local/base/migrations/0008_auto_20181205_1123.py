# Generated by Django 2.1.4 on 2018-12-05 11:23

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [("base", "0007_auto_20181205_0956")]

    operations = [
        migrations.AlterField(
            model_name="scan",
            name="seg_file",
            field=models.FileField(blank=True, null=True, upload_to=""),
        )
    ]