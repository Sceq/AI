from django.db import migrations, models

class Migration(migrations.Migration):

    dependencies = [
        ('ai_app', '0005_text_element'),
    ]

    operations = [
        migrations.AddField(
            model_name='imageelement',
            name='dominant_colors',
            field=models.JSONField(blank=True, null=True),
        ),
    ]