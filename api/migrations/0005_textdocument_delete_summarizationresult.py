# Generated by Django 4.2.5 on 2023-09-21 17:17

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0004_summarizationresult_delete_textdocument'),
    ]

    operations = [
        migrations.CreateModel(
            name='TextDocument',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('original_text', models.TextField()),
                ('summary', models.TextField()),
            ],
        ),
        migrations.DeleteModel(
            name='SummarizationResult',
        ),
    ]