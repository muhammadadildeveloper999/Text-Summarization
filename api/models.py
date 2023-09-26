from django.db import models

class TextDocument(models.Model):
    original_text = models.TextField()
    summary = models.TextField()

