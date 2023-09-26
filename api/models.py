from django.db import models

class TextDocument(models.Model):
    profile = models.ImageField(upload_to="summarization/", default="Auth/dummy.jpg")


