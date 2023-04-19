from django.db import models
from django import forms
from django.utils.translation import ugettext_lazy as _

class Image(models.Model):
    image = models.ImageField(upload_to='images')