# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models
from chinese_food_detector_backend.settings import MEDIA_URL

# Create your models here.



class Food(models.Model):
    result = models.CharField(max_length=200)
    image = models.ImageField(upload_to='food_images/')