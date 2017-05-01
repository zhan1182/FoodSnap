# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.contrib import admin
from .models import Food

# Register your models here.


class FoodAdmin(admin.ModelAdmin):
    list_display = ('id', 'result')

admin.site.register(Food, FoodAdmin)