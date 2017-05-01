from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^upload-food-image/$', views.upload_food_image_view, name='upload_food_image_view'),
    url(r'^foods/$', views.foods_view, name='foods_view'),
]