# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render
from django.http import JsonResponse, HttpResponseNotAllowed, HttpResponseBadRequest, HttpResponseServerError
from django.views.decorators.csrf import csrf_exempt
from .models import Food
from django.core.cache import cache
from AML.vgg16 import load_saved_model
from AML.data.image2array import image2array
from PIL import Image
from PIL import ImageFile
import numpy as np
import os
from .constants import FOOD_DESCRIPTION_MAPPING, FOOD_NAME_MAPPING
from chinese_food_detector_backend.settings import BASE_DIR



CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

# Create your views here.

ImageFile.LOAD_TRUNCATED_IMAGES = True
MODEL_NAME = 'MODEL' 
model = None


def get_cache_model():
    global model
    if not model:
        model = load_saved_model()
    return model

def classify(image_path):
    # image = Image.open("./broc/273.jpg")
    image = Image.open(image_path)
    array = image2array(image)
    arrayl = array.reshape((1, 224, 224, 3))
    results = get_cache_model().predict(arrayl)
    preds = np.argmax(results[0])
    return preds # 0 - 5

@csrf_exempt
def upload_food_image_view(request):
    if request.method == 'POST':
        image = request.FILES.get('file')
        if image:
            # create new food
            try:
                food = Food(image=image)
                food.save()
            except Exception as e:
                print (e)
                return HttpResponseServerError()

            image_path = BASE_DIR + food.image.url
            result = classify(image_path)
            food_name = FOOD_NAME_MAPPING.get(result, '')
            food_description = FOOD_DESCRIPTION_MAPPING.get(result, '')
            data = {
                'id': food.id,
                'image': food.image.url,
                'food_name': food_name,
                'food_description': food_description
            }
            print (data)
            return JsonResponse(data)
        else:
            return HttpResponseBadRequest()
    else:
        return HttpResponseNotAllowed(['POST'])


def foods_view(request):
    foods = Food.objects.all()
    data = []
    for food in foods:
        data.append({
            'id': food.id,
            'image': food.image.url
            })
    if request.method == 'GET':
        return JsonResponse(data, safe=False)
    else:
        return HttpResponseNotAllowed(['GET'])
