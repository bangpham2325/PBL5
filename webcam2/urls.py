from django.urls import path
from .views import video_cam2, count_vehicle, index

app_names = "webcam2"
urlpatterns = [
    path('', video_cam2, name='webcam2'),
    path('index2', index, name='index2'),
    path('count_vehicle', count_vehicle, name='count_vehicle'),
]
