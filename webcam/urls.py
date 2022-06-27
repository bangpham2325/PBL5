from django.urls import path
from views import video_cam1, count_vehicle, index

app_names = "webcam"

urlpatterns = [
    path('', video_cam1, name='webcam1'),
    path('index1', index, name='webcam1'),
    path('count_vehicle', count_vehicle, name='count_vehicle'),
]
