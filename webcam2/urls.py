from django.urls import path
from views import video_cam2, count_vehicle

app_names = "webcam2"
urlpatterns = [
    path('', video_cam2, name='webcam2'),
    path('count_vehicle', count_vehicle, name='count_vehicle'),
]
