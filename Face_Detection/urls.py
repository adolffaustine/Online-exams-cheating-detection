from django.urls import path
from .views import GazeDataAPIView, RegisterUserAPIView

urlpatterns = [
    path('api/gaze/', GazeDataAPIView.as_view(), name='gaze-data'),
    path('api/register/', RegisterUserAPIView.as_view(), name='register-user'),
    # path('api/login/', LoginUserAPIView.as_view(), name='login-user'),
]
