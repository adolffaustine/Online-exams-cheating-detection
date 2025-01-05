from rest_framework import serializers
from .models import UserProfile
from .models import GazeData


class UserProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserProfile
        fields = ['profile','face_id','email']  # Adjust fields based on your model



class GazeDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = GazeData
        fields = '__all__'

class SuccessMessageSerializer(serializers.Serializer):
    success = serializers.BooleanField(default=False)
    message = serializers.CharField(max_length=255)