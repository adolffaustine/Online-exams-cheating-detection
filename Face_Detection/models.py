from django.db import models
from django.db.models.signals import post_save

class UserProfile(models.Model):
    face_id = models.UUIDField(primary_key=True,max_length=100)
    # name = models.CharField(max_length=50)
    # address = models.CharField(max_length = 100)
    # phone = models.CharField(max_length =  10)
    email = models.CharField(max_length=50)
    profile = models.ImageField(upload_to='profile_image', blank=True)

    def __str__(self):
        return self.email
    

class GazeData(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    eye_position_right = models.CharField(max_length=10)
    face_status = models.CharField(max_length=10)
    message = models.CharField(max_length=10)
    # eye_position_left = models.CharField(max_length=10)
    # fps = models.FloatField()
   
