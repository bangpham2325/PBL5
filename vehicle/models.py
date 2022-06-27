from django.db import models



class VehicleSpeed(models.Model):
    date = models.DateField(blank=True,default=True,null=True)
    time = models.TimeField(default=True)
    camera = models.CharField(max_length=20)
    speed = models.FloatField()
    number_plate = models.TextField(max_length=20,default=True)
    vehicle_type = models.CharField(max_length=50)

    def hour(self):
        return self.time.hour