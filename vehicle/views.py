from calendar import month
from cmath import e
from dataclasses import dataclass
from time import time
from django.shortcuts import render
from numpy import arange
from .models import *
from .serializers import *
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, viewsets
from datetime import datetime
from django.db.models.functions import TruncMonth, TruncDay
from django.db.models import Count
from calendar import monthrange
import pandas as pd
import numpy as np
from django.db.models import Q
import os
# Create your views here.
import pyrebase
from dotenv import load_dotenv
from threading import Timer
from pathlib import Path
env_path = Path('.', '.env')
load_dotenv(dotenv_path=env_path)
config = {
    "apiKey": "AIzaSyBn5WdGXBiFegf_g6VD6mzDK7HwcuVXfCQ",
    "authDomain": "vehicle-1784e.firebaseapp.com",
    "databaseURL": "https://vehicle-1784e.firebaseio.com",
    "projectId": "vehicle-1784e",
    "storageBucket": "vehicle-1784e.appspot.com",
    "messagingSenderId": "939407755660",
    "appId": "1:939407755660:web:9014bdb0c480453032eea9",
    "measurementId": "G-C7WC170JGF"
}

firebase = pyrebase.initialize_app(config)
storage = firebase.storage()
# storage.child("cam1_vehicle.csv").download('', "cam1_vehicleIDdownload.csv")
getData_isready = False


def getData(delay_repeat=10):
    global getData_isready
    if not getData_isready:
        data = VehicleSpeed.objects.all()
        if data.count() >= 1:
            data.delete()
            pass
        storage.child("cam1_vehicle.csv").download('', "cam1_vehicle_download.csv")
        database = pd.read_csv('cam1_vehicle_download.csv')
        vehicle_cam1 = []
        for i in database.values:
            vehicle_cam1.append(VehicleSpeed(date=convert_datetime(i[1]),
                              time=i[2],
                              camera=i[3],
                              speed=float(i[4]),
                              number_plate="",
                              vehicle_type=i[5]))
        VehicleSpeed.objects.bulk_create(vehicle_cam1)
        storage.child("cam2_vehicle.csv").download('', "cam2_vehicleIDdownload.csv")
        database = pd.read_csv('cam2_vehicleIDdownload.csv')
        vehicle_cam2 = []
        for i in database.values:
            vehicle_cam2.append(VehicleSpeed(date=convert_datetime(i[1]),
                              time=i[2],
                              camera=i[3],
                              speed=float(i[4]),
                              number_plate="",
                              vehicle_type=i[5]))
        VehicleSpeed.objects.bulk_create(vehicle_cam2)
        getData_isready = True
        Timer(delay_repeat, getData, (delay_repeat,)).start()
        getData_isready = False


def convert_datetime(date):
    ar = str(date).split('/')
    return (str(ar[2]) + "-" + str(ar[0]) + "-" + str(ar[1]))


def cam(data, camera):
    try:
        pass
    except Exception:
        print("Loi o get camera ham camera trong file view")
    if (type(camera) == str and camera == "ALL"):
        return data
    if (type(camera) == str and camera != ""):
        data = data.filter(camera=camera)
        return data
    return data


def year(year, camera):
    data = VehicleSpeed.objects.filter(date__year=year)
    data = cam(data, camera)
    month__count = data.annotate(month=TruncMonth('date')).values('month').annotate(count=Count('id')).values('month',
                                                                                                              'count')
    month_number__count = {}
    for i in arange(1, 13):
        month_number__count[f'{i}'] = 0
    for i in month__count:
        var = i['month'].month
        month_number__count[f'{var}'] = i['count']
    print(month_number__count)
    context = get_vehicle_type(data)
    context['chart'] = month_number__count
    context['chart_tilte'] = "Month"
    return context


def filter_hour(data):
    _1 = data.filter(time__hour=1).count()
    _2 = data.filter(time__hour=2).count()
    _3 = data.filter(time__hour=3).count()
    _4 = data.filter(time__hour=4).count()
    _5 = data.filter(time__hour=5).count()
    _6 = data.filter(time__hour=6).count()
    _7 = data.filter(time__hour=7).count()
    _8 = data.filter(time__hour=8).count()
    _9 = data.filter(time__hour=9).count()
    _10 = data.filter(time__hour=10).count()
    _11 = data.filter(time__hour=11).count()
    _12 = data.filter(time__hour=12).count()
    _13 = data.filter(time__hour=13).count()
    _14 = data.filter(time__hour=14).count()
    _15 = data.filter(time__hour=15).count()
    _16 = data.filter(time__hour=16).count()
    _17 = data.filter(time__hour=17).count()
    _18 = data.filter(time__hour=18).count()
    _19 = data.filter(time__hour=19).count()
    _20 = data.filter(time__hour=20).count()
    _21 = data.filter(time__hour=21).count()
    _22 = data.filter(time__hour=22).count()
    _23 = data.filter(time__hour=23).count()
    _0 = data.filter(time__hour=0).count()
    context = {
        '0': _0,
        '1': _1,
        '2': _2,
        '3': _3,
        '4': _4,
        '5': _5,
        '6': _6,
        '7': _7,
        '8': _8,
        '9': _9,
        '10': _10,
        '11': _11,
        '12': _12,
        '13': _13,
        '14': _14,
        '15': _15,
        '16': _16,
        '17': _17,
        '18': _18,
        '19': _19,
        '20': _20,
        '21': _21,
        '22': _22,
        '23': _23,
    }
    return context


def day_month_year(date, camera):
    data = VehicleSpeed.objects.filter(date=date)
    data = cam(data, camera)
    context = get_vehicle_type(data)
    context['hour'] = filter_hour(data)
    context['chart_title'] = "Hour"
    return context


def get_vehicle_type(data):
    truck = data.filter(vehicle_type='Truck')
    car = data.filter(vehicle_type='Car')
    bus = data.filter(vehicle_type='Bus')
    motorcycle = data.filter(vehicle_type='Motorcycle')
    context = {
        "truck": truck.count(),
        "car": car.count(),
        "motorcycle": motorcycle.count(),
        "bus": bus.count(),
    }
    return context


def month_year(month, year, camera):
    data = VehicleSpeed.objects.filter(date__month=month, date__year=year)
    data = cam(data, camera)
    day__count = data.annotate(day=TruncDay('date')).values('day').annotate(count=Count('id')).values('day', 'count')
    day_number__count = {}
    for i in arange(1, (monthrange(int(year), int(month))[1]) + 1):
        day_number__count[f'{i}'] = 0
    for i in day__count:
        var = i['day'].day
        day_number__count[f'{var}'] = i['count']
    context = get_vehicle_type(data)
    context['chart'] = day_number__count
    context['chart_title'] = "Day"
    return context


class VehicleViewSet(APIView):
    def get(self, request):
        getData()
        data = VehicleSpeed.objects.values_list('camera')
        cam = []
        for i in data:
            if i[0] not in cam:
                cam.append(i[0])
        day = request.query_params.get('day')
        month = request.query_params.get('month')
        year_ = request.query_params.get('year')
        camera = request.query_params.get('camera')
        print(day, month, year_, camera)
        context = []
        if day != "" and month != "" and year_ != "":
            date = year_ + '-' + month + '-' + day
            context = day_month_year(date, camera)
        if day == "" and month == "" and year_ != "":
            context = year(year_, camera)
        if day == "" and month != "" and year_ != "":
            context = month_year((month), (year_), camera)
        return Response(context, status=status.HTTP_201_CREATED)

    def post(self, request):
        pass


class DetailView(APIView):
    def cam(self, data, camera):
        try:
            pass
        except Exception:
            print("Loi o get camera ham camera trong file view")
        if (type(camera) == str and camera == "ALL"):
            return data
        if (type(camera) == str and camera != ""):
            data = data.filter(camera=camera)
            return data
        return data

    def get_vehicle_type(self, data):
        truck = data.filter(vehicle_type='Truck').count()
        car = data.filter(vehicle_type='Car').count()
        bus = data.filter(vehicle_type='Bus').count()
        motorbycle = data.filter(vehicle_type='Motorcycle').count()
        context = {
            "truck": truck,
            "car": car,
            "bus": bus,
            "motorbycle": motorbycle,
            "all": data.count()}
        return context

    def year_filter(self, year, camera):
        data = VehicleSpeed.objects.filter(date__year=year)
        data = self.cam(data, camera)
        return self.get_vehicle_type(data)

    def month_year_filter(self, year, camera):
        data = VehicleSpeed.objects.filter(date__year=year)
        data = self.cam(data, camera)
        # month__count = data.annotate(month=TruncMonth('date')).values('month').annotate(count=Count('id')).values('month', 'count')
        month_number__count = {}
        for i in arange(1, 13):
            context_vehicle_type_month = {'data': self.get_vehicle_type(data.filter(date__month=i))}
            month_number__count[str(i)] = context_vehicle_type_month
        # print(month_number__count)
        return month_number__count

    def getHourAverage_year(self, year, cam):
        data = VehicleSpeed.objects.filter(date__year=year)
        data = self.cam(data, cam)
        # hour AM
        print(np.arange(0, 6))
        morning = {}
        afternoon = {}
        night = {}
        morning_hour = [5, 6, 7, 8, 9, 10, 11, 12]
        afternoon_hour = [13, 14, 15, 16, 17]
        night_hour = [18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4]
        for i in morning_hour:
            morning_data = data.filter(time__hour=i)
            morning[i] = get_vehicle_type(morning_data)
            morning[i]['count'] = morning_data.count()
        for i in afternoon_hour:
            afternoon_data = data.filter(time__hour=i)
            afternoon[i] = get_vehicle_type(afternoon_data)
            afternoon[i]['count'] = afternoon_data.count()
        for i in night_hour:
            night_data = data.filter(time__hour=i)
            night[i] = get_vehicle_type(night_data)
            night[i]['count'] = night_data.count()
            print(night[i]['count'])
        # season
        ___1 = data.filter(Q(date__month=1) | Q(date__month=2) | Q(date__month=3), date__year=year).count()
        ___2 = data.filter(Q(date__month=4) | Q(date__month=5) | Q(date__month=6), date__year=year).count()
        ___3 = data.filter(Q(date__month=7) | Q(date__month=8) | Q(date__month=9), date__year=year).count()
        ___4 = data.filter(Q(date__month=10) | Q(date__month=11) | Q(date__month=12), date__year=year).count()

        _4_season = [___1, ___2, ___3, ___4]
        print(_4_season)
        max_season = max(_4_season)
        min_season = min(_4_season)
        max_value = {}
        min_value = {}
        for i in np.arange(4):
            if _4_season[i] == max_season:
                max_value = {
                    'name': i + 1,
                    'quantity': max_season,
                    'percent': round((max_season / sum(_4_season)) * 100, 2)
                }
            if _4_season[i] == min_season:
                min_value = {
                    'name': i + 1,
                    'quantity': min_season,
                    'percent': round((min_season / sum(_4_season)) * 100, 2)
                }
        season = {
            'max': max_value,
            'min': min_value,
        }
        return season, morning, afternoon, night

    def get(self, request, cam="Cam1"):
        year = request.query_params.get('year')
        cam = request.query_params.get('cam')
        getData()
        list_years = VehicleSpeed.objects.dates('date', 'year')
        years = []
        context = {}
        for i in list_years:
            years.append(i.year)
            context[int(i.year)] = self.year_filter(int(i.year), cam)
        season, morning, afternoon, night = self.getHourAverage_year(year, cam)
        context['years'] = years
        context['year_now'] = year
        context['data_month'] = self.month_year_filter(camera=cam, year=int(year))
        context['morning'] = morning
        context['afternoon'] = afternoon
        context['night'] = night
        context['season'] = season
        return Response(context, status=status.HTTP_201_CREATED)