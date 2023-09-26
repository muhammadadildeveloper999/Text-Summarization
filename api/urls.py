from django.urls import path,include
from api.views import *


urlpatterns = [
    # path('SummarizeAPIView', SummarizeAPIView.as_view()),
    path('SummarizeView', SummarizeView.as_view()),

]