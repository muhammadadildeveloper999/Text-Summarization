from django.contrib import admin
from django.urls import path,include

#for images

from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('api.urls')),


]+static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)


admin.site.site_header = 'tsummarized'