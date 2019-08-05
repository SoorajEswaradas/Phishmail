
from django.contrib import admin
from django.urls import path

from .views import home
from accounts.views import login_view, register_view, logout_view, new_page


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', home),
    path('accounts/login/', login_view),
    path('accounts/register/', register_view),
    path('accounts/logout/', logout_view),
    path('accounts/newpage',  new_page,  name="my_function")

]
