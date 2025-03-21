import os
from celery import Celery
from core_stack_backend_onprem.settings import INSTALLED_APPS

# set the default Django settings module for the 'celery' program.
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core_stack_backend_onprem.settings")

app = Celery("core_stack_backend_onprem")

# Using a string here means the worker doesn't
# have to serialize the configuration object to
# child processes. - namespace='CELERY' means all
# celery-related configuration keys should
# have a `CELERY_` prefix.
app.config_from_object("django.conf:settings")

# Load task modules from all registered Django app configs.
app.autodiscover_tasks(INSTALLED_APPS)
app.conf.broker_connection_retry_on_startup = True
