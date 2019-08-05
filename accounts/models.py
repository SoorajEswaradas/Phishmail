from django.db import models

class mails(models.Model):
    frommail = models.CharField(max_length=300)
    to = models.CharField(max_length=300)
    subject = models.TextField()
    body = models.TextField()
