from django.db import models
from django.contrib.auth.models import User

class PracticeSession(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    input_text = models.TextField()
    corrected_text = models.TextField()
    session_type = models.CharField(max_length=20)  # grammar, speech, vocab
    timestamp = models.DateTimeField(auto_now_add=True)


# Create your models here.
