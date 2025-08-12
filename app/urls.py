from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("grammar/", views.grammar_check, name="grammar_check"),
    path("spell-check/", views.spell_check, name="spell_check"),
    path("translation/", views.translation, name="translation"),
    path("plagiarism-checker/", views.plagiarism_checker, name="plagiarism_checker"),
    path("text-summarization/", views.text_summarization, name="text_summarization"),
]
