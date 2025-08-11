from django.shortcuts import render
from django.http import HttpResponse
import language_tool_python
import deepl
from deep_translator import GoogleTranslator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup
from transformers import pipeline



# paraphraser = pipeline("text2text-generation", model="t5-small", use_fast=False)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def home(request):
    return render(request, "core/home.html")


def grammar_check(request):
    corrected = ""
    if request.method == "POST":
        text = request.POST["text"]
        tool = language_tool_python.LanguageTool("en-US")
        matches = tool.check(text)
        corrected = language_tool_python.utils.correct(text, matches)
    return render(request, "core/grammar.html", {"corrected": corrected})

def spell_check(request):
    corrected = ""
    errors = []
    text = ""  # Initialize text variable for all request types
    
    if request.method == "POST":
        text = request.POST.get("text", "")
        try:
            tool = language_tool_python.LanguageTool('en-US')
            # Check text for spelling and grammar issues
            matches = tool.check(text)
            # Get corrected text
            corrected = tool.correct(text)
            # Extract spelling errors for display (optional)
            errors = [
                {
                    'word': match.context[match.offset:match.offset + match.errorLength],
                    'suggestions': match.replacements,
                    'message': match.message
                }
                for match in matches if 'spelling' in match.message.lower()
            ]
        except Exception as e:
            errors = [{'message': f"Error in spell checking: {str(e)}"}]
            corrected = text  # Fallback to original text
        finally:
            tool.close()  # Close the LanguageTool instance to free resources
    
    return render(request, "core/spell_check.html", {
        "corrected": corrected,
        "errors": errors,  # Pass errors for display in template
        "original": text
    })

def translation(request):
    translated = ""
    error = ""
    text = ""
    target_lang = "en"
    
    if request.method == "POST":
        text = request.POST.get("text", "")
        target_lang = request.POST.get("target_lang", "en")
        
        if not text.strip():
            error = "Please enter some text to translate"
        else:
            try:
                from deep_translator import GoogleTranslator
                translator = GoogleTranslator(source='auto', target=target_lang)
                translated = translator.translate(text)
            except ImportError:
                error = "deep-translator not installed. Run: pip install deep-translator"
                translated = text
            except Exception as e:
                error = f"Translation error: {str(e)}"
                translated = text
    
    return render(request, "core/translation.html", {
        "translated": translated,
        "error": error,
        "original": text,
        "target_lang": target_lang
    })

def plagiarism_checker(request):
    plagiarism_report = ""
    similarity_results = []
    text = ""  # Initialize text variable for all request types
    
    if request.method == "POST":
        text = request.POST.get("text", "")
        
        # Skip plagiarism check if text is empty
        if not text.strip():
            plagiarism_report = "Please enter some text to check for plagiarism"
        else:
            try:
                # Step 1: Search for similar content online (simplified Google search)
                query = text[:100].replace(" ", "+")  # Limit query length
                url = f"https://www.google.com/search?q={query}"
                headers = {"User-Agent": "Mozilla/5.0"}
                response = requests.get(url, headers=headers)
                soup = BeautifulSoup(response.text, "html.parser")
                
                # Extract snippets from search results
                snippets = [p.text for p in soup.find_all("div", class_="BNeawe s3v9rd AP7Wnd")][:3]  # Top 3 results

                # Only proceed if we found snippets
                if snippets:
                    # Step 2: Compare input text with snippets using TF-IDF
                    documents = [text] + snippets
                    vectorizer = TfidfVectorizer()
                    tfidf_matrix = vectorizer.fit_transform(documents)
                    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
                    
                    # Step 3: Generate report
                    similarity_results = [
                        {"source": f"Web Result {i+1}", "similarity": f"{sim*100:.2f}%", "text": snippets[i]}
                        for i, sim in enumerate(similarities) if sim > 0.2  # Threshold for matches
                    ]
                    
                    if similarity_results:
                        plagiarism_report = "Potential matches found."
                    else:
                        plagiarism_report = "No significant matches found. Text appears original."
                else:
                    plagiarism_report = "No search results found to compare against so you are safe."
                    
            except requests.RequestException as e:
                plagiarism_report = f"Error fetching search results: {str(e)}"
            except ImportError:
                plagiarism_report = "Required libraries not installed. Please install: pip install requests beautifulsoup4 scikit-learn"
            except Exception as e:
                plagiarism_report = f"Error checking plagiarism: {str(e)}"
    
    return render(request, "core/plagiarism_checker.html", {
        "plagiarism_report": plagiarism_report,
        "similarity_results": similarity_results,
        "original": text
    })



# Global variables to store model pipelines (lazy-loaded)
summarizer = None

def init_summarizer():
    global summarizer
    if summarizer is None:
        try:
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        except Exception as e:
            print(f"Error loading summarizer model: {e}")
            summarizer = None
    return summarizer


def text_summarization(request):
    summary_text = ""
    error = ""
    input_text = ""
    if request.method == "POST":
        input_text = request.POST.get("text", "")
        if not input_text:
            error = "Please provide text to summarize."
        else:
            try:
                # Initialize or get summarizer model
                model = init_summarizer()
                if model is None:
                    error = "Summarization model not available."
                else:
                    result = model(input_text, max_length=150, min_length=30, do_sample=False)
                    summary_text = result[0]["summary_text"]
            except Exception as e:
                error = f"Error summarizing text: {str(e)}"
                summary_text = input_text
    return render(request, "core/text_summarization.html", {
        "summary_text": summary_text,
        "error": error,
        "input_text": input_text
    })
    

# Create your views here.
