from django.shortcuts import render
from django.http import HttpResponse
import language_tool_python
import deepl
from deep_translator import GoogleTranslator
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import os
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

def home(request):
    return render(request, "app/home.html")

def correct_grammar_with_hf_api(text, api_key):
    """Use Hugging Face Inference API for grammar correction"""
    try:
        API_URL = "https://api-inference.huggingface.co/models/vennify/t5-base-grammar-correction"
        headers = {"Authorization": f"Bearer {api_key}"}
        
        payload = {
            "inputs": f"grammar: {text}",
            "parameters": {
                "max_length": 512,
                "num_beams": 4,
                "early_stopping": True
            }
        }
        
        response = requests.post(API_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", text)
            return text
        else:
            logging.error(f"Hugging Face API error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        logging.error(f"Error calling Hugging Face API: {str(e)}")
        return None

def correct_spelling_with_hf_api(text, api_key):
    """Use Hugging Face Inference API for spelling correction"""
    try:
        API_URL = "https://api-inference.huggingface.co/models/oliverguhr/spelling-correction-english-base"
        headers = {"Authorization": f"Bearer {api_key}"}
        
        payload = {
            "inputs": text,
            "parameters": {
                "max_length": 512,
                "num_beams": 4,
                "early_stopping": True
            }
        }
        
        response = requests.post(API_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", text)
            return text
        else:
            logging.error(f"Hugging Face Spelling API error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        logging.error(f"Error calling Hugging Face Spelling API: {str(e)}")
        return None

def grammar_check(request):
    corrected = ""
    original_text = ""
    errors = []
    error_message = ""
    
    if request.method == "POST":
        original_text = request.POST.get("text", "").strip()
        logging.info(f"Received grammar check input: {original_text[:50]}...")
        
        if not original_text:
            error_message = "Please enter some text to check grammar."
            logging.warning("No input text provided for grammar check")
        else:
            try:
                # Get Hugging Face API key
                api_key = os.getenv("HUGGINGFACE_API_KEY")
                if not api_key:
                    logging.warning("HUGGINGFACE_API_KEY environment variable not set")
                    # Fallback to using language_tool_python
                    try:
                        tool = language_tool_python.LanguageTool('en-US')
                        matches = tool.check(original_text)
                        corrected = tool.correct(original_text)
                        
                        # Convert matches to errors format
                        errors = []
                        for match in matches:
                            errors.append({
                                'message': match.message,
                                'context': match.context,
                                'suggestions': match.replacements[:3],  # Limit to 3 suggestions
                                'rule': match.ruleId,
                                'category': match.category
                            })
                        
                        logging.info("Grammar correction completed using LanguageTool")
                        tool.close()
                        
                    except Exception as lt_error:
                        logging.error(f"LanguageTool error: {str(lt_error)}")
                        error_message = "Grammar correction not available. Showing original text."
                        corrected = original_text
                else:
                    # Use Hugging Face Inference API
                    logging.info("Running grammar correction via Hugging Face API...")
                    corrected = correct_grammar_with_hf_api(original_text, api_key)
                    
                    if corrected is None:
                        # Fallback to LanguageTool if API fails
                        try:
                            tool = language_tool_python.LanguageTool('en-US')
                            matches = tool.check(original_text)
                            corrected = tool.correct(original_text)
                            
                            # Convert matches to errors format
                            errors = []
                            for match in matches:
                                errors.append({
                                    'message': match.message,
                                    'context': match.context,
                                    'suggestions': match.replacements[:3],
                                    'rule': match.ruleId,
                                    'category': match.category
                                })
                            
                            logging.info("Grammar correction completed using LanguageTool fallback")
                            tool.close()
                            
                        except Exception as lt_error:
                            logging.error(f"LanguageTool fallback error: {str(lt_error)}")
                            error_message = "Grammar correction not available. Showing original text."
                            corrected = original_text
                    else:
                        logging.info("Grammar correction completed via Hugging Face API")
                        
                        # Create error format for API result
                        if corrected != original_text:
                            errors = [{
                                'message': "Grammar or style issue detected",
                                'context': original_text,
                                'suggestions': [corrected],
                                'rule': "GRAMMAR_CORRECTION",
                                'category': "Grammar"
                            }]
                        else:
                            errors = []
                    
            except Exception as e:
                logging.error(f"Grammar correction error: {str(e)}")
                error_message = f"Error checking grammar: {str(e)}"
                corrected = original_text  # Fallback to original text
                
    return render(request, "app/grammar.html", {
        "corrected": corrected,
        "original": original_text,
        "errors": errors,
        "error_message": error_message
    })

def spell_check(request):
    corrected = ""
    errors = []
    text = ""
    error_message = ""
    
    if request.method == "POST":
        text = request.POST.get("text", "").strip()
        logging.info(f"Received spell check input: {text[:50]}...")
        
        if not text:
            error_message = "Please enter some text to check spelling."
            logging.warning("No input text provided for spell check")
        else:
            try:
                # Get Hugging Face API key
                api_key = os.getenv("HUGGINGFACE_API_KEY")
                if not api_key:
                    logging.warning("HUGGINGFACE_API_KEY environment variable not set")
                    # Fallback to using language_tool_python for spelling
                    try:
                        tool = language_tool_python.LanguageTool('en-US')
                        matches = tool.check(text)
                        corrected = tool.correct(text)
                        
                        # Convert matches to errors format for spelling
                        errors = []
                        for match in matches:
                            # Focus on spelling-related errors
                            if 'SPELL' in match.ruleId or 'TYPO' in match.ruleId or match.category == 'TYPOS':
                                errors.append({
                                    'word': match.context,
                                    'suggestions': match.replacements[:5],  # Show up to 5 suggestions
                                    'message': match.message,
                                    'rule': match.ruleId,
                                    'category': match.category
                                })
                        
                        # If no spelling errors found, show all errors
                        if not errors:
                            for match in matches:
                                errors.append({
                                    'word': match.context,
                                    'suggestions': match.replacements[:3],
                                    'message': match.message,
                                    'rule': match.ruleId,
                                    'category': match.category
                                })
                        
                        logging.info("Spell check completed using LanguageTool")
                        tool.close()
                        
                    except Exception as lt_error:
                        logging.error(f"LanguageTool spelling error: {str(lt_error)}")
                        error_message = "Spell checking not available. Showing original text."
                        corrected = text
                else:
                    # Use Hugging Face Inference API for spelling
                    logging.info("Running spell check via Hugging Face API...")
                    corrected = correct_spelling_with_hf_api(text, api_key)
                    
                    if corrected is None:
                        # Fallback to LanguageTool if API fails
                        try:
                            tool = language_tool_python.LanguageTool('en-US')
                            matches = tool.check(text)
                            corrected = tool.correct(text)
                            
                            # Convert matches to errors format for spelling
                            errors = []
                            for match in matches:
                                # Focus on spelling-related errors
                                if 'SPELL' in match.ruleId or 'TYPO' in match.ruleId or match.category == 'TYPOS':
                                    errors.append({
                                        'word': match.context,
                                        'suggestions': match.replacements[:5],
                                        'message': match.message,
                                        'rule': match.ruleId,
                                        'category': match.category
                                    })
                            
                            # If no spelling errors found, show all errors
                            if not errors:
                                for match in matches:
                                    errors.append({
                                        'word': match.context,
                                        'suggestions': match.replacements[:3],
                                        'message': match.message,
                                        'rule': match.ruleId,
                                        'category': match.category
                                    })
                            
                            logging.info("Spell check completed using LanguageTool fallback")
                            tool.close()
                            
                        except Exception as lt_error:
                            logging.error(f"LanguageTool spelling fallback error: {str(lt_error)}")
                            error_message = "Spell checking not available. Showing original text."
                            corrected = text
                    else:
                        logging.info("Spell check completed via Hugging Face API")
                        
                        # Create error format for API result
                        if corrected != text:
                            errors = [{
                                'word': text,
                                'suggestions': [corrected],
                                'message': "Spelling or grammar issue detected",
                                'rule': "SPELLING_CORRECTION",
                                'category': "Spelling"
                            }]
                        else:
                            errors = []
                        
            except Exception as e:
                logging.error(f"Spell check error: {str(e)}")
                error_message = f"Error in spell checking: {str(e)}"
                corrected = text  # Fallback to original text
    
    return render(request, "app/spell_check.html", {
        "corrected": corrected,
        "errors": errors,
        "original": text,
        "error_message": error_message
    })

def translation(request):
    translated = ""
    error = ""
    text = ""
    target_lang = "en"
    
    if request.method == "POST":
        text = request.POST.get("text", "")
        target_lang = request.POST.get("target_lang", "en")
        
        if not text:
            error = "Please provide text to translate."
        else:
            try:
                translator = GoogleTranslator(source='auto', target=target_lang)
                translated = translator.translate(text)
            except Exception as e:
                error = f"Translation error: {str(e)}"
                translated = text  # Fallback to original text
    
    return render(request, "app/translation.html", {
        "translated": translated,
        "error": error,
        "original": text,
        "target_lang": target_lang
    })

def plagiarism_checker(request):
    results = []
    error = ""
    text = ""
    
    if request.method == "POST":
        text = request.POST.get("text", "")
        
        if not text:
            error = "Please provide text to check for plagiarism."
        else:
            try:
                # Simple web search approach for plagiarism detection
                search_query = text[:100]  # Use first 100 characters for search
                search_url = f"https://www.google.com/search?q=\"{search_query}\""
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                
                response = requests.get(search_url, headers=headers)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    search_results = soup.find_all('div', class_='g')
                    
                    for result in search_results[:5]:  # Limit to 5 results
                        title_elem = result.find('h3')
                        link_elem = result.find('a')
                        
                        if title_elem and link_elem:
                            results.append({
                                'title': title_elem.get_text(),
                                'url': link_elem.get('href', ''),
                                'similarity': 'Potential match found'
                            })
                    
                    if not results:
                        results.append({
                            'title': 'No matches found',
                            'url': '',
                            'similarity': 'Text appears to be original'
                        })
                        
                else:
                    error = "Unable to perform plagiarism check at this time."
                    
            except Exception as e:
                error = f"Plagiarism check error: {str(e)}"
    
    return render(request, "app/plagiarism_checker.html", {
        "results": results,
        "error": error,
        "original": text
    })

# Text summarization with fallback approach
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
                # Simple extractive summarization fallback
                sentences = input_text.split('.')
                if len(sentences) > 3:
                    # Take first and last sentences as a simple summary
                    summary_text = sentences[0] + '. ' + sentences[-2] + '.'
                else:
                    summary_text = input_text
                    
            except Exception as e:
                error = f"Error summarizing text: {str(e)}"
                summary_text = input_text[:150] + "..." if len(input_text) > 150 else input_text
    
    return render(request, "app/text_summarization.html", {
        "summary_text": summary_text,
        "error": error,
        "input_text": input_text
    })
