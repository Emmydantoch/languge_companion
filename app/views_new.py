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
    """Use Hugging Face Inference API for grammar correction with multiple model fallbacks"""
    # List of grammar correction models to try in order of preference
    models = [
        "grammarly/coedit-large",  # More comprehensive grammar correction
        "pszemraj/flan-t5-large-grammar-synthesis",  # Alternative grammar model
        "vennify/t5-base-grammar-correction"  # Original fallback
    ]
    
    for model in models:
        try:
            API_URL = f"https://api-inference.huggingface.co/models/{model}"
            headers = {"Authorization": f"Bearer {api_key}"}
            
            # Different input formats for different models
            if "coedit" in model:
                # CoEdit model expects specific format
                payload = {
                    "inputs": f"Fix grammar: {text}",
                    "parameters": {
                        "max_new_tokens": 256,
                        "temperature": 0.1,
                        "do_sample": False
                    }
                }
            elif "flan-t5" in model:
                # Flan-T5 model format
                payload = {
                    "inputs": f"Grammar: {text}",
                    "parameters": {
                        "max_length": 512,
                        "num_beams": 4,
                        "early_stopping": True,
                        "temperature": 0.1
                    }
                }
            else:
                # Original format for t5-base model
                payload = {
                    "inputs": f"grammar: {text}",
                    "parameters": {
                        "max_length": 512,
                        "num_beams": 4,
                        "early_stopping": True
                    }
                }
            
            response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    corrected_text = result[0].get("generated_text", text)
                    # Clean up the response (remove input prefix if present)
                    if corrected_text.startswith(("Fix grammar: ", "Grammar: ", "grammar: ")):
                        corrected_text = corrected_text.split(": ", 1)[1] if ": " in corrected_text else corrected_text
                    
                    # Only return if there's a meaningful change
                    if corrected_text.strip() and corrected_text.strip() != text.strip():
                        logging.info(f"Grammar correction successful with model: {model}")
                        return corrected_text.strip()
                    elif corrected_text.strip():
                        # Text is already correct
                        return corrected_text.strip()
                        
                # Try next model if no meaningful result
                logging.warning(f"Model {model} returned empty or unchanged result")
                continue
                
            elif response.status_code == 503:
                # Model is loading, try next model
                logging.warning(f"Model {model} is loading (503), trying next model")
                continue
            else:
                logging.error(f"Hugging Face API error for {model}: {response.status_code} - {response.text}")
                continue
                
        except Exception as e:
            logging.error(f"Error calling Hugging Face API for {model}: {str(e)}")
            continue
    
    # All models failed
    logging.error("All Hugging Face grammar models failed")
    return None

def correct_spelling_with_hf_api(text, api_key):
    """Use Hugging Face Inference API for spelling correction with multiple model fallbacks"""
    # List of spelling correction models to try
    models = [
        "oliverguhr/spelling-correction-english-base",
        "microsoft/DialoGPT-medium",  # Fallback that can handle spelling
        "t5-base"  # General text-to-text model
    ]
    
    for model in models:
        try:
            API_URL = f"https://api-inference.huggingface.co/models/{model}"
            headers = {"Authorization": f"Bearer {api_key}"}
            
            # Different input formats for different models
            if "spelling-correction" in model:
                payload = {
                    "inputs": text,
                    "parameters": {
                        "max_length": 512,
                        "num_beams": 4,
                        "early_stopping": True,
                        "temperature": 0.1
                    }
                }
            elif "DialoGPT" in model:
                payload = {
                    "inputs": f"Fix spelling: {text}",
                    "parameters": {
                        "max_length": len(text) + 50,
                        "temperature": 0.3,
                        "do_sample": True,
                        "pad_token_id": 50256
                    }
                }
            else:  # t5-base
                payload = {
                    "inputs": f"spelling: {text}",
                    "parameters": {
                        "max_length": 512,
                        "num_beams": 4,
                        "early_stopping": True
                    }
                }
            
            response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    corrected_text = result[0].get("generated_text", text)
                    
                    # Clean up response
                    if corrected_text.startswith(("Fix spelling: ", "spelling: ")):
                        corrected_text = corrected_text.split(": ", 1)[1] if ": " in corrected_text else corrected_text
                    
                    # Only return if there's a meaningful change or valid text
                    if corrected_text.strip():
                        logging.info(f"Spelling correction successful with model: {model}")
                        return corrected_text.strip()
                        
                # Try next model if no result
                continue
                
            elif response.status_code == 503:
                logging.warning(f"Model {model} is loading, trying next model")
                continue
            else:
                logging.error(f"Hugging Face Spelling API error for {model}: {response.status_code} - {response.text}")
                continue
                
        except Exception as e:
            logging.error(f"Error calling Hugging Face Spelling API for {model}: {str(e)}")
            continue
    
    # All models failed
    logging.error("All Hugging Face spelling models failed")
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
                    # Fallback to using language_tool_python with enhanced rules
                    try:
                        # Use LanguageTool with more comprehensive rule checking
                        tool = language_tool_python.LanguageTool('en-US')
                        
                        # Enable additional rule categories for better grammar checking
                        tool.disabled_rules = []  # Enable all rules
                        
                        matches = tool.check(original_text)
                        corrected = tool.correct(original_text)
                        
                        # Convert matches to errors format with more detail
                        errors = []
                        for match in matches:
                            errors.append({
                                'message': match.message,
                                'context': match.context,
                                'suggestions': match.replacements[:5],  # Show more suggestions
                                'rule': match.ruleId,
                                'category': match.category,
                                'offset': match.offset,
                                'length': match.errorLength
                            })
                        
                        logging.info(f"Grammar correction completed using LanguageTool. Found {len(matches)} issues.")
                        tool.close()
                        
                    except Exception as lt_error:
                        logging.error(f"LanguageTool error: {str(lt_error)}")
                        error_message = "Grammar correction not available. Please check if Java is installed for LanguageTool."
                        corrected = original_text
                else:
                    # Use Hugging Face Inference API
                    logging.info("Running grammar correction via Hugging Face API...")
                    corrected = correct_grammar_with_hf_api(original_text, api_key)
                    
                    if corrected is None:
                        # Fallback to LanguageTool if API fails
                        logging.info("Hugging Face API failed, falling back to LanguageTool")
                        try:
                            tool = language_tool_python.LanguageTool('en-US')
                            
                            # Enable comprehensive grammar checking
                            tool.disabled_rules = []  # Enable all rules for thorough checking
                            
                            matches = tool.check(original_text)
                            corrected = tool.correct(original_text)
                            
                            # Convert matches to errors format with enhanced details
                            errors = []
                            for match in matches:
                                errors.append({
                                    'message': match.message,
                                    'context': match.context,
                                    'suggestions': match.replacements[:5],  # More suggestions
                                    'rule': match.ruleId,
                                    'category': match.category,
                                    'offset': match.offset,
                                    'length': match.errorLength,
                                    'type': 'LanguageTool'
                                })
                            
                            logging.info(f"Grammar correction completed using LanguageTool fallback. Found {len(matches)} issues.")
                            tool.close()
                            
                        except Exception as lt_error:
                            logging.error(f"LanguageTool fallback error: {str(lt_error)}")
                            error_message = "Grammar correction services unavailable. Please ensure Java is installed for offline grammar checking."
                            corrected = original_text
                    else:
                        logging.info("Grammar correction completed via Hugging Face API")
                        
                        # Create enhanced error format for API result
                        if corrected != original_text:
                            errors = [{
                                'message': "Grammar and style corrections applied",
                                'context': original_text,
                                'suggestions': [corrected],
                                'rule': "AI_GRAMMAR_CORRECTION",
                                'category': "Grammar",
                                'type': 'HuggingFace_AI',
                                'improvement': 'Comprehensive grammar and style correction'
                            }]
                        else:
                            errors = [{
                                'message': "No grammar issues detected",
                                'context': original_text,
                                'suggestions': [],
                                'rule': "NO_ISSUES",
                                'category': "Grammar",
                                'type': 'HuggingFace_AI'
                            }]
                    
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
        text = request.POST.get("text", "").strip()
        target_lang = request.POST.get("target_lang", "en")
        
        if not text:
            error = "Please provide text to translate."
        else:
            try:
                # Primary translation using GoogleTranslator
                logging.info(f"Translating text to {target_lang}: {text[:50]}...")
                translator = GoogleTranslator(source='auto', target=target_lang)
                translated = translator.translate(text)
                
                if not translated or translated == text:
                    # Fallback: try with explicit source language detection
                    try:
                        from deep_translator import single_detection
                        detected_lang = single_detection(text, api_key='free')
                        if detected_lang != target_lang:
                            translator = GoogleTranslator(source=detected_lang, target=target_lang)
                            translated = translator.translate(text)
                            logging.info(f"Translation successful with detected source: {detected_lang}")
                        else:
                            translated = text  # Same language, no translation needed
                            logging.info("Source and target languages are the same")
                    except Exception as fallback_error:
                        logging.error(f"Language detection fallback failed: {str(fallback_error)}")
                        translated = text
                        error = "Translation completed but may not be optimal. Language detection failed."
                else:
                    logging.info("Translation completed successfully")
                    
            except Exception as e:
                logging.error(f"Translation error: {str(e)}")
                error = f"Translation error: {str(e)}"
                
                # Final fallback: return original text with error message
                translated = text
                
                # Try to provide a more helpful error message
                if "target language" in str(e).lower():
                    error = f"Unsupported target language '{target_lang}'. Please try a different language code."
                elif "source language" in str(e).lower():
                    error = "Could not detect source language. Please try with clearer text."
                elif "connection" in str(e).lower() or "network" in str(e).lower():
                    error = "Network error. Please check your internet connection and try again."
                else:
                    error = f"Translation service temporarily unavailable: {str(e)}"
    
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
