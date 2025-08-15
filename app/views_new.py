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

# Simple grammar rules for deployment-friendly fallback
def simple_grammar_fallback(text):
    """Simple rule-based grammar correction that works without external dependencies"""
    corrections = []
    corrected_text = text
    
    # Common grammar fixes
    grammar_rules = [
        # Subject-verb agreement fixes
        (r'\bI are\b', 'I am', 'Subject-verb agreement: "I are" should be "I am"'),
        (r'\bYou is\b', 'You are', 'Subject-verb agreement: "You is" should be "You are"'),
        (r'\bHe are\b', 'He is', 'Subject-verb agreement: "He are" should be "He is"'),
        (r'\bShe are\b', 'She is', 'Subject-verb agreement: "She are" should be "She is"'),
        (r'\bIt are\b', 'It is', 'Subject-verb agreement: "It are" should be "It is"'),
        (r'\bWe is\b', 'We are', 'Subject-verb agreement: "We is" should be "We are"'),
        (r'\bThey is\b', 'They are', 'Subject-verb agreement: "They is" should be "They are"'),
        
        # Common verb form errors
        (r'\bdon\'t\b(?=\s+(?:he|she|it))', 'doesn\'t', 'Verb form: "don\'t" should be "doesn\'t" with third person'),
        (r'\b(?:he|she|it)\s+don\'t\b', lambda m: m.group(0).replace("don't", "doesn't"), 'Third person singular'),
        (r'\bhave went\b', 'have gone', 'Past participle: "have went" should be "have gone"'),
        (r'\bhas went\b', 'has gone', 'Past participle: "has went" should be "has gone"'),
        (r'\bI seen\b', 'I saw', 'Past tense: "I seen" should be "I saw"'),
        (r'\bI done\b', 'I did', 'Past tense: "I done" should be "I did"'),
        
        # Double negatives
        (r'\bdon\'t\s+(?:have\s+)?no\b', 'don\'t have any', 'Double negative correction'),
        (r'\bcan\'t\s+get\s+no\b', 'can\'t get any', 'Double negative correction'),
        
        # Capitalization at sentence start
        (r'^([a-z])', lambda m: m.group(1).upper(), 'Sentence should start with capital letter'),
        (r'\. ([a-z])', lambda m: '. ' + m.group(1).upper(), 'Sentence should start with capital letter'),
    ]
    
    import re
    
    for pattern, replacement, message in grammar_rules:
        if callable(replacement):
            # Handle lambda functions
            matches = list(re.finditer(pattern, corrected_text, re.IGNORECASE))
            for match in reversed(matches):  # Reverse to maintain positions
                corrections.append({
                    'message': message,
                    'context': match.group(0),
                    'suggestions': [replacement(match)],
                    'rule': 'SIMPLE_GRAMMAR',
                    'category': 'Grammar',
                    'type': 'Simple_Fallback'
                })
                corrected_text = corrected_text[:match.start()] + replacement(match) + corrected_text[match.end():]
        else:
            # Handle string replacements
            if re.search(pattern, corrected_text, re.IGNORECASE):
                corrections.append({
                    'message': message,
                    'context': re.search(pattern, corrected_text, re.IGNORECASE).group(0),
                    'suggestions': [replacement],
                    'rule': 'SIMPLE_GRAMMAR',
                    'category': 'Grammar',
                    'type': 'Simple_Fallback'
                })
                corrected_text = re.sub(pattern, replacement, corrected_text, flags=re.IGNORECASE)
    
    return corrected_text, corrections

def home(request):
    return render(request, "app/home.html")

def correct_grammar_with_hf_api(text, api_key):
    """Use Hugging Face Inference API for grammar correction with verified working models"""
    # List of verified working grammar correction models
    models = [
        "microsoft/DialoGPT-medium",  # Conversational AI that can fix grammar
        "t5-base",  # Base T5 model for text-to-text generation
        "facebook/bart-base"  # BART model for text correction
    ]
    
    for model in models:
        try:
            API_URL = f"https://api-inference.huggingface.co/models/{model}"
            headers = {"Authorization": f"Bearer {api_key}"}
            
            # Different input formats for different models
            if "DialoGPT" in model:
                # DialoGPT format for grammar correction
                payload = {
                    "inputs": f"Fix the grammar in this sentence: {text}",
                    "parameters": {
                        "max_length": len(text) + 50,
                        "temperature": 0.3,
                        "do_sample": True,
                        "pad_token_id": 50256
                    }
                }
            elif "t5-base" in model:
                # T5 model format for grammar correction
                payload = {
                    "inputs": f"grammar: {text}",
                    "parameters": {
                        "max_length": 512,
                        "num_beams": 4,
                        "early_stopping": True,
                        "temperature": 0.1
                    }
                }
            elif "bart" in model:
                # BART model format
                payload = {
                    "inputs": f"Correct grammar: {text}",
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
                    prefixes_to_remove = [
                        "Fix the grammar in this sentence: ",
                        "Correct grammar: ",
                        "Fix grammar: ", 
                        "Grammar: ", 
                        "grammar: "
                    ]
                    
                    for prefix in prefixes_to_remove:
                        if corrected_text.startswith(prefix):
                            corrected_text = corrected_text[len(prefix):].strip()
                            break
                    
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
                    # Try LanguageTool first, then fallback to simple rules
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
                        logging.info("Falling back to simple grammar correction rules")
                        
                        # Use simple grammar fallback that works in any environment
                        corrected, simple_errors = simple_grammar_fallback(original_text)
                        errors = simple_errors
                        
                        if corrected != original_text:
                            logging.info(f"Simple grammar correction applied {len(errors)} fixes")
                        else:
                            logging.info("No grammar issues detected by simple rules")
                else:
                    # Use Hugging Face Inference API
                    logging.info("Running grammar correction via Hugging Face API...")
                    corrected = correct_grammar_with_hf_api(original_text, api_key)
                    
                    if corrected is None:
                        # Multi-level fallback system
                        logging.info("Hugging Face API failed, trying fallback options")
                        
                        # First try LanguageTool
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
                            logging.info("Using simple grammar rules as final fallback")
                            
                            # Final fallback: simple grammar rules that always work
                            corrected, simple_errors = simple_grammar_fallback(original_text)
                            errors = simple_errors
                            
                            if corrected != original_text:
                                logging.info(f"Simple grammar fallback applied {len(errors)} corrections")
                            else:
                                logging.info("No grammar issues detected by simple fallback rules")
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
                logging.info("Using emergency simple grammar fallback")
                
                # Emergency fallback: always works
                try:
                    corrected, simple_errors = simple_grammar_fallback(original_text)
                    errors = simple_errors
                    if corrected != original_text:
                        logging.info(f"Emergency fallback applied {len(errors)} corrections")
                except Exception as fallback_error:
                    logging.error(f"Even simple fallback failed: {str(fallback_error)}")
                    error_message = "Grammar correction temporarily unavailable. Please try again later."
                    corrected = original_text
                
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
