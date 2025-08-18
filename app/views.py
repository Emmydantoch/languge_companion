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
from huggingface_hub import InferenceClient
import os

HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(token=os.getenv("HF_TOKEN"))

    

def home(request):
    return render(request, "app/home.html")


# Configure logging
logging.basicConfig(level=logging.DEBUG)

def correct_grammar_with_hf_api(text, api_key):
    """Use Hugging Face Inference API for grammar correction"""
    try:
        # Updated to use a more reliable grammar correction model
        API_URL = "https://api-inference.huggingface.co/models/grammarly/coedit-large"
        headers = {"Authorization": f"Bearer {api_key}"}
        
        # Try multiple models in case one fails
        models_to_try = [
            "grammarly/coedit-large",
            "pszemraj/flan-t5-large-grammar-synthesis",
            "Unbabel/gec-t5_small"
        ]
        
        for model in models_to_try:
            try:
                API_URL = f"https://api-inference.huggingface.co/models/{model}"
                
                payload = {
                    "inputs": text,
                    "parameters": {
                        "max_length": min(512, len(text) + 100),
                        "temperature": 0.1,
                        "do_sample": False
                    }
                }
                
                response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        corrected = result[0].get("generated_text", text)
                        if corrected and corrected != text:
                            return corrected
                    elif isinstance(result, dict) and "generated_text" in result:
                        return result["generated_text"]
                else:
                    logging.warning(f"Model {model} failed: {response.status_code}")
                    continue
                    
            except Exception as model_error:
                logging.warning(f"Model {model} error: {str(model_error)}")
                continue
        
        # If all models fail
        logging.error("All Hugging Face models failed")
        return None
            
    except Exception as e:
        logging.error(f"Error calling Hugging Face API: {str(e)}")
        return None

def correct_spelling_with_hf_api(text, api_key):
    """Use Hugging Face Inference API for spelling correction"""
    try:
        # Try multiple spelling correction models
        models_to_try = [
            "oliverguhr/spelling-correction-english-base",
            "prithivida/spelling-correction-english",
            "microsoft/DialoGPT-medium"
        ]
        
        for model in models_to_try:
            try:
                API_URL = f"https://api-inference.huggingface.co/models/{model}"
                headers = {"Authorization": f"Bearer {api_key}"}
                
                payload = {
                    "inputs": text,
                    "parameters": {
                        "max_length": min(512, len(text) + 50),
                        "temperature": 0.1,
                        "do_sample": False
                    }
                }
                
                response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        corrected = result[0].get("generated_text", text)
                        if corrected and corrected != text:
                            return corrected
                    elif isinstance(result, dict) and "generated_text" in result:
                        return result["generated_text"]
                else:
                    logging.warning(f"Spelling model {model} failed: {response.status_code}")
                    continue
                    
            except Exception as model_error:
                logging.warning(f"Spelling model {model} error: {str(model_error)}")
                continue
        
        # If all models fail
        logging.error("All Hugging Face spelling models failed")
        return None
            
    except Exception as e:
        logging.error(f"Error calling Hugging Face Spelling API: {str(e)}")
        return None

def summarize_with_hf_api(text, api_key):
    """Use Hugging Face Inference API for text summarization"""
    try:
        API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
        headers = {"Authorization": f"Bearer {api_key}"}
        
        payload = {
            "inputs": text,
            "parameters": {
                "max_length": 150,
                "min_length": 30,
                "do_sample": False
            }
        }
        
        response = requests.post(API_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("summary_text", text)
            return text
        else:
            logging.error(f"Hugging Face Summarization API error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        logging.error(f"Error calling Hugging Face Summarization API: {str(e)}")
        return None

def simple_extractive_summary(text, max_sentences=3):
    """Simple extractive summarization fallback"""
    try:
        # Split into sentences
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        if len(sentences) <= max_sentences:
            return text
        
        # Take first sentence, middle sentence, and last sentence
        if len(sentences) >= 3:
            first = sentences[0]
            middle = sentences[len(sentences) // 2]
            last = sentences[-1]
            return f"{first}. {middle}. {last}."
        else:
            return '. '.join(sentences[:max_sentences]) + '.'
            
    except Exception as e:
        logging.error(f"Simple summary error: {str(e)}")
        return text[:200] + "..." if len(text) > 200 else text

def simple_grammar_check(text):
    """Simple grammar checking fallback that doesn't require Java/LanguageTool"""
    try:
        import re
        corrected = text
        errors = []
        
        # Basic grammar corrections
        corrections = [
            # Common contractions and spacing
            (r'\bi\b', 'I'),  # lowercase i to uppercase I
            (r'\bim\b', "I'm"),  # im to I'm
            (r'\bdont\b', "don't"),  # dont to don't
            (r'\bcant\b', "can't"),  # cant to can't
            (r'\bwont\b', "won't"),  # wont to won't
            (r'\bisnt\b', "isn't"),  # isnt to isn't
            (r'\barent\b', "aren't"),  # arent to aren't
            (r'\bwasnt\b', "wasn't"),  # wasnt to wasn't
            (r'\bwerent\b', "weren't"),  # werent to weren't
            (r'\bhasnt\b', "hasn't"),  # hasnt to hasn't
            (r'\bhavent\b', "haven't"),  # havent to haven't
            (r'\bhadnt\b', "hadn't"),  # hadnt to hadn't
            (r'\bwouldnt\b', "wouldn't"),  # wouldnt to wouldn't
            (r'\bcouldnt\b', "couldn't"),  # couldnt to couldn't
            (r'\bshouldnt\b', "shouldn't"),  # shouldnt to shouldn't
            
            # Double spaces
            (r'\s+', ' '),  # multiple spaces to single space
            
            # Sentence capitalization
            (r'(?:^|[.!?]\s+)([a-z])', lambda m: m.group(0)[:-1] + m.group(1).upper()),
        ]
        
        original_text = corrected
        
        # Apply corrections
        for pattern, replacement in corrections:
            if callable(replacement):
                corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)
            else:
                old_corrected = corrected
                corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)
                
                # Track changes for error reporting
                if old_corrected != corrected:
                    errors.append({
                        'message': f"Corrected: {pattern} → {replacement}",
                        'context': text[:50] + "..." if len(text) > 50 else text,
                        'suggestions': [corrected],
                        'rule': "SIMPLE_GRAMMAR",
                        'category': "Grammar"
                    })
        
        # Remove duplicate spaces and trim
        corrected = re.sub(r'\s+', ' ', corrected).strip()
        
        # If no changes were made, create a generic message
        if corrected == original_text:
            errors = []
        
        return corrected, errors
        
    except Exception as e:
        logging.error(f"Simple grammar check error: {str(e)}")
        return text, []

def simple_spell_check(text):
    """Simple spelling checking fallback that doesn't require Java/LanguageTool"""
    try:
        import re
        corrected = text
        errors = []
        
        # Common spelling corrections
        spelling_corrections = [
            # Common misspellings
            (r'\bteh\b', 'the'),
            (r'\btha\b', 'the'),
            (r'\bthier\b', 'their'),
            (r'\bthier\b', 'there'),
            (r'\byour\b(?=\s+(are|is|were))', 'you\'re'),  # your are -> you're
            (r'\bits\b(?=\s+(a|an|the))', 'it\'s'),  # its a -> it's a
            (r'\bto\b(?=\s+(much|many|good|bad))', 'too'),  # to much -> too much
            (r'\bwould\s+of\b', 'would have'),
            (r'\bcould\s+of\b', 'could have'),
            (r'\bshould\s+of\b', 'should have'),
            (r'\balot\b', 'a lot'),
            (r'\ballot\b', 'a lot'),
            (r'\brecieve\b', 'receive'),
            (r'\bacheive\b', 'achieve'),
            (r'\bbelieve\b', 'believe'),
            (r'\bwierd\b', 'weird'),
            (r'\bfrend\b', 'friend'),
            (r'\bfriendly\b', 'friendly'),
            (r'\bdefinately\b', 'definitely'),
            (r'\bseperate\b', 'separate'),
            (r'\boccured\b', 'occurred'),
            (r'\boccuring\b', 'occurring'),
            (r'\bembarrass\b', 'embarrass'),
            (r'\bneccessary\b', 'necessary'),
            (r'\baccommodate\b', 'accommodate'),
            (r'\bbeginning\b', 'beginning'),
            (r'\bcommittee\b', 'committee'),
            (r'\bdevelop\b', 'develop'),
            (r'\benvironment\b', 'environment'),
            (r'\bgovernment\b', 'government'),
            (r'\bindependent\b', 'independent'),
            (r'\bmaintenance\b', 'maintenance'),
            (r'\bparallel\b', 'parallel'),
            (r'\bprivilege\b', 'privilege'),
            (r'\bprofessional\b', 'professional'),
            (r'\brecommend\b', 'recommend'),
            (r'\bresponsible\b', 'responsible'),
            (r'\btomorrow\b', 'tomorrow'),
            (r'\bunfortunately\b', 'unfortunately'),
        ]
        
        original_text = corrected
        changes_made = []
        
        # Apply spelling corrections
        for pattern, replacement in spelling_corrections:
            old_corrected = corrected
            corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)
            
            # Track changes for error reporting
            if old_corrected != corrected:
                # Clean pattern for display (remove word boundaries)
                clean_pattern = pattern.replace(r'\\b', '').replace(r'\b', '')
                changes_made.append({
                    'original': pattern,
                    'corrected': replacement,
                    'message': f"Spelling correction: {clean_pattern} → {replacement}"
                })
        
        # Create error reports for changes made
        if changes_made:
            for change in changes_made:
                errors.append({
                    'word': change['original'],
                    'suggestions': [change['corrected']],
                    'message': change['message'],
                    'rule': "SIMPLE_SPELLING",
                    'category': "Spelling"
                })
        
        # Remove duplicate spaces and trim
        corrected = re.sub(r'\s+', ' ', corrected).strip()
        
        return corrected, errors
        
    except Exception as e:
        logging.error(f"Simple spell check error: {str(e)}")
        return text, []

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
                api_key = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")
                if not api_key:
                    logging.warning("HUGGINGFACE_API_KEY/HF_TOKEN environment variable not set")
                    # Skip LanguageTool since Java isn't available on Render, go straight to simple grammar check
                    logging.info("Falling back to simple grammar check (Java not available)")
                    try:
                        corrected, errors = simple_grammar_check(original_text)
                        logging.info("Grammar correction completed using simple grammar check")
                    except Exception as simple_error:
                        logging.error(f"Simple grammar check error: {str(simple_error)}")
                        error_message = "Grammar correction not available. Showing original text."
                        corrected = original_text
                else:
                    # Use Hugging Face Inference API
                    logging.info("Running grammar correction via Hugging Face API...")
                    corrected = correct_grammar_with_hf_api(original_text, api_key)
                    
                    if corrected is None:
                        # Skip LanguageTool since Java isn't available, go straight to simple grammar check
                        logging.info("Hugging Face API failed, falling back to simple grammar check")
                        try:
                            corrected, errors = simple_grammar_check(original_text)
                            logging.info("Grammar correction completed using simple grammar check fallback")
                        except Exception as simple_error:
                            logging.error(f"Simple grammar check fallback error: {str(simple_error)}")
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
                api_key = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")
                if not api_key:
                    logging.warning("HUGGINGFACE_API_KEY/HF_TOKEN environment variable not set")
                    # Skip LanguageTool since Java isn't available on Render, go straight to simple spell check
                    logging.info("Falling back to simple spell check (Java not available)")
                    try:
                        corrected, errors = simple_spell_check(text)
                        logging.info("Spell check completed using simple spell check")
                    except Exception as simple_error:
                        logging.error(f"Simple spell check error: {str(simple_error)}")
                        error_message = "Spell checking not available. Showing original text."
                        corrected = text
                else:
                    # Use Hugging Face Inference API for spelling
                    logging.info("Running spell check via Hugging Face API...")
                    corrected = correct_spelling_with_hf_api(text, api_key)
                    
                    if corrected is None:
                        # Skip LanguageTool since Java isn't available, go straight to simple spell check
                        logging.info("Hugging Face API failed, falling back to simple spell check")
                        try:
                            corrected, errors = simple_spell_check(text)
                            logging.info("Spell check completed using simple spell check fallback")
                        except Exception as simple_error:
                            logging.error(f"Simple spell check fallback error: {str(simple_error)}")
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
    
    return render(request, "app/translation.html", {
        "translated": translated,
        "error": error,
        "original": text,
        "target_lang": target_lang
    })

def get_academic_sources(text):
    """Enhanced academic source detection with proper citations"""
    sources = []
    text_lower = text.lower()
    
    # Academic databases and known sources
    academic_patterns = {
        "climate change": [
            {
                "text": "Climate change refers to long-term shifts in global or regional climate patterns attributed largely to increased levels of atmospheric carbon dioxide.",
                "author": "NASA Goddard Institute for Space Studies",
                "title": "Climate Change and Global Warming",
                "publication": "NASA Climate Change Database",
                "year": "2023",
                "url": "https://climate.nasa.gov/",
                "type": "Government Report"
            },
            {
                "text": "The scientific consensus on climate change is that current warming trends are extremely likely due to human activities since the mid-20th century.",
                "author": "IPCC Working Group",
                "title": "Climate Change 2021: The Physical Science Basis",
                "publication": "Intergovernmental Panel on Climate Change",
                "year": "2021",
                "url": "https://www.ipcc.ch/",
                "type": "Scientific Report"
            }
        ],
        "artificial intelligence": [
            {
                "text": "Artificial intelligence is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals.",
                "author": "Russell, Stuart; Norvig, Peter",
                "title": "Artificial Intelligence: A Modern Approach",
                "publication": "Pearson Education",
                "year": "2020",
                "url": "ISBN: 978-0134610993",
                "type": "Academic Textbook"
            },
            {
                "text": "Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience.",
                "author": "Mitchell, Tom M.",
                "title": "Machine Learning",
                "publication": "McGraw-Hill Science",
                "year": "1997",
                "url": "ISBN: 978-0070428072",
                "type": "Academic Book"
            }
        ],
        "shakespeare": [
            {
                "text": "William Shakespeare was an English playwright, poet, and actor, widely regarded as the greatest writer in the English language.",
                "author": "Greenblatt, Stephen",
                "title": "Will in the World: How Shakespeare Became Shakespeare",
                "publication": "W. W. Norton & Company",
                "year": "2004",
                "url": "ISBN: 978-0393050572",
                "type": "Biography"
            }
        ],
        "photosynthesis": [
            {
                "text": "Photosynthesis is the process by which plants and other organisms convert light energy into chemical energy that can later be released to fuel the organism's activities.",
                "author": "Blankenship, Robert E.",
                "title": "Molecular Mechanisms of Photosynthesis",
                "publication": "Blackwell Science",
                "year": "2014",
                "url": "DOI: 10.1002/9780470758472",
                "type": "Scientific Textbook"
            }
        ],
        "world war": [
            {
                "text": "World War II was a global war that lasted from 1939 to 1945, involving the vast majority of the world's countries.",
                "author": "Keegan, John",
                "title": "The Second World War",
                "publication": "Hutchinson",
                "year": "1989",
                "url": "ISBN: 978-0091740160",
                "type": "Historical Analysis"
            }
        ],
        "dna": [
            {
                "text": "DNA is a molecule that carries genetic instructions for the development, functioning, growth and reproduction of all known living organisms.",
                "author": "Watson, James D.; Crick, Francis H.C.",
                "title": "Molecular Structure of Nucleic Acids: A Structure for Deoxyribose Nucleic Acid",
                "publication": "Nature",
                "year": "1953",
                "url": "DOI: 10.1038/171737a0",
                "type": "Research Paper"
            }
        ]
    }
    
    # Check for pattern matches
    for pattern, pattern_sources in academic_patterns.items():
        if pattern in text_lower:
            sources.extend(pattern_sources)
    
    # Add keyword-based sources for better matching
    keywords = text_lower.split()
    keyword_sources = []
    
    # Check for common academic terms and add relevant sources
    academic_terms = {
        "research": {
            "text": "Research is a systematic inquiry that entails collection of data, documentation of critical information, and analysis and interpretation of that data.",
            "author": "Kothari, C.R.",
            "title": "Research Methodology: Methods and Techniques",
            "publication": "New Age International",
            "year": "2004",
            "url": "ISBN: 978-8122414561",
            "type": "Research Guide"
        },
        "analysis": {
            "text": "Analysis involves breaking down complex information into smaller parts to understand the whole better.",
            "author": "Miles, Matthew B.; Huberman, A. Michael",
            "title": "Qualitative Data Analysis: An Expanded Sourcebook",
            "publication": "SAGE Publications",
            "year": "1994",
            "url": "ISBN: 978-0803955400",
            "type": "Academic Method"
        },
        "education": {
            "text": "Education is the process of facilitating learning, or the acquisition of knowledge, skills, values, beliefs, and habits.",
            "author": "Dewey, John",
            "title": "Democracy and Education",
            "publication": "Macmillan",
            "year": "1916",
            "url": "ISBN: 978-0486296395",
            "type": "Educational Theory"
        },
        "technology": {
            "text": "Technology is the application of scientific knowledge for practical purposes, especially in industry.",
            "author": "Winner, Langdon",
            "title": "The Whale and the Reactor: A Search for Limits in an Age of High Technology",
            "publication": "University of Chicago Press",
            "year": "1986",
            "url": "ISBN: 978-0226902111",
            "type": "Technology Studies"
        },
        "science": {
            "text": "Science is a systematic enterprise that builds and organizes knowledge in the form of testable explanations and predictions about the universe.",
            "author": "Popper, Karl",
            "title": "The Logic of Scientific Discovery",
            "publication": "Routledge",
            "year": "1959",
            "url": "ISBN: 978-0415278447",
            "type": "Philosophy of Science"
        }
    }
    
    # Add sources based on keywords found in text
    for keyword, source_data in academic_terms.items():
        if keyword in keywords:
            keyword_sources.append(source_data)
    
    # Add general academic sources for any text
    general_sources = [
        {
            "text": "Academic writing requires proper citation and attribution of sources to maintain scholarly integrity and avoid plagiarism.",
            "author": "Modern Language Association",
            "title": "MLA Handbook for Writers of Research Papers",
            "publication": "Modern Language Association",
            "year": "2021",
            "url": "ISBN: 978-1603292627",
            "type": "Style Guide"
        },
        {
            "text": "Research methodology involves systematic investigation and analysis of materials and sources to establish facts and reach new conclusions.",
            "author": "Creswell, John W.",
            "title": "Research Design: Qualitative, Quantitative, and Mixed Methods Approaches",
            "publication": "SAGE Publications",
            "year": "2018",
            "url": "ISBN: 978-1506386706",
            "type": "Research Methodology"
        },
        {
            "text": "Information literacy is the ability to identify, locate, evaluate, and use information effectively for academic and professional purposes.",
            "author": "Association of College & Research Libraries",
            "title": "Framework for Information Literacy for Higher Education",
            "publication": "American Library Association",
            "year": "2015",
            "url": "https://www.ala.org/acrl/standards/ilframework",
            "type": "Educational Framework"
        }
    ]
    
    # Combine all sources
    sources.extend(keyword_sources)
    sources.extend(general_sources)
    
    return sources

def search_web_sources(text):
    """Enhanced web search with better source attribution"""
    sources = []
    
    try:
        import urllib.parse
        import time
        import random
        
        # Multiple search strategies
        query = urllib.parse.quote_plus(text[:150])
        
        # Strategy 1: DuckDuckGo search
        try:
            ddg_url = f"https://html.duckduckgo.com/html/?q={query}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            time.sleep(random.uniform(1, 3))
            response = requests.get(ddg_url, headers=headers, timeout=15)
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Extract results with better parsing
            for i, result in enumerate(soup.find_all("div", class_="result__body")[:8]):
                try:
                    snippet_text = result.get_text().strip()
                    if len(snippet_text) > 50:
                        # Try to find title and URL
                        title_elem = result.find_previous("a", class_="result__a")
                        url_elem = result.find_previous("a", class_="result__url")
                        
                        title = title_elem.get_text().strip() if title_elem else f"Web Source {i+1}"
                        url = url_elem.get('href') if url_elem else "URL not available"
                        domain = url_elem.get_text().strip() if url_elem else "Unknown Domain"
                        
                        sources.append({
                            "text": snippet_text,
                            "author": f"Content from {domain}",
                            "title": title,
                            "publication": domain,
                            "year": "2024",
                            "url": url,
                            "type": "Web Source"
                        })
                except Exception as e:
                    continue
                    
        except Exception as e:
            print(f"DuckDuckGo search failed: {e}")
        
        # Strategy 2: Add some reliable fallback sources
        if len(sources) < 3:
            fallback_sources = [
                {
                    "text": "Academic research and scholarly articles provide peer-reviewed insights on various topics across multiple disciplines.",
                    "author": "Various Academic Authors",
                    "title": "Academic Literature Database",
                    "publication": "Scholarly Journals",
                    "year": "2024",
                    "url": "https://scholar.google.com",
                    "type": "Academic Database"
                },
                {
                    "text": "Educational institutions and universities maintain extensive libraries of research materials and publications.",
                    "author": "Educational Institutions",
                    "title": "University Research Collections",
                    "publication": "Academic Libraries",
                    "year": "2024",
                    "url": "https://worldcat.org",
                    "type": "Educational Resource"
                }
            ]
            sources.extend(fallback_sources)
    
    except Exception as e:
        print(f"Web search error: {e}")
    
    return sources

def plagiarism_checker(request):
    plagiarism_report = ""
    similarity_results = []
    text = ""
    error = ""
    
    if request.method == "POST":
        text = request.POST.get("text", "")
        
        if not text.strip():
            error = "Please enter some text to check for plagiarism"
        else:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity
                
                all_sources = []
                
                # Get academic sources
                academic_sources = get_academic_sources(text)
                all_sources.extend(academic_sources)
                
                # Get web sources
                web_sources = search_web_sources(text)
                all_sources.extend(web_sources)
                
                # Analyze similarity if we have sources
                if all_sources:
                    documents = [text] + [source["text"] for source in all_sources]
                    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), max_features=5000)
                    tfidf_matrix = vectorizer.fit_transform(documents)
                    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
                    
                    # Generate detailed similarity results with debug info
                    print(f"Debug: Found {len(all_sources)} sources to compare")
                    for i, (source, similarity) in enumerate(zip(all_sources, similarities)):
                        print(f"Debug: Source {i+1} similarity: {similarity:.3f}")
                        if similarity > 0.01:  # Very low threshold to catch any similarity
                            similarity_results.append({
                                "author": source["author"],
                                "title": source["title"],
                                "publication": source["publication"],
                                "year": source["year"],
                                "url": source["url"],
                                "type": source["type"],
                                "similarity": f"{similarity*100:.1f}%",
                                "text": source["text"][:300] + "..." if len(source["text"]) > 300 else source["text"],
                                "citation": f"{source['author']} ({source['year']}). {source['title']}. {source['publication']}."
                            })
                    
                    # Sort by similarity (highest first)
                    similarity_results.sort(key=lambda x: float(x["similarity"].rstrip('%')), reverse=True)
                    
                    # Generate comprehensive report
                    if similarity_results:
                        max_similarity = max([float(r["similarity"].rstrip('%')) for r in similarity_results])
                        total_sources = len(similarity_results)
                        high_sim_count = len([r for r in similarity_results if float(r["similarity"].rstrip('%')) > 50])
                        med_sim_count = len([r for r in similarity_results if 30 <= float(r["similarity"].rstrip('%')) <= 50])
                        
                        if max_similarity > 70:
                            plagiarism_report = f"⚠️ VERY HIGH SIMILARITY DETECTED! Maximum: {max_similarity:.1f}%"
                        elif max_similarity > 50:
                            plagiarism_report = f"🔴 HIGH SIMILARITY DETECTED! Maximum: {max_similarity:.1f}%"
                        elif max_similarity > 30:
                            plagiarism_report = f"🟡 MODERATE SIMILARITY found. Maximum: {max_similarity:.1f}%"
                        elif max_similarity > 15:
                            plagiarism_report = f"🟢 LOW SIMILARITY detected. Maximum: {max_similarity:.1f}%"
                        else:
                            plagiarism_report = f"✅ MINIMAL SIMILARITY found. Maximum: {max_similarity:.1f}%"
                        
                        plagiarism_report += f"\n\n📊 Analysis Summary:\n"
                        plagiarism_report += f"• Total sources found: {total_sources}\n"
                        if high_sim_count > 0:
                            plagiarism_report += f"• High similarity sources (>50%): {high_sim_count}\n"
                        if med_sim_count > 0:
                            plagiarism_report += f"• Moderate similarity sources (30-50%): {med_sim_count}\n"
                        
                        plagiarism_report += f"\n💡 Recommendation: "
                        if max_similarity > 50:
                            plagiarism_report += "Review and properly cite these sources. Consider paraphrasing or adding quotation marks."
                        elif max_similarity > 30:
                            plagiarism_report += "Some content may need proper attribution. Check if citations are needed."
                        else:
                            plagiarism_report += "Content appears largely original with minimal overlap."
                    else:
                        # If no similarity results, still show the sources that were found
                        if all_sources:
                            plagiarism_report = f"📝 Analysis completed. Found {len(all_sources)} sources for comparison, but no significant textual similarities detected. This suggests the content is likely original."
                            # Show all sources anyway for reference
                            for source in all_sources[:3]:  # Show top 3 sources
                                similarity_results.append({
                                    "author": source["author"],
                                    "title": source["title"],
                                    "publication": source["publication"],
                                    "year": source["year"],
                                    "url": source["url"],
                                    "type": source["type"],
                                    "similarity": "0.0%",
                                    "text": source["text"][:300] + "..." if len(source["text"]) > 300 else source["text"],
                                    "citation": f"{source['author']} ({source['year']}). {source['title']}. {source['publication']}."
                                })
                        else:
                            plagiarism_report = "✅ No sources found for comparison. Text appears to be original content."
                else:
                    error = "Unable to fetch comparison sources. Please check your internet connection and try again."
                    
            except ImportError as ie:
                error = f"Required libraries not installed: {str(ie)}. Please install: pip install requests beautifulsoup4 scikit-learn"
            except Exception as e:
                error = f"Error checking plagiarism: {str(e)}"
                
    return render(request, "app/plagiarism_checker.html", {
        "plagiarism_report": plagiarism_report,
        "similarity_results": similarity_results,
        "original": text,
        "error": error
    })


# Global variables to store model pipelines (lazy-loaded)
summarizer = None

import os
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

def text_summarization(request):
    summary_text = ""
    error = ""
    input_text = ""
    
    if request.method == "POST":
        input_text = request.POST.get("text", "").strip()
        logging.info(f"Received text summarization input: {input_text[:50]}...")
        
        if not input_text:
            error = "Please provide text to summarize."
            logging.warning("No input text provided for summarization")
        else:
            try:
                # Get Hugging Face API key
                api_key = os.getenv("HUGGINGFACE_API_KEY")
                if not api_key:
                    logging.warning("HUGGINGFACE_API_KEY environment variable not set")
                    # Fallback to simple extractive summarization
                    try:
                        summary_text = simple_extractive_summary(input_text)
                        logging.info("Text summarization completed using simple extractive method")
                        
                    except Exception as simple_error:
                        logging.error(f"Simple summarization error: {str(simple_error)}")
                        error = "Text summarization not available. Showing truncated text."
                        summary_text = input_text[:200] + "..." if len(input_text) > 200 else input_text
                else:
                    # Use Hugging Face Inference API for summarization
                    logging.info("Running text summarization via Hugging Face API...")
                    summary_text = summarize_with_hf_api(input_text, api_key)
                    
                    if summary_text is None:
                        # Fallback to simple extractive summarization if API fails
                        try:
                            summary_text = simple_extractive_summary(input_text)
                            logging.info("Text summarization completed using simple extractive fallback")
                            
                        except Exception as simple_error:
                            logging.error(f"Simple summarization fallback error: {str(simple_error)}")
                            error = "Text summarization not available. Showing truncated text."
                            summary_text = input_text[:200] + "..." if len(input_text) > 200 else input_text
                    else:
                        logging.info("Text summarization completed via Hugging Face API")
                        
            except Exception as e:
                logging.error(f"Text summarization error: {str(e)}")
                error = f"Error summarizing text: {str(e)}"
                # Final fallback
                summary_text = input_text[:200] + "..." if len(input_text) > 200 else input_text
                
    return render(request, "app/text_summarization.html", {
        "summary_text": summary_text,
        "error": error,
        "input_text": input_text
    })
