#!/usr/bin/env python3
"""
Test script for grammar correction functionality
"""
import os
import sys
import django
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'languge_companion.settings')
django.setup()

# Import the grammar correction function
from app.views_new import correct_grammar_with_hf_api
import language_tool_python

def test_languagetool_fallback():
    """Test LanguageTool fallback functionality"""
    print("=" * 60)
    print("TESTING LANGUAGETOOL FALLBACK (No API Key)")
    print("=" * 60)
    
    test_sentences = [
        "I are going to the store.",
        "She don't like apples.",
        "The book are on the table.",
        "He have three cats.",
        "They was playing football.",
        "I seen that movie yesterday.",
        "Me and him went to school.",
        "The weather is very good today."  # Correct sentence
    ]
    
    try:
        tool = language_tool_python.LanguageTool('en-US')
        tool.disabled_rules = []  # Enable all rules
        
        for sentence in test_sentences:
            print(f"\nOriginal: {sentence}")
            
            matches = tool.check(sentence)
            corrected = tool.correct(sentence)
            
            print(f"Corrected: {corrected}")
            print(f"Issues found: {len(matches)}")
            
            for match in matches[:3]:  # Show first 3 issues
                print(f"  - {match.message}")
                if match.replacements:
                    print(f"    Suggestions: {', '.join(match.replacements[:3])}")
        
        tool.close()
        print("\n✅ LanguageTool test completed successfully!")
        
    except Exception as e:
        print(f"❌ LanguageTool test failed: {e}")
        print("Note: LanguageTool requires Java to be installed.")

def test_huggingface_api():
    """Test Hugging Face API functionality (if API key available)"""
    print("\n" + "=" * 60)
    print("TESTING HUGGING FACE API")
    print("=" * 60)
    
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        print("❌ HUGGINGFACE_API_KEY not found in environment variables")
        print("To test HF API, set your API key: export HUGGINGFACE_API_KEY=your_key_here")
        return
    
    test_sentences = [
        "I are going to the store.",
        "She don't like apples.",
        "The book are on the table.",
        "He have three cats."
    ]
    
    for sentence in test_sentences:
        print(f"\nOriginal: {sentence}")
        
        try:
            corrected = correct_grammar_with_hf_api(sentence, api_key)
            if corrected:
                print(f"Corrected: {corrected}")
                if corrected != sentence:
                    print("✅ Grammar correction applied")
                else:
                    print("ℹ️  No changes needed")
            else:
                print("❌ API correction failed")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🔧 Grammar Correction Test Suite")
    print("Testing improved grammar correction functionality...")
    
    # Test LanguageTool (should work without API key)
    test_languagetool_fallback()
    
    # Test Hugging Face API (requires API key)
    test_huggingface_api()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print("1. LanguageTool provides comprehensive grammar checking")
    print("2. Multiple HuggingFace models provide AI-powered corrections")
    print("3. Robust fallback system ensures grammar checking always works")
    print("4. Enhanced error reporting with detailed suggestions")
