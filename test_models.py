#!/usr/bin/env python3
"""
Test script to verify Groq AI models are working
"""

import os
from groq import Groq

def test_models():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ GROQ_API_KEY not found in environment")
        return
    
    client = Groq(api_key=api_key)
    
    models_to_test = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant", 
        "mixtral-8x7b-32768",
        "gemma2-9b-it"
    ]
    
    test_prompt = "What is soil pH? Answer in one sentence."
    
    for model in models_to_test:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": test_prompt}],
                max_tokens=50,
                temperature=0.3
            )
            print(f"✅ {model}: {response.choices[0].message.content[:50]}...")
        except Exception as e:
            print(f"❌ {model}: {str(e)[:100]}...")

if __name__ == "__main__":
    test_models()