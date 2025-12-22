"""
OpenAI Utilities Module
Handles all OpenAI API interactions with proper error handling and logging.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("[ERROR] OPENAI_API_KEY not found in .env file")
    raise ValueError("OPENAI_API_KEY not found in .env file")

client = OpenAI(api_key=api_key)
print("[OPENAI_UTILS] OpenAI client initialized successfully")


def call_openai_with_system_user_prompt(system_prompt: str, user_prompt: str) -> str:
    """
    Call OpenAI API with system and user prompts.
    
    Args:
        system_prompt: The system prompt that defines the agent's behavior
        user_prompt: The user's question or message
        
    Returns:
        The assistant's response as a string
        
    Raises:
        ValueError: If no valid response is received
        RuntimeError: If the API call fails
    """
    print("[OPENAI_UTILS] Calling OpenAI API...")
    print(f"[OPENAI_UTILS] System prompt: {system_prompt[:100]}...")
    print(f"[OPENAI_UTILS] User prompt: {user_prompt[:100]}...")
    
    try:
        print("[OPENAI_UTILS] Preparing API request...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        print("[OPENAI_UTILS] API request completed successfully")
        
        # Validate response
        if response.choices and len(response.choices) > 0:
            assistant_response = response.choices[0].message.content
            print(f"[OPENAI_UTILS] Assistant response received: {assistant_response[:100]}...")
            return assistant_response
        else:
            print("[OPENAI_UTILS] ERROR: No valid response received from OpenAI")
            raise ValueError("No valid response received from OpenAI.")
    
    except Exception as e:
        error_message = f"OpenAI API call failed: {str(e)}"
        print(f"[OPENAI_UTILS] ERROR: {error_message}")
        raise RuntimeError(error_message)


def call_openai_with_conversation_history(
    system_prompt: str, 
    user_prompt: str, 
    conversation_history: list = None
) -> str:
    """
    Call OpenAI API with system prompt, user prompt, and conversation history.
    
    Args:
        system_prompt: The system prompt that defines the agent's behavior
        user_prompt: The user's current question or message
        conversation_history: List of tuples (user_msg, assistant_msg) from previous exchanges
        
    Returns:
        The assistant's response as a string
        
    Raises:
        ValueError: If no valid response is received
        RuntimeError: If the API call fails
    """
    print("[OPENAI_UTILS] Calling OpenAI API with conversation history...")
    print(f"[OPENAI_UTILS] System prompt: {system_prompt[:100]}...")
    print(f"[OPENAI_UTILS] User prompt: {user_prompt[:100]}...")
    print(f"[OPENAI_UTILS] Conversation history length: {len(conversation_history) if conversation_history else 0}")
    
    try:
        # Build messages list with system prompt
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history if provided
        if conversation_history:
            print(f"[OPENAI_UTILS] Processing {len(conversation_history)} previous exchanges")
            for i, exchange in enumerate(conversation_history):
                user_msg, assistant_msg = exchange
                messages.append({"role": "user", "content": user_msg})
                messages.append({"role": "assistant", "content": assistant_msg})
                print(f"[OPENAI_UTILS] Added message pair {i+1}")
        
        # Add current user prompt
        messages.append({"role": "user", "content": user_prompt})
        print(f"[OPENAI_UTILS] Total messages prepared: {len(messages)}")
        
        print("[OPENAI_UTILS] Sending request to OpenAI API...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        print("[OPENAI_UTILS] API request completed successfully")
        
        # Validate response
        if response.choices and len(response.choices) > 0:
            assistant_response = response.choices[0].message.content
            print(f"[OPENAI_UTILS] Assistant response received: {assistant_response[:100]}...")
            return assistant_response
        else:
            print("[OPENAI_UTILS] ERROR: No valid response received from OpenAI")
            raise ValueError("No valid response received from OpenAI.")
    
    except Exception as e:
        error_message = f"OpenAI API call failed: {str(e)}"
        print(f"[OPENAI_UTILS] ERROR: {error_message}")
        raise RuntimeError(error_message)
