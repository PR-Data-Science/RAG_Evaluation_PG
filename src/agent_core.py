import os
import gradio as gr
from dotenv import load_dotenv
from openai_utils import call_openai_with_conversation_history

print("[INIT] Starting agent initialization...")

# Load environment variables from .env file
print("[INIT] Loading .env file...")
load_dotenv()

print("[INIT] OpenAI utilities imported successfully")
print("[SUCCESS] Agent initialization complete")

def chat_with_agent(user_message: str, chat_history: list = None) -> tuple:
    """
    Chat with the UTA HR Policies Agent using OpenAI API
    
    Args:
        user_message: The user's question/message
        chat_history: List of message dictionaries with 'role' and 'content' keys
        
    Returns:
        Tuple of (updated_chat_history, empty_string_for_input)
    """
    print(f"\n[CHAT] User message received: {user_message}")
    
    if chat_history is None:
        chat_history = []
        print("[CHAT] Starting new chat history")
    
    print(f"[CHAT] Chat history type: {type(chat_history)}")
    print(f"[CHAT] Chat history length: {len(chat_history)}")
    
    # System prompt for the agent
    system_prompt = """You are the UTA HR Policies Agent. Your role is to help answer questions about 
    University of Texas at Arlington (UTA) HR Policies. You are knowledgeable about:
    - Benefits policies
    - Employment policies
    - Leave and Absences
    - Staff Performance and Evaluation
    
    When answering questions, be helpful, professional, and cite relevant policy sections when possible.
    If you don't know the answer, direct the user to contact HR directly."""
    
    print("[CHAT] System prompt prepared")
    
    # Convert chat history from Gradio format to conversation format for API
    # Gradio sends chat_history as list of dictionaries: [{"role": "user"/"assistant", "content": "..."}, ...]
    conversation_history = None
    if chat_history and len(chat_history) > 0:
        print("[CHAT] Converting chat history to conversation format...")
        conversation_history = []
        i = 0
        while i < len(chat_history):
            # Get user message
            if i < len(chat_history) and isinstance(chat_history[i], dict) and chat_history[i].get("role") == "user":
                user_msg = chat_history[i].get("content", "")
                
                # Get assistant message (should be right after)
                if i + 1 < len(chat_history) and isinstance(chat_history[i+1], dict) and chat_history[i+1].get("role") == "assistant":
                    assistant_msg = chat_history[i+1].get("content", "")
                    conversation_history.append((user_msg, assistant_msg))
                    print(f"[CHAT] Added pair: User={user_msg[:40]}... | Assistant={assistant_msg[:40]}...")
                    i += 2
                else:
                    i += 1
            else:
                i += 1
    
    try:
        # Call OpenAI API with conversation history using the utility function
        print("[CHAT] Calling OpenAI utility function...")
        assistant_response = call_openai_with_conversation_history(
            system_prompt=system_prompt,
            user_prompt=user_message,
            conversation_history=conversation_history
        )
        
        # DEBUG: Print response details
        print(f"\n[CHAT] ========== RESPONSE RECEIVED FROM MODEL ==========")
        print(f"[CHAT] Response type: {type(assistant_response)}")
        print(f"[CHAT] Response length: {len(assistant_response) if assistant_response else 0}")
        print(f"[CHAT] Response content: {repr(assistant_response)}")
        print(f"[CHAT] Full response:\n{assistant_response}")
        print(f"[CHAT] ===================================================\n")
        
        print(f"[CHAT] Response received from OpenAI utility")
        
        # Append to chat history as dictionaries with role and content (Gradio Chatbot format)
        print(f"[CHAT] Adding user message to history as dict")
        chat_history.append({"role": "user", "content": user_message})
        
        print(f"[CHAT] Adding assistant response to history as dict")
        chat_history.append({"role": "assistant", "content": assistant_response})
        
        print(f"[CHAT] Chat history after append: {len(chat_history)} messages")
        print(f"[CHAT] Chat history format check:")
        for i, msg in enumerate(chat_history):
            print(f"[CHAT]   Message {i}: type={type(msg)}, keys={msg.keys() if isinstance(msg, dict) else 'N/A'}")
        print(f"[CHAT] Returning chat_history and empty string")
        
        return chat_history, ""
    
    except RuntimeError as e:
        error_message = f"❌ OpenAI Error: {str(e)}"
        print(f"[ERROR] RuntimeError occurred: {str(e)}")
        print(f"[ERROR] Adding error to chat history")
        
        # Add user message and error as dictionaries
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": error_message})
        return chat_history, ""
    
    except ValueError as e:
        error_message = f"❌ Invalid Response: {str(e)}"
        print(f"[ERROR] ValueError occurred: {str(e)}")
        print(f"[ERROR] Adding error to chat history")
        
        # Add user message and error as dictionaries
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": error_message})
        return chat_history, ""
    
    except Exception as e:
        error_message = f"❌ Unexpected Error: {str(e)}"
        print(f"[ERROR] Unexpected exception occurred: {str(e)}")
        print(f"[ERROR] Error type: {type(e).__name__}")
        print(f"[ERROR] Adding error to chat history")
        
        # Add user message and error as dictionaries
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": error_message})
        return chat_history, ""

def create_interface():
    """Create and launch the Gradio interface"""
    
    print("[UI] Creating Gradio interface...")
    
    with gr.Blocks(title="UTA HR Policies Agent") as demo:
        print("[UI] Gradio Blocks created")
        
        gr.Markdown("# UTA HR Policies Agent")
        gr.Markdown("Hi, I am the UTA HR Policies Agent. You can ask me any questions related to UTA HR Policies!")
        print("[UI] Header markdown added")
        
        chatbot = gr.Chatbot(
            label="Chat",
            height=400,
            show_label=True
        )
        print("[UI] Chatbot component created")
        
        with gr.Row():
            user_input = gr.Textbox(
                label="Your Question",
                placeholder="Ask me about UTA HR policies...",
                lines=2
            )
            submit_btn = gr.Button("Send", variant="primary")
            print("[UI] Input textbox and submit button created")
        
        # Connect submit button to chat function
        print("[UI] Connecting submit button to chat_with_agent function...")
        submit_btn.click(
            fn=chat_with_agent,
            inputs=[user_input, chatbot],
            outputs=[chatbot, user_input]
        )
        print("[UI] Button click event connected")
        
        # Also allow Enter key to submit
        print("[UI] Connecting Enter key to chat_with_agent function...")
        user_input.submit(
            fn=chat_with_agent,
            inputs=[user_input, chatbot],
            outputs=[chatbot, user_input]
        )
        print("[UI] Enter key event connected")
    
    print("[UI] Gradio interface creation complete")
    return demo

if __name__ == "__main__":
    print("\n" + "="*60)
    print("[MAIN] UTA HR Policies Agent - Starting")
    print("="*60 + "\n")
    
    try:
        print("[MAIN] Creating Gradio interface...")
        demo = create_interface()
        
        print("[MAIN] Interface created successfully")
        print("[MAIN] Launching Gradio server...")
        print("[MAIN] Server running on: http://127.0.0.1:7900")
        print("[MAIN] Press Ctrl+C to stop the server\n")
        
        demo.launch(
            server_name="127.0.0.1",
            server_port=7900,
            share=False
        )
        
    except KeyboardInterrupt:
        print("\n[MAIN] Server stopped by user (Ctrl+C)")
    except Exception as e:
        print(f"\n[ERROR] Failed to launch server: {str(e)}")
        print(f"[ERROR] Error type: {type(e).__name__}")
        raise
