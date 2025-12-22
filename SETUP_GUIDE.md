# UTA HR Policies Agent - Setup Guide

## Step-by-Step Setup Instructions

### Step 1: Verify Files Are Created
You now have these files in your project:
- ✅ `.env` - Stores your OpenAI API key
- ✅ `.gitignore` - Protects sensitive files from Git
- ✅ `requirements.txt` - Lists Python dependencies
- ✅ `src/agent_core.py` - Main agent code with Gradio UI

### Step 2: Add Your OpenAI API Key
1. Open the `.env` file in your editor
2. Replace `your_openai_api_key_here` with your actual API key:
   ```
   OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxx
   ```
3. Save the file

**To get your API key:**
- Visit: https://platform.openai.com/api-keys
- Sign in to your OpenAI account
- Click "Create new secret key"
- Copy the entire key (starts with `sk-`)
- Paste it in the `.env` file

### Step 3: Set Up Python Virtual Environment
Open terminal and run:
```bash
cd /Users/pr/Downloads/Learning_Projects/LLM/Agent_UTA_HR_Policies

# Create virtual environment
python3 -m venv venv

# Activate it (macOS/Linux)
source venv/bin/activate
```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

This installs:
- `openai` - To call OpenAI API
- `gradio` - For the web UI
- `python-dotenv` - To read .env file

### Step 5: Run the Agent
```bash
python src/agent_core.py
```

You should see:
```
Running on local URL:  http://127.0.0.1:7861
```

### Step 6: Open in Browser
Go to: **http://127.0.0.1:7861**

You'll see the chat interface!

## Testing the Agent

Try asking these questions:
- "Who are you?"
- "What can you help me with?"
- "Tell me about UTA HR policies"

The agent should respond with:
> "Hi, I am the UTA HR Policies Agent. You can ask me any questions related to UTA HR Policies!"

## File Descriptions

### `.env` File
Stores sensitive information (API keys). **Never commit to Git!**
```
OPENAI_API_KEY=your_key_here
```

### `.gitignore` File
Tells Git which files to ignore:
- `.env` (protects your API key)
- `__pycache__/` (Python cache)
- `venv/` (virtual environment)
- etc.

### `requirements.txt` File
Lists all Python packages needed:
```
openai>=1.0.0
gradio>=4.0.0
python-dotenv>=1.0.0
```

### `src/agent_core.py` File
Contains:
- OpenAI API integration
- System prompt for HR agent
- Chat function with history
- Gradio web interface

## Checklist Before Running

- [ ] .env file has your OpenAI API key
- [ ] Virtual environment created (`venv/` folder exists)
- [ ] Virtual environment activated (`(venv)` shows in terminal)
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] OpenAI API key is valid and has credits
- [ ] Port 7860 is available (not used by another app)

## Common Issues & Fixes

| Issue | Solution |
|-------|----------|
| "OPENAI_API_KEY not found" | Check .env file has your key |
| "ModuleNotFoundError: openai" | Run `pip install -r requirements.txt` |
| "Connection error" | Check internet, verify API key validity |
| "Port 7861 already in use" | Close other app using the port or modify agent_core.py |
| "API key rejected" | Verify key starts with `sk-` and is valid |

## What's Next?

After verification, you can:
1. **Test with your HR documents** - Add actual policy files to DataSources/
2. **Enhance with RAG** - Implement Retrieval Augmented Generation for accurate policy answers
3. **Deploy** - Host on a web server for team access
4. **Customize** - Modify the system prompt for different behaviors

## Security Reminders

⚠️ **Important:**
1. Never share your API key
2. Never commit `.env` to Git (already protected by .gitignore)
3. Rotate keys if exposed
4. Monitor your OpenAI usage to control costs

## Questions?

Refer to:
- OpenAI docs: https://platform.openai.com/docs/
- Gradio docs: https://gradio.app/docs
- This project's GitHub issues
