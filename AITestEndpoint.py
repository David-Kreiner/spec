import os
import uvicorn

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import openai

# Initialize FastAPI app
app = FastAPI()

# Load the OpenAI API key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")

class ExtraContextItem(BaseModel):
    description: str
    content: str

class PipelineRequest(BaseModel):
    prompt: str
    extra_context: Optional[List[ExtraContextItem]] = None

@app.post("/chat")
async def process_pipeline(request: PipelineRequest):
    try:
        response = call_openai_function(
            prompt=request.prompt,
            extra_context=request.extra_context
        )
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def gather_repo_context(folders):
    context = ""
    file_list = []
    for folder in folders:
        for root, dirs, files in os.walk(folder):
            dirs[:] = [d for d in dirs if not d.startswith('.')]  # Exclude hidden directories
            files = [f for f in files if not f.startswith('.')]  # Exclude hidden files
            
            for file in files:
                if file.endswith(('.yaml', '.yml', '.ts', '.js', '.json', '.md', '.go')):
                    file_path = os.path.join(root, file)
                    file_list.append(file_path)
                    with open(file_path, 'r') as f:
                        context += f"\n\nFile: {file_path}\n" + f.read()
    return context, file_list

def create_messages(prompt, repo_context="", extra_context=None):
    system_prompt = (
        "You are an AI programming assistant named 'AI Devops Engineer'. "
        "You are part of a chat interface for an open source CI/CD platform called 'Gitness'. "
        "Follow the user's requirements carefully & to the letter. "
        "Your expertise is strictly limited to CI/CD and DevOps topics. "
        "Follow Harness content policies. "
        "Avoid content that violates copyrights. "
        "For questions not related to CI/CD and DevOps, simply give a reminder that you are an AI Devops Engineer. "
        "Keep your answers short and impersonal. "
        "You can answer general DevOps questions and perform the following tasks through tool calls: "
        "* Generate a CI/CD pipeline configuration "
        "* Update a CI/CD pipeline configuration "
        "* Analyze failed pipeline logs and suggest a code fix "
        "First think step-by-step - describe your plan for what to build and then do it. "
        "Minimize any other prose. "
        "Avoid wrapping the whole response in triple backticks. "
        "You can understand and use context from Gitness examples, JSON schema, and YAML files. "
        "Each time you respond ensure the USER QUERY is satisfied to the best of your ability. "
        "Make no assumptions about the functions and usage of YAML files that do not exist in the context, if there is missing information, ask for that information. "
        "Do not make up an answer to questions that are not related to the query. "
        "Prioritize using tool calls to perform the tasks when possible. "
        "Do not explicitly mention the tool in the response. "
        "There is no need to provide the full code or YAML in the message response unless specifically requested by the user. "
        "If a request is ambiguous or requires additional information, ask the user for clarification. "
        "Ensure all generated configurations adhere to security best practices and are optimized for performance. "
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": f"Context from the repo: {repo_context}"}
    ]
    
    if extra_context:
        for context_item in extra_context:
            messages.append({"role": "system", "content": f"{context_item.description}: {context_item.content}"})

    user_message = prompt
    
    messages.append({"role": "user", "content": user_message})

    return messages

def create_tools():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "generate_pipeline",
                "description": "Generates a CI/CD pipeline.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pipeline_config": {
                            "type": "string",
                            "description": "The YAML configuration of the generated pipeline."
                        }
                    },
                    "required": ["pipeline_config"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "update_pipeline",
                "description": "Updates a CI/CD pipeline",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "updated_pipeline_config": {
                            "type": "string",
                            "description": "The updated YAML configuration of the pipeline."
                        }
                    },
                    "required": ["updated_pipeline_config"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "fix_failed_pipeline",
                "description": "Fix a failed pipeline.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "suggested_fix": {
                            "type": "string",
                            "description": "The suggested code fix based on the logs."
                        }
                    },
                    "required": ["suggested_fix"]
                }
            }
        }
    ]
    return tools

def call_openai_function(prompt, extra_context=None):
    folders = ['./samples', './dist']  # List of folders to gather context from
    repo_context, file_list = gather_repo_context(folders)
    
    messages = create_messages(prompt, repo_context, extra_context)
    tools = create_tools()

    client = openai.OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        n=1,
        tool_choice="auto"
    )
    if response.choices[0].message.tool_calls:
        for tool_call in response.choices[0].message.tool_calls:
            print("Made tool call:")
            print(f"Tool Name: {tool_call.function.name}")
            print(f"Arguments: {tool_call.function.arguments}")
    else:
        print("No tool call made.")

    return response.choices[0].message

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
