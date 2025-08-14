import os
import json
import redis
from datetime import timedelta
from openai import AzureOpenAI
import requests
from typing import List, Dict, Optional


# REDIS_URL = "redis://default:ZtywrXXjmu9WDgc3QvZbftStFlKOJ5SQ@redis-18829.c326.us-east-1-3.ec2.redns.redis-cloud.com:18829"
REDIS_URL="redis://default:lWemOBF6rGhWGwiSnRkOsMHfzOMX0cxB@redis-18829.c326.us-east-1-3.ec2.redns.redis-cloud.com:18829"
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)


SYSTEM_PROMPT = """
You are GenAI HelpBot, a smart, friendly, and highly professional assistant.

# Behavior Guidelines:
- Use chat history to maintain conversational flow.
- Understand and respond to follow-ups like "okay","are you sure", "and?", "what about weekends?" based on earlier messages.
- Be warm and humanlike in tone, especially when users express satisfaction or politeness.
- If you need more information to complete a request, ask clarifying questions.
- Maintain context throughout the conversation.
- either use available tool for repsonse or a normal response(followup, acknowledgement respose), STRICTLY do not respond from your own knowledge
- for IT related query, do not ask followup or clarifications questions
"""

client = AzureOpenAI(
    azure_endpoint="",
    api_key="",
    api_version="2024-08-01-preview",
)

deployment_name = "gpt-4o"


weather_function = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city to get the weather for"
                }
            },
            "required": ["city"]
        }
    }
}

service_now_function = {
    "type": "function",
    "function": {
        "name": "create_service_now_user",
        "description": "Create a new user in ServiceNow with name and age (both are mandatory fields)",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "User's full name which is a mandatory field"
                },
                "age": {
                    "type": "string",
                    "description": "User's age which is a mandatory field"
                }
            },
            "required": ["name", "age"]
        }
    }
}

it_query_function = {
    "type": "function",
    "function": {
        "name": "resolve_it_query",
        "description": "Resolve IT-related technical issues and questions",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The IT-related question to resolve"
                }
            },
            "required": ["question"]
        }
    }
}

service_now_ticket_query_function = {
    "type": "function",
    "function": {
        "name": "get_user_tickets_by_sys_id",
        "description": "Retrieve ticket numbers from ServiceNow for a user using their sys_id like '938sos912jswi293ns'",
        "parameters": {
            "type": "object",
            "properties": {
                "sys_id": {
                    "type": "string",
                    "description": "The sys_id of the user to fetch tickets for"
                }
            },
            "required": ["sys_id"]
        }
    }
}

tools = [weather_function, service_now_function, it_query_function, service_now_ticket_query_function]


def get_current_weather(city):
    """Simulated weather API call"""
    return f"The current weather in {city} is 23 degrees."

def create_service_now_user(name, age):
    """Execute ServiceNow API call"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Basic YOUR_ENCODED_CREDENTIALS"
    }
    
    try:
        response = {"message": "User created successfully with name: " + name + " and age: " + age}
        return response["message"]
    except Exception as e:
        return {"error": str(e)}

def resolve_it_query(question):
    """Handle IT query API call to the specified endpoint"""
    endpoint = "https://devallbots-hkbxevhudkgabdhq.westeurope-01.azurewebsites.net/IT/predictanswer"
    
    headers = {
        "X-API-KEY": "GN^CBB4185E5BFDAEDF7B1F172EE5F21",
        "Content-Type": "application/json"
    }
    
    payload = {"question": question}
    
    try:
        response = requests.post(
            url=endpoint,
            json=payload,
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        return response.json().get("answer", "No answer provided by IT system.")
    except requests.exceptions.RequestException as e:
        return f"Error contacting IT system: {str(e)}"

def get_bearer_token():
    """Fetch the Bearer token required to call the ServiceNow API"""
    token_url = "https://chucklebot.azurewebsites.net/getServiceNow"
    headers = {
        "X-API-KEY": "GN^CBB4185E5BFDAEDF7B1F172EE5F21"
    }

    try:
        response = requests.get(token_url, headers=headers)
        response.raise_for_status()
        token_data = response.json()
        return token_data.get("access_token")
    except Exception as e:
        return {"error": f"Failed to fetch bearer token: {str(e)}"}

def get_user_tickets_by_sys_id(sys_id):
    """Fetch ticket numbers from ServiceNow for the provided sys_id"""
    if not sys_id:
        return {"error": "Missing required parameter: sys_id"}

    access_token = get_bearer_token()
    if isinstance(access_token, dict) and "error" in access_token:
        return access_token

    query_url = (
        "https://genpactdevelop.service-now.com/api/now/table/sc_req_item?"
        f"sysparm_query=u_rpt_requested_for={sys_id}"
        "&sysparm_display_value=true&sysparm_fields=number"
    )

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.get(query_url, headers=headers)
        response.raise_for_status()
        result = response.json()

        if "result" in result:
            tickets = [item.get("number", "N/A") for item in result["result"]]
            return {"tickets": tickets} if tickets else {"message": "No tickets found."}
        else:
            return {"error": "Unexpected response format"}
    except Exception as e:
        return {"error": str(e)}


def store_conversation(session_id: str, role: str, content: str, ttl_hours: int = 24):
    """Store conversation message in Redis"""
    message = {
        "role": role,
        "content": content,
        "timestamp": str(timedelta)
    }
    redis_client.rpush(f"conversation:{session_id}", json.dumps(message))
    redis_client.expire(f"conversation:{session_id}", timedelta(hours=ttl_hours))

def get_conversation_history(session_id: str, max_messages: int = 10) -> List[Dict]:
    """Retrieve conversation history from Redis"""
    messages = redis_client.lrange(f"conversation:{session_id}", -max_messages, -1)
    return [json.loads(msg) for msg in messages]

def prepare_messages(session_id: str, user_input: str) -> List[Dict]:
    """Prepare the messages for OpenAI API with system prompt and history"""
    history = get_conversation_history(session_id)
    print("the history is", history)
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    

    print("The complete prompt messages are", messages)
    
    return messages

def run_conversation(session_id: str, user_input: str):
    """Run a conversation with context from Redis"""
  
    store_conversation(session_id, "user", user_input)
 
    messages = prepare_messages(session_id, user_input)
 
    response = client.chat.completions.create(
        model=deployment_name,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    
    response_message = response.choices[0].message
    messages.append(response_message)
    
   
    assistant_response = None
    

    if response_message.content:
        
        store_conversation(session_id, "assistant", response_message.content)
        assistant_response = response_message.content
  
    if response_message.tool_calls:
        print("Processing tool calls:", response_message.tool_calls)
  
        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            
            if function_name == "get_current_weather":
                function_response = get_current_weather(
                    city=function_args.get("city")
                )
            elif function_name == "resolve_it_query":
                function_response = resolve_it_query(
                    question=function_args.get("question")
                )
            elif function_name == "get_user_tickets_by_sys_id":
                function_response = get_user_tickets_by_sys_id(
                    sys_id=function_args.get("sys_id")
                )
            elif function_name == "create_service_now_user":
                function_response = create_service_now_user(
                    name=function_args.get("name"),
                    age=function_args.get("age")
                )
            else:
                function_response = json.dumps({"error": "Unknown function"})
            

            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": str(function_response),
            })
        
      
        final_response = client.chat.completions.create(
            model=deployment_name,
            messages=messages,
        )
        

        final_content = final_response.choices[0].message.content
        store_conversation(session_id, "assistant", final_content)
        assistant_response = final_content
    
    return assistant_response


def chat_loop():
    print("Welcome to GenAI HelpBot! Type 'quit' to exit.")
    session_id = input("Please enter a session ID (or leave blank for a new session): ") or "default_session"
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
        
        response = run_conversation(session_id, user_input)
        print(f"\nAssistant: {response}")

if __name__ == "__main__":
    chat_loop()
