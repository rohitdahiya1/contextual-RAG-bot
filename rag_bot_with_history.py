import os
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough
)
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from datetime import datetime, timedelta
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_redis import RedisChatMessageHistory
 
 
REDIS_URL = "redis://default:ZtywrXXjmu9WDgc3QvZbftStFlKOJ5SQ@redis-10545.c212.ap-south-1-1.ec2.redns.redis-cloud.com:10545"
print(f"Connecting to Redis at: {REDIS_URL}")
 
# history = RedisChatMessageHistory(session_id="user_123", redis_url=REDIS_URL)
def get_redis_history(session_id: str) -> BaseChatMessageHistory:
    return RedisChatMessageHistory(session_id, redis_url=REDIS_URL)
 
 
load_dotenv()
 
BASE_URL = os.getenv("AZURE_OPENAI_API_BASE")
API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")
API_VERSION = os.getenv("OPENAI_API_VERSION")
API_TYPE = os.getenv("OPENAI_API_TYPE")
CUSTOM_API_KEY = os.getenv("CUSTOM_API_KEY")
AZURE_OPENAI_DEPLOYMENT_NAME=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
 
 
EMBEDDINGS_OPENAI_MODEL=os.getenv("EMBEDDINGS_OPENAI_MODEL")
EMBEDDINGS_OPENAI_API_BASE=os.getenv("EMBEDDINGS_OPENAI_API_BASE")
EMBEDDINGS_OPENAI_API_KEY=os.getenv("EMBEDDINGS_OPENAI_API_KEY")
 
 
 
 
# os.environ["AZURE_OPENAI_API_KEY"] = "ebef62a4025f46939fee667e78f96bec"
# os.environ["AZURE_OPENAI_ENDPOINT"] = "https://gopenaieus2.openai.azure.com/"
 
os.environ["AZURE_OPENAI_API_KEY"]=os.getenv("AZURE_OPENAI_API_KEY")
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_API_BASE")
 
 
def check_api_key(api_key):
    if api_key == CUSTOM_API_KEY:
        return True
    else:
        return False
 
 
def initialize_llm():
    llm = AzureChatOpenAI(
        azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
        api_version=API_VERSION,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
 
    return llm
 
 
def load_vector_db():
    openai_embeddings = AzureOpenAIEmbeddings(
            model=EMBEDDINGS_OPENAI_MODEL,
            azure_endpoint=EMBEDDINGS_OPENAI_API_BASE,
            api_key=EMBEDDINGS_OPENAI_API_KEY,
            openai_api_version=API_VERSION
    )
    print("vector db loaded")
    vector_db1 = FAISS.load_local(r'IT_bot/Band_3_and_Below_25_April', openai_embeddings,allow_dangerous_deserialization=True)
    vector_db2 = FAISS.load_local(r'IT_bot/Band_2_and_Above_25_April', openai_embeddings,allow_dangerous_deserialization=True)
    return vector_db1, vector_db2
 
 
def get_chain_prompt():
    SYSTEM_PROMPT = """
    You are GenAI HelpBot, a smart, friendly, and highly professional IT assistant for Genpact employees.
 
    # Your Core Behavior:
    - Think like a human IT support agent: helpful, attentive, warm.
    - You must **always use the conversation history** to maintain context.
    - If the user’s new message is a follow-up (e.g., "okay", "what about weekends?", "and?"), understand it based on the previous interaction and answer accordingly.
    - If the user’s input indicates satisfaction (e.g., "okay", "thanks", "got it"), reply warmly, like "You're welcome! Let me know if you need anything else."
    - If the input is a **new query**, ignore prior conversation and **answer based on the given Context**.
    - If the user requests to "connect to an agent", reply with:
    > I understand you'd like to connect with a live agent. Please allow me an attempt to resolve your query.
 
    # Answer Construction Rules:
    - **Strictly avoid hallucinating** information not present in History or Context.
    - **Never ask unnecessary questions**; assume best intent.
    - **No interrogative sentences** unless absolutely necessary.
    - **Format responses with Markdown**:
    - Headings (`#`, `##`)
    - Bullet points
    - Bold/Italic for key terms
    - Proper Markdown for URLs `[text](link)`
    - Respond in a crisp, professional but friendly tone.
    - Always structure long information neatly and clearly.
 
    # Few-Shot Behavior Learning:
 
    ---
 
    ## Example 1: Simple Acknowledgment
    History:
    - User: How do I reset password?
    - Bot: Go to login page > Forgot Password > Follow email instructions.
    User: Thanks!
 
    Bot Reply:  
    You're welcome! Let me know if you need anything else.
 
    ---
 
    ## Example 2: Deepening Conversation
    History:
    - User: What are IT support hours?
    - Bot: Mon-Fri: 9 AM to 6 PM. Sat: 10 AM to 2 PM.
    User: And Sunday?
 
    Bot Reply:  
    **IT Support on Sunday**  
    - Closed on Sundays.
 
    ---
 
    ## Example 3: New Query Handling
    History: (empty)
 
    Context:
    - Leave requests must be submitted via the HR portal.
 
    User: How do I apply for leave?
 
    Bot Reply:
    ### Applying for Leave
    - Log in to the **HR Portal**.
    - Navigate to **Leave Application**.
    - Submit your request.
 
    ---
 
    Behave like a smart conversational assistant, maintaining flow naturally without breaking user immersion.
    """
 
    system_prompt = SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT)
 
    user_template = """You are GenAI HelpBot, a helpful and professional IT assistant for Genpact employees. You assist users by answering questions based only on the information provided in the context.
    Provide the answer strictly as it appears in the given context without altering, summarizing, or rephrasing any part of it.
    Do not change the wording—only apply formatting if required to improve clarity.
 
    Previous Conversation History (for reference):
    {chat_history}
 
    Current Context Information (only if needed):
    {context}
 
    User's Current Input:
    {question}
 
    # Your Task:
    - Refer to chat history first. If it answers the question or provides clues, continue from there.
    - If no relevant history, use the provided Context to answer.
    - Maintain a conversational, human tone as per the system instructions.
    - Format nicely with Markdown.
 
    Answer:
    """
 
    user_prompt = HumanMessagePromptTemplate.from_template(user_template)
 
    final_prompt = ChatPromptTemplate.from_messages([
        system_prompt,
        user_prompt
    ])
 
    return final_prompt
 
 
 
 
 
def select_vector_db(band, vector_db1, vector_db2):
    print(f"Band selected: {band}")  
    if band.lower() in ['1', '2']:
        print("vectordb_2", vector_db2)
        return vector_db2
    else:
        print("vectordb_1", vector_db1)
        return vector_db1
 
 
 
def get_qa_chain(vector_db):
    llm = initialize_llm()
    retriever = vector_db.as_retriever(search_kwargs={"k": 2})
 
    # def debug_history(inputs):
    #     print("\n--- DEBUG: Current Chat History ---")
    #     print(f"Session History: {store.get('1', 'No history')}")
    #     print("-------------------------------\n")
    #     return inputs
 
    retrieval = RunnableParallel({
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
        "chat_history": itemgetter("chat_history")
    })
 
    basic_chain = (
        retrieval
        | get_chain_prompt()   # Updated: now returns (system + user prompt)
        | llm
        | StrOutputParser()
    )
 
    return RunnableWithMessageHistory(
        basic_chain,
        get_redis_history,
        input_messages_key="question",
        history_messages_key="chat_history"
    )
 
 
 
def stream_llm_response_IT(question, vector_db):
    chain = get_qa_chain(vector_db)
    input_dict = {"question": question}  # Input is already a dict with "question" key
    for chunk in chain.stream(input_dict, {'configurable': {'session_id': '1'}}):
        yield chunk
 
 
def llm_response_IT(question, vector_db):
    chain = get_qa_chain(vector_db)
    answer=chain.invoke(question)
    return answer
