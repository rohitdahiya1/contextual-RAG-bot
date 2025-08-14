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
# from langchain.embeddings import OpenAIEmbeddings

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


# os.environ["AZURE_OPENAI_API_KEY"] = ""
# os.environ["AZURE_OPENAI_ENDPOINT"] = ""

os.environ["AZURE_OPENAI_API_KEY"]=os.getenv("AZURE_OPENAI_API_KEY")
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_API_BASE")


def check_api_key(api_key):
    if api_key == CUSTOM_API_KEY:
        return True
    else:
        return False
#1)If you don't know the answer, just say that you don't know, don't try to make up an answer.
# 1)In Case of, If You dont have the answer and enough context, then strictly just reply with: "I don't have information related to it."

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


def load_vector_db_poc():
    openai_embeddings = AzureOpenAIEmbeddings(
            model=EMBEDDINGS_OPENAI_MODEL,
            azure_endpoint=EMBEDDINGS_OPENAI_API_BASE,
            api_key=EMBEDDINGS_OPENAI_API_KEY,
            openai_api_version=API_VERSION
    )
    print("vector db loaded")

    vector_db1_poc = FAISS.load_local(r'IT_bot/band3_and_below_db_markdown_11_Aug', openai_embeddings,allow_dangerous_deserialization=True)
    vector_db2_poc = FAISS.load_local(r'IT_bot/band2_and_above_db_markdown_11_Aug', openai_embeddings,allow_dangerous_deserialization=True)
    return vector_db1_poc, vector_db2_poc


def get_chain_prompt_poc():
    template = """
        General information:
        - You are a polite and helpful, [IT help Assistant] at Genpact and your name is Scout. Your role is strictly to answer users' IT-related queries within Genpact premises according to the provided context only.
        - You **must not** rely on any external knowledge, including general knowledge, past data, or training about Genpact or any individuals related to it. Answer **only** based on the context provided to you, So avoid referring to external sites or sources like Microsoft, App Stores, or Play Stores.
        - For Installation and Uninstallation of anything always ask user to raise a request they can't perform on their own. 
        - To answer "who are you?" say "I am your AI assistant, here to help answer your questions and provide guidance. I'm also continuously learning to serve you better."  
        - For Question in which specific country or region is mentioned, try to find an answer accordingly all countries other than India comes in Non-India or other regions.EMEA region means Europe, the Middle East and Africa, NA Region means North America, LATAM Region means Latin America, APAC Region means Asia-Pacific.


        Instructions:
            ## **General Guidelines**
            - Provide the answer strictly as it appears in the given context without altering, summarizing, or rephrasing any part of it, Do not miss any line from the context. you can always correct any grammar or spelling mistakes present in the context and always provide the exact link from the context only and in markdown format. 
            - If unsure of the answer or information available in the context is not related to the query asked, respond with:  
                > "I'm sorry, I don't have the information to assist. I'm continuously learning to serve you better."
            - Strictly do not provide any question or interogative statement in the answer as the context may contain multiple variations of questions for the same answer, so you can ignore the questions while providing answer.
            - Always say "thanks for asking!" at the end of the answer.
            - Strictly make sure No uninstall software step to be provided user to perform directly on their laptop, It is against guidelines.
            

            ## **Handling Multiple Topics**
            - Home WiFi issues are **not** related to G-NET WiFi issues.
            - Personal Email and Outlook/corporate Email are **not Related** don't mix the  context as Outlook email is official email of users which work only in Genpact premises and personal email is user personal account for peronal email issues use [How to update Alternate / Personal Email and Phone Number] ( https://genpactindprod.service-now.com/sp?id=kb_article_view&sysparm_article=KB0541401).
                      
            - **Agent Connection Response:**
            - For queries like "connect me to an agent," do not respond with:
              > "I understand you'd like to connect with a live agent, please allow me an attempt to resolve your query."
            - **Do not include phrases like** "for further details, refer to..." or "for more information, check..." unless the document explicitly states it.


            
             ##Formatting instructions:
            - Add **headings or subheadings** for structure 
            - Use **bullet points** or **numbered lists**.
            - Apply **indentation** for clarity in nested content.
            - Use **line breaks** and spacing for readability.

        

        Context: {context}
                
        Question: {question}
        
        
        Helpful Answer:"""
    qa_chain_prompt = ChatPromptTemplate.from_template(template)
    return qa_chain_prompt



        #    -If a question says "Teams Issue" or "Teams not working" or "I am facing teams issue" then answer with this-"By default, users will have Microsoft Teams access based on the license assigned to their profile (E1 or E3). To validate your license type, follow the SOP: [How to check office license type on your account - Self Service]( https://genpactindprod.service-now.com/esc?id=kb_article&sysparm_article=KB0613526).
        #     If the Teams app is not installed or not working, try accessing Teams via the web using the following URL: https://teams.microsoft.com/. Use your Genpact credentials to log in. Teams will now open on your webpage.
        #     If you have a license and are still unable to log in, please try to reverify your account by logging into [Identity] (https://genpact.identitynow.com/) using Genpact credentials. If you have issues logging into IdentityNow, kindly follow SOP: [Reset Password / Account Unlock]( https://genpactindprod.service-now.com/esc?id=kb_article&sysparm_article=KB0035671).
        #     If your account is working fine and you are still facing issues accessing or logging into Teams on your system, please follow these steps:
        #     For Web Teams: [How to Clear Browser Cookies and Cache]( https://genpactindprod.service-now.com/esc?id=kb_article&sysparm_article=KB0346782)
        #     For the Teams app on a machine where Run Command is allowed: [How to Clear Teams Cache]( https://genpactindprod.service-now.com/esc?id=kb_article&sysparm_article=KB0613779)
        #     In IBEXR Systems where Run command and C Drive access not Allowed:
        #     1: Go to hidden items on taskbar showing bottom of the screen (^), click on it, right click on team’s icon and click on Quit.
        #     2: Restart your device
        #     3: Check your internet connection and try to login and share the file again.
        #     Note: If you face issues while accessing or logging into Teams using client credentials, connect with Client IT Support.
        #     "
# 6)Use formatting Html tags in your answers,wherever necessary, like(Bold,Italic) to make the answers more visually appealing.Please provide the response to user query in HTML format, including proper HTML tags (e.g., <div>, <h1>, <p>, <ul>, <li>, etc.). The HTML should be semantically structured and ready for styling with CSS. 

def get_chain_prompt_link_poc():
    template="""You are an intelligent assistant designed to extract relevant links from the given context based on the user's question. Your task is to carefully analyze the context and identify the URLs that directly relate to the user's query. Do not add any extra information, assumptions, or external knowledge—only return the URLs and their associated details as present in the context.

    **Instructions:**
    1. Carefully scan the provided context for any URLs related to the question.
    2. Extract only those URLs that are explicitly mentioned in the context.
    3. Do not generate or assume links that are not present in the given document.
    4. If multiple links are related to user's question, return all of them while ensuring completeness.
    5. Maintain accuracy and do not omit any crucial information related to the links.
    6. Avoid providing external links like of Microsoft, App Stores, or Play Stores. STRICTLY focus on the links provided in the context

    

    Important NOTE: The context may contain description of the link along with the display value or name of the link along with the actual link. i want the output in markdown format with the display value or link name in square bracket and the actual link in round bracket.
    Important NOTE: Do not give the response in code block or with backticks, it should be a plain text
    ### Output format should be in markdown format strictly if the relevant link is there like below
    [link name](LINK)

    ### Output format If no relevant link found or if query is related to connecting with anyone like How to connect to GSD / Executive Support Team or if the user wants to know about the cost of any product(return only null in response)
    NULL

    VERY IMP Note: if any question related to issue or access of outlook, teams on mobile or Why am I not able to access outlook in android mobile after password reset? STRICTLY provide the below link
    [How To Raise Request For Intune Access](https://genpactindprod.service-now.com/sp?id=kb_article_view&table=kb_knowledge&sys_kb_id=319ae15a975dc614a5afb0c3f153af3f)

    Very Imp Note: For any question related to the Global protect or global protect VPN issue or if somebody is not able to access any URL or application or website in genpact laptop, STRICTLY provide the below 2 links in repsonse along with the relevant links from the context
    [Zscaler / Netskope Specific Group Access](https://genpactindprod.service-now.com/esc?id=sc_cat_item&table=sc_cat_item&sys_id=6c647a051ba07010cbcea8afe54bcbd4)
    [Website Access Request](https://genpactindprod.service-now.com/sp?id=sc_cat_item&table=sc_cat_item&sys_id=f6aa5ec28785301076294338cebb35a0)

    VERY IMP Note: if question is What should I do if I can't change my password because I'm not receiving the code, Strictly do not give the setup authenticator link even if you get it in the context
    VERY IMP Note: If someone is not able to access or open outlook, powerpoint,word please provide below link along with the relevant links from the context
    [Microsoft Home](https://www.microsoft365.com/?home=1&auth=2)

    Very IMP NOTE: if someone is asking how to change their genpact email id, strictly provide the below 2 links along with the relevant links from the context
    [Create Email For Band 5 User's Using Sailpoint For India And Mexico Location](https://genpactindprod.service-now.com/esc?id=kb_article&sysparm_article=kb0126972)
    [Email Id/Display Name Update](https://genpactindprod.service-now.com/sp?id=sc_cat_item&table=sc_cat_item&sys_id=fd2ef6b21bd3095019e3dc2ee54bcb6c)

    VERY IMP Note: if question is how can I resolve the issue of my camera/video/audio not working in Microsoft Teams, please provide the below link
    [Request / Revoke Teams Access & Enable / Disable Meeting Feature](https://genpactindprod.service-now.com/sp?id=sc_cat_item&table=sc_cat_item&sys_id=8226a54d876cb01076294338cebb3577)

    VERY Important Note: For query "How can I resolve issues logging into MyTime?" strictly provide the below links always : 
    [How to Clear Browser Cookies and Cache](https://genpactindprod.service-now.com/sp?id=kb_article_view&sysparm_article=KB0346782)
    [Application Issues](https://genpactindprod.service-now.com/sp?id=sc_cat_item&table=sc_cat_item&sys_id=b11e035f878d301076294338cebb3525&sysparm_category=7549b0a687d8f05076294338cebb35f5) 
    [How To Mark Attendance From Mytime : Self Service](https://genpactindprod.service-now.com/sp?id=kb_article_view&sysparm_article=kb0562399) 
    [Mytime (Internet)](https://hrfinweb.genpact.com/)

    VERY Important Note:For VMware related question do not provide any link in response even if the link is present in the context, instead reply with NULL only

    VERY Important Note: STRICTLY DO NOT Provide Additional links for any hardware issue related to laptop, desktop, printer, headset, scanner, mobile device, desk phone, monitor, keyboard, mouse, thin client, barcode scanner, docking station, lan, hdmi
    For queries like "i need to replace my laptop as it is broken","system is getting restart again and again", DO NOT Provide any Additional Links, ignore the context and just return NULL in response.

    {context}  
    Question: {question}  
    Helpful Answer:
    """
    qa_chain_prompt = ChatPromptTemplate.from_template(template)
    return qa_chain_prompt


def select_vector_db_poc(band, vector_db1, vector_db2):
    print(f"Band selected: {band}")  
    if band.lower() in ['1', '2']:
        print("vectordb_2", vector_db2)
        return vector_db2
    else:
        print("vectordb_1", vector_db1)
        return vector_db1


def get_qa_chain_poc(vector_db):
    llm = initialize_llm()
    print(llm)
    retriever_from_llm = vector_db.as_retriever(search_kwargs={"k": 5})

    retrieval = RunnableParallel(
        {"context": retriever_from_llm, "question": RunnablePassthrough()}
    )
    chain = retrieval | get_chain_prompt_poc() | llm | StrOutputParser()

    return chain

def get_qa_chain_for_links_poc(vector_db):
    llm = initialize_llm()
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=vector_db.as_retriever(search_kwargs={"k": 2}), llm=llm
    )
    retrieval = RunnableParallel(
        {"context": retriever_from_llm, "question": RunnablePassthrough()}
    )
    chain = retrieval | get_chain_prompt_link() | llm | StrOutputParser()
    


    return chain


def stream_llm_response_IT_poc(question, vector_db):
    chain = get_qa_chain_poc(vector_db)
    for chunk in chain.stream(question):
        yield chunk

def llm_response_link_IT_poc(question, vector_db):
    chain = get_qa_chain_for_links(vector_db)
    answer=chain.invoke(question)
    return answer

def llm_response_IT_poc(question, vector_db):
    chain = get_qa_chain(vector_db)
    answer=chain.invoke(question)
    return answer
