import openai
import pinecone
import os
import pandas as pd
from pathlib import Path
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from langchain.chains.conversation.memory import ConversationBufferMemory
from flask import Flask, render_template, request,jsonify
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings 
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor()


openai_api_key = os.environ.get('OPENAI_API_KEY')
project_root = os.path.dirname(os.path.realpath('__file__'))
static_path = os.path.join(project_root, 'app/static')
app = Flask(__name__, template_folder= 'templates')
context_set = ""

@app.route('/')
def index():
    return render_template('index.html')

model_name = 'text-embedding-ada-002'

embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=openai_api_key
)


# # initialize pinecone
pinecone.init(
    api_key="c90dcbb6-61f6-4204-89ca-aae3d2a82d6e",
    environment="gcp-starter"
)

index = pinecone.Index("pdf")

text_field = "text"
vectorstore = Pinecone(
    index, embed, text_field
)

query = "Provide me answers from vector storage."

vectorstore.similarity_search(
    query,  # our search query
    k=3  # return 3 most relevant docs
)

# completion llm
llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model_name='gpt-3.5-turbo',
    temperature=0.1
)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 1})
)

custom_array = [
    {"question": "Hello", "answer": "Hello! How can I assist you today?"},
    {"question": "welcome", "answer": "Thank you! If you have any questions or if there's anything specific you'd like to know or discuss, feel free to let me know. I'm here to help!"},
    {"question": "thankyou", "answer": "You're welcome! If you ever have more questions or need assistance in the future, feel free to reach out. Have a great day!"},
    
]
@app.route('/query', methods=['POST', 'GET'])
def query():
    if request.method == 'GET':
        user_input = str(request.args.get('text'))
                       
        prompt_template ="You are a chatbot of given content.Please provide a user answers only from the given content. please give me answer which is asking by user -{user_input}"
       
        prompt = prompt_template.format(user_input=user_input)
        
        for entry in custom_array:
            if user_input.lower() == entry["question"].lower():
                result = entry["answer"]
                break
        else:
            try:
                # result = qa.run(prompt)
                result = executor.submit(qa.run, prompt).result()
                print(result)
            except Exception as e:
                print(e)
                result = "Unfortunately, the information is currently not accessible."
        
        return result

if __name__ == '__main__':
    app.run(debug=True)