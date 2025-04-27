import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import BitsAndBytesConfig
import torch
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
st.set_page_config(page_title="EXAONE-3.5-7.8B-RAG")
st.title("EXAONE-3.5-7.8B-RAG")
st.markdown(
    """
    🎧  
    """
)


#########################################
# 모델(EXAONE-3.5-2.4B-Instruct) 불러오기
#########################################
model_name = '../models/EXAONE-3.5-7.8B-Instruct'
bnb_config = BitsAndBytesConfig(
    
    load_in_4bit=True,
    bnb_4bit_quant_dtype="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True   
)
if "llm" not in st.session_state:
    st.session_state.llm = HuggingFacePipeline.from_model_id(
        model_id=model_name,
        task="text-generation",
        pipeline_kwargs=dict(
            max_new_tokens=4096,
            do_sample=False,
            eos_token_id = [361],
            # repetition_penalty=1.03,
        ),
        model_kwargs=dict(
            trust_remote_code=True,
            quantization_config = bnb_config
        ),
    )
llm = st.session_state.llm

#########################################
# RAG, retriever 불러오기
#########################################

if "retriever" not in st.session_state:
    persist_directory = 'C://Users//AFOC//Desktop//llm_mark2//langchain-streamlit//vectordb'
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=HuggingFaceEmbeddings(model_name='C://Users//AFOC//Desktop//llm_mark2//models//multilingual-e5-large-instruct'))
    st.session_state.retriever = vectordb.as_retriever(search_kwargs={"k": 4})

retriever = st.session_state.retriever


print(retriever)
#########################################
# sesstion_state들 초기화
#########################################

st.session_state.avatars = {'user': "./assests/icons/airforceman.png", 'assistant': "./assests/icons/aibot.png", 'system': "🍕"}

if 'user_text' not in st.session_state:
    st.session_state.user_text = None

st.session_state.system_message = '''너는 병역이행법 상담을 해주는 챗봇이야. 아래 규칙에 따라 사용자의 질문에 답변해.
1. 주어진 정보들중 필요한 정보만 이용해서 답변해.(모든 정보들을 이용할 필요 없어.)
2. 주어진 정보들로 사용자의 질문에 답할 수 없으면 너가 찾은 정보로는 답변할 수 없다고 말해.
3. 질문과 주어진 정보가 완벽히 일치하지 않는다면 절대 아는 척 하고 말하지마.
4. 주어진 질문에만 답해 추가적인 정보를 원하지 않아.'''
#########################################
# chat_history 초기화 및 리셋
#########################################
with st.sidebar:
    reset_history = st.button("Reset Chat")
    
if "chat_history" not in st.session_state or reset_history:
    st.session_state.chat_history = [{"role": "system", "content": st.session_state.system_message}]



#########################################
# llm 통해 답변을 얻는 함수
#########################################
def format_docs(docs):
    result = ""
    for doc in docs:
        result += "#**<정보>**:\n"
        result += doc.page_content
        result += '\n\n\n'
    return result

def get_response(system_message, user_text, context):
    
    
    template = '''[|system|]{system_message}.
    정보들:
    
    {context}[|endofturn|]
    
    [|user|]{user_text}.[|endofturn|]
    [|assistant|]'''
    
    prompt = PromptTemplate.from_template(template)

    chat = ( prompt | llm.bind(skip_prompt=True) | StrOutputParser(output_key='content')
    )
    
    return chat.stream({
        "user_text" : user_text,
        "system_message": system_message,
        "context": context
    })



#########################################
# chat bot UI(interface 관리)
#########################################
chat_interface = st.container(border=True)
with chat_interface:
    output_container = st.container()
    st.session_state.user_text = st.chat_input(placeholder="ChatBot에게 무엇이든지 물어보세요!")
    
    
with output_container:
    
    for message in st.session_state.chat_history:
        
        if message['role'] == 'system':
            continue       
        with st.chat_message(message['role'], avatar=st.session_state['avatars'][message['role']]):
            st.markdown(message['content'])
            
            
    # When the user enter new text:
    if st.session_state.user_text:        
        
        st.session_state.chat_history.append({"role": "user", "content": st.session_state.user_text})
        with st.chat_message("user", avatar=st.session_state.avatars['user']):
            st.markdown(st.session_state.user_text)
            
        context = format_docs(retriever.invoke(st.session_state.user_text))
        with st.sidebar:
            st.markdown(f"RAG에 이용된 문서는 다음과 같습니다.\n\n{context}")

        with st.chat_message("assistant", avatar=st.session_state.avatars['assistant']):
            
            response = st.write_stream(get_response(
                system_message=st.session_state.system_message, 
                user_text=st.session_state.user_text,
                context = context
            ))
            st.session_state.chat_history.append({"role": "assistant", "content": response})