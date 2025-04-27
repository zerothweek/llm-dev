import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import BitsAndBytesConfig
import torch
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="EXAONE-3.5-2.4B-BASELINE")
st.title("EXAONE-3.5-2.4B-BASELINE")
st.markdown(
    """
    🎧
    """
)


#########################################
# 모델(EXAONE-3.5-2.4B-Instruct) 불러오기
#########################################
model_name = '../models/EXAONE-3.5-2.4B-Instruct'
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
            max_new_tokens=2048,
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
# sesstion_state들 초기화
#########################################

st.session_state.avatars = {'user': "./assests/icons/airforceman.png", 'assistant': "./assests/icons/aibot.png"}

if 'user_text' not in st.session_state:
    st.session_state.user_text = None

st.session_state.system_message = "You are EXAONE model from LG AI Research, a helpful assistant."


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
def get_response(system_message, user_text):
    
    template = '''[|system|]{system_message}.[|endofturn|]
    [|user|]{user_text}
    [|assistant|]'''
    
    prompt = PromptTemplate.from_template(template)
    print(prompt)
    chat = prompt | llm.bind(skip_prompt=True) | StrOutputParser(output_key='content')
    
    return chat.stream({
        "user_text" : user_text,
        "system_message": system_message
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
            
            
        with st.chat_message("assistant", avatar=st.session_state.avatars['assistant']):
            response = st.write_stream(get_response(
                system_message=st.session_state.system_message, 
                user_text=st.session_state.user_text,
            ))
            st.session_state.chat_history.append({"role": "assistant", "content": response})