import os
from PIL import Image
from langchain_huggingface import HuggingFacePipeline
import streamlit as st
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers import BitsAndBytesConfig
import torch
image_path = "./icons/background.png"
background = Image.open(image_path)


model_id =  "./models/EXAONE-3.5-7.8B-Instruct-SFT/checkpoint-32"


bnb_config = BitsAndBytesConfig(
    
    load_in_4bit=True,
    bnb_4bit_quant_dtype="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
    
)




def get_llm_hf_inference(max_new_token, model_name=model_id):
    llm = HuggingFacePipeline.from_model_id(
        model_id=model_name,
        task="text-generation",
        pipeline_kwargs=dict(
            max_new_tokens=max_new_token,
            do_sample=False,
            # repetition_penalty=1.03,
        ),
        model_kwargs=dict(
            trust_remote_code=True,
            quantization_config = bnb_config
        ),
    )
    return llm

# Configure the Streamlit app
st.set_page_config(page_title="작피티", page_icon="./icons/logo.png")
st.title("작피티")

# Initialize session state for avatars
st.session_state.avatars = {'user': "./icons/airforceman.png", 'assistant': "./icons/aibot.png"}

# Initialize session state for user text input
if 'user_text' not in st.session_state:
    st.session_state.user_text = None

# Initialize session state for model parameters
if "max_response_length" not in st.session_state:
    st.session_state.max_response_length = 256

# if tuning include system_message this use
#st.session_state.system_message = "Your name is 작피티 not EXAONE and You are a friendly AI conversing with a human user."

st.session_state.system_message = "You are EXAONE model from LG AI Research, a helpful assistant."

# also tuning with
st.session_state.starter_message = "안녕하세요! 무엇을 도와드릴까요?"
    
with st.sidebar:
    st.header("Setting")

    # Model Settings
    st.session_state.max_response_length = st.number_input(
        "Max Response Length", value=128
    )
    
    # Reset Chat History
    reset_history = st.button("Reset Chat")
    
# Initialize or reset chat history
if "chat_history" not in st.session_state or reset_history:
    #st.session_state.chat_history = [{"role": "system", "content": st.session_state.system_message}, {"role" : "assistant", "content" : st.session_state.starter_message}]
    st.session_state.chat_history = [{"role": "system", "content": st.session_state.system_message}]
def get_response(system_message, chat_history, user_text, max_new_tokens=st.session_state.max_response_length):
    hf = get_llm_hf_inference(max_new_tokens)

    # Create the prompt template
    prompt = PromptTemplate.from_template(
        (
            "[INST] {system_message}"
            "\nCurrent Conversation:\n{chat_history}\n\n"
            "\nUser: {user_text}.\n [/INST]"
            "\nAI:"
        )
    )
    print(prompt)
    # Make the chain and bind the prompt
    chat = prompt | hf.bind(skip_prompt=True) | StrOutputParser(output_key='content')
    
    return chat.stream({
        "chat_history" : chat_history,
        "user_text" : user_text,
        "system_message": system_message
    })

# Chat interface
chat_interface = st.container(border=True)
with chat_interface:
    output_container = st.container()
    st.session_state.user_text = st.chat_input(placeholder="여기에 입력해주세요.")
    
# Display chat messages
with output_container:
    # For every message in the history
    for message in st.session_state.chat_history:
        # Skip the system message
        if message['role'] == 'system':
            continue
            
        # Display the chat message using the correct avatar
        with st.chat_message(message['role'], 
                             avatar=st.session_state['avatars'][message['role']]):
            st.markdown(message['content'])
            
 # When the user enter new text:
    if st.session_state.user_text:
        
        st.session_state.chat_history.append({"role": "user", "content": st.session_state.user_text})
        # Display the user's new message immediately
        with st.chat_message("user", 
                             avatar=st.session_state.avatars['user']):
            st.markdown(st.session_state.user_text)
            
        # Display a spinner status bar while waiting for the response
        with st.chat_message("assistant", 
                             avatar=st.session_state.avatars['assistant']):

            response = st.write_stream(get_response(
                system_message=st.session_state.system_message, 
                user_text=st.session_state.user_text,
                chat_history=st.session_state.chat_history,
                max_new_tokens=st.session_state.max_response_length,
            ))
            st.session_state.chat_history.append({"role": "assistant", "content": response})