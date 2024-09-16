import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.callbacks import StreamlitCallbackHandler
import os
import shelve
import pyperclip

st.set_page_config(page_title="Prompt Generator", page_icon="🤖")
st.title("PROMPT GENERATOR")

USER_AVATAR = "👤"
BOT_AVATAR = "🤖"

# Load chat history from shelve file
def load_chat_history():
    with shelve.open("chat_history") as db:
        return db.get("messages", [])

# Save chat history to shelve file
def save_chat_history(messages):
    with shelve.open("chat_history") as db:
        db["messages"] = messages

# Display chat history
def display_chat_history():
    for message in st.session_state.messages:
        avatar = USER_AVATAR if message["role"] == "human" else BOT_AVATAR
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

# Initialize session state or load chat history
if "messages" not in st.session_state or st.session_state.get('reset_history', True):
    st.session_state.messages = load_chat_history()
    st.session_state.is_first_message = True
    st.session_state.reset_history = False
if "llm_provider" not in st.session_state:
    st.session_state.llm_provider = "OpenAI"
if "openai_model" not in st.session_state:
    st.session_state.openai_model = "gpt-4"
if "gemini_model" not in st.session_state:
    st.session_state.gemini_model = "gemini-1.5-pro-exp-0827"
if "input_variables" not in st.session_state:
    st.session_state.input_variables = []
if "output_format" not in st.session_state:
    st.session_state.output_format = ""
if "response_content" not in st.session_state:
    st.session_state.response_content = ""  # To store the LLM output

# Load system prompt
def load_system_prompt():
    with open("sys_prompt.txt", "r") as file:
        return file.read()

system_prompt = load_system_prompt()

# Sidebar for LLM selection and API key input
with st.sidebar:
    st.session_state.llm_provider = st.selectbox("Select LLM Provider", ["OpenAI", "Gemini"])
    
    if st.session_state.llm_provider == "OpenAI":
        api_key = st.text_input("Enter OpenAI API Key", type="password")
        st.session_state.openai_model = st.selectbox("Select OpenAI Model", ["gpt-4o", "gpt-4"])
        os.environ["OPENAI_API_KEY"] = api_key
    else:
        api_key = st.text_input("Enter Gemini API Key", type="password")
        st.session_state.gemini_model = st.selectbox("Select Gemini Model", ["gemini-1.5-pro-exp-0827", "gemini-1.5-flash", "gemini-1.5-flash-exp-0827", "gemini-1.5-pro"])
        os.environ["GOOGLE_API_KEY"] = api_key

    # Optional Input Variables
    with st.expander("Input Variables (EXPERIMENTAL)", expanded=False):
        new_variable = st.text_input("Add new input variable")
        
        if st.button("Add Variable"):
            if new_variable and new_variable not in st.session_state.input_variables:
                st.session_state.input_variables.append(new_variable)
        
        st.write("Current Input Variables:")
        
        # Iterate over the current input variables and display them with a "Remove" button
        for idx, var in enumerate(st.session_state.input_variables):
            col1, col2 = st.columns([1, 1])  # Adjust column widths for better alignment
            with col1:
                st.write(var)  # Display the variable
            with col2:
                if st.button("Remove", key=f"remove_{idx}"):
                    st.session_state.input_variables.pop(idx)  # Remove the variable


    # Optional Output Format
    with st.expander("Output Format (EXPERIMENTAL)", expanded=False):
        st.session_state.output_format = st.text_area("Enter desired output format", st.session_state.output_format)

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.is_first_message = True
        st.session_state.response_content = ""  # Reset LLM response
        save_chat_history([])

# Display chat history
display_chat_history()

# Disable chat input if API key is not entered
input_disabled = not bool(api_key)

# Main chat interface
prompt = st.chat_input("How can I help?", disabled=input_disabled)
if prompt and not input_disabled:
    if st.session_state.is_first_message:
        # Replace the {{TASK}} placeholder in the system prompt with the user input
        full_prompt = system_prompt.replace("{{TASK}}", prompt)

        # Check if input variables are provided and append them in a new line
        if st.session_state.input_variables:
            variables_str = f"\nInput Variables: {', '.join(st.session_state.input_variables)}"
            full_prompt += variables_str  # Append input variables to the full prompt

        # Check if output format is provided and append it in a new line after input variables
        if st.session_state.output_format:
            output_format_str = f"\nOutput should strictly be in this format: {st.session_state.output_format}"
            full_prompt += output_format_str  # Append output format to the full prompt

        # Add the system message with the full prompt to the chat history
        st.session_state.messages.append({"role": "system", "content": full_prompt})
        st.session_state.is_first_message = False

    else:
        history_str = "\n".join([f'{msg["role"]}: {msg["content"]}' for msg in st.session_state.messages])
        
        full_prompt = f'Human: {prompt}\nHistory for Reference:{history_str}'


    st.session_state.messages.append({"role": "human", "content": prompt})
    with st.chat_message("human", avatar=USER_AVATAR):
        st.markdown(prompt)

    with st.chat_message("ai", avatar=BOT_AVATAR):
        message_placeholder = st.empty()
        
        if st.session_state.llm_provider == "OpenAI":
            llm = ChatOpenAI(model=st.session_state.openai_model, streaming=True)
        else:
            llm = ChatGoogleGenerativeAI(model=st.session_state.gemini_model)
        
        msgs = [SystemMessage(content=m["content"]) if m["role"] == "system" else HumanMessage(content=m["content"]) if m["role"] == "human" else AIMessage(content=m["content"]) 
                for m in st.session_state.messages] 
         
        stream_handler = StreamlitCallbackHandler(message_placeholder)
        response = llm.predict(full_prompt, callbacks=[stream_handler])
        message_placeholder.markdown(response)
        st.session_state.response_content = response  # Store the LLM response
    
    st.session_state.messages.append({"role": "ai", "content": response})

    # Save chat history after each interaction
    save_chat_history(st.session_state.messages)

# Add a button to copy the LLM output to the clipboard
if st.session_state.response_content:
    if st.button("Copy to Clipboard"):
        pyperclip.copy(st.session_state.response_content)
        st.success("Response copied to clipboard!")

# Display current input variables and output format (optional)
if st.session_state.input_variables or st.session_state.output_format:
    st.sidebar.subheader("Current Settings")
    if st.session_state.input_variables:
        st.sidebar.write(f"Input Variables: {', '.join(st.session_state.input_variables)}")
    if st.session_state.output_format:
        st.sidebar.write(f"Output Format: {st.session_state.output_format}")