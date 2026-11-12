import streamlit as st
import requests
import time

def api_request(method, url, **kwargs):
    for attempt in range(6):
        try:
            response = requests.request(
                method,
                url,
                timeout=120,
                **kwargs
            )

            if method.upper() == "GET" and response.status_code in [502, 503, 504]:
                if attempt < 5:
                    time.sleep(10)
                    continue

            return response

        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout):
            if attempt == 5:
                raise
            time.sleep(10)

if "chat_history" not in st.session_state:
  st.session_state.chat_history = []

API_URL = "https://document-intelligence-chatbot-4.onrender.com"

st.title("Document Intelligence Chatbot")

st.text("FastAPI Connection")

if st.button("Test Connection"):
  try:
    response = api_request("GET", f"{API_URL}/docs")
    if response.status_code == 200:
      st.write("Successfully connected to FastAPI!")
    else:
      st.write("FastAPI returned an error.")
      st.write("Status code:", response.status_code)
  except requests.exceptions.ConnectionError as e:
    st.error(f"Connection failed")


upload_files = st.file_uploader("Choose a files", type=["pdf", "txt"], accept_multiple_files=True)
if upload_files:
  if st.button("upload"):
    files = []
    for upload_file in upload_files:
     files.append(("files", (upload_file.name, upload_file.getvalue(), upload_file.type)))
    url = f"{API_URL}/upload"
    try:
     response = api_request("POST", url, files=files)
     if response.status_code == 200:
       st.success("Uploaded successfully!")
     else:
       st.error(f"Upload failed")
    except requests.exceptions.ConnectionError:
     st.error("Could not connect to FastAPI.")


prompt = st.chat_input("Ask a question about the files....")
try:
 if prompt:
  url = f"{API_URL}/chat"
  response = api_request("POST", url, json={"question":prompt})
  if response.status_code == 200:
    st.session_state.chat_history.append({"user":prompt})
    answer = response.json()['response']
    st.session_state.chat_history.append({"assistant": answer})
  else:
    st.error(f"{response.text}")
except requests.exceptions.ConnectionError:
  st.error("Could not connect to FastAPI.")

chat_start = False
for chat in st.session_state.chat_history:
      for key, value in chat.items():
        with st.chat_message(key):
          st.write(value)
          chat_start = True
if chat_start == True:
 if st.button("Clear Chat"):
   st.session_state.chat_history.clear()
   st.rerun()
