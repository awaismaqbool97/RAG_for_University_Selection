import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from htmlTemplates import css, bot_template, user_template
import re
import json

DATABASE_PATH = "../Data/local_faiss_index"

def get_pdf_text(pdf_docs):
    text = ""
    try:
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
                else:
                    st.warning(f"Some pages in {pdf.name} may not have text content.")
    except Exception as e:
        st.error(f"Error reading PDF '{pdf.name}': {str(e)}")
    return text

def get_csv_text(csv_docs):
    text = ""
    try:
        for csv in csv_docs:
            df = pd.read_csv(csv)
            text += df.to_string(index=False) + "\n"
    except pd.errors.EmptyDataError:
        st.error(f"CSV file '{csv.name}' is empty. Please provide a valid file.")
    except pd.errors.ParserError:
        st.error(f"CSV file '{csv.name}' contains parsing errors.")
    except Exception as e:
        st.error(f"Error reading CSV '{csv.name}': {str(e)}")
    return text


def get_json_text(json_docs):
    dynamic_chunks = []
    for json_file in json_docs:
        data = json.load(json_file)

        for university in data:
            university_info = university.get("Information", {})
            university_name = university.get("University_Name", "Unknown University")
            
            chunks_for_university = []

            for section_name, section_data in university_info.items():
                if isinstance(section_data, dict):
                    chunk = f"{section_name} for {university_name}:\n"
                    for key, value in section_data.items():
                        if isinstance(value, str):
                            chunk += f"{key}: {value}\n"
                        elif isinstance(value, list):
                            chunk += f"{key}: {', '.join(value)}\n"
                    chunks_for_university.append(chunk)

                elif isinstance(section_data, list):
                    chunk = f"{section_name} for {university_name}:\n"
                    chunk += ", ".join(section_data) + "\n"
                    chunks_for_university.append(chunk)

            dynamic_chunks.append("\n".join(chunks_for_university))

    return "\n".join(dynamic_chunks)




def get_text_chunks(text):
    try:
        splitter = CharacterTextSplitter(separator="\n", chunk_size=1200, chunk_overlap=200)
        text_chunks = splitter.split_text(text)
        if not text_chunks:
            st.warning("No text chunks generated. Ensure your documents contain enough textual content.")
        return text_chunks
    except Exception as e:
        st.error(f"Error splitting text into chunks: {str(e)}")
        return []

def save_vector_store(vectorstore):
    try:
        if os.path.exists(DATABASE_PATH):
            import shutil
            shutil.rmtree(DATABASE_PATH)

        os.makedirs(DATABASE_PATH, exist_ok=True)

        vectorstore.save_local(DATABASE_PATH)
        st.success("Database saved locally and previous data overwritten.")
    except Exception as e:
        st.error(f"Error saving vector store: {str(e)}")


def load_vector_store():
    try:
        if os.path.exists(DATABASE_PATH):
            vectorstore = FAISS.load_local(DATABASE_PATH, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
            st.success("Database loaded from local storage.")
            return vectorstore
        else:
            st.warning("No local database found. Please upload and analyze files to create one.")
            return None
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        return None


def get_vector_store(text_chunks):
    try:
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vector_store
    except ValueError as e:
        st.error(f"ValueError while creating vector store: {str(e)}")
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
    return None

def get_conversation_chain(vectorstore):
    try:
        llm = ChatOpenAI(temperature=0)
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
        return conversation_chain
    except ValueError as e:
        st.error(f"ValueError while creating conversation chain: {str(e)}")
    except Exception as e:
        st.error(f"Error creating conversation chain: {str(e)}")
    return None

def handle_userinput(user_question):
    if not st.session_state.conversation:
        st.error("No files have been uploaded and analyzed yet. Please upload and analyze files before asking a question.")
        return

    try:
        data_folder = "../Data/"
        csv_file = f"{data_folder}university_overview.csv"

        university_names = set()
        try:
            df = pd.read_csv(csv_file)
            if "University_Name" in df.columns:
                for entry in df["University_Name"].dropna():
                    names = [name.strip().lower() for name in entry.split("or")]
                    university_names.update(names)
            else:
                st.warning("The CSV file does not contain a 'University_Name' column.")
        except FileNotFoundError:
            st.error("The university overview CSV file is missing.")
            return
        except Exception as e:
            st.error(f"Error processing the CSV file: {str(e)}")
            return

        comparison_keywords = ["compare", "better", "best", "vs", "versus", "comparison"]
        comparison_intent = any(keyword in user_question.lower() for keyword in comparison_keywords)

        mentioned_universities = [
            uni for uni in university_names if re.search(rf"\b{re.escape(uni)}\b", user_question.lower())
        ]

        if mentioned_universities:
            prompt = f"Give answer based on {', '.join(mentioned_universities)} only and use previous' message context if details not mentioned."
        else:
            prompt = "Give generalized answer combining all available university data or specific university data only being used in context in previous messages."

        if comparison_intent:
            added_text = "give answer by Comparing the Data of mentioned universities only you have and do complete analysis and comparison of information and also use rankings, strengths, opportunities etc to support the answer."
            prompt += added_text

        
        full_prompt = f"You are a RAG chatbot designed to assist students in gathering information about universities and help them decide where to apply. Your goal is to answer questions clearly, with a focus on factual and helpful insights from the provided university database. {prompt} QUESTION = '{user_question}' Please ensure the information is relevant, accurate, and clearly explained."


        response = st.session_state.conversation({'question': full_prompt})
    

        st.session_state.chat_history.append({"role": "user", "content": user_question})
        st.session_state.chat_history.append({"role": "bot", "content": response['answer']})

        for message in st.session_state.chat_history:
            if message['role'] == "user":
                st.write(user_template.replace("{{MSG}}", message['content']), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message['content']), unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error handling user input: {str(e)}")


def main():
    try:
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        load_dotenv()
        api_key = st.secrets.get("OPENAI_API_KEY", None)
        if not api_key:
            st.error("OpenAI API key is missing in Streamlit secrets.")
            return

        # st.set_page_config(page_title="RAG", page_icon="::")
        st.set_page_config(page_title="University Chatbot", page_icon="🎓")
        st.write(css, unsafe_allow_html=True)

        if 'conversation' not in st.session_state or st.session_state.conversation is None:
            st.session_state.conversation = None


        st.header("University Chatbot 🎓")

        message = st.text_input("Ask a question")

        if message:
            handle_userinput(message)

        with st.sidebar:
            st.subheader('Current version, supports PDF, CSV and JSON files Only')

            docs = st.file_uploader("Upload a document", accept_multiple_files=True)

            st.subheader("Manual")
            st.write("1. Upload PDF, CSV or json files for analysis.")
            st.write("2. Click 'Analyze' to start the analysis.")
            st.write("3. The chatbot will provide answers based on the document content.")

            if st.button("Analyze"):
                with st.spinner("Analyzing..."):
                    raw_text = ""
                    if docs and len(docs) > 0:
                        for doc in docs:
                            if doc.name.endswith(".pdf"):
                                raw_text += get_pdf_text([doc])
                            elif doc.name.endswith(".csv"):
                                raw_text += get_csv_text([doc])
                            elif doc.name.endswith(".json"):
                                raw_text += get_json_text([doc])
                            else:
                                st.error(f"Unsupported file format: {doc.name}. Please upload PDF or CSV files.")

                        text_chunks = get_text_chunks(raw_text)
                        if text_chunks:
                            vectorstore = get_vector_store(text_chunks)

                        if vectorstore:
                                save_vector_store(vectorstore)
                                st.session_state.conversation = get_conversation_chain(vectorstore)
                                st.success("Analysis complete.")
                        else:
                            st.warning("No text extracted from the uploaded files.")
                    else:
                        st.error("Please upload at least one PDF or CSV file.")
            else:
                vectorstore = load_vector_store()
                if vectorstore:
                    st.session_state.conversation = get_conversation_chain(vectorstore)

    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")

if __name__ == '__main__':
    main()
