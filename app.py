# import os
# import streamlit as st
# import pickle
# import time
# from langchain import OpenAI
# from langchain.chains import RetrievalQAWithSourcesChain
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import UnstructuredURLLoader
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
# from dotenv import load_dotenv

# load_dotenv()  # take environment variables from .env (especially openai api key)

# st.title("News Research Tool 📈")
# st.sidebar.title("News Article URLs")

# urls = []
# for i in range(3):
#     url = st.sidebar.text_input(f"URL {i+1}")
#     urls.append(url)

# process_url_clicked = st.sidebar.button("Process URLs")  # Button to initiate processing of entered URLs
# file_path = "faiss_store_openai.pkl"  # File path for storing serialized FAISS index 

# main_placeholder = st.empty()  # Placeholder for main content area
# llm = OpenAI(temperature=0.9, max_tokens=500)  # Initializing OpenAI language model 

# if process_url_clicked: 
#     loader = UnstructuredURLLoader(urls=urls)
#     main_placeholder.text("Data Loading...Started...✅✅✅")  # Display loading message
#     data = loader.load()

#     # Split data into smaller documents
#     text_splitter = RecursiveCharacterTextSplitter(
#         separators=['\n\n', '\n', '.', ','],
#         chunk_size=1000
#     )
#     main_placeholder.text("Text Splitter...Started...✅✅✅")  # Display text splitting message
#     docs = text_splitter.split_documents(data)

#     # Create embeddings from documents and build FAISS index
#     embeddings = OpenAIEmbeddings()
#     vectorstore_openai = FAISS.from_documents(docs, embeddings)
#     pkl = vectorstore_openai.serialize_to_bytes()
#     main_placeholder.text("Embedding Vector Started Building...✅✅✅")  # Display embedding vector building message
#     time.sleep(2)  # Simulate processing time

#     # Save the FAISS index to a pickle file
#     with open(file_path, "wb") as f:
#         pickle.dump(pkl, f)


# # Input field for user query
# query = main_placeholder.text_input("Question: ")

# if query:
#     if os.path.exists(file_path):
#         with open(file_path, "rb") as f:
#             pkl = pickle.load(f)
#             # Deserialize the FAISS index and create a retrieval question-answering chain
#             vectorstore = FAISS.deserialize_from_bytes(embeddings=OpenAIEmbeddings(), serialized=pkl, allow_dangerous_deserialization=True)
#             chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
#             result = chain({"question": query}, return_only_outputs=True)
#             # result will be a dictionary of this format --> {"answer": "", "sources": [] }
#             st.header("Answer")  # Display header for answer
#             st.write(result["answer"])  # Display the answer

#             # Display sources, if available
#             sources = result.get("sources", "")
#             if sources:
#                 st.subheader("Sources:")  # Display subheader for sources
#                 sources_list = sources.split("\n")  # Split sources by newline
#                 for source in sources_list:
#                     st.write(source)  # Display each source



import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Initialize the Streamlit app title
st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

# Create an empty list to hold URLs from the user input
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}")
    urls.append(url)

# Button to trigger the processing of URLs
process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"  # File path for storing the FAISS index

# Placeholder to display content dynamically
main_placeholder = st.empty()

# Initialize the OpenAI language model (LLM)
llm = OpenAI(temperature=0.9, max_tokens=500)

# Process the URLs if the button is clicked
if process_url_clicked:
    # Load content from the URLs
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")  # Inform the user data is loading
    data = loader.load()

    # Split the data into smaller chunks for processing
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitting...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    # Create embeddings from the documents and build the FAISS index
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    pkl = vectorstore_openai.serialize_to_bytes()  # Serialize the FAISS index
    main_placeholder.text("Embedding Vector Building...✅✅✅")
    time.sleep(2)  # Simulate processing time

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(pkl, f)

# Input field for user query
query = main_placeholder.text_input("Ask a question about the news articles:")

# Process the query if the user enters one
if query:
    if os.path.exists(file_path):  # Check if the FAISS index file exists
        with open(file_path, "rb") as f:
            pkl = pickle.load(f)  # Load the serialized FAISS index

            # Deserialize the FAISS index and create the retrieval question-answering chain
            vectorstore = FAISS.deserialize_from_bytes(
                embeddings=OpenAIEmbeddings(),
                serialized=pkl,
                allow_dangerous_deserialization=True
            )
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())

            # Get the result from the chain
            result = chain({"question": query}, return_only_outputs=True)

            # Display the answer
            st.header("Answer")
            st.write(result["answer"])

            # Display the sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")
                for source in sources_list:
                    st.write(source)














