import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
#from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from detectron2.config import get_cfg
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain_groq import ChatGroq
# from langchain.llms import OpenAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.callbacks import get_openai_callback
import os

# Sidebar contents
with st.sidebar:
    st.title('🤗💬 LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
 
    ''')
    #add_vertical_space(5)
    #st.write('Made with ❤️ by [Himanshu]')

    load_dotenv()

    def main():
        st.header("Chat with PDF 💬")

    # upload a PDF file
        pdf = st.file_uploader("Upload your PDF", type='pdf')
        if pdf is not None:
            st.write(pdf.name)

            # Save the uploaded PDF to a temporary file
            with open(pdf.name, "wb") as f:
                f.write(pdf.getbuffer())
            
            loaders = [UnstructuredPDFLoader(pdf.name)]

        # st.write(pdf)
        if pdf is not None:
            pdf_reader = PdfReader(pdf)

            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )

            cfg = get_cfg()
            cfg.MODEL.DEVICE = 'CPU' 

            chunks = text_splitter.split_text(text=text)

             # # embeddings
            store_name = pdf.name[:-4]
            st.write(f'{store_name}')
            # st.write(chunks)
    
            if os.path.exists(f"{store_name}.pkl"):
                with open(f"{store_name}.pkl", "rb") as f:
                    VectorStore = pickle.load(f)
                st.write('Embeddings Loaded from the Disk')
            else:
                embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                VectorStore = FAISS.from_texts(chunks, embedding=embedding)
                with open(f"{store_name}.pkl", "wb") as f:
                    pickle.dump(VectorStore, f)
                    st.write('Embeddings Computation completed')

            # Accept user question or queries
            query = st.text_input("Ask your question about the PDF file")
            if query:
                model_name = "Llama3-8b-8192"
                embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                llm = ChatGroq(groq_api_key="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",model_name=model_name)
                index = VectorstoreIndexCreator(embedding=embedding).from_loaders(loaders)
                response = index.query(query, llm=llm)
                st.write(response)
                    
            #st.write(chunks)

if __name__ == '__main__':
    main()