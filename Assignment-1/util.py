from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from pathlib import Path
from uuid import uuid4
import numpy as np
import config

class Docutils:
    def __init__(self):
        self.loaded_docs = []
        self.all_doc_text = [] # to store the Document objects
        self.hf_embeddings = HuggingFaceEmbeddings(model_name = config.embedding_model_name) # HF light-weight embedding model
    
    # Extract all the .md files within directories and sub-directories
    def get_all_files(self):
        all_docs = list(Path(config.directory_path).rglob(f"*.{config.file_pattern}"))
        
        return all_docs

    # Loader to load all the .md documents
    def document_loader(self, docs):
        for item in docs:
            loader = UnstructuredMarkdownLoader(item)
            self.loaded_docs.append(loader.load())
        
        return self.loaded_docs

    # chunking the documents and creating Document objects
    def document_chunking(self, loaded_docs):
        text_splitter = CharacterTextSplitter(separator= config.separator_of_chunk,
                                              chunk_size = config.size_of_chunk,
                                              chunk_overlap = config.overlap_of_chunk)
        for ele in loaded_docs:
            all_doc_text = text_splitter.split_text(ele[0].page_content)
            for i, txt in enumerate(all_doc_text):
                document = Document(page_content= txt, metadata={"source": ele[0].metadata['source']}, id = str(i))
                self.all_doc_text.append(document)
        return self.all_doc_text
    
    # Creating a vector store and saving it locally so that it can be used during conversation
    def create_vectorstore(self, doc_text):
        uuids = [str(uuid4()) for _ in range(len(doc_text))]
        vector_store = Chroma(
        collection_name=config.name_of_collection,
        embedding_function=self.hf_embeddings,
        persist_directory=config.vector_store_path)
        vector_store.add_documents(documents=doc_text, ids=uuids)
        
        print('Documents added successfully!')

obj = Docutils()
files = obj.get_all_files()
loader = obj.document_loader(files)
doc_txt= obj.document_chunking(loader)
obj.create_vectorstore(doc_txt)
print('Vector store saved locally!')