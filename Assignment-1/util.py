from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from pathlib import Path
from uuid import uuid4
import numpy as np
import chromadb
import config


class Docutils:
    def __init__(self):
        self.loaded_docs = []
        self.all_doc_text = [] # to store the document chunks
        self.all_doc_embeddings = [] # to store the chunk embeddings
        self.all_doc_metadata = [] # to store the document metada
        self.all_doc_ids = [] # to store document id
        self.hf_embeddings = HuggingFaceEmbeddings(model_name = config.embedding_model_name)
    
    def get_all_files(self):
        all_docs = list(Path(config.directory_path).rglob(f"*.{config.file_pattern}"))
        
        return all_docs

    def document_loader(self, docs):
        for item in docs:
            loader = UnstructuredMarkdownLoader(item)
            self.loaded_docs.append(loader.load())
        
        return self.loaded_docs

    def document_chunking(self, loaded_docs):
        text_splitter = CharacterTextSplitter(separator= config.separator_of_chunk,
                                              chunk_size = config.size_of_chunk,
                                              chunk_overlap = config.overlap_of_chunk)
        for ele in loaded_docs:
            all_doc_text = text_splitter.split_text(ele[0].page_content)
            for i, txt in enumerate(all_doc_text):
                document = Document(page_content= txt, metadata={"source": ele[0].metadata['source']}, id = str(i))
                # doc_metadata = ele[0].metadata['source']
                # doc_embedding = self.hf_embeddings.embed_documents([txt])
                # self.all_doc_text.extend([txt])
                # self.all_doc_embeddings.extend(np.array(doc_embedding))
                # self.all_doc_metadata.append({'source': doc_metadata})
                # self.all_doc_ids.append(str(uuid4()))
                self.all_doc_text.append(document)
        # return self.all_doc_text, self.all_doc_embeddings, self.all_doc_metadata, self.all_doc_ids
        return self.all_doc_text
    
    def create_collection(self, doc_text, doc_embeds, doc_metadata, doc_id):
        # new_collection = self.client.create_collection(name=config.name_of_collection)
        # new_collection.add(documents=doc_text, metadatas= doc_metadata, ids = doc_id, embeddings=doc_embeds)
        # self.client.persist(config.vector_store_path)
        presistant_storage = chromadb.PersistentClient(path= config.vector_store_path)
        document_collection = presistant_storage.create_collection(name= config.name_of_collection,metadata={"hnsw:space":"cosine"})
        document_collection.add(documents=doc_text, metadatas= doc_metadata, ids = doc_id, embeddings=doc_embeds)
    
    def create_vectorstore(self, doc_text):
        uuids = [str(uuid4()) for _ in range(len(doc_text))]
        vector_store = Chroma(
        collection_name="document_collection",
        embedding_function=self.hf_embeddings,
        persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)
        vector_store.add_documents(documents=doc_text, ids=uuids)

        results = vector_store.similarity_search("How can we do Authentication through the Serial Vault?", k=2)
        print(results)

obj = Docutils()
files = obj.get_all_files()
loader = obj.document_loader(files)
doc_txt= obj.document_chunking(loader)
obj.create_vectorstore(doc_txt)
print('Collection saved locally!')