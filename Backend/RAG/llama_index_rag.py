from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from typing import Dict, List, Tuple, Union, Optional, Any
import psycopg2
import logging
import numpy as np
import ast
from datetime import datetime


class LlamaIndexRAGProcessor():
    def __init__(self):
        self.logger = logging.getLogger("LlamaIndexRAGProcessor")
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        try:
            # Get model name from environment or use default
            model_name ="nomic-ai/nomic-embed-text-v1.5"
           
            
            # Initialize embedding model with trust_remote_code=True
            self.embed_model = HuggingFaceEmbedding(
                model_name=model_name,
                trust_remote_code=True  # This fixes the trust_remote_code error
            )
            
            # Set global settings
            Settings.embed_model = self.embed_model
            
            # Initialize text splitter
            self.text_splitter = SentenceSplitter(
                chunk_size=512,
                chunk_overlap=50
            )
            
            self.logger.info(f"LlamaIndexRAGProcessor initialized successfully with model: {model_name}")
            
        except Exception as e:
            self.logger.error(f"Error initializing HuggingFaceEmbedding: {e}")
            raise RuntimeError(f"Failed to initialize LlamaIndexRAGProcessor: {e}")
        

    def preprocess_text(self, text: str) -> str:
        """Preprocess input text"""
        if not text:
            return ""
        
        # Basic text preprocessing
        text = text.strip()
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        return text
    
    def semantic_splitter(self, text: Any) -> List[str]:
        """Split text semantically into chunks"""
        try:
            if isinstance(text, str):
                preprocessed_text = self.preprocess_text(text)
                # Use LlamaIndex's SentenceSplitter
                nodes = self.text_splitter.split_text(preprocessed_text)
                return [node for node in nodes if node.strip()]
            else:
                self.logger.warning("Input text is not a string")
                return []
        except Exception as e:
            self.logger.error(f"Error in semantic splitting: {e}")
            return []
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a given text"""
        try:
            if not text or not text.strip():
                return []
            
            # Use the embedding model to generate embeddings
            embedding = self.embed_model.get_text_embedding(text.strip())
            return embedding
        except Exception as e:
            self.logger.error(f"Error generating embedding: {e}")
            return []
        
    def process_document(self, document_text: Any) -> Tuple[List[str], List[List[float]]]:
        """Process document to extract chunks and generate embeddings"""
        try:
            # Split text into chunks
            chunks = self.semantic_splitter(document_text)
            
            if not chunks:
                self.logger.warning("No chunks generated from document")
                return [], []
            
            # Generate embeddings for each chunk
            embeddings = []
            for chunk in chunks:
                embedding = self.generate_embedding(chunk)
                if embedding:
                    embeddings.append(embedding)
                else:
                    self.logger.warning(f"Failed to generate embedding for chunk: {chunk[:100]}...")
            
            self.logger.info(f"Processed document into {len(chunks)} chunks with {len(embeddings)} embeddings")
            return chunks, embeddings
            
        except Exception as e:
            self.logger.error(f"Error processing document: {e}")
            return [], []
        


    def query_embedding(self, query: str) -> List[float]:
        """Generate embedding for a query"""
        return self.generate_embedding(query)
    


    

    def save_document_to_db(self,pdf_name: str, pdf_url: str, chunks: List[str], embeddings: List[List[float]]):
            """
            Enregistre les chunks et embeddings dans la table pdf_info.
            """
            if not chunks or not embeddings or len(chunks) != len(embeddings):
                raise ValueError("Chunks et embeddings doivent être non vides et de même longueur")
            try:
                conn = psycopg2.connect(
                    host="localhost",
                    port="5432",
                    database="chatbotdb_2025",
                    user="admin",
                    password="docuBot111"
                )
                cur = conn.cursor()
                
                for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    # Convertir embedding en format vector PostgreSQL
                    embedding_str = '[' + ','.join([str(x) for x in embedding]) + ']'
                    
                    cur.execute("""
                        INSERT INTO pdf_info (pdf_name, pdf_url, chunk_content, chunk_index, embedding)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (pdf_name, pdf_url, chunk, idx, embedding_str))
                
                conn.commit()
                cur.close()
                conn.close()
                self.logger.info(f"{len(chunks)} chunks enregistrés pour {pdf_name}")
            except Exception as e:
               self.logger.error(f"Erreur lors de l'enregistrement en DB: {e}")
            print(f"✅ Document '{pdf_name}' enregistré avec {len(chunks)} chunks.")


    def get_chunks_and_embeddings(self):
        """Récupère tous les chunks et embeddings depuis la base PostgreSQL"""
        conn = psycopg2.connect(
            host="localhost",
            port="5432",
            database="chatbotdb_2025",
            user="admin",
            password="docuBot111"
        )
        cur = conn.cursor()
        cur.execute("SELECT chunk_content, embedding FROM pdf_info;")
        rows = cur.fetchall()
        cur.close()
        conn.close()

        chunks = [row[0] for row in rows]
        embeddings = [np.array(ast.literal_eval(row[1])) for row in rows]
        return chunks, embeddings
    

    def enregistrer_OA_in_Base_donnes(self,query,response):
         """enregistrer les donnes QA dans la table qa_history """
         try:
                conn = psycopg2.connect(
                    host="localhost",
                    port="5432",
                    database="chatbotdb_2025",
                    user="admin",
                    password="docuBot111"
                )
                cur = conn.cursor()
                now=datetime.now()  
                cur.execute("""
                        INSERT INTO qa_history (question, answer, created_at)
                        VALUES (%s, %s, %s)
                    """, (query,response,now))
                
                conn.commit()
                cur.close()
                conn.close()
                self.logger.info("donnes bien  enregistrés")
         except Exception as e:
               self.logger.error(f"Erreur lors de l'enregistrement en DB: {e}")



    def get_context_conversation(self):
        """Récupère les 5 dernier QA depuis la base PostgreSQL"""

        try:
                conn = psycopg2.connect(
                    host="localhost",
                    port="5432",
                    database="chatbotdb_2025",
                    user="admin",
                    password="docuBot111"
                )
                cur = conn.cursor()
                cur.execute("""
                    SELECT question, answer 
                    FROM qa_history 
                    ORDER BY id DESC 
                    LIMIT 5;
                """)
                
                rows = cur.fetchall()

                cur.close()
                conn.close()

                # Retourne une liste [(question, answer), ...]
                return [{"question": row[0], "answer": row[1]} for row in rows]

        except Exception as e:
            print(f"❌ Erreur lors de la récupération du contexte : {e}")
            return []