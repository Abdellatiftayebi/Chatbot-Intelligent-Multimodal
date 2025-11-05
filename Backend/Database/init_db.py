import psycopg2

def init_db():
    conn = psycopg2.connect(
        host="localhost",
        port="5432",
        database="chatbotdb_2025",
        user="admin",
        password="docuBot111"
    )
    cur = conn.cursor()
    # Table 1 : Informations sur les PDFs
    cur.execute("""
    CREATE TABLE IF NOT EXISTS pdf_info (
        id SERIAL PRIMARY KEY,
        pdf_name VARCHAR(255),
        pdf_url TEXT,
        chunk_content TEXT,
        chunk_index INT,
        embedding VECTOR(1536),  -- optionnel si tu veux stocker les embeddings
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # Table 2 : Questions / Réponses des clients
    cur.execute("""
    CREATE TABLE IF NOT EXISTS qa_history (
        id SERIAL PRIMARY KEY,
        question TEXT,
        answer TEXT,
        pdf_id INT REFERENCES pdf_info(id) ON DELETE CASCADE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("✅ Tables créées avec succès !")

if __name__ == "__main__":
    init_db()
