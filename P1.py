import psycopg2
from config import load_config
from sentence_transformers import SentenceTransformer

def alter_table():
    try:
        config = load_config()
        query = """
        ALTER TABLE bookcorpus ADD COLUMN embedding FLOAT8[]; 
        """
        with psycopg2.connect(**config) as conn:
            with conn.cursor() as cur:
                cur.execute(query)
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)

def process_sentence(sentence):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embedding = model.encode(sentence)
    return embedding.tolist()

def update_embedding(row_id, embedding):
    sql = """ UPDATE bookcorpus
                SET embedding = %s
                WHERE id = %s"""
    
    config = load_config()
    
    try:
        with  psycopg2.connect(**config) as conn:
            with  conn.cursor() as cur:
                # execute the UPDATE statement
                cur.execute(sql, (embedding, row_id))

            # commit the changes to the database
            conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)    


if __name__ == '__main__':
    #alter_table()

    try:
        config = load_config()
        with psycopg2.connect(**config) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id, text FROM bookcorpus")
                rows = cur.fetchall()
                    
                for row in rows:
                    row_id = row[0]
                    sentence = row[1]
                    embedding = process_sentence(sentence)
                    update_embedding(row_id, embedding)

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)


