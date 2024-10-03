from datasets import load_dataset
from config import load_config
import psycopg2

def create_table():
    try:
        config = load_config()
        query = """
        CREATE TABLE IF NOT EXISTS bookcorpus (
            id SERIAL PRIMARY KEY,
            text TEXT NOT NULL
        )
        """
        with psycopg2.connect(**config) as conn:
            with conn.cursor() as cur:
                cur.execute(query)
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)

def insert_sentences(sentences):
    try:
        config = load_config()
        with psycopg2.connect(**config) as conn:
            with conn.cursor() as cur:
                for sentence in sentences:
                    cur.execute("INSERT INTO bookcorpus (text) VALUES (%s)", (sentence,))
                conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)

if __name__ == '__main__':
    create_table()

    ds = load_dataset("williamkgao/bookcorpus100mb", split="train")
    subset = ds.select(range(10000))  
    sentences = [item['text'] for item in subset]
    
    insert_sentences(sentences)
