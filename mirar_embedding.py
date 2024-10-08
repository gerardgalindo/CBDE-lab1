import psycopg2
from config import load_config
from sentence_transformers import SentenceTransformer
import time
import statistics

minimum = float('inf')
maximum = float('-inf')
total = 0
times = []
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')




if __name__ == '__main__':
    #alter_table()

    try:
        config = load_config()
        with psycopg2.connect(**config) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT embedding FROM bookcorpus WHERE id = 1")
                print(len(cur.fetchone()[0]))
                



    except (Exception, psycopg2.DatabaseError) as error:
        print(error)


