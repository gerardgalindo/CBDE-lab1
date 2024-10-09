from datasets import load_dataset
from config import load_config
import psycopg2
import time
import statistics

minimum = float('inf')
maximum = float('-inf')
total = 0
times = []

# crear la taula
def create_table():
    try:
        config = load_config()
        query = """
        CREATE TABLE IF NOT EXISTS bookcorpus_pgvector (
            id bigserial PRIMARY KEY,
            text TEXT NOT NULL
        )
        """
        with psycopg2.connect(**config) as conn:
            with conn.cursor() as cur:
                cur.execute(query)
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)

# inserir les frases a la taula
def insert_sentences(sentences):
    try:
        config = load_config()
        with psycopg2.connect(**config) as conn:
            with conn.cursor() as cur:
                for sentence in sentences:
                    temps = time.time()
                    cur.execute("INSERT INTO bookcorpus_pgvector (text) VALUES (%s)", (sentence,))
                    temps = time.time() - temps
                    global minimum, maximum, total  
                    if temps < minimum:
                        minimum = temps
                    if temps > maximum:
                        maximum = temps
                    total += temps
                    times.append(temps)
                conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)

if __name__ == '__main__':
    create_table()

    ds = load_dataset("williamkgao/bookcorpus100mb", split="train")
    subset = ds.select(range(10000))  
    sentences = [item['text'] for item in subset]
    
    insert_sentences(sentences)

    print("Temps mínim: ", minimum)
    print("Temps màxim: ", maximum)
    print("Temps mitjà: ", total/10000)
    print("Desviació estàndard: ", statistics.stdev(times))