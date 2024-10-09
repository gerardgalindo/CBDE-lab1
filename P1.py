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

# afegir columna embeddings a la taula
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

# calcular els embeddings
def process_sentence(sentence):
    embedding = model.encode(sentence)
    return embedding.tolist()

# actualitzar la taula amb els embeddings
def update_embedding(row_id, embedding):
    sql = """ UPDATE bookcorpus
                SET embedding = %s
                WHERE id = %s"""
    
    config = load_config()
    
    try:
        with  psycopg2.connect(**config) as conn:
            with  conn.cursor() as cur:
                print("inserint embedding per a la fila: ", row_id)
                temps = time.time()
                cur.execute(sql, (embedding, row_id))
                temps = time.time() - temps
                global minimum, maximum, total
                if temps < minimum:
                    minimum = temps
                if temps > maximum:
                    maximum = temps
                total += temps
                times.append(temps)
                #print("Temps d'execució: ", time.time() - temps)

            conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)    


if __name__ == '__main__':
    alter_table()

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

        print("Temps mínim: ", minimum)
        print("Temps màxim: ", maximum)
        print("Total rows: ", len(rows))
        print("Temps mitjà: ", total/len(rows))
        print("Desviació estàndard: ", statistics.stdev(times))

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)


