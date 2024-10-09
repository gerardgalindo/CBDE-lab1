import psycopg2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from config import load_config
import time
import statistics

times = []


# Agafa 10 frases
def fetch_sentences(limit, conn):
    cursor = conn.cursor()
    cursor.execute("SELECT id, text, embedding FROM bookcorpus_pgvector ORDER BY id LIMIT %s;", (limit,))

    sentences = cursor.fetchall()

    cursor.close()
    
    return sentences

def find_similar_pgvector(metric, sentence, conn):
    id, text, embedding = sentence
    
    query = f"""
        SELECT id, text, embedding {metric} '{embedding}' AS distance
        FROM bookcorpus_pgvector
        ORDER BY embedding {metric} '{embedding}'
        LIMIT 3;
        """

    
    with conn.cursor() as cursor:
        temps = time.time()
        cursor.execute(query)
        times.append(time.time() - temps)
        results = cursor.fetchall()
    
    print(f"Comparing sentence ID {id}: '{text}'")
    print("Two most similar sentences:")
    for result in results[1:3]:  # Skip the first result as it is the sentence itself
        similar_id, similar_text, distance = result
        print(f"ID: {similar_id}, Sentence: '{similar_text}', Distance: {distance}")

def find_similar_sentences(sentences, conn):
    global times

    print("Calculating similarities with euclidean distance")
    # Euclidean distance
    for i in range(len(sentences)):
        find_similar_pgvector("<->", sentences[i], conn)

    print("---- Temps Euclidean distance ----")
    print("Temps mínim: ", min(times))
    print("Temps màxim: ", max(times))
    print("Temps mitjà: ", sum(times)/len(times))
    print("Desviació estàndard: ", statistics.stdev(times))
    print("---------------------------------")

    times = []

    print("Calculating similarities with cosine similarity")
    # Cosine Similarity
    for i in range(len(sentences)):
        find_similar_pgvector("<=>", sentences[i], conn)
       
    print("---- Temps cosine similarity ----")
    print("Temps mínim: ", min(times))
    print("Temps màxim: ", max(times))
    print("Temps mitjà: ", sum(times)/len(times))
    print("Desviació estàndard: ", statistics.stdev(times))
    print("---------------------------------")


if __name__ == '__main__':

    try:
        config = load_config()
        with psycopg2.connect(**config) as conn:
            num_frases = 10

            #Agafa 10 frases
            sentences = fetch_sentences(num_frases, conn)
            
            #Cerca les 2 frases més semblants
            similar_sentences = find_similar_sentences(sentences, conn)

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)