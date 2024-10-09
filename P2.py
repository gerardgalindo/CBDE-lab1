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
    
    cursor.execute("SELECT id, text, embedding FROM bookcorpus ORDER BY id LIMIT %s;", (limit,))

    sentences = cursor.fetchall()

    cursor.close()
    
    return sentences

# Agafa totes les frases
def fetch_all_sentences(conn):
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, text, embedding FROM bookcorpus")
    sentences = cursor.fetchall()

    cursor.close()
    
    return sentences

# Busca les dues frases mes semblants
def find_similar_senteces(sentences, all_sentences):
    global times

    # Separar les dades 
    ids, text_sentences, embeddings = zip(*sentences)
    all_ids, all_text_sentences, all_embeddings = zip(*all_sentences)

    # Passar els embeddings de tuples a numpy array
    embeddings_array = np.array([np.array(embed) for embed in embeddings])
    all_embeddings_array = np.array([np.array(embed) for embed in all_embeddings])

    # Euclidean distance, funcio de la llibreria sklearn.metrics.pairwise
    euclidean_dist_matrix = []
    for i in range(len(embeddings_array)):
        temps = time.time()
        euclidean_dist_matrix.append(euclidean_distances(embeddings_array[i].reshape(1,-1), all_embeddings_array)[0])
        times.append(time.time() - temps)

    print("---- Temps euclidean_distances ----")
    print("Temps mínim: ", min(times))
    print("Temps màxim: ", max(times))
    print("Temps mitjà: ", sum(times)/len(times))
    print("Desviació estàndard: ", statistics.stdev(times))
    print("---------------------------------")

    times = []

    # Cosine Similarity
    cos_sim_matrix = []
    for i in range(len(embeddings_array)):
        temps = time.time()
        cos_sim_matrix.append(cosine_similarity(embeddings_array[i].reshape(1,-1), all_embeddings_array)[0])
        times.append(time.time() - temps)

    print("---- Temps cosine similarity ----")
    print("Temps mínim: ", min(times))
    print("Temps màxim: ", max(times))
    print("Temps mitjà: ", sum(times)/len(times))
    print("Desviació estàndard: ", statistics.stdev(times))
    print("---------------------------------")

    similar_sentences = {}

    for idx, id in enumerate(ids):
        # Obtenir resultats similars
        euclidean_scores = euclidean_dist_matrix[idx]
        cos_sim_scores = cos_sim_matrix[idx]

        # Agafa els indexs del top-2 de les frases mes semblant (sense tenir en compte ella mateixa)
        eucl_top_indices = np.argsort(euclidean_scores)[1:3]  # Ordre ascendent
        cos_top_indices = np.argsort(-cos_sim_scores)[1:3]  # Negative for descending order

        similar_sentences[id] = {
            "sentence": text_sentences[idx],
            "euclidean_distance": [(all_ids[i], all_text_sentences[i], euclidean_scores[i]) for i in eucl_top_indices],
            "cosine_similarity": [(all_ids[i], all_text_sentences[i], cos_sim_scores[i]) for i in cos_top_indices]
        }

    return similar_sentences

if __name__ == '__main__':

    try:
        config = load_config()
        with psycopg2.connect(**config) as conn:

            num_frases = 10

            #Agafa 10 frases
            sentences = fetch_sentences(num_frases, conn)

            all_sentences = fetch_all_sentences(conn)
            
            #Cerca les 2 frases més semblants
            similar_sentences = find_similar_senteces(sentences, all_sentences)

            # Escriu els resultats
            for id, similarities in similar_sentences.items():
                # frase escollida
                print(f"Frase ID: {id}")
                print(f"Frase: \"{similarities['sentence']}\"")
                # frases semblants amb euclidean distance
                print("Top 2 frases semblants (Euclidean Distance):")
                for sim_id, sim_sentence, score in similarities["euclidean_distance"]:
                    print(f" - ID: {sim_id}, Frase: \"{sim_sentence}\", Distancia: {score:.4f}")
                # frases semblants amb cosine similarity
                print("Top 2 frases semblants (Cosine Similarity):")
                for sim_id, sim_sentence, score in similarities["cosine_similarity"]:
                    print(f" - ID: {sim_id}, Frase: \"{sim_sentence}\", Distancia: {score:.4f}")
                print("\n")

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)