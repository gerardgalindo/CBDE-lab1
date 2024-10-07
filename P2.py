import psycopg2
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from config import load_config
from scipy.spatial.distance import cdist

# Agafa 10 frases
def fetch_sentences(limit, conn):
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, sentence, embedding FROM book_sentences LIMIT %s;", (limit,))
    sentences = cursor.fetchall()

    cursor.close()
    
    return sentences

# Busca les dues frases mes semblants
def find_similar_senteces(sentences):

    # Separar les dades 
    ids, text_sentences, embeddings = zip(*sentences)

    # Passar els embeddings de tuples a numpy array
    embeddings_array = np.array([np.array(embed) for embed in embeddings])

    # Euclidean distancia
    euclidean_dist_matrix = euclidean_distances(embeddings_array)

    # Manhatann distancia 
    man_dist_matrix = cdist(embeddings_array, embeddings_array, metric='cityblock')

    similar_sentences = {}

    for idx, id in enumerate(ids):
        # Obtenir resultats similars
        euclidean_scores = euclidean_dist_matrix[idx]
        manhattan_scores = man_dist_matrix[idx]

        # Agafa els indexs del top-2 de les frases mes semblant (sense tenir en compte ella mateixa)
        eucl_top_indices = np.argsort(euclidean_scores)[1:3]  # Ordre ascendent
        manh_top_indices = np.argsort(manhattan_scores)[1:3]  # Ordre ascendent

        similar_sentences[id] = {
            "sentence": text_sentences[idx],
            "euclidean_distance": [(ids[i], text_sentences[i], euclidean_scores[i]) for i in eucl_top_indices],
            "manhattan_distance": [(ids[i], text_sentences[i], manhattan_scores[i]) for i in manh_top_indices],
        }

    return similar_sentences

if __name__ == '__main__':

    try:
        config = load_config()
        with psycopg2.connect(**config) as conn:

            num_frases = 10

            #Agafa 10 frases
            sentences = fetch_sentences(num_frases, conn)
            
            #Cerca les 2 frases més semblants
            similar_sentences = find_similar_senteces(sentences)

            # Escriu els resultats
            for id, similarities in similar_sentences.items():
                # frase escollida
                print(f"Frase ID: {id}")
                print(f"Frase: \"{similarities['sentence']}\"")
                # frases semblants amb euclidean distance
                print("Top 2 frases semblants (Euclidean Distance):")
                for sim_id, sim_sentence, score in similarities["euclidean_distance"]:
                    print(f" - ID: {sim_id}, Frase: \"{sim_sentence}\", Score: {score:.4f}")
                # frases semblants amb manhattan similarity
                print("Top 2 frases semblants (Manhattan Similarity):")
                for sim_id, sim_sentence, score in similarities["manhattan_distance"]:
                    print(f" - ID: {sim_id}, Frase: \"{sim_sentence}\", Score: {score:.4f}")
                print("\n")

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)