from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import cdist
from chromadb import Client
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import time
import statistics
from datasets import load_dataset

minimum = float('inf')
maximum = float('-inf')
total = 0
times = []

def initialize():
    global minimum, maximum, total, times
    minimum = float('inf')
    maximum = float('-inf')
    total = 0
    times = []

model = SentenceTransformer('all-MiniLM-L6-v2') 

# Funcio per conectar a Chroma
def connect_to_chroma():
    try:
        client = Client(Settings())
        print('Connected to the Chroma database.')
        return client
    
    except Exception as error:
        print(f"Error connecting to Chroma: {error}")
        return None

# Crea la collection si no existeix 
def create_collection(client, collection_name_euclidean, collection_name_cosinus):
    try:
        # necesari crear 2 collection diferents ja que es pot especificar
        client.get_or_create_collection(
            name = collection_name_euclidean,
            )
        client.get_or_create_collection(
            name = collection_name_cosinus,
            metadata={"hnsw:space": "cosine"}
            )
        print("Collection created successfully.")

    except Exception as e:
        print(f"Error creating collection: {e}")

# Inserta les frases a la collection
def insert_sentences(sentences, client, collection_name_euclidean, collection_name_cosinus):

    try:
        collection = client.get_collection(collection_name_euclidean)
        ids = [str(i) for i in range(len(sentences))]
        text = [{'text': sentence} for sentence in sentences]
        embedding = model.encode(sentences).tolist()

        # Afegeix data a la collection
        for i in range(len(sentences)):
            collection.add(ids=[ids[i]], metadatas=[text[i]], embeddings=[embedding[i]])

        collection = client.get_collection(collection_name_cosinus)
        ids = [str(i) for i in range(len(sentences))]
        text = [{'text': sentence} for sentence in sentences]
        embedding = model.encode(sentences).tolist()

        # Afegeix data a la collection
        for i in range(len(sentences)):
            collection.add(ids=[ids[i]], metadatas=[text[i]], embeddings=[embedding[i]])

        print("Sentences inserted successfully.")

    except Exception as e:
        print(f"Error inserting sentences: {e}")

# Agafa 10 frases
def fetch_sentences(client, collection_name):
    # Assuming you have a collection named 'book_sentences' in ChromaDB
    collection = client.get_collection(collection_name)

    result = collection.get(
        ids = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        include =  ["metadatas"]
    )
    
    return result

# Busca les dues frases més semblants
def find_similar_sentences(sentences, collection_name):
    collection = client.get_collection(collection_name)

    calculs = {}

    for metadata in sentences["metadatas"]:
        sentence = metadata['text']

        temps = time.time()
        results = collection.query(
            query_texts=[sentence], 
            n_results=3 
        )
        temps = time.time() - temps
        global minimum, maximum, total
        if temps < minimum:
                minimum = temps
        if temps > maximum:
                maximum = temps
        total += temps
        times.append(temps)

        calculs[sentence] = results

    return calculs

if __name__ == '__main__':
    try:
        # Conecta a Chroma
        client = connect_to_chroma()

        collection_name_euclidean = 'bookcorpus_euc'
        collection_name_cosinus = 'bookcorpus_cos'

        # Crea la collection
        create_collection(client, collection_name_euclidean, collection_name_cosinus)

        ds = load_dataset("williamkgao/bookcorpus100mb", split ="train")
        subset = ds.select(range(10000))
        sentences = [item['text'] for item in subset]
        for s in sentences:
            if s == "`` thank you for that . ''":
                print(sentences)
    
        # Actualitza les frases de la collection
        if client:
            insert_sentences(sentences, client, collection_name_euclidean, collection_name_cosinus)
        
        print()

        # Agafa 10 frases
        sentences_euc = fetch_sentences(client, collection_name_euclidean)
        sentences_cos = fetch_sentences(client, collection_name_cosinus)

        # Cerca les 2 frases més semblants
        similar_sentences_euc = find_similar_sentences(sentences_euc, collection_name_euclidean)

        # Escriu els resultats de euclidean
        for sentence, similarities in similar_sentences_euc.items():
            # frase escollida
            print(f"Frase original: \"{sentence}\"")
            # frases semblants amb euclidean distance
            print("Top 2 frases semblants (Euclidean Distance):")

            ids = similarities['ids'][0][1:3] 
            distances = similarities['distances'][0][1:3]  
            metadatas = similarities['metadatas'][0][1:3] 

            for sim_id, distance, metadata in zip(ids, distances, metadatas):
                sim_sentence = metadata['text']  # Extract the sentence from the metadata
                print(f" - ID: {sim_id}, Frase: \"{sim_sentence}\", Distancia: {distance:.4f}")
            
            print()

        print("---------------------------------")
        print("Temps mínim: ", minimum)
        print("Temps màxim: ", maximum)
        print("Temps mitjà: ", total/10000)
        print("Desviació estàndard: ", statistics.stdev(times))
        print("---------------------------------")


        initialize()
        print("Total per veure que s'ha reiniciat", total)
        similar_sentences_cos = find_similar_sentences(sentences_cos, collection_name_cosinus)

        for sentence, similarities in similar_sentences_cos.items():
            # frase escollida
            print(f"Frase original: \"{sentence}\"")
            # frases semblants amb euclidean distance
            print("Top 2 frases semblants (Cosinus Similarity):")

            ids = similarities['ids'][0][1:3] 
            distances = similarities['distances'][0][1:3]  
            metadatas = similarities['metadatas'][0][1:3] 

            for sim_id, distance, metadata in zip(ids, distances, metadatas):
                sim_sentence = metadata['text']  # Extract the sentence from the metadata
                print(f" - ID: {sim_id}, Frase: \"{sim_sentence}\", Distancia: {distance:.4f}")
            
            print()

        print("---------------------------------")
        print("Temps mínim: ", minimum)
        print("Temps màxim: ", maximum)
        print("Temps mitjà: ", total/10000)
        print("Desviació estàndard: ", statistics.stdev(times))
        print("---------------------------------")


    except Exception as error:
        print(error)