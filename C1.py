from chromadb import Client
from chromadb import Settings
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import time
import statistics

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
def create_collection(client, collection_name):
    try:
        client.get_or_create_collection(name = collection_name)
        print("Collection created successfully.")

    except Exception as e:
        print(f"Error creating collection: {e}")


# Inserta les frases a la collection
def insert_sentences(sentences, client, collection_name):
    try:
        collection = client.get_collection(collection_name)
        ids = [str(i) for i in range(len(sentences))]

        text = [{'text': sentence} for sentence in sentences]
        embedding = model.encode(sentences).tolist()

        # Afegeix data a la collection
        for i in range(len(sentences)):
            temps = time.time()
            collection.add(ids=[ids[i]], metadatas=[text[i]], embeddings=[embedding[i]])
            temps = time.time() - temps
            global minimum, maximum, total
            if temps < minimum:
                    minimum = temps
            if temps > maximum:
                    maximum = temps
            total += temps
            times.append(temps)

        print("Sentences inserted successfully.")

    except Exception as e:
        print(f"Error inserting sentences: {e}")

if __name__ == '__main__':

    # Conecta a Chroma
    client = connect_to_chroma()

    collection_name = 'bookcorpus_emb'

    # Crea la collection
    create_collection(client, collection_name)

    ds = load_dataset("williamkgao/bookcorpus100mb", split ="train")
    subset = ds.select(range(10000))
    sentences = [item['text'] for item in subset]
 
    # Actualitza les frases de la collection
    if client:
        insert_sentences(sentences, client, collection_name)
        print("Temps mínim: ", minimum)
        print("Temps màxim: ", maximum)
        print("Temps mitjà: ", total/10000)
        print("Desviació estàndard: ", statistics.stdev(times))