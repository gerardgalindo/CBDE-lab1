from datasets import load_dataset
import chromadb
from chromadb import Settings
import numpy as np
import time
import statistics

minimum = float('inf')
maximum = float('-inf')
total = 0
times = []

# Funcio per conectar a Chroma
def connect_to_chroma():
    try:
        client = chromadb.Client(Settings())
        print('Connected to the Chroma database.')
        return client
    
    except Exception as error:
        print(f"Error connecting to Chroma: {error}")
        return None
    
# Crea la collection si no existeix 
def create_collection(client, collection_name):
    try:
        client.create_collection(name = collection_name)
        print("Collection created successfully.")

    except Exception as e:
        print(f"Error creating collection: {e}")

# Inserta les frases a la collection
def insert_sentences(sentences, client, collection_name):
    try:
        collection = client.get_collection(collection_name)
        ids = [str(i) for i in range(len(sentences))]

        text = [{'text': sentence} for sentence in sentences]
        zero_embeddings = np.zeros((len(sentences), 128)).tolist()
        
        # Afegeix data a la collection
        for i in range(len(sentences)):
            try:
                # Captura el temps d'inserció
                temps = time.time()
                collection.add(ids=[ids[i]], metadatas=[text[i]], embeddings=[zero_embeddings[i]])
                temps = time.time() - temps
                
                # Actualitza els valors de temps mínim, màxim, total
                global minimum, maximum, total
                if temps < minimum:
                    minimum = temps
                if temps > maximum:
                    maximum = temps
                total += temps
                times.append(temps)
            
            except Exception as e:
                print(f"Error inserting sentence {i}: {e}")

        
        print("Sentences inserted successfully.")

    except Exception as e:
        print(f"Error inserting sentences: {e}")

if __name__ == '__main__':

    # Conecta a Chroma
    client = connect_to_chroma()

    collection_name = 'bookcorpus'

    # Crea la collection
    create_collection(client, collection_name)

    ds = load_dataset("williamkgao/bookcorpus100mb", split ="train")
    subset = ds.select(range(10000))
    sentences = [item['text'] for item in subset]
 
    # Inserta les frases a la collection
    if client:
        insert_sentences(sentences, client, collection_name)
        print("Temps mínim: ", minimum)
        print("Temps màxim: ", maximum)
        print("Temps mitjà: ", total/10000)
        print("Desviació estàndard: ", statistics.stdev(times))