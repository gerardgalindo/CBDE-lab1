import chromadb
from chromadb import Client
from chromadb import Settings
import numpy as np
import time
import statistics
from datasets import load_dataset

minimum = float('inf')
maximum = float('-inf')
total = 0
times = []

# Funcio per conectar a Chroma
def connect_to_chroma():
    try:
        client = Client(Settings())
        print('Connected to the Chroma database.')
        return client
    
    except Exception as error:
        print(f"Error connecting to Chroma: {error}")
        return None


# Inserta les frases a la collection
def insert_sentences(client, collection_name, sentences):
    collection = client.get_or_create_collection(collection_name)
    ids = [str(i) for i in range(len(sentences))]
        
    # Afegeix data a la collection
    for i in range(len(sentences)):
        temps = time.time()
        collection.upsert(ids=[ids[i]], documents=[sentences[i]])
        temps = time.time() - temps
        global minimum, maximum, total
        if temps < minimum:
            minimum = temps
        if temps > maximum:
            maximum = temps
        total += temps
        times.append(temps)

        # Mostra el progrés a la consola
        if i % 100 == 0:  # Per mostrar el progrés cada 100 insercions
            print(f"{i} sentences inserted so far...")

    print("Sentences inserted successfully.")



if __name__ == '__main__':

    # Conecta a Chroma
    client = connect_to_chroma()
    
    collection_name = 'bookcorpus'

    ds = load_dataset("williamkgao/bookcorpus100mb", split ="train")
    subset = ds.select(range(10000))
    sentences = [item['text'] for item in subset]

    # Inserta les frases a la collection
    if client:
        insert_sentences(client, collection_name, sentences)
        print("Temps mínim: ", minimum)
        print("Temps màxim: ", maximum)
        print("Temps mitjà: ", total/10000)
        print("Desviació estàndard: ", statistics.stdev(times))
