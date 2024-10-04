import psycopg2
from config import load_config

# Agafa 10 frases
def fetch_sentences(limit, conn):
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, text, embedding FROM book_sentences LIMIT %s;", (limit,))
    sentences = cursor.fetchall()
    
    cursor.close()
    
    return sentences

# Busca les dues frases mes semblants
def find_similar_senteces(sentences):

    similar_sentences = {}

    return similar_sentences

if __name__ == '__main__':

    try:
        config = load_config()
        with psycopg2.connect(**config) as conn:

            num_frases = 10

            sentences = fetch_sentences(num_frases, conn)

            find_similar_senteces(sentences)

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
