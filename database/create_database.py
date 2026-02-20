import sqlite3
import numpy as np

class KenyonDB:
    def __init__(self, db_path="kenyondb.db"):
        self.db_path = db_path
        self.create_tables()

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def create_tables(self):
        sql_statements = [ 
            """CREATE TABLE IF NOT EXISTS mnist_metadata(
                    id INTEGER PRIMARY KEY AUTOINCREMENT, 
                    label INTEGER, 
                    image_index INTEGER
                );""",

            """CREATE TABLE IF NOT EXISTS mnist_vectors (
                    neuron_id INTEGER,
                    image_id INTEGER,
                    FOREIGN KEY(image_id) REFERENCES mnist_metadata(id)
                );""",

            "CREATE INDEX IF NOT EXISTS idx_neuron ON mnist_vectors(neuron_id);"
        ]

        with self.get_connection() as conn:
            cursor = conn.cursor()
            for i in sql_statements:
                cursor.execute(i)
            conn.commit()

    def add_new_record(self, label, original_index, active_ids):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO mnist_metadata (label, image_index) VALUES (?,?)", (int(label), int(original_index))
            )

            image_id = cursor.lastrowid

            vector_data = [(int(nid), image_id) for nid in active_ids]
            cursor.executemany(
                "INSERT INTO mnist_vectors (neuron_id, image_id) VALUES (?,?)",
                vector_data
            )
            conn.commit()

    def populate_from_npz(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        vectors = data['vectors']
        labels = data['labels']

        print(f"Indexing {len(vectors)} vectors...")
        for i in range(len(vectors)):
            self.add_new_record(labels[i], i, vectors[i])
        print("Database population complete")

    def similarity_search(self, query_acive_ids, top_k=5):
        if len(query_acive_ids) == 0:
            return []

        id_list = ",".join(map(str, query_acive_ids))

        query = f"""
            SELECT v.image_id, m.label, COUNT(v.neuron_id) as overlap
            FROM mnist_vectors v
            JOIN mnist_metadata m ON v.image_id = m.id
            WHERE v.neuron_id IN ({id_list})
            GROUP BY v.image_id
            ORDER BY overlap DESC
            LIMIT ?
        """

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (top_k,))
            return cursor.fetchall()


if __name__ == "__main__":
    db = KenyonDB("kenyondb.db")
    #db.populate_from_npz("mnist_latent_vectors.npz")

    results = db.similarity_search([216, 395, 406], top_k=3)
    for res in results:
        print(f"Match: ImageID {res[0]}, Label {res[1]}, Shared Neurons: {res[2]}")