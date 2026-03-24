import sqlite3
import numpy as np
import torch
import io
from abc import ABC, abstractmethod

class DataStrategy(ABC):
    """Absztrakt osztály az adatbázis felpopulálásához"""
    def __init__(self, db_service):
        self.db_service = db_service

    @abstractmethod
    def extract_data(self, file_path):
        "Adatok kinyerése npz és pt fájlokból"
        pass

    def import_data(self, file_path, model_type):
        data = self.extract_data(file_path)

        vectors = data['vectors']
        labels = data['labels']
        images = data['images']

        print(f"Indexing {len(vectors)} records for {model_type}...")

        with self.db_service.get_connection():
            for i in range(len(vectors)):
                self.add_new_record(labels[i], i, vectors[i], model_type, images[i])

        print(f"{model_type} population complete")

class NpzStrategy(DataStrategy):
    """Npz adatok importálása"""
    def extract_data(self, file_path):
        data = np.load(file_path, allow_pickle=True)
        return data

class PtStrategy(DataStrategy):
    """Pt adatok importálása"""
    def extract_data(self, file_path):
        data = torch.load(pt_path, map_location='cpu', weights_only=False)
        return data

class KenyonDB:
    """Sqlite adatbázis latent vektorok és metadata tárolásásra"""
    def __init__(self, db_path="kenyon.db"):
        self.db_path = db_path
        self.create_tables()

    def get_connection(self):
        """Csatlakozás az adatbázishoz"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;") # Write-Ahead Logging, lehetővé teszi  konkurens olvasást és írást
        conn.execute("PRAGMA synchronous=NORMAL;") # írás sebesség gyorsítása
        conn.execute("PRAGMA temp_store=MEMORY;") # átmeneti táblák és indexek RAM-ban való tárolása
        return conn

    def create_tables(self):
        """Táblák létrehozása"""
        # Táblák: mnist_metadata, mnist_vectors
        # Indexek: idx_neuron, idx_model, idx_image_id, idx_neuron_image
        sql_statements = [ 
            """CREATE TABLE IF NOT EXISTS mnist_metadata(
                    id INTEGER PRIMARY KEY AUTOINCREMENT, 
                    label INTEGER, 
                    image_index INTEGER,
                    model_type TEXT,
                    image_blob BLOB
                );""",
            """CREATE TABLE IF NOT EXISTS mnist_vectors (
                    neuron_id INTEGER,
                    image_id INTEGER,
                    FOREIGN KEY(image_id) REFERENCES mnist_metadata(id)
                );""",
            "CREATE INDEX IF NOT EXISTS idx_neuron ON mnist_vectors(neuron_id);",
            "CREATE INDEX IF NOT EXISTS idx_model ON mnist_metadata(model_type);",
            "CREATE INDEX IF NOT EXISTS idx_image_id ON mnist_vectors(image_id);",
            "CREATE INDEX IF NOT EXISTS idx_neuron_image ON mnist_vectors(neuron_id, image_id);"
        ]
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for sql in sql_statements:
                cursor.execute(sql)
            conn.commit()

    def __serialize_image(self, img_array):
        """Átalakítja a numpy tömböt blob formátumba (Binary Large Object)"""
        if img_array is None: 
            return None
        img_bytes = io.BytesIO() # virtuálisan elmenti RAM-ba a fájlt
        np.save(img_bytes, img_array)
        return img_bytes.getvalue() # kimenti a 0-kat és 1-ket, majd visszatér egy bytes objektummal

    def __deserialize_image(self, blob):
        """Visszaalakítja a byte objektumot numpy tömbbé"""
        if blob is None:
            return None
        img_bytes = io.BytesIO(blob)
        return np.load(img_bytes)

    def add_new_record(self, label, original_index, active_ids, model_type, image_data=None):
        """Új rekord hozzáadása táblához"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            img_blob = self.__serialize_image(image_data) # lehetővé teszi a képek adatbázisban való tárolását
            
            cursor.execute(
                "INSERT INTO mnist_metadata (label, image_index, model_type, image_blob) VALUES (?,?,?,?)", 
                (int(label), int(original_index), str(model_type), img_blob)
            )
            image_id = cursor.lastrowid
            
            vector_data = [(int(nid), image_id) for nid in active_ids]
            cursor.executemany(
                "INSERT INTO mnist_vectors (neuron_id, image_id) VALUES (?,?)",
                vector_data
            )
            conn.commit()

    def clear_database_by_model(self, model_type):
        """Rekordok törlése adott modell típushoz"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                DELETE FROM mnist_vectors
                WHERE image_id IN (
                    SELECT id FROM mnist_metadata WHERE model_type = ?
                )
            """, (model_type,))

            cursor.execute("""
                DELETE FROM mnist_metadata
                WHERE model_type = ?
            """, (model_type,))

            conn.commit()

        print(f"Database cleared for model: {model_type}")


    def populate_from_npz(self, npz_path, model_type):
        """Felpopulálja az adatbázist .npz kiterjesztésű fájlból (mushroom body modell)"""
        data = np.load(npz_path, allow_pickle=True)
        vectors = data['active_kc_ids']
        labels = data['labels']
        images = data['images']

        print(f"Indexing {len(vectors)} vectors from .npz file for model: {model_type}...")

        # Ha mindegyik rekord rendben volt akkor COMMIT, ha bármikor hiba merül fel akkor ROLLBACK
        with self.get_connection():
            for i in range(len(vectors)):
                self.add_new_record(labels[i], i, vectors[i], model_type, images[i])

        print(f"npz population for {model_type} complete")

    def populate_from_pytorch(self, pt_path, model_type):
        """Felpopulálja az adatbázist .pt kiterjesztésű fájlból (echo state modellek)"""
        data = torch.load(pt_path, map_location='cpu', weights_only=False)
        vectors = data['vectors']
        labels = data['labels']
        images = data['images']

        print(f"Indexing {len(vectors)} vectors from .pt file for model: {model_type}...")

        with self.get_connection():
            for i in range(len(vectors)):
                image = images[i].numpy()            
                self.add_new_record(labels[i], i, vectors[i], model_type, image)

        print(f"pt population for {model_type} complete")

        
    def get_image_and_label(self, image_id):
        """Visszaadja az adatbázisban tárolt kép blob objektumát id alapján"""
        query = "SELECT label, image_blob FROM mnist_metadata WHERE id = ?"
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (image_id,))
            result = cursor.fetchone()
            if result:
                label, blob = result[0], result[1]
                return label, self.__deserialize_image(blob)

            return None, None

    
    def similarity_search(self, query_active_ids, model_type, top_k=5):
        """Lefuttatja a lekérdezést és kiszámítja az eredmények és a query közötti hasonlósági metrikákat"""
        if query_active_ids is None or len(query_active_ids) == 0:
            return []

        query_active_ids = sorted({int(x) for x in query_active_ids})
        query_size = len(query_active_ids)

        if query_size == 0:
            return []

        placeholders = ",".join(["?"] * query_size) # SQl-injection ellen és a dinamikusan változó query méret miatt
        params = [model_type] + query_active_ids + [query_size, query_size, query_size, top_k]

        # 1. Potenciális jelöltek kiválogatása akiknek legalább van 1 közös neuronjuk
        # 2. Jelöltek neuron mennyiségének kiszámolása
        # 3. Hasonlósági metrikák kiszámítása
        query = f"""
            WITH candidate_overlap AS (
                SELECT v.image_id, COUNT(*) AS overlap
                FROM mnist_vectors v
                JOIN mnist_metadata m ON v.image_id = m.id
                WHERE m.model_type = ?
                  AND v.neuron_id IN ({placeholders})
                GROUP BY v.image_id
            ),
            candidate_sizes AS (
                SELECT image_id, COUNT(*) AS target_size
                FROM mnist_vectors
                WHERE image_id IN (SELECT image_id FROM candidate_overlap)
                GROUP BY image_id
            )
            SELECT
                co.image_id,
                m.label,
                co.overlap,
                cs.target_size,
                CAST(co.overlap AS FLOAT) / (? + cs.target_size - co.overlap) AS jaccard,
                (2.0 * co.overlap) / (? + cs.target_size) AS dice,
                CAST(co.overlap AS FLOAT) / MIN(? , cs.target_size) AS overlap_coeff
            FROM candidate_overlap co
            JOIN candidate_sizes cs ON co.image_id = cs.image_id
            JOIN mnist_metadata m ON co.image_id = m.id
            ORDER BY jaccard DESC
            LIMIT ?
        """

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            raw_data = cursor.fetchall()

        total = len(raw_data)
        results = []

        for i, row in enumerate(raw_data):
            img_id, label, overlap, target_size, jaccard, dice, overlap_coeff = row

            results.append({
                "image_id": int(img_id),
                "label": int(label),
                "metrics": {
                    "overlap": int(overlap), # közös neuronok száma
                    "jaccard": float(jaccard), # Jaccard-index
                    "dice": float(dice), # Dice-Sorensen együttható
                    "overlap_coeff": float(overlap_coeff), # Overlap együttható
                    "target_size": int(target_size),
                    "query_size": int(query_size),
                }
            })

        return results


    def get_record_by_image_id(self, image_id):
        """Rekord lekérdezése image id alapján"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, label, image_index, model_type, image_blob
                FROM mnist_metadata
                WHERE id = ?
            """, (image_id,))
            row = cursor.fetchone()

        if row is None:
            return None

        return {
            "id": row[0],
            "label": row[1],
            "image_index": row[2],
            "model_type": row[3],
            "image": self.__deserialize_image(row[4]),
        }

    def get_all_image_ids_by_model(self, model_type):
        """Összes image id lekérdezése adott modellhez"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, label
                FROM mnist_metadata
                WHERE model_type = ?
                ORDER BY id
            """, (model_type,))
            rows = cursor.fetchall()

        return [{"image_id": int(r[0]), "label": int(r[1])} for r in rows]


    def get_active_ids_by_image_id(self, image_id):
        """Neuron id-k lekérdezése image id alapján"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT neuron_id
                FROM mnist_vectors
                WHERE image_id = ?
                ORDER BY neuron_id
            """, (image_id,))
            rows = cursor.fetchall()

        return [int(r[0]) for r in rows]


    def get_all_vectors_by_model(self, model_type):
        """Összes vektor lekérdezése adott modellhez"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT m.id, m.label, v.neuron_id
                FROM mnist_metadata m
                JOIN mnist_vectors v ON m.id = v.image_id
                WHERE m.model_type = ?
                ORDER BY m.id, v.neuron_id
            """, (model_type,))
            rows = cursor.fetchall()

        grouped = {}
        for image_id, label, neuron_id in rows:
            image_id = int(image_id)
            if image_id not in grouped:
                grouped[image_id] = {
                    "image_id": image_id,
                    "label": int(label),
                    "active_ids": [],
                }
            grouped[image_id]["active_ids"].append(int(neuron_id))

        return list(grouped.values())

    def get_database_summary(self):
        """Összesítő adatok az adatbázisról"""
        # Összes kép és vektor száma, legkisebb és legnagyobb indexek
        model_order = ["mb", "ws", "ba", "er"]
        summary = {
            "total_images": 0,
            "total_vectors": 0,
            "models": {},
        }

        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM mnist_metadata")
            summary["total_images"] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM mnist_vectors")
            summary["total_vectors"] = cursor.fetchone()[0]

            cursor.execute(
                """
                SELECT m.model_type,
                       COUNT(DISTINCT m.id) AS image_count,
                       COUNT(v.neuron_id) AS vector_count,
                       COUNT(DISTINCT m.label) AS distinct_labels,
                       MIN(m.image_index) AS min_image_index,
                       MAX(m.image_index) AS max_image_index
                FROM mnist_metadata m
                LEFT JOIN mnist_vectors v ON v.image_id = m.id
                GROUP BY m.model_type
                """
            )
            rows = cursor.fetchall()

        row_map = {
            row[0]: {
                "image_count": row[1] or 0,
                "vector_count": row[2] or 0,
                "distinct_labels": row[3] or 0,
                "min_image_index": row[4],
                "max_image_index": row[5],
            }
            for row in rows
        }

        for model_type in model_order:
            summary["models"][model_type] = row_map.get(
                model_type,
                {
                    "image_count": 0,
                    "vector_count": 0,
                    "distinct_labels": 0,
                    "min_image_index": None,
                    "max_image_index": None,
                },
            )

        return summary

