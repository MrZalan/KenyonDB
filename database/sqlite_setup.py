import sqlite3

db_name = "kenyondb.db"

sql_statements = [ 
    """CREATE TABLE IF NOT EXISTS mnist_metadata(
            id INTEGER PRIMARY KEY, 
            label INTEGER, 
            npz_filename TEXT
        );""",

    """CREATE TABLE IF NOT EXISTS mnist_vectors (
            neuron_id INTEGER,
            image_id INTEGER,
            FOREIGN KEY(image_id) REFERENCES mnist_metadata(id)
        );""",

    """CREATE INDEX IF NOT EXISTS idx_neuron ON mnist_vectors(neuron_id);
    """
]

# create a database connection
try:
    with sqlite3.connect(db_name) as conn:
        # create a cursor
        cursor = conn.cursor()

        # execute statements
        for statement in sql_statements:
            cursor.execute(statement)

        # commit the changes
        conn.commit()

        print("Tables created successfully.")
except sqlite3.OperationalError as e:
    print("Failed to create tables:", e)
