import os
import mysql.connector 
from email_feedback import EmailFeedback
from pymilvus import (
    connections,
    Collection,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType, 
)
import logging 
SURVEILLANCE_COLLECTION="surveillance_faces"

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s [%(levelname)s] %(message)s (%(filename)s:%(lineno)s)',
                    handlers=[logging.FileHandler("app.log"),
                              logging.StreamHandler()])

class Database:
    def __init__(self):
        self.mysql_conn=None
        self.mysql_cursor=None
        self.connection=None

    def initialize_mysql(self): 
        try:
            #  **MySQL Connection for Authorization**
            self.mysql_conn = mysql.connector.connect(
                host=os.getenv("MYSQL_HOST"),
                port=os.getenv("MYSQL_PORT"),
                user=os.getenv("MYSQL_USER"),
                password=os.getenv("MYSQL_PASSWORD"),
                database=os.getenv("MYSQL_DATABASE")
            )
            self.mysql_cursor = self.mysql_conn.cursor()
            self.mysql_cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id VARCHAR(64) PRIMARY KEY,
                    embedding BLOB,
                    nonce BLOB,
                    hmac BLOB,
                    det_score FLOAT,
                    model_version VARCHAR(16),
                    registration_time BIGINT
                )
            """)

            # Recognition logs table
            self.mysql_cursor.execute("""
                CREATE TABLE IF NOT EXISTS authorization_logs (
                    log_id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id VARCHAR(64),
                    similarity FLOAT,
                    det_score FLOAT,
                    model_version VARCHAR(16),
                    timestamp BIGINT,
                    status VARCHAR(15)
                );
            """)

            self.mysql_cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_images (
                    image_id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id VARCHAR(64),
                    image_path VARCHAR(255),
                    det_score FLOAT,
                    registration_time BIGINT
                );
            """)

            self.mysql_conn.commit()
            logging.info("MySQL tables initialized successfully")
            EmailFeedback.compose_email("Success", f"MySQL tables initialized successfully")
            return self.mysql_conn, self.mysql_cursor

        except Exception as e:
            error_msg = f"Unexpected error during DB initialization: {str(e)}"
            logging.critical(error_msg)
            EmailFeedback.compose_email("Error", error_msg)
            raise RuntimeError(error_msg)
    def initialize_milvus(self):
        global collection, logs_collection

        try:
            # Connect to Zilliz Cloud
            connections.connect(
                alias="default",
                uri=os.getenv("ZILLIZ_URI"),
                token=os.getenv("ZILLIZ_TOKEN"),
                secure=True
            )

            # Setup or load the surveillance collection
            if not utility.has_collection(SURVEILLANCE_COLLECTION):
                fields = [
                    FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=64, is_primary=True,),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512),
                    FieldSchema(name="det_score", dtype=DataType.FLOAT),
                    FieldSchema(name="model_version", dtype=DataType.VARCHAR, max_length=16),
                    FieldSchema(name="registration_time", dtype=DataType.INT64)
                ]
                schema = CollectionSchema(fields, description="Surveillance face templates")
                self.collection = Collection(SURVEILLANCE_COLLECTION, schema)

                # Create vector index
                index_params = {
                    "index_type": "HNSW",
                    "metric_type": "IP",
                    "params": {"M": 16, "efConstruction": 50}
                }
                self.collection.create_index(field_name="embedding", index_params=index_params)
                logging.info("INFO: Created new surveillance faces collection")
            else:
                self.collection = Collection(SURVEILLANCE_COLLECTION)
                logging.info("INFO: Loaded existing surveillance faces collection")
            EmailFeedback.compose_email("Success", f"Milvus  initialized successfully")
            self.collection.load()
            return self.collection
        
        except Exception as e:
            error_msg = f"CRITICAL: Database connection failed: {str(e)}"
            logging.critical(error_msg)
            EmailFeedback.compose_email("Error", error_msg)
            raise RuntimeError(f"DB initialization error: {str(e)}") from e

