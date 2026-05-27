"""
Client MongoDB pentru baza de date BAC Romania
"""
import logging
from datetime import datetime
from typing import Optional, Dict, List, Any

try:
    import pymongo
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "bac_romania"
COLLECTION_ELEVI = "elevi"
COLLECTION_PREDICTII = "predictii"
COLLECTION_MODELE = "modele"


class MongoDBClient:
    def __init__(self, uri: str = MONGO_URI, db_name: str = DB_NAME):
        self.uri = uri
        self.db_name = db_name
        self.client = None
        self.db = None
        self._connected = False

    def connect(self) -> bool:
        if not PYMONGO_AVAILABLE:
            logger.warning("pymongo nu este instalat. Folosind fallback CSV.")
            return False
        try:
            self.client = MongoClient(self.uri, serverSelectionTimeoutMS=3000)
            self.client.admin.command("ping")
            self.db = self.client[self.db_name]
            self._connected = True
            logger.info(f"Conectat la MongoDB: {self.uri}, DB: {self.db_name}")
            return True
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.warning(f"Nu s-a putut conecta la MongoDB: {e}")
            self._connected = False
            return False
        except Exception as e:
            logger.error(f"Eroare neasteptata la conectare MongoDB: {e}")
            self._connected = False
            return False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def insert_data(self, df: pd.DataFrame) -> bool:
        if not self._connected or self.db is None:
            logger.warning("Nu esti conectat la MongoDB.")
            return False
        try:
            collection = self.db[COLLECTION_ELEVI]
            records = df.to_dict(orient="records")
            # Sterge date existente si reinserteaza
            collection.drop()
            collection.insert_many(records)
            logger.info(f"Inserate {len(records)} inregistrari in {COLLECTION_ELEVI}")
            return True
        except Exception as e:
            logger.error(f"Eroare la insertia datelor: {e}")
            return False

    def get_all_data(self) -> Optional[pd.DataFrame]:
        if not self._connected or self.db is None:
            return None
        try:
            collection = self.db[COLLECTION_ELEVI]
            cursor = collection.find({}, {"_id": 0})
            data = list(cursor)
            if not data:
                return None
            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Eroare la preluarea datelor: {e}")
            return None

    def get_by_year(self, year: int) -> Optional[pd.DataFrame]:
        if not self._connected or self.db is None:
            return None
        try:
            collection = self.db[COLLECTION_ELEVI]
            cursor = collection.find({"an": year}, {"_id": 0})
            data = list(cursor)
            if not data:
                return None
            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Eroare la preluarea datelor pentru anul {year}: {e}")
            return None

    def get_by_county(self, judet: str) -> Optional[pd.DataFrame]:
        if not self._connected or self.db is None:
            return None
        try:
            collection = self.db[COLLECTION_ELEVI]
            cursor = collection.find({"judet": judet}, {"_id": 0})
            data = list(cursor)
            if not data:
                return None
            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Eroare la preluarea datelor pentru judetul {judet}: {e}")
            return None

    def get_statistics(self) -> Optional[Dict[str, Any]]:
        if not self._connected or self.db is None:
            return None
        try:
            collection = self.db[COLLECTION_ELEVI]
            total = collection.count_documents({})
            promovati = collection.count_documents({"promovat": 1})
            absenti = collection.count_documents({"absent": 1})

            pipeline_medie = [
                {"$group": {"_id": None, "medie": {"$avg": "$medie_generala"}}}
            ]
            medie_result = list(collection.aggregate(pipeline_medie))
            medie = medie_result[0]["medie"] if medie_result else 0

            stats = {
                "total_elevi": total,
                "total_promovati": promovati,
                "rata_promovare": promovati / total if total > 0 else 0,
                "total_absenti": absenti,
                "medie_generala": round(medie, 2)
            }
            return stats
        except Exception as e:
            logger.error(f"Eroare la calculul statisticilor: {e}")
            return None

    def insert_prediction(self, pred_data: Dict[str, Any]) -> bool:
        if not self._connected or self.db is None:
            return False
        try:
            collection = self.db[COLLECTION_PREDICTII]
            pred_data["timestamp"] = datetime.utcnow().isoformat()
            collection.insert_one(pred_data)
            logger.info(f"Predictie inserata: {pred_data}")
            return True
        except Exception as e:
            logger.error(f"Eroare la insertia predictiei: {e}")
            return False

    def get_predictions(self) -> Optional[List[Dict]]:
        if not self._connected or self.db is None:
            return None
        try:
            collection = self.db[COLLECTION_PREDICTII]
            cursor = collection.find({}, {"_id": 0}).sort("timestamp", -1).limit(100)
            return list(cursor)
        except Exception as e:
            logger.error(f"Eroare la preluarea predictiilor: {e}")
            return None

    def save_model_metadata(self, metadata: Dict[str, Any]) -> bool:
        if not self._connected or self.db is None:
            return False
        try:
            collection = self.db[COLLECTION_MODELE]
            metadata["timestamp"] = datetime.utcnow().isoformat()
            collection.insert_one(metadata)
            return True
        except Exception as e:
            logger.error(f"Eroare la salvarea metadatelor modelului: {e}")
            return False

    def close(self):
        if self.client:
            self.client.close()
            self._connected = False
            logger.info("Conexiune MongoDB inchisa.")


_client_instance: Optional[MongoDBClient] = None


def get_client() -> MongoDBClient:
    global _client_instance
    if _client_instance is None:
        _client_instance = MongoDBClient()
        _client_instance.connect()
    return _client_instance
