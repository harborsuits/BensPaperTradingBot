"""
Mock pymongo module for testing without MongoDB dependencies
"""

class MongoClient:
    def __init__(self, host=None, port=None, **kwargs):
        self.host = host or 'localhost'
        self.port = port or 27017
        self.options = kwargs
        
    def __getitem__(self, db_name):
        return Database(self, db_name)
    
    def get_database(self, name):
        return Database(self, name)
        
    def close(self):
        pass
        
class Database:
    def __init__(self, client, name):
        self.client = client
        self.name = name
        
    def __getitem__(self, collection_name):
        return Collection(self, collection_name)
    
    def get_collection(self, name):
        return Collection(self, name)
        
class Collection:
    def __init__(self, database, name):
        self.database = database
        self.name = name
        self.documents = []
        
    def insert_one(self, document):
        self.documents.append(document)
        return InsertOneResult(1)
        
    def insert_many(self, documents):
        self.documents.extend(documents)
        return InsertManyResult(len(documents))
    
    def find(self, query=None, projection=None):
        return Cursor(self.documents)
    
    def find_one(self, query=None, projection=None):
        if not self.documents:
            return None
        return self.documents[0]
    
    def update_one(self, query, update, upsert=False):
        return UpdateResult(1, 1)
    
    def update_many(self, query, update, upsert=False):
        return UpdateResult(len(self.documents), len(self.documents))
    
    def delete_one(self, query):
        return DeleteResult(1)
    
    def delete_many(self, query):
        count = len(self.documents)
        self.documents = []
        return DeleteResult(count)
    
    def count_documents(self, query=None):
        return len(self.documents)
    
class Cursor:
    def __init__(self, documents):
        self.documents = documents
        self.position = 0
        
    def __iter__(self):
        return self
        
    def __next__(self):
        if self.position >= len(self.documents):
            raise StopIteration
        document = self.documents[self.position]
        self.position += 1
        return document
        
    def limit(self, limit):
        return self
        
    def sort(self, key_or_list, direction=None):
        return self
        
    def skip(self, skip):
        return self
        
    def next(self):
        return self.__next__()
        
class InsertOneResult:
    def __init__(self, inserted_id):
        self.inserted_id = inserted_id
        
class InsertManyResult:
    def __init__(self, inserted_ids):
        self.inserted_ids = list(range(inserted_ids))
        
class UpdateResult:
    def __init__(self, matched_count, modified_count):
        self.matched_count = matched_count
        self.modified_count = modified_count
        
class DeleteResult:
    def __init__(self, deleted_count):
        self.deleted_count = deleted_count

# Special constants
ASCENDING = 1
DESCENDING = -1
