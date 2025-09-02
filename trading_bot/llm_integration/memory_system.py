"""
Hierarchical Memory System for the Trading Bot

This module implements a sophisticated three-layer memory system:
- Short-term memory: Recent price action, technical indicators, and market data
- Medium-term memory: Patterns, regime changes, and successful trading decisions
- Long-term memory: Fundamental knowledge about markets, companies, and macroeconomics

Features:
- Importance scoring
- Memory decay over time
- Vector embeddings for semantic search
- Persistence and retrieval mechanisms
"""

import os
import json
import time
import logging
import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
import sqlite3
import numpy as np
from collections import defaultdict

# For vector embeddings
try:
    import torch
    from sentence_transformers import SentenceTransformer
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False

# Initialize logger
logger = logging.getLogger("memory_system")

class MemoryType(Enum):
    """Types of memory in the system"""
    SHORT_TERM = "short_term"  # Hours to days (price action, indicators)
    MEDIUM_TERM = "medium_term"  # Days to weeks (patterns, regimes)
    LONG_TERM = "long_term"  # Permanent (fundamental knowledge)
    
@dataclass
class MemoryItem:
    """Individual memory item with metadata"""
    content: str
    memory_type: MemoryType
    importance: float  # 0.0 to 1.0
    created_at: float  # Unix timestamp
    last_accessed: float  # Unix timestamp
    access_count: int
    tags: List[str]
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    memory_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        result["memory_type"] = self.memory_type.value
        # Convert numpy arrays to lists if needed
        if isinstance(self.embedding, np.ndarray):
            result["embedding"] = self.embedding.tolist()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryItem':
        """Create from dictionary"""
        data = data.copy()
        data["memory_type"] = MemoryType(data["memory_type"])
        return cls(**data)
    
    def calculate_relevance(self, current_time: float, decay_factor: float = 0.01) -> float:
        """
        Calculate memory relevance score based on importance, recency and access frequency
        
        Args:
            current_time: Current Unix timestamp
            decay_factor: Controls how quickly memory decays with time
            
        Returns:
            Relevance score from 0.0 to 1.0
        """
        # Recency component: exponential decay based on time since creation
        time_factor = np.exp(-decay_factor * (current_time - self.created_at) / 86400)  # Days
        
        # Access frequency component: memories accessed more often are more relevant
        access_factor = np.log1p(self.access_count) / 10.0  # Log to dampen effect
        access_factor = min(access_factor, 0.5)  # Cap at 0.5
        
        # Combine factors with importance as the base
        relevance = self.importance * 0.5 + time_factor * 0.3 + access_factor * 0.2
        return float(min(relevance, 1.0))
        
    def update_access(self):
        """Update access timestamp and count"""
        self.last_accessed = time.time()
        self.access_count += 1
        
    def __repr__(self) -> str:
        return (f"MemoryItem(id={self.memory_id}, type={self.memory_type.value}, "
                f"importance={self.importance:.2f}, created={datetime.datetime.fromtimestamp(self.created_at)})")

class MemorySystem:
    """
    Hierarchical memory system for the trading bot
    
    Provides:
    - Memory storage and retrieval by type and tags
    - Semantic search using embeddings
    - Importance-based forgetting mechanism
    - Memory consolidation from short-term to long-term
    """
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        enable_embeddings: bool = True,
        short_term_limit: int = 1000,
        medium_term_limit: int = 500,
        long_term_limit: int = 5000,
        min_importance_threshold: float = 0.3,
        debug: bool = False
    ):
        """
        Initialize the memory system
        
        Args:
            db_path: Path to SQLite database for persistence
            embedding_model: Model to use for text embeddings
            enable_embeddings: Whether to use vector embeddings
            short_term_limit: Maximum items in short-term memory
            medium_term_limit: Maximum items in medium-term memory  
            long_term_limit: Maximum items in long-term memory
            min_importance_threshold: Minimum importance to keep a memory
            debug: Enable debug logging
        """
        # Set up logging
        logging_level = logging.DEBUG if debug else logging.INFO
        logging.basicConfig(level=logging_level)
        self.debug = debug
        
        # Memory configuration
        self.short_term_limit = short_term_limit
        self.medium_term_limit = medium_term_limit
        self.long_term_limit = long_term_limit
        self.min_importance_threshold = min_importance_threshold
        
        # Initialize memory stores
        self.memories = {
            MemoryType.SHORT_TERM: {},
            MemoryType.MEDIUM_TERM: {},
            MemoryType.LONG_TERM: {}
        }
        
        # Initialize embedding model if available
        self.enable_embeddings = enable_embeddings and HAS_EMBEDDINGS
        self.embedding_model = None
        if self.enable_embeddings:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
                logger.info(f"Loaded embedding model: {embedding_model}")
            except Exception as e:
                logger.error(f"Error loading embedding model: {e}")
                self.enable_embeddings = False
        
        # Set up database for persistence
        self.db_path = db_path or os.path.join(os.path.dirname(__file__), "memory.db")
        self._init_db()
        
        # Load existing memories from database
        self._load_memories()
        
        logger.info(f"Memory system initialized with {self.count_memories()} total memories")
        
    def _init_db(self):
        """Initialize the SQLite database"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            
            # Create tables if they don't exist
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT UNIQUE,
                content TEXT,
                memory_type TEXT,
                importance REAL,
                created_at REAL,
                last_accessed REAL,
                access_count INTEGER,
                source TEXT,
                metadata TEXT
            )
            ''')
            
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS memory_tags (
                memory_id TEXT,
                tag TEXT,
                PRIMARY KEY (memory_id, tag),
                FOREIGN KEY (memory_id) REFERENCES memories(memory_id) ON DELETE CASCADE
            )
            ''')
            
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS memory_embeddings (
                memory_id TEXT PRIMARY KEY,
                embedding BLOB,
                FOREIGN KEY (memory_id) REFERENCES memories(memory_id) ON DELETE CASCADE
            )
            ''')
            
            self.conn.commit()
            logger.info(f"Initialized database at {self.db_path}")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
            
    def _load_memories(self):
        """Load memories from the database"""
        try:
            # Get all memories
            self.cursor.execute("SELECT * FROM memories")
            rows = self.cursor.fetchall()
            
            for row in rows:
                # Extract memory data
                memory_id = row[1]
                content = row[2]
                memory_type = MemoryType(row[3])
                importance = row[4]
                created_at = row[5]
                last_accessed = row[6]
                access_count = row[7]
                source = row[8]
                metadata = json.loads(row[9])
                
                # Get tags for this memory
                self.cursor.execute("SELECT tag FROM memory_tags WHERE memory_id = ?", (memory_id,))
                tags = [tag[0] for tag in self.cursor.fetchall()]
                
                # Get embeddings if available
                embedding = None
                if self.enable_embeddings:
                    self.cursor.execute("SELECT embedding FROM memory_embeddings WHERE memory_id = ?", (memory_id,))
                    result = self.cursor.fetchone()
                    if result:
                        embedding = np.frombuffer(result[0], dtype=np.float32)
                
                # Create memory item
                memory = MemoryItem(
                    content=content,
                    memory_type=memory_type,
                    importance=importance,
                    created_at=created_at,
                    last_accessed=last_accessed,
                    access_count=access_count,
                    tags=tags,
                    source=source,
                    metadata=metadata,
                    embedding=embedding,
                    memory_id=memory_id
                )
                
                # Add to in-memory store
                self.memories[memory_type][memory_id] = memory
                
            logger.info(f"Loaded {len(rows)} memories from database")
        except Exception as e:
            logger.error(f"Error loading memories: {e}")
            raise
            
    def _generate_memory_id(self, memory: MemoryItem) -> str:
        """Generate a unique ID for a memory item"""
        import hashlib
        hash_input = f"{memory.content}-{memory.memory_type.value}-{memory.created_at}"
        return hashlib.md5(hash_input.encode()).hexdigest()
            
    def _save_memory(self, memory: MemoryItem):
        """Save a memory to the database"""
        try:
            # Generate memory ID if not present
            if not memory.memory_id:
                memory.memory_id = self._generate_memory_id(memory)
            
            # Insert or update memory
            self.cursor.execute('''
            INSERT OR REPLACE INTO memories 
            (memory_id, content, memory_type, importance, created_at, last_accessed, 
             access_count, source, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                memory.memory_id,
                memory.content,
                memory.memory_type.value,
                memory.importance,
                memory.created_at,
                memory.last_accessed,
                memory.access_count,
                memory.source,
                json.dumps(memory.metadata)
            ))
            
            # Delete existing tags and add new ones
            self.cursor.execute("DELETE FROM memory_tags WHERE memory_id = ?", (memory.memory_id,))
            for tag in memory.tags:
                self.cursor.execute("INSERT INTO memory_tags (memory_id, tag) VALUES (?, ?)",
                                  (memory.memory_id, tag))
            
            # Save embedding if available
            if memory.embedding is not None:
                if isinstance(memory.embedding, list):
                    embedding_bytes = np.array(memory.embedding, dtype=np.float32).tobytes()
                else:
                    embedding_bytes = memory.embedding.tobytes()
                    
                self.cursor.execute('''
                INSERT OR REPLACE INTO memory_embeddings (memory_id, embedding)
                VALUES (?, ?)
                ''', (memory.memory_id, embedding_bytes))
            
            self.conn.commit()
            
            if self.debug:
                logger.debug(f"Saved memory {memory.memory_id} to database")
        except Exception as e:
            logger.error(f"Error saving memory: {e}")
            self.conn.rollback()
            raise
            
    def _delete_memory(self, memory_id: str):
        """Delete a memory from the database"""
        try:
            self.cursor.execute("DELETE FROM memories WHERE memory_id = ?", (memory_id,))
            self.cursor.execute("DELETE FROM memory_tags WHERE memory_id = ?", (memory_id,))
            self.cursor.execute("DELETE FROM memory_embeddings WHERE memory_id = ?", (memory_id,))
            self.conn.commit()
            
            if self.debug:
                logger.debug(f"Deleted memory {memory_id} from database")
        except Exception as e:
            logger.error(f"Error deleting memory: {e}")
            self.conn.rollback()
            raise
            
    def _create_embedding(self, text: str) -> Optional[np.ndarray]:
        """Create a vector embedding for text"""
        if not self.enable_embeddings or not self.embedding_model:
            return None
            
        try:
            return self.embedding_model.encode(text, show_progress_bar=False)
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            return None
    
    def add_memory(
        self,
        content: str,
        memory_type: MemoryType,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        source: str = "system",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a new memory to the system
        
        Args:
            content: The text content of the memory
            memory_type: Type of memory (short/medium/long term)
            importance: Importance score from 0.0 to 1.0
            tags: List of tags for categorization
            source: Source of the memory (e.g., "user", "market", "news")
            metadata: Additional structured data
            
        Returns:
            memory_id: ID of the created memory
        """
        # Validate inputs
        if not content or not content.strip():
            raise ValueError("Memory content cannot be empty")
            
        importance = max(0.0, min(1.0, importance))
        tags = tags or []
        metadata = metadata or {}
        
        # Create embedding if enabled
        embedding = self._create_embedding(content) if self.enable_embeddings else None
        
        # Create memory item
        memory = MemoryItem(
            content=content,
            memory_type=memory_type,
            importance=importance,
            created_at=time.time(),
            last_accessed=time.time(),
            access_count=1,
            tags=tags,
            source=source,
            metadata=metadata,
            embedding=embedding
        )
        
        # Generate a unique ID
        memory.memory_id = self._generate_memory_id(memory)
        
        # Add to in-memory store and database
        self.memories[memory_type][memory.memory_id] = memory
        self._save_memory(memory)
        
        # Check memory limits and possibly forget least important memories
        self._enforce_memory_limits()
        
        logger.info(f"Added {memory_type.value} memory: {memory.memory_id} (importance: {importance:.2f})")
        return memory.memory_id
        
    def update_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        importance: Optional[float] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update an existing memory
        
        Args:
            memory_id: ID of the memory to update
            content: New content (if changing)
            importance: New importance score (if changing)
            tags: New tags (if changing)
            metadata: New or updated metadata (merged with existing)
            
        Returns:
            bool: True if update successful
        """
        # Find the memory
        memory = None
        memory_type = None
        for mem_type, memories in self.memories.items():
            if memory_id in memories:
                memory = memories[memory_id]
                memory_type = mem_type
                break
                
        if not memory:
            logger.warning(f"Memory {memory_id} not found for update")
            return False
            
        # Update fields
        if content is not None:
            memory.content = content
            # Update embedding if content changed
            if self.enable_embeddings:
                memory.embedding = self._create_embedding(content)
                
        if importance is not None:
            memory.importance = max(0.0, min(1.0, importance))
            
        if tags is not None:
            memory.tags = tags
            
        if metadata is not None:
            memory.metadata.update(metadata)
            
        # Mark as accessed
        memory.update_access()
        
        # Save to database
        self._save_memory(memory)
        
        logger.info(f"Updated memory {memory_id}")
        return True
        
    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory
        
        Args:
            memory_id: ID of the memory to delete
            
        Returns:
            bool: True if deletion successful
        """
        # Find and remove the memory
        for memory_type, memories in self.memories.items():
            if memory_id in memories:
                del memories[memory_id]
                self._delete_memory(memory_id)
                logger.info(f"Deleted memory {memory_id}")
                return True
                
        logger.warning(f"Memory {memory_id} not found for deletion")
        return False
        
    def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """
        Retrieve a specific memory by ID
        
        Args:
            memory_id: ID of the memory to retrieve
            
        Returns:
            MemoryItem or None if not found
        """
        # Find the memory
        for memory_type, memories in self.memories.items():
            if memory_id in memories:
                memory = memories[memory_id]
                memory.update_access()
                self._save_memory(memory)  # Save updated access stats
                return memory
                
        return None
        
    def query_memories(
        self,
        text_query: Optional[str] = None,
        memory_types: Optional[List[MemoryType]] = None,
        tags: Optional[List[str]] = None,
        min_importance: float = 0.0,
        sources: Optional[List[str]] = None,
        limit: int = 10,
        use_embeddings: bool = True
    ) -> List[MemoryItem]:
        """
        Query memories based on various criteria
        
        Args:
            text_query: Text to search for (semantic search if embeddings enabled)
            memory_types: Types of memories to include
            tags: Filter by tags (ANY match)
            min_importance: Minimum importance score
            sources: Filter by sources
            limit: Maximum results to return
            use_embeddings: Whether to use vector embeddings for search
            
        Returns:
            List of matching MemoryItems, ordered by relevance
        """
        memory_types = memory_types or list(MemoryType)
        results = []
        
        # If we have a text query and embeddings are enabled, do semantic search
        if text_query and self.enable_embeddings and use_embeddings and self.embedding_model:
            query_embedding = self._create_embedding(text_query)
            if query_embedding is not None:
                # Calculate similarity scores for all memories
                scored_memories = []
                
                for memory_type in memory_types:
                    for memory in self.memories[memory_type].values():
                        # Skip memories without embeddings
                        if memory.embedding is None:
                            continue
                            
                        # Apply filters
                        if memory.importance < min_importance:
                            continue
                            
                        if tags and not any(tag in memory.tags for tag in tags):
                            continue
                            
                        if sources and memory.source not in sources:
                            continue
                            
                        # Calculate cosine similarity
                        embedding = memory.embedding
                        if isinstance(embedding, list):
                            embedding = np.array(embedding)
                            
                        similarity = np.dot(query_embedding, embedding) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
                        )
                        
                        scored_memories.append((memory, float(similarity)))
                
                # Sort by similarity score
                scored_memories.sort(key=lambda x: x[1], reverse=True)
                results = [memory for memory, _ in scored_memories[:limit]]
        else:
            # Regular filtering without semantic search
            candidate_memories = []
            
            for memory_type in memory_types:
                for memory in self.memories[memory_type].values():
                    # Apply filters
                    if memory.importance < min_importance:
                        continue
                        
                    if tags and not any(tag in memory.tags for tag in tags):
                        continue
                        
                    if sources and memory.source not in sources:
                        continue
                        
                    if text_query and text_query.lower() not in memory.content.lower():
                        continue
                        
                    candidate_memories.append(memory)
            
            # Sort by relevance (recency and importance)
            current_time = time.time()
            candidate_memories.sort(
                key=lambda m: m.calculate_relevance(current_time),
                reverse=True
            )
            results = candidate_memories[:limit]
        
        # Update access counts for returned memories
        for memory in results:
            memory.update_access()
            self._save_memory(memory)
            
        return results
        
    def get_memories_by_type(self, memory_type: MemoryType, limit: int = 100) -> List[MemoryItem]:
        """
        Get memories of a specific type, sorted by importance and recency
        
        Args:
            memory_type: Type of memories to retrieve
            limit: Maximum number to return
            
        Returns:
            List of memories
        """
        memories = list(self.memories[memory_type].values())
        current_time = time.time()
        
        # Sort by relevance
        memories.sort(
            key=lambda m: m.calculate_relevance(current_time),
            reverse=True
        )
        
        return memories[:limit]
        
    def get_memories_by_tags(self, tags: List[str], limit: int = 100) -> List[MemoryItem]:
        """
        Get memories with specific tags
        
        Args:
            tags: List of tags to match (ANY match)
            limit: Maximum number to return
            
        Returns:
            List of memories
        """
        return self.query_memories(tags=tags, limit=limit)
        
    def _enforce_memory_limits(self):
        """
        Enforce memory limits by forgetting least important memories
        """
        for memory_type, limit in [
            (MemoryType.SHORT_TERM, self.short_term_limit),
            (MemoryType.MEDIUM_TERM, self.medium_term_limit),
            (MemoryType.LONG_TERM, self.long_term_limit)
        ]:
            memories = self.memories[memory_type]
            if len(memories) <= limit:
                continue
                
            # Calculate current relevance for all memories
            current_time = time.time()
            scored_memories = [
                (memory_id, memory.calculate_relevance(current_time))
                for memory_id, memory in memories.items()
            ]
            
            # Sort by relevance (ascending)
            scored_memories.sort(key=lambda x: x[1])
            
            # Identify memories to forget
            to_forget = scored_memories[:(len(memories) - limit)]
            
            # Forget memories
            for memory_id, relevance in to_forget:
                if relevance < self.min_importance_threshold:
                    logger.info(f"Forgetting memory {memory_id} with relevance {relevance:.3f}")
                    self.delete_memory(memory_id)
        
    def consolidate_memories(self):
        """
        Consolidate important short-term memories into medium-term,
        and important medium-term into long-term.
        
        This process simulates human memory consolidation during periods of low activity.
        """
        current_time = time.time()
        
        # Short-term to medium-term (memories older than 1 day with high importance)
        short_term = self.memories[MemoryType.SHORT_TERM].copy()
        day_ago = current_time - 86400
        
        for memory_id, memory in short_term.items():
            # Check if memory is old enough and important enough
            if memory.created_at < day_ago and memory.importance > 0.7:
                logger.info(f"Consolidating short-term memory {memory_id} to medium-term")
                
                # Create new medium-term memory
                new_memory = MemoryItem(
                    content=memory.content,
                    memory_type=MemoryType.MEDIUM_TERM,
                    importance=memory.importance,
                    created_at=memory.created_at,
                    last_accessed=memory.last_accessed,
                    access_count=memory.access_count,
                    tags=memory.tags + ["consolidated_from_short_term"],
                    source=memory.source,
                    metadata=memory.metadata,
                    embedding=memory.embedding
                )
                
                # Add to medium-term and remove from short-term
                new_memory.memory_id = self._generate_memory_id(new_memory)
                self.memories[MemoryType.MEDIUM_TERM][new_memory.memory_id] = new_memory
                self._save_memory(new_memory)
                self.delete_memory(memory_id)
        
        # Medium-term to long-term (memories older than 7 days with high importance and access)
        medium_term = self.memories[MemoryType.MEDIUM_TERM].copy()
        week_ago = current_time - 604800
        
        for memory_id, memory in medium_term.items():
            # Check if memory is old enough, important enough, and accessed frequently
            if memory.created_at < week_ago and memory.importance > 0.8 and memory.access_count > 5:
                logger.info(f"Consolidating medium-term memory {memory_id} to long-term")
                
                # Create new long-term memory
                new_memory = MemoryItem(
                    content=memory.content,
                    memory_type=MemoryType.LONG_TERM,
                    importance=memory.importance,
                    created_at=memory.created_at,
                    last_accessed=memory.last_accessed,
                    access_count=memory.access_count,
                    tags=memory.tags + ["consolidated_from_medium_term"],
                    source=memory.source,
                    metadata=memory.metadata,
                    embedding=memory.embedding
                )
                
                # Add to long-term and remove from medium-term
                new_memory.memory_id = self._generate_memory_id(new_memory)
                self.memories[MemoryType.LONG_TERM][new_memory.memory_id] = new_memory
                self._save_memory(new_memory)
                self.delete_memory(memory_id)
                
        logger.info("Memory consolidation complete")
        
    def count_memories(self) -> Dict[str, int]:
        """Count memories by type"""
        return {
            memory_type.value: len(memories)
            for memory_type, memories in self.memories.items()
        }
        
    def summarize_memories(self, memory_type: Optional[MemoryType] = None) -> str:
        """Generate a summary of memories for reporting"""
        if memory_type:
            memory_counts = {memory_type.value: len(self.memories[memory_type])}
        else:
            memory_counts = self.count_memories()
            
        lines = ["Memory System Summary:"]
        total = 0
        
        for type_name, count in memory_counts.items():
            lines.append(f"- {type_name}: {count} memories")
            total += count
            
        lines.append(f"Total: {total} memories")
        
        # Add tag statistics
        tag_counts = defaultdict(int)
        for memories in self.memories.values():
            for memory in memories.values():
                for tag in memory.tags:
                    tag_counts[tag] += 1
                    
        if tag_counts:
            lines.append("\nTop tags:")
            top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            for tag, count in top_tags:
                lines.append(f"- {tag}: {count}")
                
        return "\n".join(lines)
        
    def close(self):
        """Close database connection"""
        if hasattr(self, 'conn'):
            self.conn.close()
            logger.info("Memory system database connection closed")
