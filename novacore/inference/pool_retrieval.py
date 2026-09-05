"""
Pool-based Retrieval for NovaCore

Loads pool data directly from disk for retrieval-based generation.
This provides much better responses than pattern-only generation.
"""
import json
import os
import re
import random
from typing import List, Tuple, Dict, Optional
from collections import Counter


class PoolRetriever:
    """
    Retrieves similar examples from pool data files.
    
    Pool files are JSONL format with {uid, text} per line.
    The text field contains tagged data like:
    <instruction>...</instruction>
    <answer>...</answer>
    """
    
    def __init__(self, pool_dirs=None):
        self.pool_dirs = pool_dirs or []
        self.pool_data = []
        self.instruction_index = {}  # instruction -> response mapping
        self._loaded = False
        
    def load_pools(self, pool_dirs=None):
        """Load all pool data from specified directories."""
        if pool_dirs:
            self.pool_dirs = pool_dirs
        
        if not self.pool_dirs:
            # Auto-discover pool directories
            base_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'hf_cache', 'pool')
            if os.path.exists(base_path):
                for entry in os.listdir(base_path):
                    full_path = os.path.join(base_path, entry)
                    if os.path.isdir(full_path):
                        self.pool_dirs.append(full_path)
        
        for pool_dir in self.pool_dirs:
            pool_file = os.path.join(pool_dir, 'pool.jsonl')
            if os.path.exists(pool_file):
                self._load_pool_file(pool_file)
        
        self._loaded = True
        return len(self.pool_data)
    
    def _load_pool_file(self, pool_file):
        """Load a single pool JSONL file."""
        try:
            with open(pool_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if line_num >= 5000:  # Limit for memory
                        break
                    try:
                        entry = json.loads(line.strip())
                        text = entry.get('text', '')
                        if text:
                            self.pool_data.append(text)
                            # Extract instruction-response pair
                            instruction, response = self._extract_pair(text)
                            if instruction and response:
                                self.instruction_index[instruction.lower()] = response
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"[WARN] Error loading pool {pool_file}: {e}")
    
    def _extract_pair(self, text):
        """Extract instruction and response from tagged text."""
        # Look for <instruction>...</instruction> and <answer>...</answer>
        instruction_match = re.search(r'<instruction>(.*?)</instruction>', text, re.DOTALL)
        answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        
        instruction = instruction_match.group(1).strip() if instruction_match else ""
        response = answer_match.group(1).strip() if answer_match else ""
        
        # Clean artifacts
        instruction = self._clean_artifacts(instruction)
        response = self._clean_artifacts(response)
        
        return instruction, response
    
    def _clean_artifacts(self, text):
        """Remove common artifacts from training data."""
        if not text:
            return ""
        
        artifacts_to_remove = [
            r'Below is an instruction that describes a task.*?###\s*Response:\s*',
            r'Write a response that appropriately completes the request\.\s*',
            r'### Instruction:\s*',
            r'### Response:\s*',
            r'Instruction:\s*',
            r'Response:\s*',
            r'Answer:\s*',
        ]
        
        for artifact in artifacts_to_remove:
            text = re.sub(artifact, '', text, flags=re.DOTALL)
        
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def retrieve_similar(self, query: str, k: int = 5) -> List[Tuple[str, float, str]]:
        """
        Retrieve similar examples to the query.
        
        Returns list of (instruction, similarity_score, response)
        """
        if not self._loaded:
            self.load_pools()
        
        if not self.instruction_index:
            return []
        
        query_lower = query.lower().strip()
        query_words = set(query_lower.split())
        
        # Remove stop words
        stop_words = {'what', 'who', 'where', 'when', 'why', 'how', 'is', 'are', 'do', 'does',
                      'can', 'could', 'would', 'should', 'tell', 'me', 'about', 'the', 'a', 'an',
                      'please', 'help', 'want', 'need', 'like', 'to', 'of', 'in', 'for', 'on',
                      'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during', 'before',
                      'after', 'above', 'below', 'between', 'out', 'off', 'over', 'under'}
        
        query_words -= stop_words
        
        results = []
        for instruction, response in self.instruction_index.items():
            instruction_lower = instruction.lower()
            instruction_words = set(instruction_lower.split())
            instruction_words -= stop_words
            
            if not query_words or not instruction_words:
                continue
            
            # Calculate word overlap similarity
            overlap = query_words & instruction_words
            similarity = 0.0
            
            if overlap:
                # Jaccard similarity
                union = query_words | instruction_words
                similarity = len(overlap) / len(union) if union else 0
                
                # Strong bonus for exact phrase match
                if query_lower in instruction_lower:
                    similarity += 0.5
                
                # Bonus for matching key words (excluding common words)
                key_words = query_words - {'what', 'is', 'are', 'the', 'a', 'an', 'do', 'can', 'how', 'tell', 'explain'}
                if key_words and len(key_words & instruction_words) / len(key_words) > 0.5:
                    similarity += 0.3
                
                # Bonus for exact instruction match
                if query_lower == instruction_lower:
                    similarity += 1.0
            
            # Penalty for responses that are too short or too generic
            if len(response.split()) < 10:
                similarity *= 0.5
            
            if similarity > 0.1:
                results.append((instruction, min(1.0, similarity), response))
        
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:k]
    
    def get_random_response(self) -> str:
        """Get a random response from the pool data."""
        if not self.instruction_index:
            return ""
        
        responses = list(self.instruction_index.values())
        if responses:
            return random.choice(responses)
        return ""
    
    def get_greeting_response(self) -> str:
        """Get a greeting-related response if available."""
        greeting_keywords = ['hello', 'hi', 'hey', 'greetings', 'welcome']
        
        for instruction, response in self.instruction_index.items():
            if any(keyword in instruction.lower() for keyword in greeting_keywords):
                if response and len(response) > 10:
                    return response
        
        return ""
    
    def get_question_response(self, question_type: str = "") -> str:
        """Get a response for a specific question type."""
        question_keywords = {
            'what': ['what is', 'what are', 'what do'],
            'how': ['how to', 'how do', 'how can'],
            'why': ['why do', 'why is', 'why are'],
            'when': ['when do', 'when is', 'when are'],
            'who': ['who is', 'who are', 'who do'],
        }
        
        keywords = question_keywords.get(question_type.lower(), [])
        
        for instruction, response in self.instruction_index.items():
            if any(keyword in instruction.lower() for keyword in keywords):
                if response and len(response) > 20:
                    return response
        
        return ""
