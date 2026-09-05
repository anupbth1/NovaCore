"""
NovaCore Knowledge Extractor - Learns from data at encoding time.
Zero training, zero backprop, just smart pattern extraction.
"""

import re
import json
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional, Tuple
import math


class KnowledgeExtractor:
    """
    Extracts knowledge from training data at encoding time.
    No training, no backprop, just statistical pattern extraction.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Knowledge stores
        self.fact_patterns = {}          # Facts: "X is Y"
        self.qa_patterns = {}            # Q&A patterns
        self.entity_relations = {}       # Entity relationships
        self.definitions = {}            # Word definitions
        self.procedures = {}             # How-to procedures
        self.causal_relations = {}       # Cause-effect
        self.comparisons = {}            # Comparisons
        self.examples = {}               # Examples
        self.code_patterns = {}          # Code snippets
        self.math_formulas = {}          # Math formulas
        
        # Statistics
        self.stats = {
            'facts_extracted': 0,
            'qa_pairs': 0,
            'entities': 0,
            'definitions': 0,
            'procedures': 0,
            'causal': 0,
            'comparisons': 0,
            'examples': 0,
            'code': 0,
            'math': 0,
        }
    
    def extract_from_text(self, text: str) -> Dict[str, Any]:
        """Extract all knowledge from a single text."""
        knowledge = {
            'facts': [],
            'qa_pairs': [],
            'definitions': [],
            'procedures': [],
            'causal': [],
            'comparisons': [],
            'examples': [],
            'code': [],
            'math': [],
        }
        
        # Clean text
        clean_text = self._clean_for_extraction(text)
        
        # Extract different knowledge types
        knowledge['facts'] = self._extract_facts(clean_text)
        self.stats['facts_extracted'] += len(knowledge['facts'])
        
        knowledge['qa_pairs'] = self._extract_qa_pairs(clean_text)
        self.stats['qa_pairs'] += len(knowledge['qa_pairs'])
        
        knowledge['definitions'] = self._extract_definitions(clean_text)
        self.stats['definitions'] += len(knowledge['definitions'])
        
        knowledge['procedures'] = self._extract_procedures(clean_text)
        self.stats['procedures'] += len(knowledge['procedures'])
        
        knowledge['causal'] = self._extract_causal(clean_text)
        self.stats['causal'] += len(knowledge['causal'])
        
        knowledge['comparisons'] = self._extract_comparisons(clean_text)
        self.stats['comparisons'] += len(knowledge['comparisons'])
        
        knowledge['examples'] = self._extract_examples(clean_text)
        self.stats['examples'] += len(knowledge['examples'])
        
        knowledge['code'] = self._extract_code(clean_text)
        self.stats['code'] += len(knowledge['code'])
        
        knowledge['math'] = self._extract_math(clean_text)
        self.stats['math'] += len(knowledge['math'])
        
        return knowledge
    
    def _clean_for_extraction(self, text: str) -> str:
        """Clean text for knowledge extraction."""
        # Remove training artifacts
        text = re.sub(r'<instruction>.*?</instruction>', '', text, flags=re.DOTALL)
        text = re.sub(r'<answer>.*?</answer>', '', text, flags=re.DOTALL)
        text = re.sub(r'<text>.*?</text>', '', text, flags=re.DOTALL)
        text = re.sub(r'<user>.*?</user>', '', text, flags=re.DOTALL)
        text = re.sub(r'<assistant>.*?</assistant>', '', text, flags=re.DOTALL)
        text = re.sub(r'###\s*Instruction:.*?###\s*Response:', '', text, flags=re.DOTALL)
        text = re.sub(r'Below is an instruction.*?###\s*Response:', '', text, flags=re.DOTALL)
        text = re.sub(r'Write a response that appropriately completes the request\.\s*', '', text, flags=re.DOTALL)
        text = re.sub(r'The request\.\s*', '', text, flags=re.DOTALL)
        
        return text.strip()
    
    def _extract_facts(self, text: str) -> List[Dict]:
        """Extract factual statements: 'X is Y', 'X was Y', etc."""
        facts = []
        
        # Pattern: "X is a Y", "X is Y", "X was Y", "X are Y"
        patterns = [
            r'(\w+(?:\s+\w+){0,5})\s+is\s+(?:a\s+)?(\w+(?:\s+\w+){0,10})[.\n]',
            r'(\w+(?:\s+\w+){0,5})\s+was\s+(?:a\s+)?(\w+(?:\s+\w+){0,10})[.\n]',
            r'(\w+(?:\s+\w+){0,5})\s+are\s+(?:a\s+)?(\w+(?:\s+\w+){0,10})[.\n]',
            r'(\w+(?:\s+\w+){0,5})\s+means\s+(\w+(?:\s+\w+){0,10})[.\n]',
            r'(\w+(?:\s+\w+){0,5})\s+refers\s+to\s+(\w+(?:\s+\w+){0,10})[.\n]',
            r'(\w+(?:\s+\w+){0,5})\s+is\s+defined\s+as\s+(\w+(?:\s+\w+){0,10})[.\n]',
            r'(\w+(?:\s+\w+){0,5})\s+equals\s+(\w+(?:\s+\w+){0,10})[.\n]',
            r'(\w+(?:\s+\w+){0,5})\s+consists\s+of\s+(\w+(?:\s+\w+){0,10})[.\n]',
            r'(\w+(?:\s+\w+){0,5})\s+contains\s+(\w+(?:\s+\w+){0,10})[.\n]',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                if len(groups) >= 2:
                    subject = groups[0].strip()
                    obj = groups[1].strip()  # Second group (first capture group after first)
                    
                    # Filter out noise
                    if len(subject) > 2 and len(obj) > 2:
                        if not self._is_noise(subject) and not self._is_noise(obj):
                            facts.append({
                                'subject': subject,
                                'object': obj,
                                'pattern': pattern,
                                'confidence': 0.7
                            })
        
        return facts
    
    def _extract_qa_pairs(self, text: str) -> List[Dict]:
        """Extract Q&A pairs from text."""
        qa_pairs = []
        
        # Pattern: "Q: ... A: ..." or "Question: ... Answer: ..."
        patterns = [
            r'(?:Q|Question):\s*([^?]+\?)\s*(?:A|Answer):\s*([^.\n]+[.\n])',
            r'Question:\s*([^?]+\?)\s*Answer:\s*([^.\n]+[.\n])',
            r'What\s+is\s+([^?]+\?)\s*([^.\n]+[.\n])',
            r'How\s+to\s+([^?]+\?)\s*([^.\n]+[.\n])',
            r'Why\s+([^?]+\?)\s*([^.\n]+[.\n])',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) >= 2:
                    question = match.group(1).strip()
                    answer = match.group(2).strip()
                    
                    if len(question) > 5 and len(answer) > 5:
                        qa_pairs.append({
                            'question': question,
                            'answer': answer,
                            'confidence': 0.8
                        })
        
        return qa_pairs
    
    def _extract_definitions(self, text: str) -> List[Dict]:
        """Extract definitions: 'X is defined as Y', 'X means Y'."""
        definitions = []
        
        patterns = [
            r'(\w+(?:\s+\w+){0,5})\s+is\s+(?:defined\s+as|defined\s+by)\s+([^.\n]+[.\n])',
            r'(\w+(?:\s+\w+){0,5})\s+means\s+([^.\n]+[.\n])',
            r'(\w+(?:\s+\w+){0,5})\s+refers\s+to\s+([^.\n]+[.\n])',
            r'(?:Definition|define)\s+of\s+(\w+(?:\s+\w+){0,5})\s+is\s+([^.\n]+[.\n])',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) >= 2:
                    term = match.group(1).strip()
                    definition = match.group(2).strip()
                    
                    if len(term) > 2 and len(definition) > 5:
                        definitions.append({
                            'term': term,
                            'definition': definition,
                            'confidence': 0.85
                        })
        
        return definitions
    
    def _extract_procedures(self, text: str) -> List[Dict]:
        """Extract how-to procedures and steps."""
        procedures = []
        
        # Pattern: "Step 1: ...", "1. ...", "First, ...", "To X, do Y"
        patterns = [
            r'(?:Step|step)\s+\d+:?\s*([^.\n]+[.\n])',
            r'\d+\.\s*([^.\n]+[.\n])',
            r'(?:First|Second|Third|Finally),?\s*([^.\n]+[.\n])',
            r'To\s+(\w+(?:\s+\w+){0,5}),?\s*([^.\n]+[.\n])',
            r'In\s+order\s+to\s+(\w+(?:\s+\w+){0,5}),?\s*([^.\n]+[.\n])',
        ]
        
        steps = []
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) >= 1:
                    step = match.group(1).strip()
                    if len(step) > 10:
                        steps.append(step)
        
        if len(steps) >= 2:
            procedures.append({
                'steps': steps,
                'confidence': 0.75
            })
        
        return procedures
    
    def _extract_causal(self, text: str) -> List[Dict]:
        """Extract cause-effect relationships."""
        causal = []
        
        patterns = [
            r'(\w+(?:\s+\w+){0,10})\s+(?:causes?|leads?\s+to|results?\s+in)\s+(\w+(?:\s+\w+){0,10})[.\n]',
            r'Because\s+(\w+(?:\s+\w+){0,10}),?\s*(\w+(?:\s+\w+){0,10})[.\n]',
            r'(\w+(?:\s+\w+){0,10})\s+(?:since|as|because)\s+(\w+(?:\s+\w+){0,10})[.\n]',
            r'If\s+(\w+(?:\s+\w+){0,10}),?\s*then\s+(\w+(?:\s+\w+){0,10})[.\n]',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) >= 2:
                    cause = match.group(1).strip()
                    effect = match.group(2).strip()
                    
                    if len(cause) > 3 and len(effect) > 3:
                        causal.append({
                            'cause': cause,
                            'effect': effect,
                            'confidence': 0.7
                        })
        
        return causal
    
    def _extract_comparisons(self, text: str) -> List[Dict]:
        """Extract comparisons."""
        comparisons = []
        
        patterns = [
            r'(\w+(?:\s+\w+){0,5})\s+(?:is\s+)?(?:better|worse|faster|slower|larger|smaller|greater|less)\s+than\s+(\w+(?:\s+\w+){0,5})[.\n]',
            r'(\w+(?:\s+\w+){0,5})\s+vs\.?\s+(\w+(?:\s+\w+){0,5})[.\n]',
            r'Compared\s+to\s+(\w+(?:\s+\w+){0,5}),?\s*(\w+(?:\s+\w+){0,5})\s+is\s+(\w+(?:\s+\w+){0,10})[.\n]',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) >= 2:
                    comparisons.append({
                        'subject1': match.group(1).strip(),
                        'subject2': match.group(2).strip(),
                        'relation': 'comparison',
                        'confidence': 0.65
                    })
        
        return comparisons
    
    def _extract_examples(self, text: str) -> List[Dict]:
        """Extract examples: 'For example, ...', 'e.g., ...'."""
        examples = []
        
        patterns = [
            r'(?:For\s+example|Example|e\.g\.|i\.e\.),?\s*([^.\n]+[.\n])',
            r'Such\s+as\s+([^.\n]+[.\n])',
            r'Including\s+([^.\n]+[.\n])',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) >= 1:
                    example = match.group(1).strip()
                    if len(example) > 10:
                        examples.append({
                            'example': example,
                            'confidence': 0.6
                        })
        
        return examples
    
    def _extract_code(self, text: str) -> List[Dict]:
        """Extract code snippets."""
        code = []
        
        # Code blocks
        patterns = [
            r'```(?:python|java|javascript|js|cpp|c\+\+|c#|go|rust)?\n(.*?)```',
            r'`([^`]+)`',
            r'(?:def|class|import|from|print|if\s|for\s|while\s)\s+\w+',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.DOTALL)
            for match in matches:
                snippet = match.group(1) if len(match.groups()) >= 1 else match.group(0)
                snippet = snippet.strip()
                if len(snippet) > 5 and len(snippet) < 2000:
                    code.append({
                        'code': snippet,
                        'confidence': 0.8
                    })
        
        return code
    
    def _extract_math(self, text: str) -> List[Dict]:
        """Extract math formulas."""
        math_formulas = []
        
        patterns = [
            r'(\w+(?:\s+\w+){0,3})\s*=\s*([^.\n]+[.\n])',
            r'(\d+(?:\.\d+)?)\s*[\+\-\*/]\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                if len(match.groups()) >= 2:
                    math_formulas.append({
                        'formula': match.group(0).strip(),
                        'confidence': 0.6
                    })
        
        return math_formulas
    
    def _is_noise(self, text: str) -> bool:
        """Check if text is noise (common words that aren't facts)."""
        noise_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'shall', 'can',
            'need', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
            'from', 'this', 'that', 'it', 'or', 'and', 'but', 'not',
            'i', 'you', 'we', 'they', 'he', 'she', 'me', 'him', 'her',
            'us', 'them', 'my', 'your', 'our', 'their', 'his', 'her',
            'there', 'here', 'where', 'when', 'why', 'how', 'what',
            'who', 'which', 'that', 'this', 'these', 'those'
        }
        return text.lower() in noise_words
    
    def merge_knowledge(self, new_knowledge: Dict[str, Any]):
        """Merge newly extracted knowledge into knowledge base."""
        for category, items in new_knowledge.items():
            if category in ['facts', 'qa_pairs', 'definitions', 'procedures', 
                           'causal', 'comparisons', 'examples', 'code', 'math']:
                if items:
                    getattr(self, f'{category}').append(items)
                    self.stats[f'{category}_extracted'] = self.stats.get(f'{category}_extracted', 0) + len(items)
    
    def get_knowledge_base(self) -> Dict[str, Any]:
        """Get complete knowledge base."""
        return {
            'facts': self.fact_patterns,
            'qa_pairs': self.qa_patterns,
            'definitions': self.definitions,
            'procedures': self.procedures,
            'causal_relations': self.causal_relations,
            'comparisons': self.comparisons,
            'examples': self.examples,
            'code_patterns': self.code_patterns,
            'math_formulas': self.math_formulas,
            'stats': self.stats
        }
    
    def save(self, filepath: str):
        """Save knowledge base to disk."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.get_knowledge_base(), f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, filepath: str):
        """Load knowledge base from disk."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        extractor = cls()
        extractor.fact_patterns = data.get('facts', [])
        extractor.qa_patterns = data.get('qa_pairs', [])
        extractor.definitions = data.get('definitions', [])
        extractor.procedures = data.get('procedures', [])
        extractor.causal_relations = data.get('causal_relations', [])
        extractor.comparisons = data.get('comparisons', [])
        extractor.examples = data.get('examples', [])
        extractor.code_patterns = data.get('code_patterns', [])
        extractor.math_formulas = data.get('math_formulas', [])
        extractor.stats = data.get('stats', {})
        
        return extractor


class KnowledgeIndexer:
    """
    Indexes extracted knowledge for fast retrieval at inference time.
    """
    
    def __init__(self):
        self.keyword_index = defaultdict(set)  # keyword -> knowledge_ids
        self.entity_index = defaultdict(set)   # entity -> knowledge_ids
        self.category_index = defaultdict(set) # category -> knowledge_ids
        self.knowledge_store = []              # List of knowledge items
    
    def index_knowledge(self, knowledge: Dict[str, Any], doc_id: str = "") -> int:
        """Index knowledge items for retrieval. Returns number of items indexed."""
        indexed_count = 0
        
        for category, items in knowledge.items():
            if not items:
                continue
            
            for item in items:
                knowledge_id = len(self.knowledge_store)
                
                item['knowledge_id'] = knowledge_id
                item['category'] = category
                item['doc_id'] = doc_id
                
                self.knowledge_store.append(item)
                self.category_index[category].add(knowledge_id)
                
                # Index by keywords
                if category == 'facts':
                    keywords = set(item.get('subject', '').lower().split() + 
                                  item.get('object', '').lower().split())
                elif category == 'qa_pairs':
                    keywords = set(item.get('question', '').lower().split() + 
                                  item.get('answer', '').lower().split())
                elif category == 'definitions':
                    keywords = set(item.get('term', '').lower().split() + 
                                  item.get('definition', '').lower().split())
                elif category == 'procedures':
                    steps_text = ' '.join(item.get('steps', []))
                    keywords = set(steps_text.lower().split())
                elif category == 'causal':
                    keywords = set(item.get('cause', '').lower().split() + 
                                  item.get('effect', '').lower().split())
                else:
                    text = str(item)
                    keywords = set(text.lower().split())
                
                for kw in keywords:
                    if len(kw) > 2:
                        self.keyword_index[kw].add(knowledge_id)
                
                indexed_count += 1
        
        return indexed_count
    
    def search(self, query: str, top_k: int = 10, category: str = None) -> List[Dict]:
        """Search indexed knowledge for query."""
        query_words = set(query.lower().split())
        
        # Score knowledge items
        scores = defaultdict(float)
        
        for word in query_words:
            if word in self.keyword_index:
                for kid in self.keyword_index[word]:
                    scores[kid] += 1.0
        
        # Filter by category if specified
        if category:
            valid_ids = self.category_index.get(category, set())
            scores = {k: v for k, v in scores.items() if k in valid_ids}
        
        # Sort by score
        sorted_ids = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        
        results = []
        for kid, score in sorted_ids:
            if kid < len(self.knowledge_store):
                item = self.knowledge_store[kid].copy()
                item['relevance_score'] = score
                results.append(item)
        
        return results
    
    def get_by_category(self, category: str, top_k: int = 100) -> List[Dict]:
        """Get all knowledge items of a category."""
        ids = self.category_index.get(category, set())
        items = [self.knowledge_store[i] for i in sorted(ids)[:top_k]]
        return items
    
    def stats(self) -> Dict[str, int]:
        """Get index statistics."""
        return {
            'total_items': len(self.knowledge_store),
            'categories': {cat: len(ids) for cat, ids in self.category_index.items()},
            'unique_keywords': len(self.keyword_index),
        }


# Integration with NovaCore Encoding
class EncodingTimeLearner:
    """
    Learns from data DURING encoding process.
    No training, just knowledge extraction + indexing.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.extractor = KnowledgeExtractor(config)
        self.indexer = KnowledgeIndexer()
        self.total_docs_processed = 0
    
    def learn_from_document(self, text: str, doc_id: str = "") -> Dict[str, Any]:
        """Extract and index knowledge from a document."""
        # Extract knowledge
        knowledge = self.extractor.extract_from_text(text)
        
        # Index for retrieval
        indexed = self.indexer.index_knowledge(knowledge, doc_id)
        
        self.total_docs_processed += 1
        
        return knowledge
    
    def learn_from_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Learn from batch of documents."""
        results = []
        for i, text in enumerate(texts):
            knowledge = self.learn_from_document(text, f"doc_{i}")
            results.append(knowledge)
        return results
    
    def query_knowledge(self, query: str, top_k: int = 5, 
                       category: str = None) -> List[Dict]:
        """Query learned knowledge at inference time."""
        return self.indexer.search(query, top_k, category)
    
    def get_knowledge_base(self) -> Dict[str, Any]:
        """Get complete knowledge base."""
        return {
            'extractor': self.extractor.get_knowledge_base(),
            'indexer_stats': self.indexer.stats(),
            'total_docs': self.total_docs_processed
        }
    
    def save(self, base_path: str):
        """Save knowledge base and index."""
        self.extractor.save(f"{base_path}_knowledge.json")
        
        # Save indexer
        index_data = {
            'knowledge_store': self.indexer.knowledge_store,
            'keyword_index': {k: list(v) for k, v in self.indexer.keyword_index.items()},
            'entity_index': {k: list(v) for k, v in self.indexer.entity_index.items()},
            'category_index': {k: list(v) for k, v in self.indexer.category_index.items()},
        }
        with open(f"{base_path}_index.json", 'w', encoding='utf-8') as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, base_path: str):
        """Load knowledge base and index."""
        learner = cls()
        learner.extractor = KnowledgeExtractor.load(f"{base_path}_knowledge.json")
        
        with open(f"{base_path}_index.json", 'r', encoding='utf-8') as f:
            index_data = json.load(f)
        
        learner.indexer.knowledge_store = index_data['knowledge_store']
        learner.indexer.keyword_index = defaultdict(set, 
            {k: set(v) for k, v in index_data['keyword_index'].items()})
        learner.indexer.entity_index = defaultdict(set,
            {k: set(v) for k, v in index_data['entity_index'].items()})
        learner.indexer.category_index = defaultdict(set,
            {k: set(v) for k, v in index_data['category_index'].items()})
        
        learner.total_docs_processed = len(index_data['knowledge_store'])
        
        return learner


# Usage Example
if __name__ == "__main__":
    # Create learner
    learner = EncodingTimeLearner()
    
    # Sample training text
    texts = [
        """Machine learning is a subset of artificial intelligence that enables 
        systems to learn from data. For example, linear regression is a method 
        for predicting values. Step 1: collect data. Step 2: train model. 
        Because data is noisy, we need regularization.""",
        
        """Python is a programming language. It is used for web development, 
        data science, and automation. Python means easy to read code. 
        For example: def hello(): print('Hello World').""",
        
        """Photosynthesis is the process by which plants convert light energy 
        into chemical energy. It occurs in chloroplasts. Chloroplasts contain 
        chlorophyll which captures light energy."""
    ]
    
    # Learn from texts
    for text in texts:
        learner.learn_from_document(text)
    
    # Query knowledge
    queries = [
        "What is machine learning?",
        "How does photosynthesis work?",
        "Python code example",
        "What causes regularization?",
    ]
    
    for query in queries:
        results = learner.query_knowledge(query, top_k=3)
        print(f"\nQuery: {query}")
        for r in results:
            print(f"  [{r['category']}] {r}")
    
    # Print stats
    print(f"\nStats: {learner.get_knowledge_base()['extractor']['stats']}")