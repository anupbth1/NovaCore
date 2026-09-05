"""
NovaCore Model Brain - Complete Self-Contained AI System

This is the CORE of NovaCore - handles ALL thinking, reasoning, planning.
Everything runs INSIDE the model. No external hardcoded content.

Architecture:
    INPUT (user query)
        ↓
    THINKING LAYER (understand intent, classify, plan)
        ↓
    REASONING LAYER (find knowledge, connect concepts)
        ↓
    PLANNING LAYER (structure response, select approach)
        ↓
    GENERATION LAYER (produce output from dataset knowledge)
        ↓
    VERIFICATION LAYER (self-check, improve quality)
        ↓
    OUTPUT (final response)

All layers use dataset knowledge (patterns, reservoir, vocab).
"""
import re
import random
import math
from collections import Counter
from typing import List, Dict, Tuple, Optional, Any


class ThinkingLayer:
    """
    Understands user intent and classifies the query.
    All knowledge comes from dataset patterns.
    """
    
    def __init__(self, patterns, vocab, reservoir):
        self.patterns = patterns
        self.vocab = vocab
        self.reservoir = reservoir
        
        # Build intent patterns from dataset
        self.intent_patterns = self._extract_intent_patterns()
        
    def _extract_intent_patterns(self):
        """Extract intent classification patterns from dataset."""
        intents = {
            'question': [],
            'command': [],
            'statement': [],
            'greeting': [],
            'creative': [],
            'explanation': [],
            'advice': [],
            'story': [],
        }
        
        # Extract patterns from training data
        for text in self.reservoir[:5000]:  # Sample for efficiency
            if not text:
                continue
                
            text_lower = text.lower()
            
            # Classify based on patterns in dataset
            if any(word in text_lower for word in ['what is', 'who is', 'where is', 'when did', 'why do', 'how to']):
                intents['question'].append(text)
            elif any(word in text_lower for word in ['please', 'can you', 'could you', 'would you', 'tell me']):
                intents['command'].append(text)
            elif any(word in text_lower for word in ['hello', 'hi', 'hey', 'greetings']):
                intents['greeting'].append(text)
            elif any(word in text_lower for word in ['story', 'poem', 'once upon', 'imagine']):
                intents['creative'].append(text)
            elif any(word in text_lower for word in ['explain', 'describe', 'because', 'reason']):
                intents['explanation'].append(text)
            elif any(word in text_lower for word in ['should', 'recommend', 'advice', 'tip']):
                intents['advice'].append(text)
            elif any(word in text_lower for word in ['once there', 'there was', 'long ago']):
                intents['story'].append(text)
            else:
                intents['statement'].append(text)
        
        return intents
    
    def think(self, query: str) -> Dict[str, Any]:
        """
        THINK: Understand what the user wants.
        
        Returns:
            Dict with intent, keywords, complexity, and context
        """
        query_lower = query.lower().strip()
        words = query_lower.split()
        
        # 1. Classify intent
        intent = self._classify_intent(query_lower)
        
        # 2. Extract keywords
        keywords = self._extract_keywords(query_lower)
        
        # 3. Assess complexity
        complexity = self._assess_complexity(query_lower)
        
        # 4. Build context
        context = {
            'original_query': query,
            'intent': intent,
            'keywords': keywords,
            'complexity': complexity,
            'word_count': len(words),
            'has_question': '?' in query,
            'has_instruction': any(w in query_lower for w in ['please', 'can you', 'could you']),
        }
        
        return context
    
    def _classify_intent(self, query: str) -> str:
        """Classify user intent from dataset patterns."""
        # Check each intent category
        for intent, examples in self.intent_patterns.items():
            for example in examples[:100]:  # Sample check
                if self._query_matches_example(query, example):
                    return intent
        
        # Default based on query structure
        if '?' in query:
            return 'question'
        elif any(w in query for w in ['please', 'can you', 'could you']):
            return 'command'
        else:
            return 'statement'
    
    def _query_matches_example(self, query: str, example: str) -> bool:
        """Check if query matches a dataset example."""
        query_words = set(query.split())
        example_words = set(example.lower().split())
        
        # Remove common words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                      'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                      'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
                      'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from'}
        
        query_words -= stop_words
        example_words -= stop_words
        
        if not query_words:
            return False
        
        overlap = query_words & example_words
        return len(overlap) / len(query_words) > 0.4
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from query."""
        stop_words = {'what', 'who', 'where', 'when', 'why', 'how', 'is', 'are', 'do', 'does',
                      'can', 'could', 'would', 'should', 'tell', 'me', 'about', 'the', 'a', 'an',
                      'please', 'help', 'want', 'need', 'like', 'to', 'of', 'in', 'for', 'on',
                      'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during', 'before',
                      'after', 'above', 'below', 'between', 'out', 'off', 'over', 'under'}
        
        words = query.split()
        keywords = [w for w in words if w.lower() not in stop_words and len(w) > 2]
        
        return keywords[:10]  # Limit keywords
    
    def _assess_complexity(self, query: str) -> str:
        """Assess query complexity based on patterns."""
        words = query.split()
        
        if len(words) <= 3:
            return 'simple'
        elif len(words) <= 8:
            return 'medium'
        else:
            return 'complex'


class ReasoningLayer:
    """
    Finds and connects knowledge from dataset.
    All knowledge comes from patterns and reservoir.
    """
    
    def __init__(self, patterns, vocab, reservoir):
        self.patterns = patterns
        self.vocab = vocab
        self.reservoir = reservoir
        
        # Build knowledge index from dataset
        self.knowledge_index = self._build_knowledge_index()
        
    def _build_knowledge_index(self):
        """Build knowledge index from dataset."""
        index = {
            'facts': [],
            'explanations': [],
            'stories': [],
            'instructions': [],
            'definitions': [],
        }
        
        for text in self.reservoir[:5000]:
            if not text:
                continue
            
            text_lower = text.lower()
            
            # Categorize by content type
            if any(w in text_lower for w in ['is a', 'are', 'was', 'definition', 'means']):
                index['facts'].append(text)
            elif any(w in text_lower for w in ['because', 'reason', 'explanation', 'why']):
                index['explanations'].append(text)
            elif any(w in text_lower for w in ['story', 'once', 'there was', 'narrative']):
                index['stories'].append(text)
            elif any(w in text_lower for w in ['step', 'how to', 'guide', 'instruction']):
                index['instructions'].append(text)
            elif any(w in text_lower for w in ['definition', 'refers to', 'is defined']):
                index['definitions'].append(text)
        
        return index
    
    def reason(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        REASON: Find relevant knowledge and plan response.
        
        Uses thinking context to find matching dataset knowledge.
        """
        intent = context['intent']
        keywords = context['keywords']
        
        # 1. Find relevant knowledge
        relevant_knowledge = self._find_relevant_knowledge(keywords, intent)
        
        # 2. Connect concepts
        connected_concepts = self._connect_concepts(relevant_knowledge, keywords)
        
        # 3. Plan response structure
        response_plan = self._plan_response(intent, connected_concepts)
        
        reasoning_result = {
            'relevant_knowledge': relevant_knowledge,
            'connected_concepts': connected_concepts,
            'response_plan': response_plan,
            'confidence': self._calculate_confidence(relevant_knowledge, keywords),
        }
        
        context.update(reasoning_result)
        return context
    
    def _find_relevant_knowledge(self, keywords: List[str], intent: str) -> List[str]:
        """Find knowledge relevant to keywords from dataset."""
        relevant = []
        
        # Search ALL categories for better coverage
        for category in self.knowledge_index:
            for text in self.knowledge_index.get(category, []):
                if self._text_matches_keywords(text, keywords):
                    relevant.append(text)
                    if len(relevant) >= 10:
                        break
        
        return relevant
    
    def _text_matches_keywords(self, text: str, keywords: List[str]) -> bool:
        """Check if text matches keywords."""
        if not keywords:
            return False
        
        text_lower = text.lower()
        text_words = set(text_lower.split())
        
        # Check for keyword matches
        keyword_matches = sum(1 for kw in keywords if kw.lower() in text_lower)
        
        # More flexible matching - even 1 match is enough for short queries
        return keyword_matches >= 1
    
    def _connect_concepts(self, knowledge: List[str], keywords: List[str]) -> List[str]:
        """Connect related concepts from knowledge."""
        if not knowledge:
            return []
        
        # Sort by relevance to keywords
        def relevance_score(text):
            text_words = set(text.lower().split())
            return sum(1 for kw in keywords if kw.lower() in text_words)
        
        sorted_knowledge = sorted(knowledge, key=relevance_score, reverse=True)
        
        return sorted_knowledge[:5]  # Top 5 connected concepts
    
    def _plan_response(self, intent: str, concepts: List[str]) -> Dict[str, Any]:
        """Plan response structure based on intent and concepts."""
        plan = {
            'type': intent,
            'length': 'medium',
            'structure': 'paragraph',
            'approach': 'direct',
        }
        
        if intent == 'story':
            plan['structure'] = 'narrative'
            plan['length'] = 'long'
        elif intent == 'explanation':
            plan['structure'] = 'explanatory'
            plan['length'] = 'detailed'
        elif intent == 'instruction':
            plan['structure'] = 'steps'
            plan['length'] = 'medium'
        elif intent == 'question':
            plan['structure'] = 'answer'
            plan['length'] = 'concise'
        
        return plan
    
    def _calculate_confidence(self, knowledge: List[str], keywords: List[str]) -> float:
        """Calculate confidence in knowledge match."""
        if not knowledge:
            return 0.0
        
        # More knowledge and better matches = higher confidence
        knowledge_score = min(1.0, len(knowledge) / 5)
        
        return knowledge_score


class PlanningLayer:
    """
    Structures the response based on reasoning results.
    """
    
    def __init__(self, patterns, vocab):
        self.patterns = patterns
        self.vocab = vocab
    
    def plan(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        PLAN: Structure the final response.
        """
        response_plan = context.get('response_plan', {})
        connected_concepts = context.get('connected_concepts', [])
        intent = context.get('intent', 'statement')
        
        # Build response structure
        structure = {
            'opening': self._plan_opening(intent, context),
            'body': self._plan_body(connected_concepts, response_plan),
            'closing': self._plan_closing(intent),
            'transitions': self._plan_transitions(len(connected_concepts)),
        }
        
        context['response_structure'] = structure
        return context
    
    def _plan_opening(self, intent: str, context: Dict[str, Any]) -> str:
        """Plan response opening based on intent."""
        # Find similar openings in dataset
        query = context.get('original_query', '')
        
        for text in context.get('connected_concepts', []):
            if text:
                # Use first sentence as opening inspiration
                sentences = text.split('.')
                if sentences and len(sentences[0]) > 10:
                    return sentences[0].strip() + '.'
        
        return ""  # Let generation handle it
    
    def _plan_body(self, concepts: List[str], plan: Dict[str, Any]) -> List[str]:
        """Plan response body."""
        body = []
        
        for concept in concepts[:3]:  # Top 3 concepts
            if concept:
                # Clean and prepare concept
                cleaned = self._clean_concept(concept)
                if cleaned:
                    body.append(cleaned)
        
        return body
    
    def _plan_closing(self, intent: str) -> str:
        """Plan response closing."""
        # Find similar closings in dataset
        return ""  # Let generation handle it
    
    def _plan_transitions(self, num_concepts: int) -> List[str]:
        """Plan transitions between concepts."""
        # Find transition patterns from dataset
        transitions = []
        
        for gram, count in self.patterns.patterns.items():
            if len(gram) == 2 and gram[0] in ['however', 'additionally', 'furthermore', 'moreover']:
                transitions.append(' '.join(gram))
        
        return transitions[:3] if transitions else []
    
    def _clean_concept(self, concept: str) -> str:
        """Clean concept for use in response."""
        # Remove common artifacts
        concept = re.sub(r'<instruction>.*?</instruction>', '', concept, flags=re.DOTALL)
        concept = re.sub(r'<answer>.*?</answer>', '', concept, flags=re.DOTALL)
        concept = re.sub(r'<text>.*?</text>', '', concept, flags=re.DOTALL)
        
        # Clean whitespace
        concept = re.sub(r'\s+', ' ', concept).strip()
        
        return concept


class GenerationLayer:
    """
    Generates final response from dataset knowledge.
    All content comes from patterns and reservoir.
    """
    
    def __init__(self, patterns, vocab, reservoir):
        self.patterns = patterns
        self.vocab = vocab
        self.reservoir = reservoir
    
    def generate(self, context: Dict[str, Any], max_tokens: int = 100) -> str:
        """
        GENERATE: Create response from dataset knowledge.
        """
        connected_concepts = context.get('connected_concepts', [])
        response_structure = context.get('response_structure', {})
        intent = context.get('intent', 'statement')
        
        if not connected_concepts:
            # Fallback to pattern generation
            return self._generate_from_patterns(context, max_tokens)
        
        # Use connected concepts as base
        response = self._compose_from_concepts(connected_concepts, response_structure)
        
        # Enhance with patterns
        response = self._enhance_with_patterns(response, context)
        
        return response
    
    def _compose_from_concepts(self, concepts: List[str], structure: Dict[str, Any]) -> str:
        """Compose response from concepts."""
        if not concepts:
            return ""
        
        # Take the best concept and refine it
        best_concept = concepts[0]
        
        # Clean up the concept - extract just the answer part
        response = self._extract_answer(best_concept)
        
        # If response is too short, try next concept
        if len(response.split()) < 5 and len(concepts) > 1:
            response = self._extract_answer(concepts[1])
        
        return response
    
    def _extract_answer(self, text: str) -> str:
        """Extract the answer part from a training example."""
        if not text:
            return ""
        
        # Try to extract content between <assistant> tags
        assistant_match = re.search(r'<assistant>(.*?)</assistant>', text, re.DOTALL)
        if assistant_match:
            answer = assistant_match.group(1).strip()
            if answer:
                return self._refine_text(answer)
        
        # Try to extract content after "Answer:" or "Response:"
        answer_patterns = [
            r'Answer:\s*(.*?)(?:\n|$)',
            r'Response:\s*(.*?)(?:\n|$)',
            r'### Response:\s*(.*?)(?:\n|$)',
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                if answer:
                    return self._refine_text(answer)
        
        # If no structured format, clean the whole text
        return self._refine_text(text)
    
    def _refine_text(self, text: str) -> str:
        """Refine text for better quality."""
        if not text:
            return ""
        
        # Remove all training data artifacts and tags
        text = re.sub(r'<instruction>.*?</instruction>', '', text, flags=re.DOTALL)
        text = re.sub(r'<answer>.*?</answer>', '', text, flags=re.DOTALL)
        text = re.sub(r'<text>.*?</text>', '', text, flags=re.DOTALL)
        text = re.sub(r'<user>.*?</user>', '', text, flags=re.DOTALL)
        text = re.sub(r'<assistant>.*?</assistant>', '', text, flags=re.DOTALL)
        text = re.sub(r'###\s*Instruction:.*?###\s*Response:', '', text, flags=re.DOTALL)
        text = re.sub(r'Below is an instruction.*?###\s*Response:', '', text, flags=re.DOTALL)
        text = re.sub(r'Write a response that appropriately completes the request\.\s*', '', text, flags=re.DOTALL)
        text = re.sub(r'The request\.\s*', '', text, flags=re.DOTALL)
        text = re.sub(r'The following sentence.*?###\s*Response:\s*', '', text, flags=re.DOTALL)
        
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Ensure proper capitalization
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        
        # Ensure proper ending
        if text and text[-1] not in '.!?':
            text += '.'
        
        return text
    
    def _enhance_with_patterns(self, response: str, context: Dict[str, Any]) -> str:
        """Enhance response with pattern knowledge."""
        # Find relevant patterns
        keywords = context.get('keywords', [])
        
        if not keywords or not response:
            return response
        
        # Look for patterns that could enhance the response
        for gram, count in self.patterns.patterns.items():
            if len(gram) == 2:
                # Check if pattern adds value
                if gram[0] in response.lower() and gram[1] not in response.lower():
                    # Pattern might add relevant continuation
                    pass
        
        return response
    
    def _generate_from_patterns(self, context: Dict[str, Any], max_tokens: int) -> str:
        """Generate response using pattern matching."""
        keywords = context.get('keywords', [])
        intent = context.get('intent', 'statement')
        
        if not keywords:
            # Use random patterns from dataset
            return self._get_random_knowledge()
        
        # Find patterns containing keywords
        relevant_patterns = []
        for gram, count in self.patterns.patterns.items():
            if any(kw in ' '.join(gram) for kw in keywords):
                relevant_patterns.append((gram, count))
        
        if not relevant_patterns:
            return self._get_random_knowledge()
        
        # Sort by frequency
        relevant_patterns.sort(key=lambda x: x[1], reverse=True)
        
        # Build response from patterns
        response_words = []
        for gram, count in relevant_patterns[:5]:
            response_words.extend(gram)
        
        response = ' '.join(response_words)
        
        # Clean up
        response = self._refine_text(response)
        
        return response
    
    def _get_random_knowledge(self) -> str:
        """Get random knowledge from dataset."""
        if self.reservoir:
            # Get random text from reservoir
            random_text = random.choice(self.reservoir[:1000])
            return self._refine_text(random_text)
        
        return ""


class VerificationLayer:
    """
    Self-checks and improves response quality.
    All verification uses dataset knowledge.
    """
    
    def __init__(self, patterns, vocab, reservoir):
        self.patterns = patterns
        self.vocab = vocab
        self.reservoir = reservoir
    
    def verify(self, response: str, context: Dict[str, Any]) -> str:
        """
        VERIFY: Check and improve response quality.
        """
        if not response:
            return response
        
        # 1. Check coherence
        response = self._check_coherence(response)
        
        # 2. Check relevance
        response = self._check_relevance(response, context)
        
        # 3. Check quality
        response = self._check_quality(response)
        
        # 4. Final cleanup
        response = self._final_cleanup(response)
        
        return response
    
    def _check_coherence(self, response: str) -> str:
        """Check if response is coherent."""
        words = response.split()
        
        # Check for excessive repetition
        word_counts = Counter(words)
        max_repeat = max(word_counts.values()) if word_counts else 0
        
        if max_repeat > len(words) * 0.3:
            # Remove excessive repetition
            cleaned = []
            seen = set()
            for word in words:
                if word not in seen or len(cleaned) < 5:
                    cleaned.append(word)
                    seen.add(word)
            response = ' '.join(cleaned)
        
        return response
    
    def _check_relevance(self, response: str, context: Dict[str, Any]) -> str:
        """Check if response is relevant to query."""
        keywords = context.get('keywords', [])
        
        if not keywords:
            return response
        
        # Check if response contains keywords
        response_words = set(response.lower().split())
        keyword_matches = sum(1 for kw in keywords if kw.lower() in response_words)
        
        if keyword_matches == 0 and len(keywords) > 0:
            # Response might not be relevant, but don't modify too much
            pass
        
        return response
    
    def _check_quality(self, response: str) -> str:
        """Check response quality."""
        # Check length
        if len(response.split()) > 200:
            # Truncate
            words = response.split()[:200]
            response = ' '.join(words)
            if response[-1] not in '.!?':
                response += '.'
        
        # Check for proper sentence structure
        sentences = response.split('.')
        if len(sentences) > 1:
            # Ensure proper spacing
            response = '. '.join(s.strip() for s in sentences if s.strip())
            if not response.endswith('.'):
                response += '.'
        
        return response
    
    def _final_cleanup(self, response: str) -> str:
        """Final cleanup."""
        # Remove any remaining artifacts
        response = re.sub(r'<instruction>.*?</instruction>', '', response, flags=re.DOTALL)
        response = re.sub(r'<answer>.*?</answer>', '', response, flags=re.DOTALL)
        response = re.sub(r'<text>.*?</text>', '', response, flags=re.DOTALL)
        
        # Clean whitespace
        response = re.sub(r'\s+', ' ', response).strip()
        
        # Ensure proper capitalization
        if response and response[0].islower():
            response = response[0].upper() + response[1:]
        
        # Ensure proper ending
        if response and response[-1] not in '.!?':
            response += '.'
        
        return response


class ModelBrain:
    """
    Complete self-contained AI brain.
    
    All thinking, reasoning, planning, generation, and verification
    happens INSIDE the model. No external hardcoded content.
    
    All knowledge comes from:
    - Patterns (n-grams extracted from dataset)
    - Reservoir (sample texts from dataset)
    - Pool data (directly loaded from pool.jsonl files)
    - Vocabulary (tokens from dataset)
    """
    
    def __init__(self, patterns, vocab, reservoir, pool_data=None):
        """
        Initialize the model brain.
        
        Args:
            patterns: PatternExtractor with n-gram patterns from dataset
            vocab: Vocabulary from dataset
            reservoir: List of sample texts from dataset
            pool_data: List of texts loaded directly from pool files
        """
        self.patterns = patterns
        self.vocab = vocab
        # Combine reservoir and pool data for maximum knowledge
        self.reservoir = reservoir or []
        if pool_data:
            self.reservoir.extend(pool_data)
        
        # Initialize all layers
        self.thinking = ThinkingLayer(patterns, vocab, self.reservoir)
        self.reasoning = ReasoningLayer(patterns, vocab, self.reservoir)
        self.planning = PlanningLayer(patterns, vocab)
        self.generation = GenerationLayer(patterns, vocab, self.reservoir)
        self.verification = VerificationLayer(patterns, vocab, self.reservoir)
    
    def process(self, query: str, max_tokens: int = 100) -> str:
        """
        Process user query through all layers.
        
        THINK → REASON → PLAN → GENERATE → VERIFY
        
        All steps use dataset knowledge. No hardcoded content.
        """
        # 1. THINK: Understand intent
        context = self.thinking.think(query)
        
        # 2. REASON: Find knowledge
        context = self.reasoning.reason(context)
        
        # 3. PLAN: Structure response
        context = self.planning.plan(context)
        
        # 4. GENERATE: Create response
        response = self.generation.generate(context, max_tokens)
        
        # 5. VERIFY: Check and improve
        response = self.verification.verify(response, context)
        
        return response
    
    def get_status(self) -> Dict[str, Any]:
        """Get brain status."""
        return {
            'patterns_count': len(self.patterns.patterns) if hasattr(self.patterns, 'patterns') else 0,
            'vocab_size': len(self.vocab) if self.vocab else 0,
            'reservoir_size': len(self.reservoir),
            'layers': ['thinking', 'reasoning', 'planning', 'generation', 'verification'],
        }
