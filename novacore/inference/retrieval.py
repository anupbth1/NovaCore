"""
Retrieval-Based Text Generator for NovaCore

Uses reservoir samples to find similar training examples and generate
context-aware responses. This is the key improvement for better output.

Architecture:
1. Store reservoir samples with embeddings
2. At query time, find similar examples using cosine similarity
3. Extract response patterns from similar examples
4. Generate context-aware responses using pattern mixing
"""
import numpy as np
import re
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Optional


class RetrievalGenerator:
    """
    Generates text by retrieving similar training examples and mixing their patterns.
    
    Key improvements over basic n-gram:
    1. Context-aware: finds examples similar to the user's query
    2. Response-focused: extracts the response part, not just any text
    3. Pattern mixing: combines patterns from multiple similar examples
    4. Quality scoring: ranks responses by relevance and coherence
    """
    
    def __init__(self, vocab, processor, reservoir_samples=None, patterns=None):
        self.vocab = vocab
        self.processor = processor
        self.reservoir = reservoir_samples or []
        self.patterns = patterns or {}
        
        # Build index for fast retrieval
        self.sample_embeddings = []
        self.sample_texts = []
        self.sample_responses = []
        self.sample_instructions = []
        
        # Precompute embeddings for reservoir samples
        self._build_index()
        
        # Build response patterns from training data
        self.response_patterns = self._extract_response_patterns()
        
        # Common response starters from training data
        self.response_starters = self._extract_response_starters()
        
    def _build_index(self):
        """Build embedding index for reservoir samples."""
        if not self.reservoir:
            return
            
        for text in self.reservoir:
            if not isinstance(text, str) or not text.strip():
                continue
                
            # Store the full text
            self.sample_texts.append(text)
            
            # Extract instruction and response parts
            instruction, response = self._extract_instruction_response(text)
            self.sample_instructions.append(instruction)
            self.sample_responses.append(response)
            
            # Compute embedding for the instruction part
            if instruction:
                embedding = self.processor.text_to_vector(instruction, normalize=True)
                self.sample_embeddings.append(embedding)
            else:
                # Use first 100 chars as fallback
                embedding = self.processor.text_to_vector(text[:100], normalize=True)
                self.sample_embeddings.append(embeding)
    
    def _extract_instruction_response(self, text):
        """Extract instruction and response from tagged text."""
        # Look for <instruction>...</instruction> and <answer>...</answer>
        instruction_match = re.search(r'<instruction>(.*?)</instruction>', text, re.DOTALL)
        answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        
        instruction = instruction_match.group(1).strip() if instruction_match else ""
        response = answer_match.group(1).strip() if answer_match else ""
        
        # If no tags, try to split by common patterns
        if not instruction and not response:
            # Try splitting by "###" or other markers
            parts = re.split(r'###\s*(Instruction|Response|Answer):', text)
            if len(parts) >= 3:
                instruction = parts[2].strip()  # After "### Instruction:"
                response = parts[4].strip() if len(parts) > 4 else ""  # After "### Response:"
            else:
                # Use the whole text as both instruction and response
                instruction = text[:200]
                response = text
        
        # Clean up the instruction and response
        instruction = self._clean_artifacts(instruction)
        response = self._clean_artifacts(response)
        
        return instruction, response
    
    def _clean_artifacts(self, text):
        """Remove common artifacts from training data."""
        if not text:
            return ""
        
        # Remove common prefixes and suffixes
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
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _extract_response_patterns(self):
        """Extract common response patterns from training data."""
        patterns = {
            'greetings': [],
            'explanations': [],
            'instructions': [],
            'examples': [],
            'descriptions': [],
            'advice': [],
            'lists': [],
            'definitions': [],
        }
        
        for response in self.sample_responses:
            if not response:
                continue
                
            response_lower = response.lower()
            
            # Categorize by keywords
            if any(word in response_lower for word in ['hello', 'hi', 'hey', 'welcome']):
                patterns['greetings'].append(response)
            elif any(word in response_lower for word in ['because', 'reason', 'explanation', 'why']):
                patterns['explanations'].append(response)
            elif any(word in response_lower for word in ['step', 'follow', 'how to', 'guide']):
                patterns['instructions'].append(response)
            elif any(word in response_lower for word in ['example', 'for instance', 'such as']):
                patterns['examples'].append(response)
            elif any(word in response_lower for word in ['is', 'are', 'describe', 'about']):
                patterns['descriptions'].append(response)
            elif any(word in response_lower for word in ['should', 'recommend', 'advice', 'tip']):
                patterns['advice'].append(response)
            elif any(word in response_lower for word in ['1.', '2.', '3.', 'first', 'second', 'third']):
                patterns['lists'].append(response)
            elif any(word in response_lower for word in ['definition', 'means', 'refers to']):
                patterns['definitions'].append(response)
        
        return patterns
    
    def _extract_response_starters(self):
        """Extract common response starters from training data."""
        starters = Counter()
        
        for response in self.sample_responses:
            if not response:
                continue
                
            # Get first few words
            words = response.split()[:5]
            if words:
                starter = ' '.join(words)
                starters[starter] += 1
        
        # Return top starters
        return [starter for starter, count in starters.most_common(20)]
    
    def retrieve_similar(self, query: str, k: int = 5) -> List[Tuple[str, float, str]]:
        """
        Retrieve similar training examples to the query.
        
        Returns list of (instruction, similarity_score, response)
        """
        if not self.sample_embeddings:
            return []
        
        # Compute query embedding
        query_embedding = self.processor.text_to_vector(query, normalize=True)
        
        # Compute similarities
        similarities = []
        for i, emb in enumerate(self.sample_embeddings):
            if emb is not None and query_embedding is not None:
                # Cosine similarity (already normalized)
                sim = np.dot(query_embedding, emb)
                similarities.append((self.sample_instructions[i], sim, self.sample_responses[i]))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:k]
    
    def classify_query(self, query: str) -> str:
        """Classify the type of query to determine response style."""
        query_lower = query.lower()
        
        # Greeting patterns
        if re.match(r'^(hi|hello|hey|howdy|greetings|good\s+(morning|afternoon|evening))', query_lower):
            return 'greeting'
        
        # Question patterns
        if re.match(r'^(what|who|where|when|why|how|can|could|do|does|is|are|will|would)', query_lower):
            return 'question'
        
        # Command patterns
        if re.match(r'^(tell|explain|describe|list|give|show|write|create|make|help)', query_lower):
            return 'command'
        
        # Story/creative patterns
        if re.match(r'(story|tell.*story|write.*story|creative|poem|poetry)', query_lower):
            return 'creative'
        
        # Default
        return 'general'
    
    def generate_response(self, query: str, max_tokens: int = 80, temperature: float = 0.8) -> str:
        """
        Generate a response to the query using retrieval + pattern mixing.
        
        Steps:
        1. Classify the query type
        2. Retrieve similar examples
        3. Extract relevant patterns
        4. Generate response using pattern mixing
        5. Apply quality checks
        """
        if not self.sample_responses:
            return self._fallback_response(query)
        
        # Step 1: Classify query
        query_type = self.classify_query(query)
        
        # Step 2: Retrieve similar examples
        similar_examples = self.retrieve_similar(query, k=5)
        
        if not similar_examples:
            return self._fallback_response(query)
        
        # Step 3: Extract relevant patterns based on query type
        if query_type == 'greeting':
            response = self._generate_greeting_response(query, similar_examples)
        elif query_type == 'question':
            response = self._generate_question_response(query, similar_examples)
        elif query_type == 'command':
            response = self._generate_command_response(query, similar_examples)
        elif query_type == 'creative':
            response = self._generate_creative_response(query, similar_examples)
        else:
            response = self._generate_general_response(query, similar_examples)
        
        # Step 4: Apply quality checks
        response = self._apply_quality_checks(response, query)
        
        # Step 5: Limit length
        words = response.split()
        if len(words) > max_tokens:
            response = ' '.join(words[:max_tokens])
        
        return response
    
    def _generate_greeting_response(self, query: str, examples: List) -> str:
        """Generate a greeting response."""
        # Use actual greeting patterns from training data
        greeting_patterns = [
            "Hello! How can I help you today?",
            "Hi there! What can I do for you?",
            "Hey! How are you doing?",
            "Greetings! How may I assist you?",
            "Hello! What would you like to know?",
        ]
        
        # Try to find similar greeting responses
        for instruction, score, response in examples:
            if score > 0.4 and response and len(response.split()) > 3:
                # Clean up the response
                response = self._clean_response(response)
                if response and len(response) > 10:
                    return response
        
        # Use query-specific greeting
        query_lower = query.lower().strip()
        if 'hello' in query_lower or 'hi' in query_lower or 'hey' in query_lower:
            # Make it more natural
            return "Hello! I'm here to help you. What would you like to know?"
        
        # Fallback to common greeting
        return greeting_patterns[0]
    
    def _generate_question_response(self, query: str, examples: List) -> str:
        """Generate a response to a question."""
        # Look for similar question-answer pairs
        for instruction, score, response in examples:
            if score > 0.4 and response:
                # Check if the response actually answers the question
                if self._response_relevance(query, response) > 0.3:
                    return self._clean_response(response)
        
        # Extract the main topic from the query
        topic = self._extract_topic(query)
        
        # Generate a response based on the topic
        if topic:
            return f"Based on the training data, here's what I can tell you about {topic}: " + \
                   self._find_relevant_info(topic, examples)
        
        return self._fallback_response(query)
    
    def _generate_command_response(self, query: str, examples: List) -> str:
        """Generate a response to a command."""
        # Look for similar instruction-following examples
        for instruction, score, response in examples:
            if score > 0.4 and response:
                # Check if it's a list or step-by-step response
                if any(marker in response for marker in ['1.', '2.', '3.', 'Step', 'First']):
                    return self._clean_response(response)
        
        # Generate a step-by-step response
        return self._generate_step_response(query, examples)
    
    def _generate_creative_response(self, query: str, examples: List) -> str:
        """Generate a creative response like a story or poem."""
        # Look for creative examples in training data
        creative_examples = [ex for ex in examples if ex[1] > 0.3 and ex[2]]
        
        if creative_examples:
            # Mix patterns from multiple creative examples
            return self._mix_creative_patterns(query, creative_examples)
        
        return self._fallback_response(query)
    
    def _generate_general_response(self, query: str, examples: List) -> str:
        """Generate a general response."""
        # Use the most similar example
        if examples:
            instruction, score, response = examples[0]
            if score > 0.3 and response:
                return self._clean_response(response)
        
        return self._fallback_response(query)
    
    def _response_relevance(self, query: str, response: str) -> float:
        """Check how relevant a response is to the query."""
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        if not query_words or not response_words:
            return 0.0
        
        # Jaccard similarity
        intersection = query_words & response_words
        union = query_words | response_words
        
        return len(intersection) / len(union) if union else 0.0
    
    def _extract_topic(self, query: str) -> str:
        """Extract the main topic from a query."""
        # Remove common question words
        stop_words = {'what', 'who', 'where', 'when', 'why', 'how', 'is', 'are', 'do', 'does', 
                      'can', 'could', 'would', 'should', 'tell', 'me', 'about', 'the', 'a', 'an'}
        
        words = query.lower().split()
        topic_words = [w for w in words if w not in stop_words and len(w) > 2]
        
        if topic_words:
            return ' '.join(topic_words[:3])
        return ""
    
    def _find_relevant_info(self, topic: str, examples: List) -> str:
        """Find relevant information about a topic from examples."""
        for instruction, score, response in examples:
            if topic.lower() in instruction.lower() and response:
                return self._clean_response(response)
        
        return f"I don't have specific information about {topic} in my training data."
    
    def _generate_step_response(self, query: str, examples: List) -> str:
        """Generate a step-by-step response."""
        # Extract the action from the query
        action = self._extract_topic(query)
        
        if action:
            return f"Here are the steps to {action}:\n" + \
                   "1. First, understand the requirements\n" + \
                   "2. Then, plan your approach\n" + \
                   "3. Finally, execute step by step\n" + \
                   "Would you like me to elaborate on any specific step?"
        
        return "I'd be happy to help you with that. Could you provide more details?"
    
    def _mix_creative_patterns(self, query: str, examples: List) -> str:
        """Mix patterns from multiple creative examples."""
        if not examples:
            return self._fallback_response(query)
        
        # Take the best example and adapt it
        instruction, score, response = examples[0]
        
        if response:
            # Clean and adapt the response
            response = self._clean_response(response)
            
            # Make it more relevant to the query
            if query.lower() not in response.lower():
                # Add query context at the beginning
                query_words = query.split()[:5]
                prefix = ' '.join(query_words)
                response = f"About {prefix}: {response}"
            
            return response
        
        return self._fallback_response(query)
    
    def _clean_response(self, response: str) -> str:
        """Clean up a response by removing artifacts and formatting."""
        if not response:
            return ""
        
        # Remove common artifacts
        artifacts = [
            r'<instruction>.*?</instruction>',
            r'<answer>.*?</answer>',
            r'<text>.*?</text>',
            r'###\s*Instruction:.*?###\s*Response:',
            r'Below is an instruction.*?###\s*Response:',
        ]
        
        for pattern in artifacts:
            response = re.sub(pattern, '', response, flags=re.DOTALL)
        
        # Clean up whitespace
        response = re.sub(r'\s+', ' ', response).strip()
        
        # Remove leading/trailing punctuation that doesn't make sense
        response = response.strip('.,;:')
        
        return response
    
    def _apply_quality_checks(self, response: str, query: str) -> str:
        """Apply quality checks to the response."""
        if not response:
            return self._fallback_response(query)
        
        # Check if response is too short
        if len(response.split()) < 3:
            # Try to extend it
            response = self._extend_response(response, query)
        
        # Check if response is too long
        if len(response.split()) > 100:
            # Truncate to reasonable length
            words = response.split()[:100]
            response = ' '.join(words)
        
        # Check if response is coherent (basic check)
        if not self._is_coherent(response):
            # Try to fix it
            response = self._make_coherent(response)
        
        return response
    
    def _extend_response(self, response: str, query: str) -> str:
        """Extend a short response with more context."""
        topic = self._extract_topic(query)
        
        if topic:
            extensions = [
                f"This is related to {topic}.",
                f"Here's what I know about {topic}.",
                f"Let me tell you about {topic}.",
            ]
            
            # Pick the most appropriate extension
            for ext in extensions:
                if ext not in response:
                    return f"{response} {ext}"
        
        return response
    
    def _is_coherent(self, response: str) -> bool:
        """Basic coherence check."""
        words = response.split()
        
        # Check if it has reasonable length
        if len(words) < 2:
            return False
        
        # Check if it doesn't have too many repeated words
        word_counts = Counter(words)
        max_repeat = max(word_counts.values()) if word_counts else 0
        if max_repeat > len(words) * 0.3:
            return False
        
        return True
    
    def _make_coherent(self, response: str) -> str:
        """Try to make a response more coherent."""
        # Remove excessive repetition
        words = response.split()
        if not words:
            return response
        
        # Remove consecutive duplicates
        cleaned = [words[0]]
        for word in words[1:]:
            if word != cleaned[-1]:
                cleaned.append(word)
        
        return ' '.join(cleaned)
    
    def _fallback_response(self, query: str) -> str:
        """Generate a fallback response when no good match is found."""
        # Try to find any relevant pattern
        for pattern_type, patterns in self.response_patterns.items():
            if patterns:
                # Pick a random pattern from this type
                import random
                pattern = random.choice(patterns)
                if pattern and len(pattern) > 10:
                    return self._clean_response(pattern)
        
        # Try to find any response from training data
        if self.sample_responses:
            import random
            # Get a few random responses and pick the best one
            random_responses = random.sample(self.sample_responses, min(5, len(self.sample_responses)))
            for response in random_responses:
                if response and len(response) > 20:
                    # Clean and return
                    cleaned = self._clean_response(response)
                    if cleaned:
                        return cleaned
        
        # Generate a more helpful fallback based on query type
        query_lower = query.lower()
        
        # Check for common patterns
        if any(word in query_lower for word in ['what', 'who', 'where', 'when', 'why', 'how']):
            return "That's an interesting question. Based on my training data, I can provide some information about that topic."
        elif any(word in query_lower for word in ['tell', 'explain', 'describe', 'list']):
            return "I'd be happy to help explain that. Let me share what I know from my training."
        elif any(word in query_lower for word in ['write', 'create', 'make', 'generate']):
            return "I can help you with that. Let me provide some relevant information from my knowledge base."
        else:
            return "I understand your request. Let me provide some relevant information from my training data."
