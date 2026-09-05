"""Novel Creative Content Engine for NovaCore.

Extracts creative structures (story arcs, poem patterns, dialogue flows)
from dataset at encoding time. At inference, COMPOSES novel content —
not copy-paste, but recombination of learned patterns.

Zero training. All from dataset knowledge.
"""
import re
import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional


class CreativePatternExtractor:
    """Extracts creative writing patterns from training data."""

    def __init__(self):
        self.story_openings = []     # "Once upon...", "There was..."
        self.story_middles = []      # plot developments
        self.story_endings = []      # resolutions
        self.poem_structures = []    # rhyme schemes, meter patterns
        self.metaphors = []          # X is Y (creative)
        self.descriptions = []       # vivid descriptions
        self.dialogue_patterns = []  # conversation flows
        self.emotional_arcs = []     # emotion progressions

    def extract_from_text(self, text: str):
        """Extract creative patterns from a text."""
        self._extract_story_structure(text)
        self._extract_poetry(text)
        self._extract_metaphors(text)
        self._extract_descriptions(text)
        self._extract_dialogue(text)

    def _extract_story_structure(self, text: str):
        """Extract story openings, middles, endings."""
        # Openings
        open_pat = [
            r'^([Oo]nce\s+(?:upon\s+)?(?:a\s+time|there\s+was)[^.]*\.)',
            r'^([Tt]here\s+(?:was|lived|once)[^.]*\.)',
            r'^([Ll]ong\s+(?:ago|before)[^.]*\.)',
            r'^([Ii]n\s+(?:a|the)\s+\w+\s+(?:land|world|kingdom|village|city)[^.]*\.)',
            r'^([Tt]he\s+\w+\s+(?:sun|moon|stars|wind|rain)[^.]*\.)',
        ]
        for pat in open_pat:
            for m in re.finditer(pat, text, re.MULTILINE):
                self.story_openings.append(m.group(1).strip())

        # Endings
        end_pat = [
            r'([^.]*and\s+(?:they\s+)?(?:lived\s+happily|were\s+happy|found\s+(?:peace|happiness|love|joy))[^.]*\.)',
            r'([^.]*the\s+end[^.]*\.)',
            r'([^.]*forever\s+(?:after|and\s+ever)[^.]*\.)',
            r'([^.]*never\s+(?:forgot|forgot\s+that)[^.]*\.)',
            r'([^.]*all\s+(?:was\s+well|turned\s+out\s+(?:well|fine|okay))[^.]*\.)',
        ]
        for pat in end_pat:
            for m in re.finditer(pat, text, re.IGNORECASE):
                self.story_endings.append(m.group(1).strip())

        # Middles (plot developments)
        mid_pat = [
            r'([^.]*(?:but|however|suddenly|then|however)[^.]*\.)',
            r'([^.]*(?:decided|chose|wanted|needed|tried|attempted)[^.]*\.)',
            r'([^.]*(?:discovered|found|learned|realized|understood)[^.]*\.)',
        ]
        for pat in mid_pat:
            for m in re.finditer(pat, text, re.IGNORECASE):
                s = m.group(1).strip()
                if len(s) > 20:
                    self.story_middles.append(s)

    def _extract_poetry(self, text: str):
        """Extract poetry patterns and structures."""
        lines = text.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # Check if line looks like poetry (short, rhythmic)
            words = line.split()
            if 3 <= len(words) <= 10:
                # Check for rhyme-like endings
                if re.match(r'.*(?:tion|sion|ment|ness|ful|less|ing|ed|ly|er|est|ize|ise)$', line, re.IGNORECASE):
                    self.poem_structures.append({
                        'line': line,
                        'position': 'end_rhyme',
                        'word_count': len(words)
                    })

            # Metaphor detection
            if re.match(r'.*\s+(?:is|are|was|were)\s+(?:a|an|the)\s+.*', line, re.IGNORECASE):
                if any(w in line.lower() for w in ['like', 'as', 'just as']):
                    self.poem_structures.append({
                        'line': line,
                        'position': 'simile',
                        'word_count': len(words)
                    })

    def _extract_metaphors(self, text: str):
        """Extract metaphors and creative comparisons."""
        patterns = [
            r'(\w+(?:\s+\w+){0,5})\s+(?:is|are|was|were)\s+(?:a|an)\s+([^.]+)',
            r'(\w+(?:\s+\w+){0,5})\s+(?:is|are|like|resembles)\s+like\s+([^.]+)',
            r'(?:imagine|picture|think\s+of)\s+(\w+(?:\s+\w+){0,5})\s+as\s+([^.]+)',
        ]
        for pat in patterns:
            for m in re.finditer(pat, text, re.IGNORECASE):
                if len(m.groups()) >= 2:
                    source = m.group(1).strip()
                    target = m.group(2).strip()
                    if len(source) > 2 and len(target) > 5:
                        # Filter non-creative (factual statements)
                        if not any(w in target.lower() for w in ['subset', 'type of', 'defined as']):
                            self.metaphors.append({
                                'source': source,
                                'target': target,
                                'full': m.group(0).strip()
                            })

    def _extract_descriptions(self, text: str):
        """Extract vivid descriptions."""
        patterns = [
            r'((?:the\s+)?\w+\s+(?:was|were|is|are)\s+(?:very\s+)?\w+ly\s+\w+[^.]*\.)',
            r'((?:bright|dark|soft|hard|warm|cold|loud|quiet|gentle|fierce|beautiful|ugly|ancient|modern)\s+\w+[^.]*\.)',
        ]
        for pat in patterns:
            for m in re.finditer(pat, text, re.IGNORECASE):
                desc = m.group(1).strip() if m.lastindex else m.group(0).strip()
                if len(desc) > 15 and len(desc) < 200:
                    self.descriptions.append(desc)

    def _extract_dialogue(self, text: str):
        """Extract dialogue patterns."""
        patterns = [
            r'"([^"]{5,100})"',
            r'"([^"]{5,100})"',
        ]
        for pat in patterns:
            for m in re.finditer(pat, text):
                self.dialogue_patterns.append(m.group(1).strip())

    def stats(self):
        return {
            'story_openings': len(self.story_openings),
            'story_middles': len(self.story_middles),
            'story_endings': len(self.story_endings),
            'poem_structures': len(self.poem_structures),
            'metaphors': len(self.metaphors),
            'descriptions': len(self.descriptions),
            'dialogue_patterns': len(self.dialogue_patterns),
        }


class CreativeEngine:
    """
    Novel creative content generation.
    Combines extracted patterns to create NEW content — not copy-paste.
    """

    def __init__(self):
        self.patterns = CreativePatternExtractor()

    def load_patterns(self, data_path: str):
        """Load extracted creative patterns from disk."""
        if os.path.exists(data_path):
            try:
                with open(data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.patterns.story_openings = data.get('story_openings', [])
                self.patterns.story_middles = data.get('story_middles', [])
                self.patterns.story_endings = data.get('story_endings', [])
                self.patterns.poem_structures = data.get('poem_structures', [])
                self.patterns.metaphors = data.get('metaphors', [])
                self.patterns.descriptions = data.get('descriptions', [])
                self.patterns.dialogue_patterns = data.get('dialogue_patterns', [])
            except Exception:
                pass

    def save_patterns(self, data_path: str):
        """Save extracted creative patterns to disk."""
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump({
                'story_openings': self.patterns.story_openings[:2000],
                'story_middles': self.patterns.story_middles[:5000],
                'story_endings': self.patterns.story_endings[:2000],
                'poem_structures': [p for p in self.patterns.poem_structures[:3000]],
                'metaphors': self.patterns.metaphors[:3000],
                'descriptions': self.patterns.descriptions[:5000],
                'dialogue_patterns': self.patterns.dialogue_patterns[:5000],
            }, f, ensure_ascii=False)

    def generate_story(self, topic: str = '', max_paragraphs: int = 3) -> str:
        """Generate a novel story by composing patterns."""
        paragraphs = []

        # Opening
        if self.patterns.story_openings:
            opening = random.choice(self.patterns.story_openings)
        else:
            opening = "Once upon a time, there was a curious spirit."

        # Add topic context
        if topic:
            topic_lower = topic.lower()
            if 'about' in topic_lower:
                topic = topic_lower.replace('about', '').strip()
            opening = f"{opening.rstrip('.')} in a world of {topic}."

        paragraphs.append(opening)

        # Middle paragraphs (plot development)
        used_middles = set()
        middle_pool = [m for m in self.patterns.story_middles if len(m) > 20]
        random.shuffle(middle_pool)
        middle_idx = 0
        
        for i in range(max_paragraphs - 1):
            while middle_idx < len(middle_pool) and middle_pool[middle_idx] in used_middles:
                middle_idx += 1
            if middle_idx < len(middle_pool):
                middle = middle_pool[middle_idx]
                used_middles.add(middle)
                paragraphs.append(middle)
                middle_idx += 1
            elif self.patterns.descriptions:
                desc_pool = [d for d in self.patterns.descriptions if len(d) > 20]
                if desc_pool:
                    desc = random.choice(desc_pool)
                    paragraphs.append(desc)

        # Ending
        if self.patterns.story_endings:
            ending = random.choice(self.patterns.story_endings)
        else:
            ending = "And from that day forward, everything changed for the better."
        paragraphs.append(ending)

        return '\n\n'.join(paragraphs)

    def generate_poem(self, topic: str = '', lines: int = 8) -> str:
        """Generate a novel poem by composing patterns."""
        poem_lines = []
        
        # Build pools
        metaphor_pool = [m['full'] for m in self.patterns.metaphors if len(m['full']) > 10]
        desc_pool = [d for d in self.patterns.descriptions if len(d) > 15]
        struct_pool = [p['line'] for p in self.patterns.poem_structures if len(p['line']) > 10]
        
        # Topic filtering
        if topic:
            topic_words = set(topic.lower().split())
            metaphor_pool = [m for m in metaphor_pool if topic_words & set(m.lower().split())] or metaphor_pool
            desc_pool = [d for d in desc_pool if topic_words & set(d.lower().split())] or desc_pool
            struct_pool = [s for s in struct_pool if topic_words & set(s.lower().split())] or struct_pool

        all_pool = metaphor_pool + desc_pool + struct_pool
        
        if not all_pool:
            all_pool = [
                f"The {topic} whispers soft and low,",
                "A gentle breeze begins to blow.",
                "Through fields of green and skies so blue,",
                "The world reveals its wonders true.",
            ]

        # Generate unique lines
        used = set()
        while len(poem_lines) < lines and all_pool:
            random.shuffle(all_pool)
            for line in all_pool:
                clean = line.strip().rstrip('.')
                if clean and clean not in used:
                    # Trim to poetic length
                    words = clean.split()
                    if len(words) > 10:
                        clean = ' '.join(words[:10])
                    if len(words) < 3:
                        continue
                    used.add(clean)
                    poem_lines.append(clean.capitalize())
                    break
            else:
                break

        return '\n'.join(poem_lines) if poem_lines else '\n'.join(all_pool[:lines])

    def detect_creative_intent(self, query: str) -> str:
        """Detect if query wants creative content and what type."""
        q = query.lower()

        # Story intents
        if any(w in q for w in ['story', 'tale', 'narrative', 'once upon']):
            return 'story'
        if any(w in q for w in ['fairy tale', 'fable', 'legend']):
            return 'story'

        # Poetry intents
        if any(w in q for w in ['poem', 'poetry', 'verse', 'haiku', 'rhyme', 'sonnet']):
            return 'poem'
        if any(w in q for w in ['limerick', 'ode', 'ballad', 'stanza']):
            return 'poem'

        # Dialogue intents
        if any(w in q for w in ['dialogue', 'conversation', 'chat between']):
            return 'dialogue'

        # Description intents
        if any(w in q for w in ['describe', 'depict', 'portray', 'paint a picture']):
            return 'description'

        return 'none'

    def generate(self, query: str, reservoir=None) -> Optional[str]:
        """
        Generate novel creative content based on query intent.
        Returns None if not a creative request.
        """
        intent = self.detect_creative_intent(query)

        if intent == 'story':
            # Extract topic from query
            topic = re.sub(r'(write|tell|create|make|give)\s+(me\s+)?(a\s+)?(story|tale|narrative)\s*(about)?\s*', '', query, flags=re.IGNORECASE).strip()
            return self.generate_story(topic, max_paragraphs=3)

        elif intent == 'poem':
            topic = re.sub(r'(write|create|make|give)\s+(me\s+)?(a\s+)?(poem|poetry|verse|haiku)\s*(about)?\s*', '', query, flags=re.IGNORECASE).strip()
            return self.generate_poem(topic, lines=8)

        return None
