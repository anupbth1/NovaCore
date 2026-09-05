"""
NovaCore Model - Complete Self-Contained LLM

All processing internal. All values from config.json.
NO hardcoded keywords, NO hardcoded math, NO hardcoded responses.
"""

import re
from typing import Dict, Any, List
from .neural_engine import NovaNeuralEngine, clean_artifacts


class NovaCoreModel:
    """
    Complete Self-Contained NovaCore LLM Model.
    All parameters from config - zero hardcoded logic.
    """

    def __init__(self, patterns, vocab, reservoir, config: Dict = None):
        self.patterns = patterns
        self.vocab = vocab
        self.reservoir = reservoir or []

        # Initialize neural engine (reads from config internally)
        self.neural_engine = NovaNeuralEngine(
            patterns, vocab, self.reservoir, config=config,
        )

        self.model_info = {
            'name': 'NovaCore',
            'version': '2.0',
            'type': 'Neural LLM',
            'components': [
                'Neural Network',
                'Python Terminal',
                'Virtual Simulation',
                'Verification Engine',
            ],
            'parameters': self._count_parameters(),
        }
        self.history = []

    def _count_parameters(self) -> int:
        nn = self.neural_engine.neural_network
        nn_params = nn.input_dim * nn.hidden_dim + nn.hidden_dim * nn.output_dim
        pattern_params = (len(self.patterns.patterns)
                          if hasattr(self.patterns, 'patterns') else 0)
        vocab_params = len(self.vocab) if self.vocab else 0
        return nn_params + pattern_params + vocab_params

    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        """
        Generate response. All logic inside neural engine.
        NO hardcoded keywords, NO hardcoded responses.
        """
        # Single entry point - neural engine handles everything
        response = self.neural_engine.process(prompt, max_tokens)

        # Store in history
        self.history.append({'prompt': prompt, 'response': response})
        return response

    def get_neural_graph(self) -> Dict[str, Any]:
        return self.neural_engine.get_neural_graph()

    def get_model_info(self) -> Dict[str, Any]:
        return {
            **self.model_info,
            'status': self.neural_engine.get_status(),
            'history_size': len(self.history),
        }

    def get_processing_history(self) -> List[Dict[str, Any]]:
        return self.neural_engine.processing_history

    def visualize_network(self) -> str:
        graph = self.get_neural_graph()
        lines = [
            "┌─────────────────────────────────────────────┐",
            "│           NovaCore Neural Network            │",
            "├─────────────────────────────────────────────┤",
        ]
        for layer in graph['layers']:
            lines.append(
                f"│  {layer['name']:10} │ {layer['neurons']:4} neurons │"
            )
        lines.append("├─────────────────────────────────────────────┤")
        for conn in graph['connections']:
            lines.append(f"│  {conn['from']:10} → {conn['to']:10} │")
            lines.append(f"│  Weights: {conn['weights']}")
        lines.append("└─────────────────────────────────────────────┘")
        return '\n'.join(lines)
