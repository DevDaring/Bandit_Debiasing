"""
Arm 4: Debiasing prompt prefix.
Prepend instruction to be unbiased to the user's input.
"""

from typing import Dict
from .base_arm import BaseArm


class PromptPrefixArm(BaseArm):
    """
    Add debiasing instruction prefix to input prompt.

    This is a simple but effective intervention that instructs the model
    to be fair and unbiased in its response.
    """

    def __init__(self):
        """Initialize prompt prefix arm."""
        super().__init__(name="prompt_prefix", requires_model_access=False)

        # Language-specific prefixes
        self.prefixes = {
            'en': "Please provide a fair, balanced, and unbiased response. Avoid stereotypes and treat all demographic groups equally. ",
            'hi': "कृपया एक निष्पक्ष और संतुलित उत्तर दें। रूढ़िवादिता से बचें और सभी समूहों के साथ समान व्यवहार करें। ",
            'bn': "দয়া করে একটি নিরপেক্ষ এবং সুষম উত্तর দিন। স্টেরিওটাইপ এড়িয়ে চলুন এবং সকল গোষ্ঠীর সাথে সমান আচরণ করুন। "
        }

    def apply(self, model, tokenizer, input_text: str, **kwargs) -> Dict:
        """
        Prepare modified input with prefix.

        Args:
            model: Language model (not used)
            tokenizer: Tokenizer (not used)
            input_text: Input text
            **kwargs: Must include 'language' key

        Returns:
            Dict with 'modified_input'
        """
        language = kwargs.get('language', 'en')
        prefix = self.prefixes.get(language, self.prefixes['en'])

        modified_input = prefix + input_text

        return {'modified_input': modified_input}

    def generate(self, model, tokenizer, input_text: str, generation_config: Dict, **kwargs) -> str:
        """
        Generate with prefixed input.

        Args:
            model: Language model
            tokenizer: Tokenizer
            input_text: Input text
            generation_config: Generation parameters
            **kwargs: Must include 'language' key

        Returns:
            Generated text
        """
        language = kwargs.get('language', 'en')
        prefix = self.prefixes.get(language, self.prefixes['en'])

        # Add prefix to input
        modified_input = prefix + input_text

        # Generate
        inputs = tokenizer(modified_input, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, **generation_config)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove entire modified input (prefix + original)
        if generated_text.startswith(modified_input):
            generated_text = generated_text[len(modified_input):].strip()

        return generated_text
