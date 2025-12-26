"""
Content Enhancer v3 - Accurate Content Generation
==================================================
Hybrid approach prioritizing EXTRACTION over GENERATION for accuracy.

Key improvements:
1. Extraction-first approach (more accurate)
2. Context-aware AI prompts (when AI is used)
3. Smart sentence scoring and selection
4. Better fallback mechanisms
5. Improved text preprocessing

Modes:
- Extraction-only (default): Fast, accurate, no AI needed
- AI-enhanced: Uses Flan-T5 for additional elaboration
"""
import re
import math
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter
from pathlib import Path


@dataclass
class GeneratedContent:
    """Container for all generated/extracted content"""
    original: str
    simplified_explanation: str
    key_takeaways: List[str]
    elaboration: str
    examples: List[str]
    faq: List[Dict[str, str]]
    vocabulary: List[Dict[str, str]]
    
    def to_dict(self) -> Dict:
        return {
            'simplified': self.simplified_explanation,
            'takeaways': self.key_takeaways,
            'elaboration': self.elaboration,
            'examples': self.examples,
            'faq': self.faq,
            'vocabulary': self.vocabulary
        }


class ContentEnhancer:
    """
    Content enhancement using extraction-first approach.
    
    This class extracts and structures content from the source text,
    optionally using AI models for additional elaboration.
    
    The extraction-first approach is more accurate because it uses
    the actual content rather than generating potentially irrelevant text.
    """
    
    # Stopwords for filtering
    STOPWORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
        'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them', 'their',
        'we', 'us', 'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her',
        'i', 'me', 'my', 'as', 'if', 'so', 'than', 'such', 'when', 'where',
        'which', 'who', 'what', 'how', 'all', 'each', 'every', 'both', 'few',
        'more', 'most', 'other', 'some', 'any', 'no', 'not', 'only', 'same',
        'just', 'also', 'very', 'even', 'back', 'now', 'well', 'also', 'just',
        'like', 'really', 'want', 'going', 'something', 'actually', 'thing',
        'things', 'way', 'yeah', 'yes', 'okay', 'ok', 'um', 'uh', 'basically',
    }
    
    # Important sentence indicators (for scoring)
    IMPORTANCE_SIGNALS = {
        'high': [
            'in conclusion', 'to summarize', 'the main point', 'most importantly',
            'the key is', 'crucial', 'essential', 'fundamental', 'primarily',
            'significantly', 'notably', 'the purpose', 'the goal', 'therefore',
            'as a result', 'in summary', 'ultimately', 'in essence', 'the bottom line',
            'this means', 'this shows', 'this demonstrates', 'importantly',
        ],
        'medium': [
            'for example', 'for instance', 'such as', 'including', 'specifically',
            'first', 'second', 'third', 'finally', 'additionally', 'moreover',
            'furthermore', 'however', 'because', 'since', 'due to', 'leads to',
            'according to', 'research shows', 'studies indicate', 'in other words',
        ],
        'definition': [
            'is defined as', 'refers to', 'means that', 'is called', 'known as',
            'is a', 'is an', 'are a', 'are an', 'can be described as',
        ],
        'example': [
            'for example', 'for instance', 'such as', 'like', 'including',
            'consider', 'imagine', 'suppose', 'take the case of', 'as an example',
        ],
    }
    
    def __init__(self, use_ai: bool = False, model_name: str = "google/flan-t5-base"):
        """
        Initialize content enhancer
        
        Args:
            use_ai: Whether to use AI model for additional generation
            model_name: HuggingFace model to use if AI enabled
        """
        self.use_ai = use_ai
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self.cache_dir = Path.home() / ".cache" / "echonotes"
        self._model_loaded = False
    
    def _load_model(self) -> bool:
        """Load AI model if needed and not already loaded"""
        if not self.use_ai:
            return False
        
        if self._model_loaded:
            return True
        
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            import torch
            
            print(f"[ContentEnhancer] Loading model: {self.model_name}")
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            ).to(device)
            
            self.device = device
            self._model_loaded = True
            print(f"[ContentEnhancer] Model loaded on {device}")
            return True
            
        except ImportError:
            print("[ContentEnhancer] AI disabled - transformers/torch not installed")
            self.use_ai = False
            return False
        except Exception as e:
            print(f"[ContentEnhancer] Error loading model: {e}")
            self.use_ai = False
            return False
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize input text"""
        lines = text.split('\n')
        clean_lines = []
        
        # Skip patterns for metadata
        skip_patterns = [
            r'^={2,}', r'^-{2,}', r'^\[.*\]$',
            r'^transcript:', r'^audio:', r'^duration:',
            r'^confidence:', r'^words:', r'^timestamps:',
            r'^generated:', r'^echonotes', r'^recording',
        ]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip metadata
            skip = False
            for pattern in skip_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    skip = True
                    break
            if skip:
                continue
            
            # Remove timestamp markers
            line = re.sub(r'\[\d{1,2}:\d{2}(?::\d{2})?\]', '', line).strip()
            
            if len(line) > 5:
                clean_lines.append(line)
        
        text = ' '.join(clean_lines)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        
        return text.strip()
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Protect abbreviations
        text = re.sub(r'\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|e\.g|i\.e)\.\s+', r'\1<DOT> ', text)
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Restore periods
        sentences = [s.replace('<DOT>', '.').strip() for s in sentences]
        
        # Filter by length
        return [s for s in sentences if len(s) >= 20 and len(s.split()) >= 4]
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        return [w for w in words if w not in self.STOPWORDS]
    
    def _score_sentence(self, sentence: str, position: int, total: int) -> float:
        """Score sentence by importance using multiple signals"""
        score = 0.0
        sent_lower = sentence.lower()
        
        # Position score (first/last sentences are important)
        rel_pos = position / max(1, total - 1)
        if rel_pos < 0.15:
            score += 0.3
        elif rel_pos > 0.85:
            score += 0.2
        
        # Length score (prefer medium length)
        words = sentence.split()
        if 12 <= len(words) <= 35:
            score += 0.2
        elif len(words) < 8 or len(words) > 50:
            score -= 0.1
        
        # Importance signal score
        for phrase in self.IMPORTANCE_SIGNALS['high']:
            if phrase in sent_lower:
                score += 0.4
                break
        
        for phrase in self.IMPORTANCE_SIGNALS['medium']:
            if phrase in sent_lower:
                score += 0.2
                break
        
        # Definition pattern score
        for phrase in self.IMPORTANCE_SIGNALS['definition']:
            if phrase in sent_lower:
                score += 0.3
                break
        
        return score
    
    def _extract_key_sentences(self, sentences: List[str], max_count: int = 5) -> List[str]:
        """Extract most important sentences"""
        if len(sentences) <= max_count:
            return sentences
        
        # Score all sentences
        scored = []
        for i, sent in enumerate(sentences):
            score = self._score_sentence(sent, i, len(sentences))
            scored.append((i, sent, score))
        
        # Sort by score and select top
        scored.sort(key=lambda x: -x[2])
        selected = scored[:max_count]
        
        # Sort by position for coherent reading
        selected.sort(key=lambda x: x[0])
        
        return [sent for _, sent, _ in selected]
    
    def _extract_examples(self, sentences: List[str], max_count: int = 3) -> List[str]:
        """Extract example sentences from text"""
        examples = []
        
        for sent in sentences:
            sent_lower = sent.lower()
            for phrase in self.IMPORTANCE_SIGNALS['example']:
                if phrase in sent_lower:
                    # Clean and add
                    example = sent.strip()
                    if example not in examples and len(example) > 20:
                        examples.append(example)
                        break
            
            if len(examples) >= max_count:
                break
        
        return examples
    
    def _extract_definitions(self, sentences: List[str]) -> List[Dict[str, str]]:
        """Extract term definitions from text"""
        definitions = []
        
        # Patterns for definitions
        patterns = [
            r'(\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+(?:is|are)\s+(?:a|an|the)?\s*(.+?)(?:\.|,)',
            r'(\b[A-Z][a-zA-Z]+(?:\s+[a-zA-Z]+)*)\s+refers to\s+(.+?)(?:\.|,)',
            r'(\b[A-Z][a-zA-Z]+(?:\s+[a-zA-Z]+)*)\s+means\s+(.+?)(?:\.|,)',
            r'(\b[A-Z][a-zA-Z]+(?:\s+[a-zA-Z]+)*)\s+can be defined as\s+(.+?)(?:\.|,)',
        ]
        
        for sent in sentences:
            for pattern in patterns:
                matches = re.findall(pattern, sent)
                for term, definition in matches:
                    term = term.strip()
                    definition = definition.strip()
                    
                    # Validate
                    if (len(term) > 2 and len(definition) > 10 and 
                        len(definition) < 200 and
                        term.lower() not in self.STOPWORDS):
                        
                        # Check not already added
                        if not any(d['term'].lower() == term.lower() for d in definitions):
                            definitions.append({
                                'term': term,
                                'meaning': definition[0].upper() + definition[1:] + '.' if not definition.endswith('.') else definition[0].upper() + definition[1:]
                            })
        
        return definitions[:10]
    
    def _extract_vocabulary(self, text: str, sentences: List[str], max_terms: int = 8) -> List[Dict[str, str]]:
        """Extract key vocabulary terms with context"""
        # First try to find explicit definitions
        definitions = self._extract_definitions(sentences)
        
        # Find additional key terms
        text_lower = text.lower()
        
        # Extract capitalized terms (likely important)
        terms = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
        term_freq = Counter(terms)
        
        # Also extract technical terms
        technical_patterns = [
            r'\b([a-z]+(?:tion|ment|ity|ism|ology|graphy))\b',
            r'\b([a-z]+\s+(?:system|method|process|model|theory|approach))\b',
        ]
        
        for pattern in technical_patterns:
            matches = re.findall(pattern, text_lower)
            for term in matches:
                term_freq[term.title()] += 1
        
        # Build vocabulary list
        vocabulary = list(definitions)  # Start with found definitions
        seen_terms = {d['term'].lower() for d in definitions}
        
        # Add high-frequency terms without definitions
        for term, freq in term_freq.most_common(20):
            if term.lower() in seen_terms or term.lower() in self.STOPWORDS:
                continue
            if len(term) < 3 or freq < 2:
                continue
            
            # Find context sentence for this term
            context = ""
            for sent in sentences:
                if term.lower() in sent.lower():
                    context = sent
                    break
            
            if context:
                # Truncate if too long
                if len(context) > 150:
                    context = context[:150].rsplit(' ', 1)[0] + '...'
                
                vocabulary.append({
                    'term': term,
                    'meaning': context
                })
                seen_terms.add(term.lower())
        
        return vocabulary[:max_terms]
    
    def _generate_faq(self, text: str, sentences: List[str], concepts: List[str]) -> List[Dict[str, str]]:
        """Generate FAQ based on content extraction"""
        faq = []
        
        # Q1: What is the main topic?
        if sentences:
            # Use first substantial sentence as answer
            main_answer = sentences[0] if len(sentences[0]) < 200 else sentences[0][:200] + '...'
            main_topic = concepts[0] if concepts else "this topic"
            faq.append({
                'q': f"What is {main_topic}?",
                'a': main_answer
            })
        
        # Q2: Find a "why" answer
        for sent in sentences:
            sent_lower = sent.lower()
            if any(w in sent_lower for w in ['because', 'reason', 'purpose', 'important', 'significant']):
                faq.append({
                    'q': "Why is this important?",
                    'a': sent if len(sent) < 200 else sent[:200] + '...'
                })
                break
        
        # Q3: Find a "how" answer
        for sent in sentences:
            sent_lower = sent.lower()
            if any(w in sent_lower for w in ['by', 'through', 'using', 'process', 'method', 'step']):
                faq.append({
                    'q': "How does this work?",
                    'a': sent if len(sent) < 200 else sent[:200] + '...'
                })
                break
        
        # Q4: Key features/benefits
        for sent in sentences:
            sent_lower = sent.lower()
            if any(w in sent_lower for w in ['feature', 'benefit', 'advantage', 'allows', 'enables', 'provides']):
                faq.append({
                    'q': "What are the key benefits?",
                    'a': sent if len(sent) < 200 else sent[:200] + '...'
                })
                break
        
        # Q5: Example
        for sent in sentences:
            sent_lower = sent.lower()
            if any(w in sent_lower for w in ['example', 'instance', 'such as', 'like']):
                faq.append({
                    'q': "Can you give an example?",
                    'a': sent if len(sent) < 200 else sent[:200] + '...'
                })
                break
        
        return faq[:5]
    
    def _ai_generate(self, prompt: str, max_length: int = 150) -> str:
        """Generate text using AI model (if available)"""
        if not self._load_model():
            return ""
        
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(self.device)
            
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                temperature=0.3,
                do_sample=True,
                early_stopping=True
            )
            
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return result.strip()
            
        except Exception as e:
            print(f"[ContentEnhancer] AI generation error: {e}")
            return ""
    
    def simplify(self, text: str) -> str:
        """Create simplified explanation from text"""
        clean_text = self._clean_text(text)
        sentences = self._split_sentences(clean_text)
        
        if not sentences:
            return ""
        
        # Extract key sentences for summary
        key_sents = self._extract_key_sentences(sentences, max_count=3)
        
        if key_sents:
            # Join into coherent summary
            summary = ' '.join(key_sents)
            
            # If AI available, try to simplify further
            if self.use_ai and len(summary) > 50:
                ai_result = self._ai_generate(
                    f"Explain this simply in 2-3 sentences: {summary[:500]}",
                    max_length=150
                )
                if ai_result and len(ai_result) > 30:
                    return ai_result
            
            return summary
        
        return sentences[0] if sentences else ""
    
    def generate_takeaways(self, text: str, num_points: int = 5) -> List[str]:
        """Extract key takeaways from text"""
        clean_text = self._clean_text(text)
        sentences = self._split_sentences(clean_text)
        
        if not sentences:
            return []
        
        # Score and extract best sentences
        key_sents = self._extract_key_sentences(sentences, max_count=num_points + 2)
        
        takeaways = []
        for sent in key_sents:
            # Clean up the sentence
            takeaway = sent.strip()
            if takeaway and len(takeaway) > 15:
                # Ensure proper formatting
                if not takeaway[0].isupper():
                    takeaway = takeaway[0].upper() + takeaway[1:]
                if not takeaway.endswith(('.', '!', '?')):
                    takeaway += '.'
                takeaways.append(takeaway)
        
        return takeaways[:num_points]
    
    def elaborate(self, text: str) -> str:
        """Create elaboration of the content"""
        clean_text = self._clean_text(text)
        sentences = self._split_sentences(clean_text)
        
        if not sentences:
            return ""
        
        # Find sentences with definitions or explanations
        elaboration_sents = []
        for sent in sentences:
            sent_lower = sent.lower()
            if any(phrase in sent_lower for phrase in self.IMPORTANCE_SIGNALS['definition']):
                elaboration_sents.append(sent)
            elif any(phrase in sent_lower for phrase in ['means', 'explains', 'describes', 'shows']):
                elaboration_sents.append(sent)
        
        if elaboration_sents:
            return ' '.join(elaboration_sents[:3])
        
        # Fallback: return first few sentences
        return ' '.join(sentences[:2])
    
    def generate_examples(self, text: str, num_examples: int = 3) -> List[str]:
        """Extract examples from text"""
        clean_text = self._clean_text(text)
        sentences = self._split_sentences(clean_text)
        
        examples = self._extract_examples(sentences, num_examples)
        
        # If not enough examples found, add relevant sentences
        if len(examples) < num_examples:
            for sent in sentences:
                if sent not in examples and len(sent) > 30:
                    examples.append(sent)
                if len(examples) >= num_examples:
                    break
        
        return examples[:num_examples]
    
    def generate_faq(self, text: str, num_questions: int = 5) -> List[Dict[str, str]]:
        """Generate FAQ from text"""
        clean_text = self._clean_text(text)
        sentences = self._split_sentences(clean_text)
        
        # Extract main concepts
        concepts = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
        concepts = list(dict.fromkeys(concepts))[:5]  # Unique, max 5
        
        return self._generate_faq(clean_text, sentences, concepts)
    
    def extract_vocabulary(self, text: str, num_terms: int = 8) -> List[Dict[str, str]]:
        """Extract vocabulary terms with definitions"""
        clean_text = self._clean_text(text)
        sentences = self._split_sentences(clean_text)
        
        return self._extract_vocabulary(clean_text, sentences, num_terms)
    
    def enhance_content(self, text: str, title: str = "Content") -> GeneratedContent:
        """
        Generate all enhanced content for a text
        
        Uses extraction-first approach for accuracy,
        with optional AI enhancement.
        """
        print(f"\n🔄 Generating enhanced content...")
        
        clean_text = self._clean_text(text)
        
        if not clean_text or len(clean_text) < 20:
            print("   ⚠️ Not enough content to enhance")
            return GeneratedContent(
                original=text,
                simplified_explanation="",
                key_takeaways=[],
                elaboration="",
                examples=[],
                faq=[],
                vocabulary=[]
            )
        
        print(f"   📄 Processing {len(clean_text)} characters...")
        
        print("   📝 Extracting simplified explanation...")
        simplified = self.simplify(clean_text)
        
        print("   🎯 Extracting key takeaways...")
        takeaways = self.generate_takeaways(clean_text, 5)
        
        print("   📖 Creating elaboration...")
        elaboration = self.elaborate(clean_text)
        
        print("   💡 Finding examples...")
        examples = self.generate_examples(clean_text, 3)
        
        print("   ❓ Generating FAQ...")
        faq = self.generate_faq(clean_text, 5)
        
        print("   📚 Extracting vocabulary...")
        vocabulary = self.extract_vocabulary(clean_text, 8)
        
        print("   ✅ Content enhancement complete!")
        
        return GeneratedContent(
            original=text,
            simplified_explanation=simplified,
            key_takeaways=takeaways,
            elaboration=elaboration,
            examples=examples,
            faq=faq,
            vocabulary=vocabulary
        )


class OfflineContentGenerator:
    """Alias for backward compatibility"""
    
    def __init__(self, model_name: str = "google/flan-t5-base"):
        self._enhancer = ContentEnhancer(use_ai=True, model_name=model_name)
    
    def simplify(self, text: str, target_level: str = "high school") -> str:
        return self._enhancer.simplify(text)
    
    def elaborate(self, text: str) -> str:
        return self._enhancer.elaborate(text)
    
    def explain_like_im_5(self, text: str) -> str:
        return self._enhancer.simplify(text)
    
    def generate_takeaways(self, text: str, num_points: int = 5) -> List[str]:
        return self._enhancer.generate_takeaways(text, num_points)
    
    def generate_examples(self, concept: str, num_examples: int = 3) -> List[str]:
        return self._enhancer.generate_examples(concept, num_examples)
    
    def generate_faq(self, text: str, num_questions: int = 5) -> List[Dict[str, str]]:
        return self._enhancer.generate_faq(text, num_questions)
    
    def extract_vocabulary(self, text: str, num_terms: int = 10) -> List[Dict[str, str]]:
        return self._enhancer.extract_vocabulary(text, num_terms)
    
    def enhance_content(self, text: str, title: str = "Content") -> GeneratedContent:
        return self._enhancer.enhance_content(text, title)


def get_content_enhancer(use_ai: bool = False) -> ContentEnhancer:
    """
    Factory function to get content enhancer
    
    Args:
        use_ai: Whether to enable AI-powered generation
        
    Returns:
        ContentEnhancer instance
    """
    return ContentEnhancer(use_ai=use_ai)
