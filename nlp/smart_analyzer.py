"""
Smart Content Analyzer - Enhanced NLP Pipeline v2
==================================================
Improved extraction algorithms for:
- Executive Summary (multi-signal sentence scoring)
- Key Sentences (hybrid TextRank + importance scoring)
- Key Concepts (context-aware definition extraction)
- Study Questions (intelligent template selection)
- Related Topics (semantic matching)

Works for BOTH meetings AND lectures/general content.
"""
import re
import math
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import Counter


@dataclass
class ExtractedConcept:
    """A key concept with context"""
    term: str
    definition: str
    frequency: int
    importance_score: float
    related_terms: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'term': self.term,
            'definition': self.definition,
            'frequency': self.frequency,
            'importance': round(self.importance_score, 2),
            'related': self.related_terms
        }


@dataclass
class GeneratedQuestion:
    """An auto-generated study question"""
    question: str
    answer_hint: str
    question_type: str  # factual, conceptual, analytical
    difficulty: str     # easy, medium, hard
    source_sentence: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'question': self.question,
            'hint': self.answer_hint,
            'type': self.question_type,
            'difficulty': self.difficulty
        }


@dataclass
class ContentAnalysis:
    """Complete content analysis result"""
    title: str
    executive_summary: str
    key_sentences: List[str]
    concepts: List[ExtractedConcept]
    questions: List[GeneratedQuestion]
    related_topics: List[str]
    
    # Statistics
    word_count: int
    sentence_count: int
    reading_time_minutes: float
    
    # For meetings
    action_items: List[str] = field(default_factory=list)
    decisions: List[str] = field(default_factory=list)
    deadlines: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'title': self.title,
            'executive_summary': self.executive_summary,
            'key_sentences': self.key_sentences,
            'concepts': [c.to_dict() for c in self.concepts],
            'questions': [q.to_dict() for q in self.questions],
            'related_topics': self.related_topics,
            'statistics': {
                'words': self.word_count,
                'sentences': self.sentence_count,
                'reading_time': self.reading_time_minutes
            },
            'meeting_items': {
                'actions': self.action_items,
                'decisions': self.decisions,
                'deadlines': self.deadlines
            }
        }


class SmartAnalyzer:
    """
    Enhanced Content Analyzer with improved NLP
    
    Uses:
    - Multi-signal sentence scoring (position, length, TF-IDF, cue phrases)
    - Improved TextRank with damping
    - Context-window concept definition extraction
    - Intelligent question generation
    """
    
    # Expanded stopwords
    STOPWORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
        'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them', 'their',
        'we', 'us', 'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her',
        'i', 'me', 'my', 'as', 'if', 'then', 'so', 'than', 'such', 'when',
        'where', 'which', 'who', 'what', 'how', 'all', 'each', 'every',
        'both', 'few', 'more', 'most', 'other', 'some', 'any', 'no', 'not',
        'only', 'own', 'same', 'just', 'also', 'very', 'even', 'back', 'now',
        'new', 'first', 'last', 'long', 'great', 'little', 'own', 'other',
        'old', 'right', 'big', 'high', 'different', 'small', 'large', 'next',
        'use', 'used', 'using', 'make', 'made', 'get', 'got', 'go', 'went',
        'come', 'came', 'take', 'took', 'see', 'saw', 'know', 'knew', 'think',
        'said', 'tell', 'told', 'ask', 'asked', 'well', 'much', 'thing', 'things',
        'like', 'really', 'want', 'way', 'going', 'something', 'actually',
        'yeah', 'yes', 'okay', 'ok', 'um', 'uh', 'basically', 'literally',
    }
    
    # Cue phrases that indicate important sentences
    IMPORTANCE_CUE_PHRASES = {
        'high': [
            'in conclusion', 'to summarize', 'the main point', 'most importantly',
            'the key', 'crucial', 'essential', 'fundamental', 'primary',
            'significantly', 'notably', 'particularly', 'especially',
            'the purpose', 'the goal', 'the objective', 'therefore',
            'as a result', 'consequently', 'in summary', 'to conclude',
            'the bottom line', 'ultimately', 'in essence',
        ],
        'medium': [
            'for example', 'for instance', 'such as', 'including',
            'first', 'second', 'third', 'finally', 'additionally',
            'moreover', 'furthermore', 'however', 'although', 'while',
            'because', 'since', 'due to', 'leads to', 'results in',
            'according to', 'research shows', 'studies indicate',
        ],
    }
    
    # Definition patterns for extracting what something IS
    DEFINITION_PATTERNS = [
        r'{term}\s+(?:is|are|refers to|means|describes|represents)\s+(.+?)(?:\.|,\s+(?:which|that|and))',
        r'{term}\s*[:\-]\s*(.+?)(?:\.|$)',
        r'(?:called|known as|termed)\s+{term}\s*[,.]?\s*(.+?)(?:\.|$)',
        r'(.+?)\s+(?:is|are)\s+(?:called|known as|termed)\s+{term}',
        r'{term}\s+(?:can be defined as|is defined as)\s+(.+?)(?:\.|$)',
    ]
    
    # Question templates with quality scoring
    QUESTION_TEMPLATES = {
        'definition': {
            'templates': [
                "What is {concept} and why is it important?",
                "Define {concept} in your own words.",
                "Explain the concept of {concept}.",
            ],
            'difficulty': 'easy',
            'type': 'factual'
        },
        'explanation': {
            'templates': [
                "How does {concept} work?",
                "Describe the process or mechanism of {concept}.",
                "What are the key components of {concept}?",
            ],
            'difficulty': 'medium',
            'type': 'conceptual'
        },
        'significance': {
            'templates': [
                "Why is {concept} significant in this context?",
                "What is the importance of {concept}?",
                "What role does {concept} play?",
            ],
            'difficulty': 'medium',
            'type': 'conceptual'
        },
        'comparison': {
            'templates': [
                "Compare and contrast {concept1} with {concept2}.",
                "What are the key differences between {concept1} and {concept2}?",
                "How does {concept1} relate to {concept2}?",
            ],
            'difficulty': 'hard',
            'type': 'analytical'
        },
        'application': {
            'templates': [
                "Give a real-world example of {concept}.",
                "How would you apply {concept} in practice?",
                "In what situations would {concept} be most useful?",
            ],
            'difficulty': 'hard',
            'type': 'application'
        },
        'analysis': {
            'templates': [
                "What are the advantages and disadvantages of {concept}?",
                "Analyze the potential impact of {concept}.",
                "What are the implications of {concept}?",
            ],
            'difficulty': 'hard',
            'type': 'analytical'
        },
    }
    
    # Topic keywords for related topics
    TOPIC_KEYWORDS = {
        'Social Media': ['instagram', 'facebook', 'twitter', 'tiktok', 'youtube', 'social', 'post', 'share', 'followers', 'likes', 'content', 'viral', 'influencer', 'platform', 'feed', 'story', 'reel'],
        'Technology': ['software', 'hardware', 'computer', 'digital', 'internet', 'app', 'platform', 'system', 'data', 'algorithm', 'ai', 'machine', 'code', 'programming', 'device', 'technology'],
        'Business': ['company', 'market', 'revenue', 'profit', 'customer', 'sales', 'product', 'service', 'brand', 'strategy', 'business', 'investment', 'growth', 'management'],
        'Education': ['learn', 'student', 'teach', 'school', 'course', 'study', 'knowledge', 'training', 'skill', 'education', 'class', 'lecture', 'university'],
        'Science': ['research', 'experiment', 'theory', 'study', 'discovery', 'scientific', 'hypothesis', 'evidence', 'data', 'analysis', 'method'],
        'Health': ['health', 'medical', 'disease', 'treatment', 'patient', 'doctor', 'medicine', 'symptom', 'therapy', 'diagnosis', 'wellness'],
        'Communication': ['message', 'share', 'connect', 'network', 'communicate', 'interact', 'conversation', 'speak', 'listen', 'feedback'],
        'Media': ['photo', 'video', 'image', 'content', 'media', 'upload', 'download', 'stream', 'watch', 'view', 'camera'],
        'Marketing': ['marketing', 'advertising', 'promotion', 'brand', 'campaign', 'audience', 'engagement', 'reach', 'target', 'conversion'],
        'Finance': ['money', 'payment', 'finance', 'bank', 'invest', 'budget', 'cost', 'price', 'income', 'expense', 'profit', 'loss'],
        'Psychology': ['behavior', 'psychology', 'mental', 'emotion', 'cognitive', 'mind', 'perception', 'motivation', 'personality'],
        'Environment': ['environment', 'climate', 'nature', 'sustainable', 'green', 'pollution', 'energy', 'ecosystem', 'conservation'],
    }
    
    def __init__(self):
        """Initialize the analyzer"""
        self._tfidf_cache = {}
    
    def analyze(self, text: str, title: str = "Document") -> ContentAnalysis:
        """
        Analyze text and extract structured content
        """
        # Clean and preprocess
        clean_text = self._clean_text(text)
        sentences = self._split_sentences(clean_text)
        
        if not sentences:
            return self._empty_analysis(title)
        
        # Calculate TF-IDF for the document
        tfidf_scores = self._calculate_tfidf(sentences)
        
        # Extract components with improved algorithms
        key_sentences = self._extract_key_sentences(sentences, tfidf_scores, max_sentences=5)
        executive_summary = self._generate_summary(sentences, tfidf_scores, key_sentences)
        concepts = self._extract_concepts(clean_text, sentences, tfidf_scores, max_concepts=8)
        questions = self._generate_questions(concepts, sentences, key_sentences, max_questions=8)
        related_topics = self._find_related_topics(clean_text, max_topics=6)
        
        # Meeting-specific extraction
        action_items = self._extract_actions(clean_text)
        decisions = self._extract_decisions(clean_text)
        deadlines = self._extract_deadlines(clean_text)
        
        # Statistics
        word_count = len(clean_text.split())
        reading_time = word_count / 200
        
        return ContentAnalysis(
            title=title,
            executive_summary=executive_summary,
            key_sentences=key_sentences,
            concepts=concepts,
            questions=questions,
            related_topics=related_topics,
            word_count=word_count,
            sentence_count=len(sentences),
            reading_time_minutes=round(reading_time, 1),
            action_items=action_items,
            decisions=decisions,
            deadlines=deadlines
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        lines = text.split('\n')
        content_lines = []
        
        skip_patterns = [
            r'^={2,}', r'^-{2,}', r'^\[.*\]$',
            r'^transcript:', r'^audio:', r'^duration:',
            r'^confidence:', r'^words:', r'^timestamps:',
            r'^generated:', r'^echonotes',
        ]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            skip = False
            for pattern in skip_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    skip = True
                    break
            if skip:
                continue
            
            line = re.sub(r'\[\d{1,2}:\d{2}(?::\d{2})?\]', '', line).strip()
            
            if line and len(line) > 5:
                content_lines.append(line)
        
        text = ' '.join(content_lines)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        
        return text.strip()
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences with better handling"""
        text = re.sub(r'\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|e\.g|i\.e)\.\s+', r'\1<DOT> ', text)
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        sentences = [s.replace('<DOT>', '.') for s in sentences]
        
        result = []
        for s in sentences:
            s = s.strip()
            words = s.split()
            if len(s) >= 20 and len(words) >= 4:
                result.append(s)
        
        return result
    
    def _calculate_tfidf(self, sentences: List[str]) -> Dict[str, float]:
        """Calculate TF-IDF scores for terms"""
        term_doc_freq = Counter()
        term_freq = Counter()
        
        for sent in sentences:
            words = self._tokenize(sent)
            unique_words = set(words)
            for word in unique_words:
                term_doc_freq[word] += 1
            for word in words:
                term_freq[word] += 1
        
        n_docs = len(sentences)
        tfidf = {}
        
        for term, tf in term_freq.items():
            df = term_doc_freq[term]
            idf = math.log((n_docs + 1) / (df + 1)) + 1
            tfidf[term] = tf * idf
        
        if tfidf:
            max_score = max(tfidf.values())
            tfidf = {k: v / max_score for k, v in tfidf.items()}
        
        return tfidf
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize and filter text"""
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        return [w for w in words if w not in self.STOPWORDS]
    
    def _score_sentence(self, sentence: str, position: int, total_sentences: int, tfidf_scores: Dict[str, float]) -> float:
        """Score a sentence using multiple signals"""
        score = 0.0
        
        # 1. Position score
        relative_pos = position / max(1, total_sentences - 1)
        if relative_pos < 0.2:
            position_score = 1.0 - (relative_pos * 2)
        elif relative_pos > 0.8:
            position_score = (relative_pos - 0.8) * 2 + 0.4
        else:
            position_score = 0.3
        score += position_score * 0.2
        
        # 2. Length score
        words = sentence.split()
        word_count = len(words)
        if 15 <= word_count <= 40:
            length_score = 1.0
        elif word_count < 15:
            length_score = word_count / 15
        else:
            length_score = max(0.3, 1.0 - (word_count - 40) / 60)
        score += length_score * 0.15
        
        # 3. TF-IDF score
        tokens = self._tokenize(sentence)
        if tokens:
            tfidf_score = sum(tfidf_scores.get(t, 0) for t in tokens) / len(tokens)
            score += tfidf_score * 0.4
        
        # 4. Cue phrase score
        sentence_lower = sentence.lower()
        cue_score = 0.0
        for phrase in self.IMPORTANCE_CUE_PHRASES['high']:
            if phrase in sentence_lower:
                cue_score = 1.0
                break
        if cue_score == 0:
            for phrase in self.IMPORTANCE_CUE_PHRASES['medium']:
                if phrase in sentence_lower:
                    cue_score = 0.5
                    break
        score += cue_score * 0.25
        
        return score
    
    def _extract_key_sentences(self, sentences: List[str], tfidf_scores: Dict[str, float], max_sentences: int = 5) -> List[str]:
        """Extract key sentences using multi-signal scoring"""
        if len(sentences) <= max_sentences:
            return sentences
        
        scored_sentences = []
        for i, sent in enumerate(sentences):
            score = self._score_sentence(sent, i, len(sentences), tfidf_scores)
            scored_sentences.append((i, sent, score))
        
        textrank_scores = self._textrank_sentences(sentences)
        for i, (idx, sent, score) in enumerate(scored_sentences):
            if idx < len(textrank_scores):
                combined = score * 0.7 + textrank_scores[idx] * 0.3
                scored_sentences[i] = (idx, sent, combined)
        
        scored_sentences.sort(key=lambda x: -x[2])
        selected = scored_sentences[:max_sentences]
        selected.sort(key=lambda x: x[0])
        
        return [sent for _, sent, _ in selected]
    
    def _textrank_sentences(self, sentences: List[str], damping: float = 0.85) -> List[float]:
        """Calculate TextRank scores for sentences"""
        n = len(sentences)
        if n == 0:
            return []
        if n == 1:
            return [1.0]
        
        tokenized = [set(self._tokenize(s)) for s in sentences]
        matrix = [[0.0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                if i != j and tokenized[i] and tokenized[j]:
                    intersection = len(tokenized[i] & tokenized[j])
                    union = len(tokenized[i] | tokenized[j])
                    if union > 0:
                        matrix[i][j] = intersection / union
        
        for i in range(n):
            row_sum = sum(matrix[i])
            if row_sum > 0:
                matrix[i] = [x / row_sum for x in matrix[i]]
        
        scores = [1.0 / n] * n
        for _ in range(50):
            new_scores = []
            for i in range(n):
                score = (1 - damping) / n
                for j in range(n):
                    score += damping * matrix[j][i] * scores[j]
                new_scores.append(score)
            
            diff = sum(abs(new_scores[i] - scores[i]) for i in range(n))
            scores = new_scores
            if diff < 1e-6:
                break
        
        if scores:
            max_score = max(scores)
            if max_score > 0:
                scores = [s / max_score for s in scores]
        
        return scores
    
    def _generate_summary(self, sentences: List[str], tfidf_scores: Dict[str, float], key_sentences: List[str], max_words: int = 100) -> str:
        """Generate executive summary from key sentences"""
        if not key_sentences:
            return sentences[0] if sentences else ""
        
        summary_parts = []
        word_count = 0
        
        for sent in key_sentences:
            words = sent.split()
            if word_count + len(words) <= max_words:
                summary_parts.append(sent)
                word_count += len(words)
            elif word_count == 0:
                truncated = ' '.join(words[:max_words])
                if not truncated.endswith('.'):
                    truncated += '...'
                summary_parts.append(truncated)
                break
        
        return ' '.join(summary_parts)
    
    def _extract_concepts(self, text: str, sentences: List[str], tfidf_scores: Dict[str, float], max_concepts: int = 8) -> List[ExtractedConcept]:
        """Extract key concepts with context-aware definitions"""
        concepts = {}
        text_lower = text.lower()
        
        # Method 1: Proper noun phrases
        proper_nouns = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
        for term in proper_nouns:
            term_lower = term.lower()
            if term_lower not in self.STOPWORDS and len(term) > 2:
                freq = len(re.findall(re.escape(term_lower), text_lower))
                if freq >= 1:
                    score = freq * 0.3 + sum(tfidf_scores.get(w, 0) for w in term_lower.split())
                    concepts[term_lower] = {'display': term, 'score': score, 'freq': freq, 'definition': ''}
        
        # Method 2: High TF-IDF terms
        top_terms = sorted(tfidf_scores.items(), key=lambda x: -x[1])[:30]
        for term, score in top_terms:
            if term not in concepts and len(term) > 3:
                freq = len(re.findall(r'\b' + re.escape(term) + r'\b', text_lower))
                if freq >= 2:
                    concepts[term] = {'display': term.title(), 'score': score, 'freq': freq, 'definition': ''}
        
        # Method 3: Compound terms
        compound_patterns = [
            r'\b(\w+\s+(?:system|service|platform|network|method|process|model|theory|approach|technique|strategy|concept|principle|framework|algorithm|protocol))\b',
            r'\b((?:data|user|content|social|digital|online|mobile)\s+\w+)\b',
        ]
        for pattern in compound_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if match not in concepts and len(match.split()) <= 3:
                    freq = text_lower.count(match)
                    if freq >= 1:
                        concepts[match] = {'display': match.title(), 'score': 0.5 + freq * 0.1, 'freq': freq, 'definition': ''}
        
        # Extract definitions
        for term, data in concepts.items():
            definition = self._extract_definition(term, sentences, text)
            if definition:
                data['definition'] = definition
                data['score'] += 0.3
        
        sorted_concepts = sorted(concepts.items(), key=lambda x: -x[1]['score'])
        
        result = []
        seen_terms = set()
        
        for term, data in sorted_concepts:
            if len(result) >= max_concepts:
                break
            
            skip = False
            for seen in seen_terms:
                if term in seen or seen in term:
                    skip = True
                    break
            if skip:
                continue
            
            definition = data['definition']
            if not definition:
                definition = self._generate_fallback_definition(term, sentences)
            
            result.append(ExtractedConcept(
                term=data['display'],
                definition=definition,
                frequency=data['freq'],
                importance_score=min(1.0, data['score'])
            ))
            seen_terms.add(term)
        
        return result
    
    def _extract_definition(self, term: str, sentences: List[str], text: str) -> str:
        """Extract a proper definition for a term"""
        term_escaped = re.escape(term)
        
        for pattern_template in self.DEFINITION_PATTERNS:
            pattern = pattern_template.format(term=term_escaped)
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                definition = match.group(1).strip()
                if 10 < len(definition) < 300:
                    definition = definition[0].upper() + definition[1:]
                    if not definition.endswith('.'):
                        definition += '.'
                    return definition
        
        best_sentence = None
        best_score = 0
        
        for sent in sentences:
            if term in sent.lower():
                score = 0
                sent_lower = sent.lower()
                
                if re.search(rf'\b{term_escaped}\s+(?:is|are|means|refers to)', sent_lower):
                    score += 3
                if 'defined as' in sent_lower or 'known as' in sent_lower:
                    score += 2
                if any(phrase in sent_lower for phrase in ['for example', 'such as', 'including']):
                    score += 1
                
                term_pos = sent_lower.find(term)
                if term_pos < len(sent) * 0.3:
                    score += 1
                
                if 40 < len(sent) < 200:
                    score += 1
                
                if score > best_score:
                    best_score = score
                    best_sentence = sent
        
        if best_sentence and best_score >= 1:
            if len(best_sentence) > 250:
                best_sentence = best_sentence[:250].rsplit(' ', 1)[0] + '...'
            return best_sentence
        
        return ""
    
    def _generate_fallback_definition(self, term: str, sentences: List[str]) -> str:
        """Generate a fallback definition"""
        for sent in sentences:
            if term in sent.lower():
                if len(sent) > 250:
                    sent = sent[:250].rsplit(' ', 1)[0] + '...'
                return sent
        
        return f"A key concept discussed in this content related to {term}."
    
    def _generate_questions(self, concepts: List[ExtractedConcept], sentences: List[str], key_sentences: List[str], max_questions: int = 8) -> List[GeneratedQuestion]:
        """Generate intelligent study questions"""
        questions = []
        
        if not concepts:
            return questions
        
        # Definition questions
        for i, concept in enumerate(concepts[:3]):
            template_data = self.QUESTION_TEMPLATES['definition']
            template = template_data['templates'][i % len(template_data['templates'])]
            
            questions.append(GeneratedQuestion(
                question=template.format(concept=concept.term),
                answer_hint=self._truncate_hint(concept.definition),
                question_type=template_data['type'],
                difficulty=template_data['difficulty'],
                source_sentence=concept.definition
            ))
        
        # Explanation question
        if concepts:
            template_data = self.QUESTION_TEMPLATES['explanation']
            questions.append(GeneratedQuestion(
                question=template_data['templates'][0].format(concept=concepts[0].term),
                answer_hint=self._truncate_hint(concepts[0].definition),
                question_type=template_data['type'],
                difficulty=template_data['difficulty']
            ))
        
        # Significance question
        if len(concepts) >= 2:
            template_data = self.QUESTION_TEMPLATES['significance']
            questions.append(GeneratedQuestion(
                question=template_data['templates'][0].format(concept=concepts[1].term),
                answer_hint="Consider the role and impact discussed in the content.",
                question_type=template_data['type'],
                difficulty=template_data['difficulty']
            ))
        
        # Comparison question
        if len(concepts) >= 2:
            template_data = self.QUESTION_TEMPLATES['comparison']
            questions.append(GeneratedQuestion(
                question=template_data['templates'][0].format(concept1=concepts[0].term, concept2=concepts[1].term),
                answer_hint="Compare their definitions, purposes, and relationships.",
                question_type=template_data['type'],
                difficulty=template_data['difficulty']
            ))
        
        # Application question
        if concepts:
            template_data = self.QUESTION_TEMPLATES['application']
            questions.append(GeneratedQuestion(
                question=template_data['templates'][0].format(concept=concepts[0].term),
                answer_hint="Think about practical scenarios where this applies.",
                question_type=template_data['type'],
                difficulty=template_data['difficulty']
            ))
        
        # Analysis question
        if len(concepts) >= 1:
            template_data = self.QUESTION_TEMPLATES['analysis']
            questions.append(GeneratedQuestion(
                question=template_data['templates'][0].format(concept=concepts[min(2, len(concepts)-1)].term),
                answer_hint="Consider both benefits and limitations.",
                question_type=template_data['type'],
                difficulty=template_data['difficulty']
            ))
        
        return questions[:max_questions]
    
    def _truncate_hint(self, text: str, max_length: int = 150) -> str:
        """Truncate hint to reasonable length"""
        if len(text) <= max_length:
            return text
        return text[:max_length].rsplit(' ', 1)[0] + '...'
    
    def _find_related_topics(self, text: str, max_topics: int = 6) -> List[str]:
        """Find related topics based on keyword matching"""
        text_lower = text.lower()
        topic_scores = {}
        
        for topic, keywords in self.TOPIC_KEYWORDS.items():
            score = 0
            matched_keywords = 0
            for kw in keywords:
                count = len(re.findall(r'\b' + re.escape(kw) + r'\b', text_lower))
                if count > 0:
                    score += min(count, 5)
                    matched_keywords += 1
            
            if matched_keywords >= 2:
                topic_scores[topic] = score
        
        sorted_topics = sorted(topic_scores.items(), key=lambda x: -x[1])
        return [topic for topic, _ in sorted_topics[:max_topics]]
    
    def _extract_actions(self, text: str) -> List[str]:
        """Extract action items from text"""
        patterns = [
            r'(?:we|I|you|they)\s+(?:will|should|need to|must|have to)\s+([^.!?]+)',
            r'action\s*(?:item)?[:\s]+([^.!?]+)',
            r'todo[:\s]+([^.!?]+)',
            r'(?:please|kindly)\s+([^.!?]+)',
            r'(?:make sure|ensure|remember)\s+to\s+([^.!?]+)',
        ]
        
        actions = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                action = match.strip()
                if 15 < len(action) < 200:
                    action = action[0].upper() + action[1:]
                    actions.append(action)
        
        seen = set()
        unique_actions = []
        for a in actions:
            a_lower = a.lower()
            if a_lower not in seen:
                seen.add(a_lower)
                unique_actions.append(a)
        
        return unique_actions[:5]
    
    def _extract_decisions(self, text: str) -> List[str]:
        """Extract decisions from text"""
        patterns = [
            r'(?:we|they|team)\s+decided\s+(?:to\s+)?([^.!?]+)',
            r'decision[:\s]+([^.!?]+)',
            r'(?:we|they)\s+agreed\s+(?:to|on|that)\s+([^.!?]+)',
            r"let'?s\s+(?:go with|use|do)\s+([^.!?]+)",
            r'(?:final|the)\s+(?:decision|choice)\s+(?:is|was)\s+(?:to\s+)?([^.!?]+)',
        ]
        
        decisions = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                decision = match.strip()
                if 10 < len(decision) < 200:
                    decision = decision[0].upper() + decision[1:]
                    decisions.append(decision)
        
        return list(set(decisions))[:5]
    
    def _extract_deadlines(self, text: str) -> List[str]:
        """Extract deadlines from text"""
        patterns = [
            r'(?:by|before|until)\s+((?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)[^.!?]*)',
            r'(?:by|before|until)\s+(tomorrow|today|tonight|end of (?:day|week|month))',
            r'(?:by|before|until)\s+(next\s+(?:week|month|monday|tuesday|wednesday|thursday|friday)[^.!?]*)',
            r'deadline[:\s]+([^.!?]+)',
            r'due\s+(?:by|on|date)?[:\s]*([^.!?]+)',
            r'((?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?(?:\s*,?\s*\d{4})?)',
        ]
        
        deadlines = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                deadline = match.strip()
                if 3 < len(deadline) < 100:
                    deadline = deadline[0].upper() + deadline[1:]
                    deadlines.append(deadline)
        
        return list(set(deadlines))[:5]
    
    def _empty_analysis(self, title: str) -> ContentAnalysis:
        """Return empty analysis for edge cases"""
        return ContentAnalysis(
            title=title,
            executive_summary="No content available for analysis.",
            key_sentences=[],
            concepts=[],
            questions=[],
            related_topics=[],
            word_count=0,
            sentence_count=0,
            reading_time_minutes=0.0
        )
