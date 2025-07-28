"""
Intelligent Document Analyzer for Adobe Hackathon Round 1B
Theme: "Connect What Matters — For the User Who Matters"

This system acts as an intelligent document analyst, extracting and prioritizing 
the most relevant sections from a collection of documents based on a specific 
persona and their job-to-be-done.

Constraints:
- Must run on CPU only
- Model size ≤ 1GB
- Processing time ≤ 60 seconds for document collection (3-5 documents)
- No internet access allowed during execution
"""

import json
import fitz  # PyMuPDF
import os
import hashlib
import pickle
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging
import time
import sys
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DocumentSection:
    """Represents a section of a document with relevance scoring"""
    document: str
    page_number: int
    section_title: str
    content: str
    importance_rank: int = 0
    confidence_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "document": self.document,
            "page_number": self.page_number,
            "section_title": self.section_title,
            "importance_rank": self.importance_rank,
            "confidence_score": round(self.confidence_score, 4)
        }

@dataclass
class SubSection:
    """Represents a refined sub-section with detailed analysis"""
    document: str
    section_title: str
    refined_text: str
    page_number: int
    relevance_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "document": self.document,
            "section_title": self.section_title,
            "refined_text": self.refined_text,
            "page_number": self.page_number,
            "relevance_score": round(self.relevance_score, 4)
        }

class PersonaProfileBuilder:
    """Builds comprehensive persona profiles for relevance matching"""
    
    # Domain-specific keywords mapping
    DOMAIN_KEYWORDS = {
        'travel planner': [
            'itinerary', 'accommodation', 'hotel', 'restaurant', 'attractions', 
            'activities', 'budget', 'transportation', 'dining', 'schedule',
            'sightseeing', 'culture', 'history', 'tips', 'recommendations'
        ],
        'food contractor': [
            'menu', 'catering', 'buffet', 'serving', 'portions', 'dietary',
            'ingredients', 'preparation', 'cooking', 'recipes', 'nutrition',
            'allergens', 'presentation', 'cost', 'bulk', 'corporate'
        ],
        'researcher': [
            'methodology', 'analysis', 'literature', 'data', 'results', 
            'findings', 'conclusion', 'study', 'research', 'investigation',
            'evidence', 'hypothesis', 'theory', 'experiment'
        ],
        'student': [
            'study', 'exam', 'concepts', 'theory', 'practice', 'learning',
            'understanding', 'knowledge', 'education', 'curriculum',
            'assignment', 'homework', 'lesson', 'tutorial'
        ],
        'analyst': [
            'trends', 'metrics', 'performance', 'comparison', 'insights',
            'evaluation', 'assessment', 'statistics', 'data analysis',
            'reporting', 'forecasting', 'modeling'
        ],
        'journalist': [
            'facts', 'sources', 'story', 'reporting', 'investigation',
            'interview', 'news', 'article', 'coverage', 'information',
            'verification', 'documentation'
        ]
    }
    
    @classmethod
    def create_profile(cls, persona_data: Dict[str, Any], job_data: Dict[str, Any]) -> str:
        """Create a comprehensive persona profile for relevance scoring"""
        role = persona_data.get('role', '').lower().strip()
        task = job_data.get('task', '').lower().strip()
        
        profile_parts = [
            f"Role: {role}",
            f"Task: {task}"
        ]
        
        # Add domain-specific keywords
        domain_keywords = cls._get_domain_keywords(role)
        if domain_keywords:
            profile_parts.append(f"Domain expertise: {' '.join(domain_keywords)}")
        
        # Extract and add task-specific keywords
        task_keywords = cls._extract_task_keywords(task)
        if task_keywords:
            profile_parts.append(f"Task focus: {' '.join(task_keywords)}")
        
        # Add enhanced dietary/domain context based on task
        enhanced_context = cls._extract_dietary_context(task)
        if enhanced_context:
            profile_parts.append(f"Specific requirements: {enhanced_context}")
        
        return " ".join(profile_parts)
    
    @classmethod
    def _get_domain_keywords(cls, role: str) -> List[str]:
        """Get domain-specific keywords based on role"""
        for domain, keywords in cls.DOMAIN_KEYWORDS.items():
            if domain in role or any(word in role for word in domain.split()):
                return keywords
        return []
    
    @classmethod
    def _extract_task_keywords(cls, task: str) -> List[str]:
        """Extract meaningful keywords from task description"""
        # Common stop words to filter out
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
            'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must'
        }
        
        # Extract words and filter
        words = re.findall(r'\b\w+\b', task.lower())
        keywords = [
            word for word in words 
            if word not in stop_words and len(word) > 2
        ]
        
        return keywords[:15]  # Limit to top 15 keywords
    
    @classmethod
    def _extract_dietary_context(cls, task: str) -> str:
        """Extract dietary and contextual requirements from task"""
        context_parts = []
        
        # Dietary restrictions
        if 'vegetarian' in task.lower():
            context_parts.append('vegetarian plant-based no-meat cheese vegetables beans tofu quinoa')
        if 'vegan' in task.lower():
            context_parts.append('vegan plant-based no-dairy no-meat vegetables beans nuts')
        if 'gluten-free' in task.lower():
            context_parts.append('gluten-free rice corn potatoes naturally-gluten-free celiac-safe')
        
        # Serving style
        if 'buffet' in task.lower():
            context_parts.append('buffet self-serve large-quantities crowd-pleasing easy-serving')
        if 'corporate' in task.lower():
            context_parts.append('corporate professional business-appropriate presentable')
        if 'gathering' in task.lower():
            context_parts.append('group event party social catering')
        
        return ' '.join(context_parts)

class PDFProcessor:
    """Handles PDF text extraction and section identification"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text from PDF with intelligent section detection"""
        self.logger.info(f"Extracting text from {pdf_path}")
        
        if not os.path.exists(pdf_path):
            self.logger.error(f"PDF file not found: {pdf_path}")
            return []
        
        try:
            doc = fitz.open(pdf_path)
            extracted_data = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Extract text with formatting information
                text_dict = page.get_text("dict")
                page_text = self._extract_formatted_text(text_dict)
                
                if page_text.strip():
                    sections = self._identify_sections(page_text, page_num + 1)
                    extracted_data.extend(sections)
            
            doc.close()
            self.logger.info(f"Extracted {len(extracted_data)} sections from {pdf_path}")
            return extracted_data
            
        except Exception as e:
            self.logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            return []
    
    def _extract_formatted_text(self, text_dict: Dict) -> str:
        """Extract text while preserving some formatting cues"""
        text_parts = []
        
        for block in text_dict.get("blocks", []):
            if "lines" in block:
                block_text = ""
                for line in block["lines"]:
                    line_text = ""
                    for span in line["spans"]:
                        text = span.get("text", "")
                        # Preserve line breaks for better section detection
                        line_text += text
                    if line_text.strip():
                        block_text += line_text + "\n"
                
                if block_text.strip():
                    text_parts.append(block_text)
        
        return "\n".join(text_parts)
    
    def _identify_sections(self, page_text: str, page_num: int) -> List[Dict[str, Any]]:
        """Identify logical sections within a page"""
        sections = []
        
        # Split by multiple newlines to get potential sections
        text_blocks = [block.strip() for block in page_text.split('\n\n') if block.strip()]
        
        current_section = ""
        current_title = self._generate_meaningful_title(page_text, page_num)
        section_count = 0
        
        for block in text_blocks:
            if self._is_likely_heading(block):
                # Save previous section if substantial
                if current_section.strip() and len(current_section.strip()) > 100:
                    sections.append({
                        'page_number': page_num,
                        'section_title': current_title,
                        'content': current_section.strip()
                    })
                
                # Start new section
                current_title = self._clean_title(block)
                current_section = ""
                section_count += 1
            else:
                current_section += block + "\n\n"
        
        # Add final section
        if current_section.strip() and len(current_section.strip()) > 100:
            sections.append({
                'page_number': page_num,
                'section_title': current_title,
                'content': current_section.strip()
            })
        
        # If no sections were found, create one for the entire page
        if not sections and page_text.strip():
            sections.append({
                'page_number': page_num,
                'section_title': self._generate_meaningful_title(page_text, page_num),
                'content': page_text.strip()
            })
        
        return sections
    
    def _is_likely_heading(self, text: str) -> bool:
        """Determine if text block is likely a section heading"""
        text = text.strip()
        
        # Skip empty or very long text
        if not text or len(text) > 300:
            return False
        
        # Very short text is likely a heading
        if len(text) < 15:
            return True
        
        # Check for heading patterns
        heading_patterns = [
            r'^[A-Z][A-Z\s]+$',  # ALL CAPS
            r'^\d+\.\s*[A-Z]',   # Numbered headings
            r'^[A-Z][a-z\s]+:',  # Title case with colon
            r'^[A-Z][a-z\s]+\.$', # Title case ending with period
        ]
        
        for pattern in heading_patterns:
            if re.match(pattern, text):
                return True
        
        # Check if text has title-like characteristics
        words = text.split()
        if len(words) <= 8 and all(word[0].isupper() for word in words if word):
            return True
        
        return False
    
    def _clean_title(self, title: str) -> str:
        """Clean and format section title"""
        title = title.strip()
        # Remove excessive whitespace
        title = re.sub(r'\s+', ' ', title)
        # Remove common prefixes that make titles messy
        title = re.sub(r'^Page \d+ - ', '', title)
        # Limit length
        if len(title) > 150:
            title = title[:147] + "..."
        return title
    
    def _clean_document_title(self, title: str) -> str:
        """Clean document title for better presentation"""
        # Remove .pdf extension
        title = re.sub(r'\.pdf$', '', title, flags=re.IGNORECASE)
        # Clean up common patterns
        title = title.strip()
        return title
    
    def _generate_meaningful_title(self, page_text: str, page_num: int) -> str:
        """Generate meaningful section title from page content"""
        lines = page_text.strip().split('\n')
        
        # Look for potential titles in first few lines
        for line in lines[:5]:
            line = line.strip()
            if line and len(line) < 100 and len(line) > 10:
                # Clean and use as title
                clean_line = re.sub(r'[^\w\s-]', '', line).strip()
                if clean_line and not clean_line.isdigit():
                    return clean_line[:80]  # Limit length
        
        # Look for key phrases that could be good titles
        key_phrases = []
        for line in lines[:10]:
            line = line.strip()
            # Look for lines that start with capital letters and have key words
            if (line and line[0].isupper() and 
                any(word in line.lower() for word in ['create', 'convert', 'edit', 'export', 'sign', 'share', 'ai', 'pdf', 'acrobat'])):
                clean_line = re.sub(r'[^\w\s-]', '', line).strip()
                if len(clean_line) < 80 and len(clean_line) > 5:
                    key_phrases.append(clean_line)
        
        # Return best phrase or fallback
        if key_phrases:
            return key_phrases[0]
        
        return f"Content and Features"

class EmbeddingManager:
    """Manages text embeddings with caching for performance"""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Use better model for improved semantic understanding (~420MB, still <1GB)
        self.model = SentenceTransformer('all-mpnet-base-v2')  # Better than all-MiniLM-L6-v2
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("Initialized embedding model: all-mpnet-base-v2")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get text embedding with caching"""
        # Create cache key
        cache_key = hashlib.md5(text.encode('utf-8')).hexdigest()
        cache_path = self.cache_dir / f"embedding_{cache_key}.pkl"
        
        # Try to load from cache
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cached embedding: {e}")
        
        # Generate new embedding
        embedding = self.model.encode([text])[0]
        
        # Cache the embedding
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            self.logger.warning(f"Failed to cache embedding: {e}")
        
        return embedding

class RelevanceScorer:
    """Scores document sections based on persona relevance"""
    
    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def score_sections(self, sections: List[Dict[str, Any]], 
                      persona_profile: str) -> List[DocumentSection]:
        """Score sections based on relevance to persona profile with domain filtering"""
        self.logger.info(f"Scoring {len(sections)} sections for relevance")
        
        if not sections:
            return []
        
        # Get persona embedding
        persona_embedding = self.embedding_manager.get_embedding(persona_profile)
        
        scored_sections = []
        
        for section in sections:
            content = section['content']
            
            # Skip very short sections
            if len(content.strip()) < 50:
                continue
            
            # Get section embedding
            content_embedding = self.embedding_manager.get_embedding(content)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(
                [persona_embedding], 
                [content_embedding]
            )[0][0]
            
            # Apply content quality boost
            quality_score = self._assess_content_quality(content)
            
            # Apply domain-specific filtering and boosting
            domain_score = self._apply_domain_filtering(content, persona_profile)
            
            # Combine scores with enhanced weighting
            final_score = similarity * 0.6 + quality_score * 0.2 + domain_score * 0.2
            
            doc_section = DocumentSection(
                document=section.get('document', ''),
                page_number=section['page_number'],
                section_title=section['section_title'],
                content=content,
                confidence_score=float(final_score)
            )
            
            scored_sections.append(doc_section)
        
        # Sort by score and assign ranks
        scored_sections.sort(key=lambda x: x.confidence_score, reverse=True)
        
        for i, section in enumerate(scored_sections):
            section.importance_rank = i + 1
        
        self.logger.info(f"Scored and ranked {len(scored_sections)} sections")
        return scored_sections
    
    def _assess_content_quality(self, content: str) -> float:
        """Assess the quality of content for boosting scores"""
        score = 0.0
        
        # Length factor (prefer substantial content)
        if 200 <= len(content) <= 2000:
            score += 0.3
        elif len(content) > 2000:
            score += 0.2
        
        # Information density (keywords, numbers, etc.)
        word_count = len(content.split())
        if word_count > 0:
            # Check for informative content indicators
            numbers = len(re.findall(r'\d+', content))
            caps_words = len(re.findall(r'\b[A-Z][a-z]+\b', content))
            
            info_density = (numbers + caps_words) / word_count
            score += min(info_density * 0.5, 0.3)
        
        # Structure indicators (lists, paragraphs)
        if '\n' in content or '•' in content or re.search(r'\d+\.', content):
            score += 0.2
        
        return min(score, 1.0)
    
    def _apply_domain_filtering(self, content: str, persona_profile: str) -> float:
        """Apply domain-specific filtering and boosting"""
        content_lower = content.lower()
        profile_lower = persona_profile.lower()
        score = 0.0
        
        # Vegetarian filtering - heavily penalize meat content
        if 'vegetarian' in profile_lower:
            meat_keywords = ['beef', 'pork', 'chicken', 'turkey', 'duck', 'fish', 'salmon', 
                           'tuna', 'shrimp', 'crab', 'lobster', 'meat', 'bacon', 'ham', 
                           'sausage', 'pepperoni', 'anchovy', 'prosciutto']
            
            meat_count = sum(1 for keyword in meat_keywords if keyword in content_lower)
            if meat_count > 0:
                score -= 0.5  # Heavy penalty for meat content
            
            # Boost vegetarian-friendly content
            veg_keywords = ['cheese', 'vegetable', 'quinoa', 'bean', 'lentil', 'tofu', 
                          'hummus', 'avocado', 'spinach', 'tomato', 'cucumber', 'pepper',
                          'mushroom', 'onion', 'garlic', 'herb', 'salad', 'pasta']
            
            veg_count = sum(1 for keyword in veg_keywords if keyword in content_lower)
            score += min(veg_count * 0.1, 0.5)  # Boost for vegetarian ingredients
        
        # Gluten-free filtering
        if 'gluten-free' in profile_lower or 'gluten free' in profile_lower:
            # Penalize gluten-containing items
            gluten_keywords = ['wheat', 'flour', 'bread', 'pasta', 'noodle', 'barley', 
                             'rye', 'oats', 'couscous', 'seitan', 'soy sauce']
            gluten_count = sum(1 for keyword in gluten_keywords if keyword in content_lower)
            if gluten_count > 0:
                score -= 0.3
            
            # Boost naturally gluten-free items
            gf_keywords = ['rice', 'corn', 'potato', 'quinoa', 'naturally gluten-free',
                         'gluten-free', 'vegetables', 'fruit', 'cheese', 'eggs']
            gf_count = sum(1 for keyword in gf_keywords if keyword in content_lower)
            score += min(gf_count * 0.1, 0.3)
        
        # Buffet/serving style boost
        if 'buffet' in profile_lower:
            buffet_keywords = ['serve', 'portion', 'bowl', 'platter', 'easy', 'self-serve',
                             'large', 'crowd', 'group', 'batch', 'quantity']
            buffet_count = sum(1 for keyword in buffet_keywords if keyword in content_lower)
            score += min(buffet_count * 0.05, 0.2)
        
        # Corporate/professional boost
        if 'corporate' in profile_lower:
            corp_keywords = ['professional', 'presentation', 'elegant', 'clean', 'simple',
                           'business', 'appropriate', 'neat', 'organized']
            corp_count = sum(1 for keyword in corp_keywords if keyword in content_lower)
            score += min(corp_count * 0.05, 0.2)
        
        return max(-1.0, min(1.0, score))  # Clamp between -1 and 1

class SubSectionGenerator:
    """Generates refined sub-sections from top-ranked sections"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def generate_subsections(self, top_sections: List[DocumentSection], 
                           max_subsections: int = 15) -> List[SubSection]:
        """Generate refined sub-sections from top sections"""
        self.logger.info(f"Generating sub-sections from {len(top_sections)} top sections")
        
        subsections = []
        
        for section in top_sections[:max_subsections]:
            # Split content into logical chunks
            chunks = self._split_content_intelligently(section.content)
            
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) > 150:  # Only substantial chunks
                    refined_text = self._refine_text(chunk)
                    
                    subsection = SubSection(
                        document=section.document,
                        section_title=f"{section.section_title} (Part {i+1})" if len(chunks) > 1 else section.section_title,
                        refined_text=refined_text,
                        page_number=section.page_number,
                        relevance_score=section.confidence_score
                    )
                    
                    subsections.append(subsection)
                    
                    if len(subsections) >= max_subsections:
                        break
            
            if len(subsections) >= max_subsections:
                break
        
        self.logger.info(f"Generated {len(subsections)} sub-sections")
        return subsections
    
    def _split_content_intelligently(self, content: str) -> List[str]:
        """Split content into logical, meaningful chunks"""
        # First try to split by double newlines (paragraphs)
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        if len(paragraphs) <= 1:
            # Fallback to sentence-based splitting
            sentences = re.split(r'[.!?]+', content)
            return self._group_sentences(sentences)
        
        # Group paragraphs into reasonable chunks
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk + paragraph) <= 800:  # Target chunk size
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _group_sentences(self, sentences: List[str]) -> List[str]:
        """Group sentences into logical chunks"""
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if len(current_chunk + sentence) <= 600:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _refine_text(self, text: str) -> str:
        """Clean and refine text content"""
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove excessive repetition
        sentences = text.split('.')
        unique_sentences = []
        seen = set()
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and sentence not in seen:
                unique_sentences.append(sentence)
                seen.add(sentence)
        
        refined = '. '.join(unique_sentences)
        
        # Ensure proper ending
        if refined and not refined.endswith(('.', '!', '?')):
            refined += '.'
        
        # Limit length if necessary
        if len(refined) > 1500:
            sentences = refined.split('.')
            if len(sentences) > 3:
                # Keep first two and last sentence
                refined = '. '.join(sentences[:2] + [sentences[-1]]) + '.'
        
        return refined

class IntelligentDocumentAnalyzer:
    """Main analyzer class that orchestrates the document analysis process"""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        self.embedding_manager = EmbeddingManager(cache_dir)
        self.pdf_processor = PDFProcessor()
        self.relevance_scorer = RelevanceScorer(self.embedding_manager)
        self.subsection_generator = SubSectionGenerator()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        self.logger.info("Initialized IntelligentDocumentAnalyzer")
    
    def analyze_documents(self, input_data: Dict[str, Any], 
                         input_dir: str = "input") -> Dict[str, Any]:
        """Main analysis function that processes documents and returns results"""
        start_time = time.time()
        self.logger.info("Starting document analysis")
        
        try:
            # Extract input components
            challenge_info = input_data.get('challenge_info', {})
            documents = input_data.get('documents', [])
            persona = input_data.get('persona', {})
            job_to_be_done = input_data.get('job_to_be_done', {})
            
            # Create persona profile
            persona_profile = PersonaProfileBuilder.create_profile(persona, job_to_be_done)
            self.logger.info(f"Created persona profile: {persona_profile[:100]}...")
            
            # Process all documents
            all_sections = []
            processed_docs = []
            
            for doc_info in documents:
                filename = doc_info['filename']
                title = doc_info.get('title', filename)
                
                # Clean document title - remove .pdf extension and clean formatting
                clean_title = self._clean_document_title(title)
                
                pdf_path = os.path.join(input_dir, filename)
                
                if os.path.exists(pdf_path):
                    sections = self.pdf_processor.extract_text_from_pdf(pdf_path)
                    
                    # Add document metadata to sections
                    for section in sections:
                        section['document'] = clean_title
                    
                    all_sections.extend(sections)
                    processed_docs.append(clean_title)
                    self.logger.info(f"Processed {filename}: {len(sections)} sections")
                else:
                    self.logger.warning(f"Document not found: {pdf_path}")
            
            if not all_sections:
                self.logger.warning("No sections extracted from any document")
                return self._create_empty_result(input_data, start_time)
            
            # Score sections for relevance
            scored_sections = self.relevance_scorer.score_sections(all_sections, persona_profile)
            
            # Get top sections (limit for performance)
            top_sections = scored_sections[:25]
            
            # Generate sub-sections
            subsections = self.subsection_generator.generate_subsections(
                top_sections, max_subsections=20
            )
            
            # Create final output
            result = {
                "metadata": {
                    "input_documents": processed_docs,
                    "persona": persona,
                    "job_to_be_done": job_to_be_done,
                    "processing_timestamp": datetime.now().isoformat(),
                    "processing_time_seconds": round(time.time() - start_time, 2),
                    "total_sections_found": len(all_sections),
                    "top_sections_analyzed": len(top_sections),
                    "subsections_generated": len(subsections)
                },
                "extracted_sections": [section.to_dict() for section in top_sections],
                "sub_section_analysis": [subsection.to_dict() for subsection in subsections]
            }
            
            processing_time = time.time() - start_time
            self.logger.info(f"Analysis completed in {processing_time:.2f} seconds")
            self.logger.info(f"Found {len(top_sections)} relevant sections")
            self.logger.info(f"Generated {len(subsections)} sub-sections")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error during analysis: {str(e)}")
            return self._create_error_result(input_data, start_time, str(e))
    
    def _create_empty_result(self, input_data: Dict[str, Any], start_time: float) -> Dict[str, Any]:
        """Create empty result when no content is found"""
        return {
            "metadata": {
                "input_documents": [doc.get('title', doc.get('filename', '')) 
                                  for doc in input_data.get('documents', [])],
                "persona": input_data.get('persona', {}),
                "job_to_be_done": input_data.get('job_to_be_done', {}),
                "processing_timestamp": datetime.now().isoformat(),
                "processing_time_seconds": round(time.time() - start_time, 2),
                "status": "no_content_found"
            },
            "extracted_sections": [],
            "sub_section_analysis": []
        }
    
    def _create_error_result(self, input_data: Dict[str, Any], 
                           start_time: float, error_msg: str) -> Dict[str, Any]:
        """Create error result when processing fails"""
        return {
            "metadata": {
                "input_documents": [doc.get('title', doc.get('filename', '')) 
                                  for doc in input_data.get('documents', [])],
                "persona": input_data.get('persona', {}),
                "job_to_be_done": input_data.get('job_to_be_done', {}),
                "processing_timestamp": datetime.now().isoformat(),
                "processing_time_seconds": round(time.time() - start_time, 2),
                "status": "error",
                "error_message": error_msg
            },
            "extracted_sections": [],
            "sub_section_analysis": []
        }
    
    def _clean_document_title(self, title: str) -> str:
        """Clean document title for better presentation"""
        # Remove .pdf extension
        title = re.sub(r'\.pdf$', '', title, flags=re.IGNORECASE)
        # Clean up common patterns
        title = title.strip()
        return title

def main():
    """Main execution function"""
    if len(sys.argv) < 2:
        print("Usage: python intelligent_document_analyzer.py <input_json_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)
    
    try:
        # Load input data
        with open(input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        
        # Initialize analyzer
        analyzer = IntelligentDocumentAnalyzer()
        
        # Process documents
        result = analyzer.analyze_documents(input_data)
        
        # Create output filename
        output_file = input_file.replace('input', 'output')
        if not output_file.endswith('_output.json'):
            output_file = output_file.replace('.json', '_output.json')
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print(f"\n{'='*60}")
        print("INTELLIGENT DOCUMENT ANALYSIS COMPLETED")
        print(f"{'='*60}")
        print(f"Input file: {input_file}")
        print(f"Output file: {output_file}")
        print(f"Processing time: {result['metadata']['processing_time_seconds']} seconds")
        print(f"Documents processed: {len(result['metadata']['input_documents'])}")
        print(f"Relevant sections found: {len(result['extracted_sections'])}")
        print(f"Sub-sections generated: {len(result['sub_section_analysis'])}")
        
        if result['extracted_sections']:
            print(f"Top section confidence: {result['extracted_sections'][0]['confidence_score']:.4f}")
        
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
