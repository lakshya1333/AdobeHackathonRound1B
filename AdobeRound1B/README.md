# Adobe Hackathon Round 1B - Intelligent Document Analyzer

**Connect What Matters — For the User Who Matters**

An advanced AI-powered document analyzer that intelligently extracts and personalizes content from PDF documents based on user personas and context.

## 🚀 Project Overview

This system addresses the challenge of information overload by providing persona-aware document analysis. It intelligently understands user context and delivers the most relevant content tailored to their specific needs and professional background.

### Key Innovation
- **Persona-Aware Intelligence**: Automatically profiles users and filters content based on their professional context
- **Advanced Semantic Analysis**: Uses state-of-the-art transformer models for deep content understanding
- **Domain-Specific Filtering**: Intelligently applies relevance scoring with domain context awareness
- **Real-Time Processing**: Optimized for sub-60 second processing while maintaining accuracy

## 🎯 Features

### Core Capabilities
- **PDF Document Processing**: Robust text extraction and section detection
- **Intelligent Content Analysis**: Semantic understanding of document structure and meaning
- **Persona Profiling**: Automatic user context detection from queries and preferences
- **Relevance Scoring**: Advanced filtering with domain-specific adjustments
- **Content Personalization**: Tailored recommendations based on user personas

### Technical Highlights
- **Advanced ML Model**: sentence-transformers/all-mpnet-base-v2 (768-dimensional embeddings)
- **Efficient Caching**: Smart embedding cache management for performance
- **Scalable Architecture**: Modular design supporting various document types
- **Domain Intelligence**: Built-in knowledge of professional contexts and preferences

## 📋 Requirements

- **Python**: 3.12+
- **Memory**: <1GB RAM usage
- **Processing**: CPU-only (no GPU required)
- **Performance**: <60 seconds processing time
- **Dependencies**: See `requirements.txt`

## 🛠️ Quick Start

### Local Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run the analyzer
python intelligent_document_analyzer.py
```

### Docker Setup
```bash
# Build container
docker build -t intelligent-analyzer .

# Run container
docker run -v ./input:/app/input -v ./output:/app/output intelligent-analyzer
```

## 📁 Project Structure

```
├── intelligent_document_analyzer.py  # Main application
├── requirements.txt                  # Python dependencies
├── Dockerfile                       # Container configuration
├── input/                          # Input PDF documents
├── output/                         # Generated analysis results
└── cache/                          # ML model and embedding cache
```

## 🧠 How It Works

### 1. Document Processing
- Extracts text from PDF documents using PyMuPDF
- Identifies and segments different document sections
- Preprocesses content for semantic analysis

### 2. Persona Profiling
- Analyzes user queries and context
- Builds comprehensive user profiles with domain keywords
- Identifies professional context and preferences

### 3. Intelligent Analysis
- Generates 768-dimensional semantic embeddings
- Performs similarity matching between user needs and content
- Applies domain-specific relevance scoring

### 4. Content Personalization
- Filters content based on persona relevance
- Prioritizes information matching user context
- Delivers tailored recommendations and insights

## 🎨 Example Use Cases

### Food Service Professional
**Input**: "vegetarian buffet options"
**Output**: Curated vegetarian recipes, plant-based ingredients, dietary alternatives

### HR Professional  
**Input**: "digital form workflows"
**Output**: E-signature processes, document management, compliance features

### Design Professional
**Input**: "creative layout tools"
**Output**: Design features, layout options, creative workflows

## 🏆 Hackathon Submission

This project demonstrates:
- **Innovation**: Novel approach to persona-aware document analysis
- **Technical Excellence**: Advanced ML integration with optimized performance
- **User Focus**: Solves real problem of information overload
- **Scalability**: Architecture supports various domains and use cases

### Performance Metrics
- **Processing Speed**: <60 seconds per document
- **Memory Efficiency**: <1GB RAM usage
- **Accuracy**: High relevance scoring with domain filtering
- **Scalability**: Handles multiple document types and personas

## 🔧 Configuration

The system automatically configures optimal settings, but key parameters can be adjusted:

- **Model**: all-mpnet-base-v2 (768-dim embeddings)
- **Similarity Threshold**: 0.3 for content filtering
- **Cache Management**: Automatic embedding persistence
- **Domain Filters**: Customizable relevance adjustments

## 🚦 Status

✅ **Core System**: Fully functional  
✅ **ML Integration**: Advanced transformer model  
✅ **Persona Profiling**: Intelligent user context detection  
✅ **Performance**: Optimized for hackathon requirements  
✅ **Testing**: Validated on multiple use cases  

---

`If the docker doesnt work but the python files are working properly please refer the instruction below:`

<img width="1280" height="686" alt="image" src="https://github.com/user-attachments/assets/1dcd8439-99bf-4a5e-aea5-798ea5dd5222" />

<img width="1280" height="691" alt="image" src="https://github.com/user-attachments/assets/5cba87c7-b325-441b-bd69-ffa89f34406a" />

<img width="1280" height="692" alt="image" src="https://github.com/user-attachments/assets/ae70a8e3-df66-4c12-960c-5ecf029432d1" />

<img width="1280" height="671" alt="image" src="https://github.com/user-attachments/assets/686123f8-e510-4321-8029-6bf551ac5c36" />

<img width="1280" height="696" alt="image" src="https://github.com/user-attachments/assets/56948d59-de97-4a4d-8cc0-1a093b6fde4f" />

<img width="1279" height="671" alt="image" src="https://github.com/user-attachments/assets/173a9b00-09b0-47b4-a3cd-d88d5172c447" />









**Built for Adobe Hackathon Round 1B - "Connect What Matters — For the User Who Matters"**

*Delivering personalized intelligence through advanced document analysis*

