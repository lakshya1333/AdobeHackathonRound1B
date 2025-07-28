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

PS C:\Users\nayus\Adobe_Project\AdobeHackathonRound1A\Adobe_Round-1A - Copy\Adobe_Round-1A> cd "c:\Users\nayus\Adobe_Project\AdobeHackathonRound1A\Adobe_Round-1A - Copy\Adobe_Round-1A" && python verify-submission.py                                                 
🔍 Adobe Hackathon Round 1B - Submission Verification
============================================================
✅ Main application: intelligent_document_analyzer.py
✅ Dependencies: requirements.txt
✅ Container config: Dockerfile
✅ Documentation: README.md
✅ Input directory: input/
✅ Output directory: output/
✅ Cache directory: cache/

📦 Python Dependencies:
✅ PyMuPDF: Available
2025-07-28 20:50:01.295326: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-07-28 20:50:06.134586: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
WARNING:tensorflow:From C:\Users\nayus\AppData\Local\Programs\Python\Python312\Lib\site-packages\tf_keras\src\losses.py:2976: The name tf.losses.sparse_softmax_cross_entropy is deprecated. Please use tf.compat.v1.losses.sparse_softmax_cross_entropy instead.       

✅ sentence-transformers: Available
✅ torch: Available
✅ numpy: Available
✅ scikit-learn: Available

✅ Main module: Importable
✅ Main module: Importable

============================================================      
🚀 SUBMISSION READY! All checks passed.

To test the system:
1. Place PDF files in the 'input/' directory
2. Run: python intelligent_document_analyzer.py
3. Check results in the 'output/' directory

For Docker:
1. Build: docker build -t intelligent-analyzer .
2. Run: docker run -v ./input:/app/input -v ./output:/app/output intelligent-analyzer
PS C:\Users\nayus\Adobe_Project\AdobeHackathonRound1A\Adobe_Round-1A - Copy\Adobe_Round-1A> ^C
PS C:\Users\nayus\Adobe_Project\AdobeHackathonRound1A\Adobe_Round-1A - Copy\Adobe_Round-1A> cd "c:\Users\nayus\Adobe_Project\AdobeHackathonRound1A\Adobe_Round-1A - Copy\Adobe_Round-1A" && python verify-submission.py
============================================================
🏆 ADOBE HACKATHON ROUND 1B - JUDGE VERIFICATION
============================================================      
Testing: Intelligent Document Analyzer
Theme: Connect What Matters — For the User Who Matters
============================================================      

📁 Checking Core Files...
❌ Main application: intelligent_document_analyzer.py (MISSING)   
❌ Dependencies: requirements.txt (MISSING)
✅ Container config: Dockerfile
❌ Documentation: README.md (MISSING)
❌ Input directory: input/ (MISSING)
❌ Output directory: output/ (MISSING)

📦 Checking Python Dependencies...
✅ PyMuPDF: Available
2025-07-28 21:11:59.131186: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-07-28 21:12:01.840904: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
WARNING:tensorflow:From C:\Users\nayus\AppData\Local\Programs\Python\Python312\Lib\site-packages\tf_keras\src\losses.py:2976: The name tf.losses.sparse_softmax_cross_entropy is deprecated. Please use tf.compat.v1.losses.sparse_softmax_cross_entropy instead.       

✅ sentence-transformers: Available
✅ torch: Available
✅ numpy: Available
✅ scikit-learn: Available

🧠 Testing System Analysis...
✅ Main module: Importable

🧪 Testing Judge Scenario (No Cache)...
--------------------------------------------------
  Backed up existing cache for testing
🚀 Running analyzer without cache...
❌ FAILED: C:\Users\nayus\AppData\Local\Programs\Python\Python312\python.exe: can't open file 'C:\\Users\\nayus\\Adobe_Project\\AdobeHackathonRound1A\\Adobe_Round-1A - Copy\\Adobe_Round-1A\\intelligent_document_analyzer.py': [Errno 2] No such file or directory    

📝 STDOUT:
📦 Restored original cache

🐳 Testing Docker Build...
--------------------------------------------------
🔨 Building Docker container...
⚠️  Docker not available (skipping Docker test)

============================================================      
📋 FINAL VERIFICATION RESULTS
============================================================      
📁 Core Files: FAIL
📦 Dependencies: PASS
🧠 Import Test: PASS
🎯 Judge Scenario (No Cache): FAIL
🐳 Docker Build: SKIPPED (Docker not available)
============================================================      
📋 FINAL VERIFICATION RESULTS
============================================================      
📁 Core Files: FAIL
📦 Dependencies: PASS
🧠 Import Test: PASS
🎯 Judge Scenario (No Cache): FAIL
🐳 Docker Build: SKIPPED (Docker not available)
📋 FINAL VERIFICATION RESULTS
============================================================      
📁 Core Files: FAIL
📦 Dependencies: PASS
🧠 Import Test: PASS
🎯 Judge Scenario (No Cache): FAIL
🐳 Docker Build: SKIPPED (Docker not available)
============================================================      
📁 Core Files: FAIL
📦 Dependencies: PASS
🧠 Import Test: PASS
🎯 Judge Scenario (No Cache): FAIL
🐳 Docker Build: SKIPPED (Docker not available)
📁 Core Files: FAIL
📦 Dependencies: PASS
🧠 Import Test: PASS
🎯 Judge Scenario (No Cache): FAIL
🐳 Docker Build: SKIPPED (Docker not available)
============================================================      
❌ SUBMISSION NEEDS ATTENTION
Please fix the issues above before submission
📦 Dependencies: PASS
🧠 Import Test: PASS
🎯 Judge Scenario (No Cache): FAIL
🐳 Docker Build: SKIPPED (Docker not available)
============================================================      
❌ SUBMISSION NEEDS ATTENTION
Please fix the issues above before submission
PS C:\Users\nayus\Adobe_Project\AdobeHackathonRound1A\Adobe_Round-1A - Copy\Ad^Ce_Round-1A>
PS C:\Users\nayus\Adobe_Project\AdobeHackathonRound1A\Adobe_Round-1A - Copy\Adobe_Round-1A> cd "c:\Users\nayus\Adobe_Project\AdobeHackathonRound1A\Adobe_Round-1A - Copy\AdobeRound1B" && python intelligent_document_analyzer.py "input/challenge1b_input (1).json"
2025-07-28 22:24:28.509752: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-07-28 22:24:32.797792: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
WARNING:tensorflow:From C:\Users\nayus\AppData\Local\Programs\Python\Python312\Lib\site-packages\tf_keras\src\losses.py:2976: The name tf.losses.sparse_softmax_cross_entropy is deprecated. Please use tf.compat.v1.losses.sparse_softmax_cross_entropy instead.

2025-07-28 22:24:55,798 - INFO - Use pytorch device_name: cpu
2025-07-28 22:24:55,798 - INFO - Load pretrained SentenceTransformer: all-mpnet-base-v2
2025-07-28 22:25:09,821 - INFO - Initialized embedding model: all-mpnet-base-v2
2025-07-28 22:25:09,821 - INFO - Initialized IntelligentDocumentAnalyzer
2025-07-28 22:25:09,821 - INFO - Starting document analysis
2025-07-28 22:25:09,821 - INFO - Created persona profile: Role: hr professional Task: create and manage fillable forms for onboarding and compliance. Task foc...
2025-07-28 22:25:09,828 - INFO - Extracting text from input\Learn Acrobat - Create and Convert_1.pdf
2025-07-28 22:25:10,538 - INFO - Extracted 77 sections from input\Learn Acrobat - Create and Convert_1.pdf
2025-07-28 22:25:10,541 - INFO - Processed Learn Acrobat - Create and Convert_1.pdf: 77 sections
2025-07-28 22:25:10,541 - INFO - Extracting text from input\Learn Acrobat - Create and Convert_2.pdf
2025-07-28 22:25:10,702 - INFO - Extracted 54 sections from input\Learn Acrobat - Create and Convert_2.pdf
2025-07-28 22:25:10,705 - INFO - Processed Learn Acrobat - Create and Convert_2.pdf: 54 sections
2025-07-28 22:25:10,705 - INFO - Extracting text from input\Learn Acrobat - Edit_1.pdf
2025-07-28 22:25:10,999 - INFO - Extracted 20 sections from input\Learn Acrobat - Edit_1.pdf
2025-07-28 22:25:11,001 - INFO - Processed Learn Acrobat - Edit_1.pdf: 20 sections
2025-07-28 22:25:11,001 - INFO - Extracting text from input\Learn Acrobat - Edit_2.pdf
2025-07-28 22:25:11,254 - INFO - Extracted 28 sections from input\Learn Acrobat - Edit_2.pdf
2025-07-28 22:25:11,262 - INFO - Processed Learn Acrobat - Edit_2.pdf: 28 sections
2025-07-28 22:25:11,263 - INFO - Extracting text from input\Learn Acrobat - Export_1.pdf
2025-07-28 22:25:11,507 - INFO - Extracted 66 sections from input\Learn Acrobat - Export_1.pdf
2025-07-28 22:25:11,507 - INFO - Processed Learn Acrobat - Export_1.pdf: 66 sections
2025-07-28 22:25:11,513 - INFO - Extracting text from input\Learn Acrobat - Export_2.pdf
2025-07-28 22:25:11,567 - INFO - Extracted 7 sections from input\Learn Acrobat - Export_2.pdf
2025-07-28 22:25:11,571 - INFO - Processed Learn Acrobat - Export_2.pdf: 7 sections
2025-07-28 22:25:11,571 - WARNING - Document not found: input\Learn Acrobat - Fill and Sign.pdf     
2025-07-28 22:25:11,571 - INFO - Extracting text from input\Learn Acrobat - Generative AI_1.pdf     
2025-07-28 22:25:12,747 - INFO - Extracted 47 sections from input\Learn Acrobat - Generative AI_1.pdf
2025-07-28 22:25:12,747 - INFO - Processed Learn Acrobat - Generative AI_1.pdf: 47 sections
2025-07-28 22:25:12,753 - INFO - Extracting text from input\Learn Acrobat - Generative AI_2.pdf     
2025-07-28 22:25:13,216 - INFO - Extracted 23 sections from input\Learn Acrobat - Generative AI_2.pdf
2025-07-28 22:25:13,221 - INFO - Processed Learn Acrobat - Generative AI_2.pdf: 23 sections
2025-07-28 22:25:13,222 - INFO - Extracting text from input\Learn Acrobat - Request e-signatures_1.pdf
2025-07-28 22:25:14,105 - INFO - Extracted 48 sections from input\Learn Acrobat - Request e-signatures_1.pdf
2025-07-28 22:25:14,105 - INFO - Processed Learn Acrobat - Request e-signatures_1.pdf: 48 sections
2025-07-28 22:25:14,112 - INFO - Extracting text from input\Learn Acrobat - Request e-signatures_2.pdf
2025-07-28 22:25:14,447 - INFO - Extracted 17 sections from input\Learn Acrobat - Request e-signatures_2.pdf
2025-07-28 22:25:14,449 - INFO - Processed Learn Acrobat - Request e-signatures_2.pdf: 17 sections
2025-07-28 22:25:14,449 - INFO - Extracting text from input\Learn Acrobat - Share_1.pdf
2025-07-28 22:25:14,631 - INFO - Extracted 23 sections from input\Learn Acrobat - Share_1.pdf
2025-07-28 22:25:14,637 - INFO - Processed Learn Acrobat - Share_1.pdf: 23 sections
2025-07-28 22:25:14,639 - INFO - Extracting text from input\Learn Acrobat - Share_2.pdf
2025-07-28 22:25:15,239 - INFO - Extracted 30 sections from input\Learn Acrobat - Share_2.pdf
2025-07-28 22:25:15,239 - INFO - Processed Learn Acrobat - Share_2.pdf: 30 sections
2025-07-28 22:25:15,239 - INFO - Extracting text from input\Test Your Acrobat Exporting Skills.pdf  
2025-07-28 22:25:15,289 - INFO - Extracted 2 sections from input\Test Your Acrobat Exporting Skills.pdf
2025-07-28 22:25:15,289 - INFO - Processed Test Your Acrobat Exporting Skills.pdf: 2 sections
2025-07-28 22:25:15,289 - INFO - Extracting text from input\The Ultimate PDF Sharing Checklist.pdf  
2025-07-28 22:25:15,337 - INFO - Extracted 9 sections from input\The Ultimate PDF Sharing Checklist.pdf
2025-07-28 22:25:15,337 - INFO - Processed The Ultimate PDF Sharing Checklist.pdf: 9 sections
2025-07-28 22:25:15,337 - INFO - Scoring 451 sections for relevance
2025-07-28 22:25:20,322 - INFO - Scored and ranked 450 sections
2025-07-28 22:25:20,326 - INFO - Generating sub-sections from 25 top sections
2025-07-28 22:25:20,328 - INFO - Generated 20 sub-sections
2025-07-28 22:25:20,328 - INFO - Analysis completed in 10.51 seconds
2025-07-28 22:25:20,328 - INFO - Found 25 relevant sections
2025-07-28 22:25:20,328 - INFO - Generated 20 sub-sections

============================================================
INTELLIGENT DOCUMENT ANALYSIS COMPLETED
============================================================
Input file: input/challenge1b_input (1).json
Output file: output/challenge1b_output (1)_output.json
2025-07-28 22:25:15,337 - INFO - Processed The Ultimate PDF Sharing Checklist.pdf: 9 sections       
2025-07-28 22:25:15,337 - INFO - Scoring 451 sections for relevance
2025-07-28 22:25:20,322 - INFO - Scored and ranked 450 sections
2025-07-28 22:25:20,326 - INFO - Generating sub-sections from 25 top sections
2025-07-28 22:25:20,328 - INFO - Generated 20 sub-sections
2025-07-28 22:25:20,328 - INFO - Analysis completed in 10.51 seconds
2025-07-28 22:25:20,328 - INFO - Found 25 relevant sections
2025-07-28 22:25:20,328 - INFO - Generated 20 sub-sections

============================================================
INTELLIGENT DOCUMENT ANALYSIS COMPLETED
============================================================
Input file: input/challenge1b_input (1).json
Output file: output/challenge1b_output (1)_output.json
2025-07-28 22:25:20,328 - INFO - Analysis completed in 10.51 seconds
2025-07-28 22:25:20,328 - INFO - Found 25 relevant sections
2025-07-28 22:25:20,328 - INFO - Generated 20 sub-sections

============================================================
INTELLIGENT DOCUMENT ANALYSIS COMPLETED
============================================================
Input file: input/challenge1b_input (1).json
Output file: output/challenge1b_output (1)_output.json
INTELLIGENT DOCUMENT ANALYSIS COMPLETED
============================================================
Input file: input/challenge1b_input (1).json
Output file: output/challenge1b_output (1)_output.json
Input file: input/challenge1b_input (1).json
Output file: output/challenge1b_output (1)_output.json
Output file: output/challenge1b_output (1)_output.json
Processing time: 10.51 seconds
Documents processed: 14
Relevant sections found: 25
Sub-sections generated: 20
Top section confidence: 0.3019
============================================================
PS C:\Users\nayus\Adobe_Project\AdobeHackathonRound1A\Adobe_Round-1A - Copy\AdobeRound1B>









Processing time: 10.51 seconds
Documents processed: 14
Relevant sections found: 25
Sub-sections generated: 20
Top section confidence: 0.3019
============================================================
Processing time: 10.51 seconds
Documents processed: 14
Relevant sections found: 25
Sub-sections generated: 20
Top section confidence: 0.3019
Processing time: 10.51 seconds
Documents processed: 14
Relevant sections found: 25
Processing time: 10.51 seconds
Documents processed: 14
Processing time: 10.51 seconds
Documents processed: 14
Processing time: 10.51 seconds
Documents processed: 14
Documents processed: 14
Relevant sections found: 25
Sub-sections generated: 20
Top section confidence: 0.3019
============================================================

**Built for Adobe Hackathon Round 1B - "Connect What Matters — For the User Who Matters"**

*Delivering personalized intelligence through advanced document analysis*

