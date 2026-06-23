# Data Directory

This directory contains the core datasets used by the HUSC RAG Chatbot.

## Files

### 📄 **QA.csv**
- **Purpose**: Q&A pairs for academic inquiries
- **Format**: CSV with `question` and `answer` columns
- **Encoding**: UTF-8 with BOM
- **Sample size**: ~X entries (update this number)
- **Topics**: Academic regulations, course policies, student procedures
- **Sample**: See `sample_QA.csv` for format reference

### 📚 **quyche.pdf** 
- **Purpose**: Official student handbook and regulations
- **Content**: 
  - Student Affairs Regulations (Pages 6-41)
  - Academic Regulations (Pages 42-85)
  - Student Policies & Benefits (Pages 86-98)
  - Scholarship Policies (Pages 99+)
- **Processing**: Structure-aware chunking by chapters/articles
- **Language**: Vietnamese

### 📝 **sample_QA.csv**
- **Purpose**: Sample format for Q&A data contribution
- **Usage**: Template for adding new Q&A pairs
- **Format**: Same as QA.csv (question,answer columns)

## Data Processing Pipeline

1. **CSV Processing**:
   - Hierarchical chunking (256 tokens)
   - Metadata extraction from Q&A pairs
   - Preprocess text normalization

2. **PDF Processing**:
   - Structure-aware chunking (1024 tokens)
   - Legal document parsing (Chương/Điều/Khoản/Mục)
   - Page-range document family inference
   - Cross-reference mapping

## Usage

The RAG engine automatically loads these files based on environment configuration:

```bash
CSV_FILE=data/QA.csv
PDF_FILE=data/quyche.pdf
```

## Contributing Data

### Adding Q&A Pairs
1. Use `sample_QA.csv` as template
2. Ensure UTF-8 encoding with BOM
3. Follow existing question-answer style
4. Test with RAG engine before committing

### Updating PDF Documents  
1. Ensure official source and permissions
2. Maintain same file naming convention
3. Update page ranges in README if structure changes
4. Rebuild vector database after updates

## Statistics

| Metric | Value |
|--------|--------|
| Total Q&A pairs | ~X |
| PDF pages | ~X |
| Document families | 4 |
| Avg chunk size | 256-1024 tokens |
| Vector dimensions | 1024 (BGE-M3) |

*Note: Update statistics after database rebuild*

## Data Security & Compliance

- **Sensitive Info**: Remove any personal data before committing
- **Licensing**: Ensure compliance with university data policies  
- **Updates**: Periodically sync with latest official documents
- **Access Control**: Use appropriate file permissions for sensitive content