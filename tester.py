from parser import extract_text_from_pdf, RobustPDFParser, PDFParserError

def main():
    try:
        print("Starting PDF processing...")
        
        # Simple usage (backward compatible)
        print("Extracting text from PDF...")
        text = extract_text_from_pdf("Uploads/Oracle.pdf", 36, 81, "output.txt")
        print(f"Extracted {len(text)} characters of text")
        
        # Advanced usage with configuration
        print("Creating parser and extracting tables...")
        parser = RobustPDFParser(max_file_size_mb=200)
        tables = parser.extract_tables_with_pymupdf("Uploads/Oracle.pdf", min_table_rows=3)
        print(f"Extracted {len(tables)} tables")
        
        # With error handling - FIXED regex patterns for your format (1.01, 1.02, etc.)
        print("Chunking text by control patterns...")
        try:
            # Updated patterns for your control ID format
            control_patterns = [
                r"\d+\.\d+",        # Matches 1.01, 1.02, 7.12, etc.
                r"\d+\.\d+\.\d+",   # Matches 1.01.1, 1.02.3, etc. (if needed)
            ]
            
            chunks = parser.chunk_text_by_patterns(text, control_patterns)
            print(f"Created {len(chunks)} chunks")
            
            # Display first few chunks as sample
            if not chunks.empty:
                print("\nFirst 3 chunks:")
                for i, row in chunks.head(3).iterrows():
                    print(f"Control ID: {row['Control ID']}")
                    print(f"Content preview: {row['Content'][:100]}...")
                    print("-" * 50)
            
        except PDFParserError as e:
            print(f"Processing failed: {e}")
        
        # Process tables if any were found
        if tables:
            print(f"\nProcessing {len(tables)} tables...")
            processed_tables = parser.process_tables(tables)
            print(f"Successfully processed {len(processed_tables)} tables")
            
            # Show table info
            for i, table in enumerate(processed_tables):
                print(f"Table {i+1}: {table.shape[0]} rows, {table.shape[1]} columns")
                print(f"Columns: {list(table.columns)}")
        
        print("\nProcessing completed successfully!")
        
    except FileNotFoundError:
        print("Error: Could not find 'Uploads/Oracle.pdf'. Please check the file path.")
    except PDFParserError as e:
        print(f"PDF parsing error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()