#!/usr/bin/env python3
"""
server.py – Production FastAPI service for framework mapping with CortexGRC integration.
UPDATED: Returns processed report data (not just raw LLM results) for Supabase integration.

Pipeline: RAG → LLM → Report → Return Processed Report Data

Requires:
    pip install fastapi uvicorn python-multipart pandas openpyxl

Environment:
    export OPENAI_API_KEY="sk-…"
"""

# ── stdlib ───────────────────────────────────────────────────────────────
from pathlib import Path
import tempfile
import logging
import os
import sys
import time
import shutil
import uuid
from datetime import datetime
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import json

# ── third-party ───────────────────────────────────────────────────────────
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

# ── local project modules ────────────────────────────────────────────────
try:
    from llm import FrameworkOverlapAnalyzer, AnalysisConfig
    import report as report
    
    try:
        from rag import MaximumRecallMatcher
        HAS_RAG_MATCHER = True
    except ImportError:
        logging.warning("RAG similarity matcher not found - will use direct framework analysis")
        HAS_RAG_MATCHER = False
        
except ImportError as e:
    logging.error(f"Failed to import required modules: {e}")
    logging.error("Make sure 'llm.py' and 'report.py' are in the same directory")
    sys.exit(1)
import nltk
nlkt.download("punkt_tab")
# ── logging setup with UTF-8 encoding ────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("server.log", encoding='utf-8')
    ]
)
log = logging.getLogger("framework-mapping-server")

# ── constants ─────────────────────────────────────────────────────────────
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {'.xlsx', '.xls', '.csv'}

# ── Pydantic Models for JSON API (Updated for processed report data) ───────
class Control(BaseModel):
    id: str
    framework_id: Optional[str] = None
    ID: str
    Domain: Optional[str] = None
    Sub_Domain: Optional[str] = None
    Controls: str

class FrameworkData(BaseModel):
    framework_name: str
    framework_id: str
    controls: List[Control]

class FrameworkCompareRequest(BaseModel):
    source_framework: FrameworkData
    target_framework: FrameworkData
    top_k: int = 5
    workers: int = 5
    generate_excel: bool = True

class ProcessedMapping(BaseModel):
    """Processed mapping data from report.py - ready for Supabase"""
    framework_a_control_id: str
    framework_a_control_domain: str
    framework_a_control_sub_domain: str
    framework_a_control_statement: str
    framework_b_control_id: str
    framework_b_control: str
    mapping_score: int
    detailed_mapping_analysis: str
    mapping_status: str
    source_control_uuid: Optional[str] = None
    target_control_uuid: Optional[str] = None

class FrameworkCompareResponse(BaseModel):
    status: str
    processing_time: float
    source_framework_name: str
    target_framework_name: str
    total_mappings: int
    excel_path: Optional[str] = None
    processed_mappings: Optional[List[ProcessedMapping]] = None

# ── Enhanced MaximumRecallMatcher ─────────────────────────────────────────
class EnhancedMaximumRecallMatcher(MaximumRecallMatcher if HAS_RAG_MATCHER else object):
    """Enhanced matcher that can work with JSON data from Supabase"""
    
    def __init__(self, *args, **kwargs):
        if HAS_RAG_MATCHER:
            super().__init__(*args, **kwargs)
        else:
            self.openai_api_key = kwargs.get('openai_api_key', '')
    
    def load_json_controls(self, controls_data: List[Dict]) -> List[Dict]:
        """Load controls from JSON data (Supabase format)"""
        controls = []
        for ctrl_data in controls_data:
            control_text = str(ctrl_data.get('Controls', '')).strip()
            control_id = str(ctrl_data.get('ID', ctrl_data.get('id', ''))).strip()
            
            if control_text and control_text.lower() != 'nan':
                controls.append({
                    'id': control_id,
                    'raw_text': control_text,
                    'domain': ctrl_data.get('Domain', ''),
                    'sub_domain': ctrl_data.get('Sub_Domain', ''),
                    'framework_id': str(ctrl_data.get('framework_id', ''))
                })
        
        return controls
    
    def match_controls_from_json(self, source_controls_json: List[Dict], 
                                target_controls_json: List[Dict], top_k: int = 5) -> pd.DataFrame:
        """Main matching pipeline for JSON data"""
        if not HAS_RAG_MATCHER:
            return self._fallback_match_controls(source_controls_json, target_controls_json, top_k)
        
        source_controls = self.load_json_controls(source_controls_json)
        target_controls = self.load_json_controls(target_controls_json)
        
        return self.match_controls(source_controls, target_controls, top_k)
    
    def _fallback_match_controls(self, source_controls_json: List[Dict], 
                               target_controls_json: List[Dict], top_k: int = 5) -> pd.DataFrame:
        """Simple fallback matching when RAG matcher is not available"""
        log.warning("Using fallback matching - RAG matcher not available")
        
        results = []
        for source_ctrl in source_controls_json:
            source_text = source_ctrl.get('Controls', '').lower()
            source_id = source_ctrl.get('ID', source_ctrl.get('id', ''))
            source_uuid = source_ctrl.get('id', '')
            
            matches = []
            for target_ctrl in target_controls_json:
                target_text = target_ctrl.get('Controls', '').lower()
                target_id = target_ctrl.get('ID', target_ctrl.get('id', ''))
                target_uuid = target_ctrl.get('id', '')
                
                source_words = set(source_text.split())
                target_words = set(target_text.split())
                common_words = source_words & target_words
                
                if len(common_words) > 2:
                    score = len(common_words) / max(len(source_words), len(target_words))
                    matches.append({
                        'target_id': target_id,
                        'target_text': target_ctrl.get('Controls', ''),
                        'target_uuid': target_uuid,
                        'score': score
                    })
            
            matches.sort(key=lambda x: x['score'], reverse=True)
            
            for rank, match in enumerate(matches[:top_k]):
                results.append({
                    'source_id': source_id,
                    'source_text': source_ctrl.get('Controls', ''),
                    'target_id': match['target_id'],
                    'target_text': match['target_text'],
                    'rank': rank + 1,
                    'source_control_uuid': source_uuid,
                    'target_control_uuid': match['target_uuid']
                })
        
        return pd.DataFrame(results)

def validate_framework_data(framework_data: FrameworkData) -> None:
    """Validate framework data structure"""
    if not framework_data.controls:
        raise HTTPException(400, f"No controls provided for framework '{framework_data.framework_name}'")
    
    for i, control in enumerate(framework_data.controls):
        if not control.Controls or not control.Controls.strip():
            raise HTTPException(400, f"Control {i+1} in framework '{framework_data.framework_name}' has empty Controls text")
        
        if not control.ID or not control.ID.strip():
            raise HTTPException(400, f"Control {i+1} in framework '{framework_data.framework_name}' has empty ID")

def validate_excel_file(file: UploadFile, content: bytes) -> None:
    """Validate uploaded Excel file"""
    if not file.filename:
        raise HTTPException(400, "No filename provided")
    
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Only Excel/CSV files are supported. Got: {file_ext}")
    
    if len(content) > MAX_FILE_SIZE:
        size_mb = len(content) / (1024 * 1024)
        raise HTTPException(400, f"File size ({size_mb:.1f}MB) exceeds limit of {MAX_FILE_SIZE/(1024*1024):.0f}MB")
    
    if file_ext in ['.xlsx', '.xls']:
        if not content.startswith(b'PK\x03\x04'):
            raise HTTPException(400, f"File '{file.filename}' does not appear to be a valid Excel file.")

def load_processed_report_data(final_report_path: Path, source_controls: List[Dict]) -> List[ProcessedMapping]:
    """
    Load the processed, filtered, cleaned report data from report.py output.
    This includes all the filtering and cleaning that report.py performs.
    """
    try:
        log.info(f"Loading processed report data from {final_report_path}")
        
        # Read the final report Excel file
        df_report = pd.read_excel(final_report_path, sheet_name="Framework Mapping")
        
        # Create UUID lookup for source controls
        uuid_lookup = {ctrl.get('ID', ''): ctrl.get('id', '') for ctrl in source_controls}
        
        processed_mappings = []
        
        for _, row in df_report.iterrows():
            # Extract data from the report (which has already been filtered and cleaned)
            framework_a_id = str(row.get('Framework A Control ID', ''))
            source_uuid = uuid_lookup.get(framework_a_id, '')
            
            # For target UUID, we need to extract from Framework B Control ID if it's not 'N/A'
            framework_b_id = str(row.get('Framework B Control ID', ''))
            target_uuid = ''  # We'll need to look this up from the original target controls if needed
            
            processed_mapping = ProcessedMapping(
                framework_a_control_id=framework_a_id,
                framework_a_control_domain=str(row.get('Framework A Control Domain', '')),
                framework_a_control_sub_domain=str(row.get('Framework A Control Sub-Domain', '')),
                framework_a_control_statement=str(row.get('Framework A Control Statement', '')),
                framework_b_control_id=framework_b_id,
                framework_b_control=str(row.get('Framework B Control', '')),
                mapping_score=int(row.get('Mapping Score', 0)),
                detailed_mapping_analysis=str(row.get('Detailed Mapping Analysis', '')),
                mapping_status=str(row.get('Mapping Status', '')),
                source_control_uuid=source_uuid,
                target_control_uuid=target_uuid
            )
            
            processed_mappings.append(processed_mapping)
        
        log.info(f"Loaded {len(processed_mappings)} processed mappings from report")
        return processed_mappings
        
    except Exception as e:
        log.error(f"Error loading processed report data: {e}")
        return []

def run_pipeline_from_json(
    source_controls: List[Dict],
    target_controls: List[Dict],
    *,
    top_k: int = 5,
    workers: int = 5,
    tmp_dir: Path,
) -> Tuple[Path, List[ProcessedMapping]]:
    """
    Execute the complete framework mapping pipeline: RAG → LLM → Report
    Returns: (final_report_path, processed_mappings_from_report)
    """
    pairs_file = tmp_dir / "framework_pairs.xlsx"
    analyzed_file = tmp_dir / "analyzed_overlaps.xlsx"
    final_report = tmp_dir / "Framework_Mapping_Report.xlsx"

    try:
        if not source_controls or not target_controls:
            raise ValueError("Both source and target controls are required")

        # Stage 1: RAG - Create framework pairs for analysis
        log.info(f"Stage 1/3: RAG - Creating framework pairs (top-{top_k})")
        
        if HAS_RAG_MATCHER:
            try:
                matcher = EnhancedMaximumRecallMatcher(openai_api_key=os.environ["OPENAI_API_KEY"])
                df_pairs = matcher.match_controls_from_json(source_controls, target_controls, top_k)
                df_pairs.to_excel(pairs_file, index=False)
                log.info(f"RAG matching created {len(df_pairs)} control pairs")
            except Exception as e:
                log.warning(f"Enhanced matching failed: {e}. Falling back to basic matching.")
                matcher = EnhancedMaximumRecallMatcher(openai_api_key=os.environ.get("OPENAI_API_KEY", ""))
                df_pairs = matcher._fallback_match_controls(source_controls, target_controls, top_k)
                df_pairs.to_excel(pairs_file, index=False)
        else:
            log.info("Using basic framework pairing (no RAG matcher)")
            pairs = []
            for source_ctrl in source_controls[:50]:
                for target_ctrl in target_controls[:20]:
                    pairs.append({
                        'source_id': source_ctrl.get('ID', source_ctrl.get('id', '')),
                        'source_text': source_ctrl.get('Controls', ''),
                        'target_id': target_ctrl.get('ID', target_ctrl.get('id', '')),
                        'target_text': target_ctrl.get('Controls', ''),
                        'source_control_uuid': source_ctrl.get('id', ''),
                        'target_control_uuid': target_ctrl.get('id', '')
                    })
            
            df_pairs = pd.DataFrame(pairs)
            df_pairs.to_excel(pairs_file, index=False)

        # Stage 2: LLM - Framework overlap analysis
        log.info(f"Stage 2/3: LLM - Analyzing framework overlaps ({workers} workers)")
        analyzer = FrameworkOverlapAnalyzer(
            api_key=os.environ["OPENAI_API_KEY"],
            model="o4-mini",
            config=AnalysisConfig(max_workers=workers, rate_limit_delay=0.1),
        )
        
        df_pairs = pd.read_excel(pairs_file)
        df_analyzed = analyzer.analyze_dataframe(df_pairs, "source_text", "target_text")
        df_analyzed.to_excel(analyzed_file, index=False)
        log.info(f"LLM analysis completed: {len(df_analyzed)} analyzed pairs")
        
        analyzer.cleanup()

        # Stage 3: Report - Generate filtered, cleaned report
        log.info("Stage 3/3: Report - Generating formatted framework mapping report")
        
        # Create temporary framework file for the report generator
        framework_df = pd.DataFrame([
            {
                'ID': ctrl.get('ID', ctrl.get('id', '')),
                'Domain': ctrl.get('Domain', 'Unknown'),
                'Sub-Domain': ctrl.get('Sub_Domain', 'Unknown'),
                'Controls': ctrl.get('Controls', '')
            }
            for ctrl in source_controls
        ])
        
        temp_framework_file = tmp_dir / "temp_framework.xlsx"
        framework_df.to_excel(temp_framework_file, index=False)
        
        success, message = report.generate_report(
            str(analyzed_file), 
            str(temp_framework_file), 
            str(final_report)
        )
        if not success:
            raise Exception(f"Report generation failed: {message}")
        
        # Load the processed, filtered report data (this is what we want to return)
        processed_mappings = load_processed_report_data(final_report, source_controls)
        
        log.info("Framework mapping pipeline completed successfully")
        log.info(f"Final report: {final_report}")
        log.info(f"Processed mappings: {len(processed_mappings)} items")
        
        return final_report, processed_mappings

    except Exception as e:
        log.error(f"Pipeline failed: {str(e)}")
        raise

def validate_framework_file(file_path: Path) -> bool:
    """Validate that uploaded file has required framework structure"""
    try:
        if file_path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)
        
        required_patterns = {
            'id': ['id', 'control_id', 'control id', 'identifier'],
            'text': ['control', 'controls', 'description', 'statement', 'text', 'control_text', 'control text'],
        }
        
        columns_lower = [col.lower() for col in df.columns]
        
        found_id = False
        found_text = False
        
        for pattern in required_patterns['id']:
            if any(pattern in col for col in columns_lower):
                found_id = True
                break
                
        for pattern in required_patterns['text']:
            if any(pattern in col for col in columns_lower):
                found_text = True
                break
        
        if not found_id or not found_text:
            log.warning(f"File {file_path.name} missing required columns. Found: {list(df.columns)}")
            return False
            
        return True
        
    except Exception as e:
        log.error(f"Error validating framework file {file_path}: {e}")
        return False

def prepare_framework_pairs(source_file: Path, target_file: Path, output_file: Path, max_pairs: int = 1000) -> Path:
    """Create framework pairs for analysis when RAG matcher is not available"""
    try:
        if source_file.suffix.lower() in ['.xlsx', '.xls']:
            source_df = pd.read_excel(source_file)
        else:
            source_df = pd.read_csv(source_file)
            
        if target_file.suffix.lower() in ['.xlsx', '.xls']:
            target_df = pd.read_excel(target_file)
        else:
            target_df = pd.read_csv(target_file)
        
        def find_column(df, patterns):
            columns_lower = [col.lower() for col in df.columns]
            for pattern in patterns:
                for i, col in enumerate(columns_lower):
                    if pattern in col:
                        return df.columns[i]
            return None
        
        source_id_col = find_column(source_df, ['id', 'control_id', 'identifier'])
        source_text_col = find_column(source_df, ['control', 'description', 'statement', 'text'])
        
        target_id_col = find_column(target_df, ['id', 'control_id', 'identifier'])
        target_text_col = find_column(target_df, ['control', 'description', 'statement', 'text'])
        
        if not all([source_id_col, source_text_col, target_id_col, target_text_col]):
            raise ValueError("Could not identify required columns in framework files")
        
        log.info(f"Source: ID='{source_id_col}', Text='{source_text_col}'")
        log.info(f"Target: ID='{target_id_col}', Text='{target_text_col}'")
        
        pairs = []
        max_source = min(len(source_df), 50)
        max_target_per_source = min(len(target_df), max_pairs // max_source)
        
        log.info(f"Creating pairs: {max_source} source × {max_target_per_source} target = {max_source * max_target_per_source} pairs")
        
        for i, source_row in source_df.head(max_source).iterrows():
            for j, target_row in target_df.head(max_target_per_source).iterrows():
                pairs.append({
                    'source_id': str(source_row[source_id_col]),
                    'source_text': str(source_row[source_text_col]),
                    'target_id': str(target_row[target_id_col]),
                    'target_text': str(target_row[target_text_col])
                })
        
        pairs_df = pd.DataFrame(pairs)
        pairs_df.to_excel(output_file, index=False)
        
        log.info(f"Created {len(pairs)} framework pairs for analysis")
        return output_file
        
    except Exception as e:
        log.error(f"Error creating framework pairs: {e}")
        raise

def run_pipeline_from_files(
    source_excel_path: Path,
    target_excel_path: Path,
    *,
    top_k: int = 5,
    workers: int = 5,
    tmp_dir: Path,
) -> Path:
    """Execute the framework mapping pipeline from Excel files (original implementation)"""
    pairs_file = tmp_dir / "framework_pairs.xlsx"
    analyzed_file = tmp_dir / "analyzed_overlaps.xlsx"
    final_report = tmp_dir / "Framework_Mapping_Report.xlsx"

    try:
        if not validate_framework_file(source_excel_path):
            raise ValueError(f"Source file {source_excel_path.name} does not appear to be a valid framework file")
        if not validate_framework_file(target_excel_path):
            raise ValueError(f"Target file {target_excel_path.name} does not appear to be a valid framework file")

        # Stage 1: Create framework pairs for analysis
        if HAS_RAG_MATCHER:
            log.info(f"Stage 1/3: Running similarity matching (top-{top_k})")
            try:
                matcher = MaximumRecallMatcher(os.environ["OPENAI_API_KEY"])
                df_pairs = matcher.run_pipeline(
                    str(source_excel_path),
                    str(target_excel_path),
                    str(tmp_dir),
                    top_k,
                    None,
                )
                df_pairs.to_excel(pairs_file, index=False)
            except Exception as e:
                log.warning(f"RAG matching failed: {e}. Falling back to direct pairing.")
                prepare_framework_pairs(source_excel_path, target_excel_path, pairs_file)
        else:
            log.info("Stage 1/3: Creating framework pairs (no RAG matcher)")
            prepare_framework_pairs(source_excel_path, target_excel_path, pairs_file)

        # Stage 2: Framework overlap analysis
        log.info(f"Stage 2/3: Analyzing framework overlaps ({workers} workers)")
        analyzer = FrameworkOverlapAnalyzer(
            api_key=os.environ["OPENAI_API_KEY"],
            model="o4-mini",
            config=AnalysisConfig(max_workers=workers, rate_limit_delay=0.1),
        )
        
        df_pairs = pd.read_excel(pairs_file)
        df_analyzed = analyzer.analyze_dataframe(df_pairs, "source_text", "target_text")
        df_analyzed.to_excel(analyzed_file, index=False)
        
        analyzer.cleanup()

        # Stage 3: Generate formatted report
        log.info("Stage 3/3: Generating formatted framework mapping report")
        success, message = report.generate_report(
            str(analyzed_file), 
            str(source_excel_path), 
            str(final_report)
        )
        if not success:
            raise Exception(f"Report generation failed: {message}")
        
        log.info("Framework mapping report generated successfully")
        return final_report

    except Exception as e:
        log.error(f"Pipeline failed: {str(e)}")
        raise

def create_mock_report() -> Path:
    """Create a realistic mock framework mapping report for testing"""
    mock_data = {
        'Sr. No.': range(1, 21),
        'Framework A Control ID': [f"CIS-{i//4 + 1}.{i%4 + 1}" for i in range(20)],
        'Framework A Control Domain': ['Asset Management'] * 5 + ['Access Control'] * 5 + ['System Security'] * 5 + ['Incident Response'] * 5,
        'Framework A Control Sub-Domain': ['Hardware Assets', 'Software Assets', 'Data Flow', 'Asset Inventory', 'Asset Configuration'] * 4,
        'Framework A Control Statement': [f'Mock Framework A control statement for control {i//4 + 1}.{i%4 + 1}' for i in range(20)],
        'Framework B Control ID': [f'NIST-{i+100}' if i % 3 != 0 else 'N/A' for i in range(20)],
        'Framework B Control': [f'Mock Framework B control {i+100}' if i % 3 != 0 else 'No equivalent controls found in Framework B.' for i in range(20)],
        'Mapping Score': [100 if i % 3 == 0 else 50 if i % 3 == 1 else 0 for i in range(20)],
        'Detailed Mapping Analysis': [f'Mock framework mapping analysis for control {i+1}' for i in range(20)],
        'Mapping Status': ['Direct Mapping' if i % 3 == 0 else 'Partial Mapping' if i % 3 == 1 else 'No Mapping' for i in range(20)]
    }
    
    df = pd.DataFrame(mock_data)
    temp_file = Path(tempfile.gettempdir()) / f"mock_framework_report_{int(time.time())}.xlsx"
    df.to_excel(temp_file, index=False, sheet_name='Framework Mapping')
    
    return temp_file

# ── FastAPI app ───────────────────────────────────────────────────────────
app = FastAPI(
    title="Framework Mapping API",
    description="Framework mapping with processed report data for Supabase integration.",
    version="4.3",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "openai_key_configured": bool(os.getenv("OPENAI_API_KEY")),
        "max_file_size_mb": MAX_FILE_SIZE / (1024 * 1024),
        "allowed_extensions": list(ALLOWED_EXTENSIONS),
        "test_mode_available": True,
        "rag_matcher_available": HAS_RAG_MATCHER,
        "pipeline_flow": "RAG → LLM → Report → Return Processed Report Data",
        "version": "4.3",
        "features": ["Processed report data", "Supabase integration", "Filtered mappings"]
    }

@app.post("/process-json", summary="Map frameworks and return processed report data")
async def process_frameworks_json(request: FrameworkCompareRequest) -> FrameworkCompareResponse:
    """
    Process frameworks and return the processed, filtered report data (not raw LLM results).
    This includes all filtering and cleaning performed by report.py.
    """
    start_time = time.time()
    
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(500, "OPENAI_API_KEY environment variable missing")

    try:
        validate_framework_data(request.source_framework)
        validate_framework_data(request.target_framework)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Invalid framework data: {str(e)}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        try:
            log.info(f"Processing {request.source_framework.framework_name} vs {request.target_framework.framework_name}")
            
            source_controls = [ctrl.dict() for ctrl in request.source_framework.controls]
            target_controls = [ctrl.dict() for ctrl in request.target_framework.controls]
            
            # Run the complete pipeline: RAG → LLM → Report
            final_report_path, processed_mappings = run_pipeline_from_json(
                source_controls=source_controls,
                target_controls=target_controls,
                top_k=request.top_k,
                workers=request.workers,
                tmp_dir=tmp_dir,
            )
            
            # Handle Excel file generation
            excel_path = None
            if request.generate_excel and final_report_path.exists():
                timestamp = int(time.time())
                download_filename = f"Framework_Mapping_{request.source_framework.framework_name}_vs_{request.target_framework.framework_name}_{timestamp}.xlsx"
                persistent_file = Path(tempfile.gettempdir()) / download_filename
                shutil.copy2(final_report_path, persistent_file)
                excel_path = str(persistent_file)
                log.info(f"Excel report saved to: {excel_path}")
            
            processing_time = time.time() - start_time
            log.info(f"Successfully processed frameworks in {processing_time:.1f} seconds")
            
            return FrameworkCompareResponse(
                status="completed",
                processing_time=processing_time,
                source_framework_name=request.source_framework.framework_name,
                target_framework_name=request.target_framework.framework_name,
                total_mappings=len(processed_mappings),
                excel_path=excel_path,
                processed_mappings=processed_mappings  # ✅ Processed, filtered report data!
            )
            
        except Exception as exc:
            log.exception("Framework mapping pipeline failed")
            error_msg = str(exc)
            if len(error_msg) > 300:
                error_msg = error_msg[:300] + "..."
            raise HTTPException(500, f"Framework analysis failed: {error_msg}")

@app.post("/test", summary="Test mode - mock framework mapping")
async def test_process_files(
    framework_a: UploadFile = File(...),
    framework_b: UploadFile = File(...),
):
    """Test endpoint that returns a pre-generated mock report"""
    start_time = time.time()
    
    try:
        framework_a_content = await framework_a.read()
        framework_b_content = await framework_b.read()
        validate_excel_file(framework_a, framework_a_content)
        validate_excel_file(framework_b, framework_b_content)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Failed to read uploaded files: {str(e)}")

    await asyncio.sleep(2)
    
    try:
        log.info(f"TEST MODE: Received {framework_a.filename} and {framework_b.filename}")
        
        mock_file = create_mock_report()
        
        base_name_a = Path(framework_a.filename).stem
        base_name_b = Path(framework_b.filename).stem
        timestamp = int(time.time())
        download_filename = f"TEST_Framework_Map_{base_name_a}_vs_{base_name_b}_{timestamp}.xlsx"
        
        processing_time = time.time() - start_time
        log.info(f"TEST MODE: Successfully generated mock report in {processing_time:.1f} seconds")
        
        return FileResponse(
            mock_file,
            filename=download_filename,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f"attachment; filename={download_filename}",
                "X-Processing-Time": f"{processing_time:.1f}s",
                "X-Test-Mode": "true",
                "X-Report-Source": "mock",
            }
        )
        
    except Exception as exc:
        log.exception("Test mode failed")
        raise HTTPException(500, f"Test processing failed: {str(exc)}")

@app.post("/process", summary="Map framework files (original endpoint)")
async def process_frameworks(
    framework_a: UploadFile = File(...),
    framework_b: UploadFile = File(...),
    top_k: int = Form(5, ge=1, le=20),
    workers: int = Form(5, ge=1, le=10),
):
    """Original endpoint for processing uploaded Excel files"""
    start_time = time.time()
    
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(500, "OPENAI_API_KEY environment variable missing")

    try:
        framework_a_content = await framework_a.read()
        framework_b_content = await framework_b.read()
        validate_excel_file(framework_a, framework_a_content)
        validate_excel_file(framework_b, framework_b_content)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Failed to read uploaded files: {str(e)}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        framework_a_path = tmp_dir / framework_a.filename
        framework_b_path = tmp_dir / framework_b.filename
        framework_a_path.write_bytes(framework_a_content)
        framework_b_path.write_bytes(framework_b_content)

        try:
            log.info(f"Processing {framework_a.filename} vs {framework_b.filename}")
            
            output_xlsx = run_pipeline_from_files(
                source_excel_path=framework_a_path,
                target_excel_path=framework_b_path,
                top_k=top_k,
                workers=workers,
                tmp_dir=tmp_dir,
            )
            
            base_name_a = Path(framework_a.filename).stem
            base_name_b = Path(framework_b.filename).stem
            timestamp = int(time.time())
            download_filename = f"Framework_Mapping_{base_name_a}_vs_{base_name_b}_{timestamp}.xlsx"
            
            processing_time = time.time() - start_time
            log.info(f"Successfully processed frameworks in {processing_time:.1f} seconds")
            
            persistent_file = Path(tempfile.gettempdir()) / download_filename
            shutil.copy2(output_xlsx, persistent_file)
            
        except Exception as exc:
            log.exception("Framework mapping pipeline failed")
            error_msg = str(exc)
            if len(error_msg) > 300:
                error_msg = error_msg[:300] + "..."
            raise HTTPException(500, f"Framework analysis failed: {error_msg}")

        return FileResponse(
            persistent_file,
            filename=download_filename,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f"attachment; filename={download_filename}",
                "X-Processing-Time": f"{processing_time:.1f}s",
                "X-Analysis-Type": "framework-mapping",
                "X-RAG-Used": str(HAS_RAG_MATCHER),
                "X-Pipeline-Version": "4.3-processed-data",
            }
        )

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Framework Mapping Analysis API",
        "version": "4.3",
        "description": "Returns processed, filtered report data ready for Supabase integration.",
        "pipeline_flow": "RAG → LLM → Report → Return Processed Report Data",
        "features": [
            "Processed report data (not raw LLM results)",
            "Filtered mappings (removes poor matches)",
            "Cleaned data ready for Supabase",
            "All report.py filtering and cleaning included"
        ],
        "endpoints": {
            "health": "/health (GET)",
            "process_json": "/process-json (POST) - Returns processed report data",
            "process": "/process (POST) - Excel file uploads",
            "test": "/test (POST) - Mock processing"
        }
    }

if __name__ == "__main__":
    import uvicorn

    log.info("Starting Framework Mapping Analysis Server v4.3...")
    log.info("Pipeline: RAG → LLM → Report → Return Processed Report Data")
    log.info("Returns filtered, cleaned mappings ready for Supabase")
    
    uvicorn.run(
        "server:app", 
        host="0.0.0.0", 
        port=8001, 
        reload=True,
        access_log=True,
        log_level="info"
    )
