#!/usr/bin/env python3
"""
LLM-based Framework Overlap Analyzer - Framework Comparison Edition
Uses OpenAI o4-mini to analyze conceptual overlap between cybersecurity framework controls
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import openai
from openai import OpenAI
from tqdm import tqdm
import argparse
import time
import signal
from datetime import datetime
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict
import yaml
import re
import traceback
from functools import wraps
import html
from concurrent.futures import ThreadPoolExecutor, as_completed

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('framework_analysis_audit.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global shutdown flag for graceful cleanup
shutdown_requested = False

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global shutdown_requested
    logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
    shutdown_requested = True

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

@dataclass
class AnalysisConfig:
    """Configuration for framework overlap analysis parameters"""
    max_retries: int = 3
    base_timeout: int = 30
    base_backoff: float = 1.0
    max_backoff: float = 30.0
    rate_limit_delay: float = 0.0  # Removed artificial delay
    max_workers: int = 5  # Number of concurrent threads
    batch_multiple_pairs: bool = False  # Analyze multiple pairs per API call
    
    @classmethod
    def from_file(cls, config_path: Path) -> 'AnalysisConfig':
        """Load configuration from YAML file"""
        if config_path.exists():
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)
                return cls(**data)
        return cls()
    
    def save_to_file(self, config_path: Path):
        """Save configuration to YAML file"""
        with open(config_path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

@dataclass
class OverlapResult:
    """Result of framework control overlap analysis"""
    overlap_type: str  # 'equivalent', 'overlapping', 'distinct'
    confidence: float  # 0.0 to 1.0
    justification: str
    key_points: List[str]
    gaps_identified: List[str]
    conceptual_strength: str
    audit_notes: str
    
    def __post_init__(self):
        """Validate and clean up the result"""
        # Ensure overlap_type is valid
        if self.overlap_type not in ['equivalent', 'overlapping', 'distinct']:
            logger.warning(f"Invalid overlap_type: {self.overlap_type}, defaulting to distinct")
            self.overlap_type = 'distinct'
        
        # Ensure confidence is in valid range
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        
        # Ensure lists are actually lists
        if not isinstance(self.key_points, list):
            self.key_points = [str(self.key_points)] if self.key_points else []
        if not isinstance(self.gaps_identified, list):
            self.gaps_identified = [str(self.gaps_identified)] if self.gaps_identified else []

@dataclass
class AuditEvent:
    """Audit trail event"""
    timestamp: str
    event_type: str
    framework_a_id: str
    framework_b_id: str
    details: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

def retry(max_attempts: int = 3, base_delay: float = 1.0, max_delay: float = 30.0, 
          exponential_base: float = 2.0):
    """
    Retry decorator with exponential backoff for synchronous functions
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                if shutdown_requested:
                    raise InterruptedError("Shutdown requested")
                
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        break
                    
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
            
            raise last_exception
        return wrapper
    return decorator

class PromptConstructor:
    """Isolated prompt construction for framework overlap analysis"""
    
    @staticmethod
    def sanitize_text(text: str) -> str:
        """Sanitize input text to prevent prompt injection and JSON issues"""
        if not isinstance(text, str):
            text = str(text)
        
        # HTML escape to prevent injection
        text = html.escape(text)
        
        # Remove or escape problematic characters for JSON
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.strip()
        
        # Truncate if too long (prevent token limit issues)
        if len(text) > 4000:
            text = text[:4000] + "... [truncated]"
        
        return text
    
    @classmethod
    def create_overlap_analysis_prompt(cls, framework_a_text: str, framework_b_text: str) -> str:
        """Create a streamlined prompt for framework overlap analysis"""
        # Sanitize inputs
        safe_framework_a = cls.sanitize_text(framework_a_text)
        safe_framework_b = cls.sanitize_text(framework_b_text)
        
        prompt = f"""
You are a **senior cybersecurity-framework analyst** with deep experience mapping controls across NIST CSF, ISO 27001/27002, CIS 18, SOC 2, PCI-DSS, and other global standards.  
Your objective is to deliver a precise, defensible judgement about the *conceptual* relationship between **one control from Framework A** and **one control from Framework B**.

---
## INPUT CONTROLS  
### FRAMEWORK A CONTROL  
{safe_framework_a}

### FRAMEWORK B CONTROL  
{safe_framework_b}

---
## OVERLAP RATING SCALE  

| Rating | Conceptual Overlap | Guidance | Typical Evidence |
|--------|-------------------|----------|------------------|
| **EQUIVALENT** | **≥ 90 %** | Controls pursue the same cybersecurity objective, address identical risk domains, and impose materially indistinguishable requirements. | Direct phrase or requirement parity; interchangeable in audits. |
| **OVERLAPPING** | **40–89 %** | Controls share substantial intent but differ in scope, depth, or governance expectations. | Same domain & threat coverage but one adds/omits sub-requirements. |
| **DISTINCT** | **< 40 %** | Controls target different objectives or risk domains. | Little or no alignment in purpose; only incidental wording overlap. |

---
## ANALYSIS CHECKLIST  
1. **De-bias** — Ignore explicit technology names/tools; focus on the control’s *outcome* (e.g., “maintain an accurate asset inventory” vs. “run a vulnerability scanner”).  
2. **Objective & Risk Domain** — Identify the primary security objective, CIA pillar(s) impacted, and risk domain (e.g., Asset Mgmt, Vulnerability Mgmt, IAM, Incident Response).  
3. **Minimum Required Capabilities** — List what must be in place to *fully* satisfy each control (people, process, governance, technology).  
4. **Tool vs. Control Test** — A tool **enables** a control but does **not** equal the control itself. For example, **vulnerability-management platforms provide discovery data but do *not* constitute an authoritative asset inventory.**  
5. **Granularity & Governance** — Compare depth (frequency, documentation, approvals, measurement, continuous monitoring).  
6. **Complementary/Competing** — Note if controls enhance one another or create obligations that might conflict.  
7. **Edge Cases** — Consider partially scoped controls (e.g., limited to “critical” assets) and note resulting gaps.

---
## REQUIRED OUTPUT  
Respond **only** with valid, minified JSON — no comments, no extra keys, no markdown:

```json
{{
  "overlap_type": "equivalent|overlapping|distinct",
  "confidence": 0.00-1.00,
  "justification": "Concise rationale (≤400 chars)",
  "key_points": ["Up to 3 bullet points on conceptual alignment"],
  "gaps_identified": ["Specific unaligned areas if overlap≠equivalent"],
  "conceptual_strength": "strong|moderate|weak",
  "audit_notes": "Brief notes for mapping workpapers (≤250 chars)"
}}


Provide professional framework analysis as JSON only."""
        
        return prompt

class ResponseParser:
    """Enhanced response parsing for framework analysis results"""
    
    @staticmethod
    def extract_json_from_response(response_text: str) -> str:
        """Extract JSON from response with multiple fallback strategies"""
        if not response_text:
            raise ValueError("Empty response text")
        
        # Strategy 1: Look for markdown code blocks
        json_patterns = [
            r'```json\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
        ]
        
        for pattern in json_patterns:
            match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
            if match:
                json_candidate = match.group(1).strip()
                if json_candidate.startswith('{') and json_candidate.endswith('}'):
                    return json_candidate
        
        # Strategy 2: Look for JSON object boundaries
        json_object_pattern = r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})'
        matches = re.findall(json_object_pattern, response_text, re.DOTALL)
        
        for match in matches:
            if '"overlap_type"' in match:  # Must contain our expected field
                return match.strip()
        
        # Strategy 3: Line-by-line assembly
        lines = response_text.split('\n')
        json_lines = []
        in_json = False
        brace_count = 0
        
        for line in lines:
            stripped = line.strip()
            if not in_json and stripped.startswith('{'):
                in_json = True
                brace_count = 0
            
            if in_json:
                json_lines.append(line)
                brace_count += stripped.count('{') - stripped.count('}')
                
                if brace_count <= 0 and stripped.endswith('}'):
                    break
        
        if json_lines:
            candidate = '\n'.join(json_lines)
            if candidate.strip().startswith('{') and candidate.strip().endswith('}'):
                return candidate.strip()
        
        # Strategy 4: Return the most JSON-like portion
        if '{' in response_text and '}' in response_text:
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            return response_text[start:end]
        
        raise ValueError("No valid JSON found in response")
    
    @classmethod
    def parse_overlap_response(cls, response_text: str, framework_a_id: str = None, framework_b_id: str = None) -> OverlapResult:
        """Parse the LLM response for framework overlap analysis"""
        context = f"[{framework_a_id}->{framework_b_id}]" if framework_a_id and framework_b_id else ""
        
        try:
            # Extract JSON from response
            json_str = cls.extract_json_from_response(response_text)
            
            # Clean up common JSON issues
            json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
            json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
            
            # Parse JSON
            data = json.loads(json_str)
            
            # Validate required fields
            required_fields = ['overlap_type', 'confidence', 'justification', 'key_points']
            for field in required_fields:
                if field not in data:
                    logger.warning(f"{context} Missing required field: {field}")
                    data[field] = cls._get_default_value(field)
            
            return OverlapResult(
                overlap_type=data.get('overlap_type', 'distinct'),
                confidence=data.get('confidence', 0.0),
                justification=data.get('justification', 'No justification provided'),
                key_points=data.get('key_points', []),
                gaps_identified=data.get('gaps_identified', []),
                conceptual_strength=data.get('conceptual_strength', 'unknown'),
                audit_notes=data.get('audit_notes', '')
            )
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"{context} JSON parsing error: {e}")
            logger.debug(f"{context} Response preview: {response_text[:300]}...")
            return cls._fallback_parse(response_text, framework_a_id, framework_b_id)
        
        except Exception as e:
            logger.error(f"{context} Unexpected parsing error: {e}")
            logger.debug(f"{context} Full response: {response_text}")
            return cls._fallback_parse(response_text, framework_a_id, framework_b_id)
    
    @staticmethod
    def _get_default_value(field: str) -> Any:
        """Get default value for missing fields"""
        defaults = {
            'overlap_type': 'distinct',
            'confidence': 0.0,
            'justification': 'Parsing error occurred',
            'key_points': ['Parsing error - manual review required'],
            'gaps_identified': ['Unable to parse response'],
            'conceptual_strength': 'unknown',
            'audit_notes': 'Response parsing failed'
        }
        return defaults.get(field)
    
    @classmethod
    def _fallback_parse(cls, response_text: str, framework_a_id: str = None, framework_b_id: str = None) -> OverlapResult:
        """Fallback parsing when JSON parsing fails"""
        context = f"[{framework_a_id}->{framework_b_id}]" if framework_a_id and framework_b_id else ""
        logger.info(f"{context} Attempting fallback parsing")
        
        lower_text = response_text.lower()
        
        # Extract overlap type using multiple strategies
        overlap_type = 'distinct'
        if any(phrase in lower_text for phrase in ['equivalent', 'same objectives', 'identical purpose']):
            overlap_type = 'equivalent'
        elif any(phrase in lower_text for phrase in ['overlapping', 'partially', 'some overlap']):
            overlap_type = 'overlapping'
        
        # Extract confidence using regex
        confidence = 0.3  # Default fallback confidence
        confidence_patterns = [
            r'confidence["\s:]+([0-9.]+)',
            r'confidence.*?([0-9.]+)',
            r'([0-9.]+).*confidence'
        ]
        
        for pattern in confidence_patterns:
            match = re.search(pattern, lower_text)
            if match:
                try:
                    confidence = float(match.group(1))
                    break
                except ValueError:
                    continue
        
        return OverlapResult(
            overlap_type=overlap_type,
            confidence=confidence,
            justification=f"Fallback parsing used due to response format error. Original response preview: {response_text[:200]}...",
            key_points=['Fallback parsing used', 'Manual review recommended'],
            gaps_identified=['Unable to parse detailed gaps due to format error'],
            conceptual_strength='unknown',
            audit_notes=f'{context} Response parsing failed - requires manual validation'
        )

class FrameworkOverlapAnalyzer:
    def __init__(self, api_key: str, model: str = "o4-mini", config: AnalysisConfig = None):
        """
        Initialize the analyzer for framework overlap analysis
        
        Args:
            api_key: OpenAI API key
            model: Model to use (default: o4-mini)
            config: Analysis configuration
        """
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.config = config or AnalysisConfig()
        
        # Framework analysis statistics
        self.stats = {
            'total_processed': 0,
            'successful_analyses': 0,
            'equivalent_matches': 0,
            'overlapping_matches': 0,
            'distinct_controls': 0,
            'api_calls': 0,
            'api_errors': 0,
            'parsing_errors': 0,
            'retry_attempts': 0,
            'cache_hits': 0
        }
        
        # Simple cache for identical control pairs
        self.analysis_cache = {}
        
        # Audit trail for compliance
        self.audit_trail: List[AuditEvent] = []
        
        # Prompt constructor and parser
        self.prompt_constructor = PromptConstructor()
        self.response_parser = ResponseParser()
    
    def _log_audit_event(self, event_type: str, framework_a_id: str, framework_b_id: str, details: Dict[str, Any]):
        """Log an audit event with context"""
        event = AuditEvent(
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            framework_a_id=framework_a_id or 'unknown',
            framework_b_id=framework_b_id or 'unknown',
            details=details
        )
        self.audit_trail.append(event)
    
    @retry(max_attempts=3, base_delay=1.0, max_delay=30.0)
    def analyze_single(self, framework_a_text: str, framework_b_text: str, 
                      framework_a_id: str = None, framework_b_id: str = None) -> OverlapResult:
        """Analyze overlap between a single framework control pair"""
        context = f"[{framework_a_id}->{framework_b_id}]" if framework_a_id and framework_b_id else ""
        
        if shutdown_requested:
            raise InterruptedError("Shutdown requested")
        
        # Check cache
        cache_key = f"{hash(framework_a_text)}_{hash(framework_b_text)}"
        if cache_key in self.analysis_cache:
            logger.debug(f"{context} Cache hit")
            self.stats['cache_hits'] += 1
            return self.analysis_cache[cache_key]
        
        # Create prompt
        prompt = self.prompt_constructor.create_overlap_analysis_prompt(framework_a_text, framework_b_text)
        
        # Log API call attempt
        self._log_audit_event('api_call', framework_a_id, framework_b_id, {
            'model': self.model,
            'prompt_length': len(prompt)
        })
        
        # Make API call
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{'role': 'user', 'content': prompt}],
            max_completion_tokens=2000,
            timeout=self.config.base_timeout
        )
        
        self.stats['api_calls'] += 1
        response_text = response.choices[0].message.content
        
        # Parse response
        analysis = self.response_parser.parse_overlap_response(
            response_text, framework_a_id, framework_b_id
        )
        
        # Cache result
        self.analysis_cache[cache_key] = analysis
        
        # Log successful analysis
        self._log_audit_event('analysis_success', framework_a_id, framework_b_id, {
            'overlap_type': analysis.overlap_type,
            'confidence': analysis.confidence
        })
        
        logger.debug(f"{context} Analyzed: {analysis.overlap_type} (confidence: {analysis.confidence:.2f})")
        
        # Add rate limiting delay only if configured
        if self.config.rate_limit_delay > 0:
            time.sleep(self.config.rate_limit_delay)
        
        return analysis
    
    def analyze_dataframe(self, df: pd.DataFrame, 
                         framework_a_col: str = 'source_text',
                         framework_b_col: str = 'target_text') -> pd.DataFrame:
        """
        Analyze all framework control pairs in a dataframe - concurrent processing for speed
        
        Args:
            df: DataFrame with framework control pairs
            framework_a_col: Column name for framework A control text
            framework_b_col: Column name for framework B control text
        
        Returns:
            DataFrame with overlap analysis results
        """
        logger.info(f"Starting concurrent framework overlap analysis of {len(df)} control pairs")
        logger.info(f"Using {self.config.max_workers} concurrent threads")
        
        analyzed_rows = []
        
        # Prepare tasks for concurrent execution
        tasks = []
        for _, row in df.iterrows():
            task = {
                'framework_a_text': row[framework_a_col],
                'framework_b_text': row[framework_b_col],
                'framework_a_id': row['source_id'],
                'framework_b_id': row['target_id'],
                'original_row': row
            }
            tasks.append(task)
        
        # Process with ThreadPoolExecutor for concurrent API calls
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            future_to_task = {}
            for task in tasks:
                if shutdown_requested:
                    break
                future = executor.submit(
                    self.analyze_single,
                    task['framework_a_text'],
                    task['framework_b_text'],
                    task['framework_a_id'],
                    task['framework_b_id']
                )
                future_to_task[future] = task
            
            # Process completed futures with progress bar
            with tqdm(total=len(future_to_task), desc="Analyzing framework overlap") as pbar:
                for future in as_completed(future_to_task):
                    if shutdown_requested:
                        break
                    
                    task = future_to_task[future]
                    
                    try:
                        # Get analysis result
                        analysis = future.result()
                        
                        # Update statistics
                        self.stats['total_processed'] += 1
                        self.stats['successful_analyses'] += 1
                        
                        if analysis.overlap_type == 'equivalent':
                            self.stats['equivalent_matches'] += 1
                        elif analysis.overlap_type == 'overlapping':
                            self.stats['overlapping_matches'] += 1
                        else:
                            self.stats['distinct_controls'] += 1
                        
                        # Create result row with framework analysis columns
                        row_dict = {
                            'source_id': task['framework_a_id'],
                            'source_text': task['framework_a_text'],
                            'target_id': task['framework_b_id'], 
                            'target_text': task['framework_b_text'],
                            'equivalence_type': analysis.overlap_type,
                            'mapping_justification': analysis.justification,
                            'overlapping_concepts': '; '.join(analysis.key_points) if analysis.key_points else '',
                            'distinct_concepts': '; '.join(analysis.gaps_identified) if analysis.gaps_identified else '',
                            'mapping_notes': analysis.audit_notes
                        }
                        
                        analyzed_rows.append(row_dict)
                        
                    except Exception as e:
                        logger.error(f"Analysis failed for {task['framework_a_id']}->{task['framework_b_id']}: {e}")
                        self.stats['api_errors'] += 1
                        
                        # Log the error
                        self._log_audit_event('analysis_error', task['framework_a_id'], task['framework_b_id'], {
                            'error': str(e),
                            'error_type': type(e).__name__
                        })
                        
                        # Create failure result
                        failure_result = OverlapResult(
                            overlap_type='distinct',
                            confidence=0.0,
                            justification=f'Analysis failed: {str(e)}',
                            key_points=['System failure'],
                            gaps_identified=['Technical analysis failure'],
                            conceptual_strength='unknown',
                            audit_notes=f'Analysis failed: {str(e)}'
                        )
                        
                        # Create failure result row
                        row_dict = {
                            'source_id': task['framework_a_id'],
                            'source_text': task['framework_a_text'],
                            'target_id': task['framework_b_id'],
                            'target_text': task['framework_b_text'],
                            'equivalence_type': failure_result.overlap_type,
                            'mapping_justification': failure_result.justification,
                            'overlapping_concepts': '; '.join(failure_result.key_points),
                            'distinct_concepts': '; '.join(failure_result.gaps_identified),
                            'mapping_notes': failure_result.audit_notes
                        }
                        
                        analyzed_rows.append(row_dict)
                        self.stats['total_processed'] += 1
                    
                    pbar.update(1)
        
        # Create result dataframe
        if analyzed_rows:
            analyzed_df = pd.DataFrame(analyzed_rows)
            # Sort by overlap quality (using the source_id and equivalence_type)
            overlap_order = {'equivalent': 0, 'overlapping': 1, 'distinct': 2}
            analyzed_df['overlap_order'] = analyzed_df['equivalence_type'].map(overlap_order)
            analyzed_df = analyzed_df.sort_values(
                ['source_id', 'overlap_order'],
                ascending=[True, True]
            ).drop('overlap_order', axis=1)
        else:
            logger.warning("No analysis results to return")
            analyzed_df = pd.DataFrame(columns=[
                'source_id', 'source_text', 'target_id', 'target_text', 
                'equivalence_type', 'mapping_justification', 'overlapping_concepts', 
                'distinct_concepts', 'mapping_notes'
            ])
        
        logger.info(f"Framework overlap analysis complete. Processed {len(analyzed_rows)} results")
        return analyzed_df
    
    def print_statistics(self):
        """Print framework overlap analysis statistics"""
        total = max(1, self.stats['total_processed'])
        
        print("\n" + "=" * 60)
        print("FRAMEWORK OVERLAP ANALYSIS STATISTICS")
        print("=" * 60)
        print(f"Total Control Pairs Processed: {self.stats['total_processed']}")
        print(f"Successful Analyses: {self.stats['successful_analyses']}")
        print(f"Equivalent Controls: {self.stats['equivalent_matches']} ({self.stats['equivalent_matches']/total*100:.1f}%)")
        print(f"Overlapping Controls: {self.stats['overlapping_matches']} ({self.stats['overlapping_matches']/total*100:.1f}%)")
        print(f"Distinct Controls: {self.stats['distinct_controls']} ({self.stats['distinct_controls']/total*100:.1f}%)")
        
        print(f"\nPerformance:")
        print(f"  Concurrent Workers: {self.config.max_workers}")
        print(f"  Total API Calls: {self.stats['api_calls']}")
        print(f"  API Errors: {self.stats['api_errors']}")
        print(f"  Parsing Errors: {self.stats['parsing_errors']}")
        print(f"  Cache Hits: {self.stats['cache_hits']}")
        print(f"  Rate Limit Delay: {self.config.rate_limit_delay}s")
        
        print(f"\nAudit Trail Events: {len(self.audit_trail)}")
        print("=" * 60)
    
    def save_audit_trail(self, output_path: Path):
        """Save comprehensive audit trail for framework analysis"""
        audit_data = {
            'analysis_metadata': {
                'timestamp': datetime.now().isoformat(),
                'model_used': self.model,
                'configuration': asdict(self.config),
                'final_statistics': self.stats
            },
            'audit_events': [event.to_dict() for event in self.audit_trail],
            'performance_summary': {
                'total_processed': self.stats['total_processed'],
                'success_rate': self.stats['successful_analyses'] / max(1, self.stats['total_processed']),
                'cache_hit_rate': self.stats['cache_hits'] / max(1, self.stats['api_calls']) if self.stats['api_calls'] > 0 else 0,
            }
        }
        
        audit_file = output_path / 'framework_analysis_audit_trail.json'
        with open(audit_file, 'w') as f:
            json.dump(audit_data, f, indent=2, default=str)
        
        logger.info(f"Audit trail saved to {audit_file}")
    
    def cleanup(self):
        """Clean up resources and save final audit trail"""
        logger.info("Performing cleanup...")
        
        # Final audit event
        self._log_audit_event('cleanup', 'system', 'system', {
            'final_stats': self.stats,
            'cache_size': len(self.analysis_cache),
            'shutdown_requested': shutdown_requested
        })

def create_detailed_report(df: pd.DataFrame, output_path: Path, analyzer: FrameworkOverlapAnalyzer):
    """Create a detailed framework overlap analysis report"""
    try:
        with pd.ExcelWriter(output_path / 'framework_overlap_report.xlsx', engine='openpyxl') as writer:
            # All results
            df.to_excel(writer, sheet_name='All Results', index=False)
            
            # Summary by overlap type
            if len(df) > 0:
                summary = df['equivalence_type'].value_counts().reset_index()
                summary.columns = ['Overlap Type', 'Count']
                summary['Percentage'] = (summary['Count'] / len(df) * 100).round(1)
                summary.to_excel(writer, sheet_name='Summary by Overlap Type', index=False)
                
                # Equivalent controls
                equivalent_controls = df[df['equivalence_type'] == 'equivalent']
                if len(equivalent_controls) > 0:
                    equivalent_controls.to_excel(writer, sheet_name='Equivalent Controls', index=False)
                
                # Overlapping controls  
                overlapping_controls = df[df['equivalence_type'] == 'overlapping']
                if len(overlapping_controls) > 0:
                    overlapping_controls.to_excel(writer, sheet_name='Overlapping Controls', index=False)
                    
                # Distinct controls
                distinct_controls = df[df['equivalence_type'] == 'distinct']
                if len(distinct_controls) > 0:
                    distinct_controls.to_excel(writer, sheet_name='Distinct Controls', index=False)
            
            # Statistics
            stats_df = pd.DataFrame([
                ['Total Processed', analyzer.stats['total_processed']],
                ['API Calls', analyzer.stats['api_calls']],
                ['API Errors', analyzer.stats['api_errors']],
                ['Parsing Errors', analyzer.stats['parsing_errors']],
                ['Cache Hits', analyzer.stats['cache_hits']]
            ], columns=['Metric', 'Value'])
            stats_df.to_excel(writer, sheet_name='Performance Stats', index=False)
        
        logger.info(f"Detailed report saved to {output_path / 'framework_overlap_report.xlsx'}")
        
    except Exception as e:
        logger.error(f"Failed to create detailed report: {e}")

def main():
    parser = argparse.ArgumentParser(
        description='Fast Concurrent Framework Overlap Analyzer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python framework_analyzer.py framework_pairs.csv --output analyzed_overlaps.csv
  python framework_analyzer.py framework_pairs.csv --workers 10 --create-report
  python framework_analyzer.py framework_pairs.csv --config config.yaml --workers 3
        """
    )
    
    parser.add_argument('input_file', help='Input CSV/XLSX file with framework control pairs')
    parser.add_argument('--output', '-o', help='Output CSV file (default: analyzed_<input_file>)')
    parser.add_argument('--model', default='o4-mini', help='OpenAI model to use (default: o4-mini)')
    parser.add_argument('--config', help='YAML configuration file for analysis parameters')
    parser.add_argument('--create-report', action='store_true', help='Create detailed Excel report')
    parser.add_argument('--api-key', help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--framework-a-col', default='source_text', help='Framework A control text column')
    parser.add_argument('--framework-b-col', default='target_text', help='Framework B control text column')
    parser.add_argument('--workers', type=int, default=5, help='Number of concurrent workers (default: 5)')
    
    args = parser.parse_args()
    
    try:
        # Get API key
        api_key = args.api_key or os.environ.get('OPENAI_API_KEY')
        if not api_key:
            print("❌ Error: OpenAI API key not found!")
            print("Set it with --api-key or export OPENAI_API_KEY='your-key-here'")
            sys.exit(1)
        
        # Check input file
        input_path = Path(args.input_file)
        if not input_path.exists():
            print(f"❌ Error: Input file not found: {args.input_file}")
            sys.exit(1)
        
        # Set output path
        if not args.output:
            args.output = input_path.parent / f"analyzed_{input_path.name}"
        output_path = Path(args.output).parent
        output_path.mkdir(exist_ok=True)
        
        # Load configuration
        config = AnalysisConfig()
        if args.config:
            config_path = Path(args.config)
            if config_path.exists():
                config = AnalysisConfig.from_file(config_path)
                logger.info(f"Loaded configuration from {config_path}")
            else:
                logger.warning(f"Config file not found: {config_path}, using defaults")
                config.save_to_file(config_path)
                logger.info(f"Saved default config to {config_path}")
        
        # Override with command line arguments
        if args.workers:
            config.max_workers = args.workers
        
        # Load data
        print(f"\n📁 Loading framework control pairs from {args.input_file}")
        if input_path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(args.input_file)
        else:
            df = pd.read_csv(args.input_file)
        
        print(f"✓ Loaded {len(df)} framework control pairs")
        
        # Validate required columns
        required_cols = ['source_id', 'source_text', 'target_id', 'target_text']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ Error: Missing required columns: {missing_cols}")
            print(f"Available columns: {list(df.columns)}")
            sys.exit(1)
        
        # Initialize analyzer
        print(f"\n🤖 Initializing framework overlap analyzer with model: {args.model}")
        print(f"🚀 Performance optimizations: {config.max_workers} workers, streamlined prompts, no artificial delays")
        analyzer = FrameworkOverlapAnalyzer(
            api_key=api_key,
            model=args.model,
            config=config
        )
        
        print(f"🔍 Starting concurrent framework overlap analysis")
        print(f"⚙️  Configuration: max_workers={config.max_workers}, max_retries={config.max_retries}, rate_limit_delay={config.rate_limit_delay}s")
        
        start_time = time.time()
        
        # Run analysis
        analyzed_df = analyzer.analyze_dataframe(
            df,
            framework_a_col=args.framework_a_col,
            framework_b_col=args.framework_b_col
        )
        
        if shutdown_requested:
            print("\n⚠️  Analysis interrupted by shutdown request")
        
        elapsed_time = time.time() - start_time
        print(f"\n✓ Framework overlap analysis completed in {elapsed_time:.1f} seconds")
        
        # Save results
        if len(analyzed_df) > 0:
            analyzed_df.to_excel(args.output, index=False)
            print(f"✓ Saved {len(analyzed_df)} analyzed results to {args.output}")
        else:
            print("⚠️  No results to save")
        
        # Save audit trail
        analyzer.save_audit_trail(output_path)
        
        # Create report if requested
        if args.create_report and len(analyzed_df) > 0:
            print("\n📊 Creating detailed framework overlap report...")
            create_detailed_report(analyzed_df, output_path, analyzer)
        
        # Print statistics
        analyzer.print_statistics()
        
        # Cleanup
        analyzer.cleanup()
        
        print(f"\n✅ Framework overlap analysis complete! Check {output_path} for all output files.")
        
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        if 'analyzer' in locals():
            analyzer.cleanup()
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        logger.error(f"Fatal error: {traceback.format_exc()}")
        if 'analyzer' in locals():
            analyzer.cleanup()
        sys.exit(1)

if __name__ == "__main__":
    main()