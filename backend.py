"""
Backend logic for NutriSense - Smart Soil Analytics Platform
Contains all business logic, database operations, AI integrations, and utility functions
"""

import os
import json
import logging
import traceback
import time
from datetime import datetime
from typing import Dict, Optional, Any
from groq import Groq
import pandas as pd
import sqlite3
from pydantic import BaseModel, field_validator
import hashlib
import streamlit as st

# Enhanced Logging Configuration - LOCAL ONLY
def setup_logging():
    """Configure single JSON file logging for NutriSense application - LOCAL DEVELOPMENT ONLY"""
    
    # Skip logging setup if running in production/cloud environment
    if os.getenv('STREAMLIT_SHARING') or os.getenv('STREAMLIT_CLOUD') or os.getenv('RAILWAY_ENVIRONMENT'):
        return logging.getLogger('nutrisense_disabled')
    
    # Only enable logging for local development
    if not os.path.exists('logs') and os.access('.', os.W_OK):
        try:
            # Create logs directory
            log_dir = "logs"
            os.makedirs(log_dir, exist_ok=True)
        except:
            # If can't create logs directory, disable logging
            return logging.getLogger('nutrisense_disabled')
    elif not os.path.exists('logs'):
        # No logs directory and can't create one - disable logging
        return logging.getLogger('nutrisense_disabled')
    
    # Configure main logger
    logger = logging.getLogger('nutrisense')
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create single JSON log file path
    log_file = os.path.join('logs', 'nutrisense_realtime.json')
    
    try:
        # Initialize JSON log file if it doesn't exist
        if not os.path.exists(log_file):
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump({"logs": [], "metadata": {"created": datetime.now().isoformat(), "version": "1.0"}}, f, indent=2)
        
        # Create custom JSON handler
        class JSONFileHandler(logging.Handler):
            def __init__(self, filename):
                super().__init__()
                self.filename = filename
                
            def emit(self, record):
                try:
                    # Create log entry
                    log_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "level": record.levelname,
                        "logger": record.name,
                        "function": record.funcName,
                        "line": record.lineno,
                        "message": record.getMessage(),
                        "module": record.module if hasattr(record, 'module') else 'unknown'
                    }
                    
                    # Add structured data if present
                    if hasattr(record, 'event_type'):
                        log_entry["event_type"] = record.event_type
                        log_entry["event_message"] = getattr(record, 'event_message', '')
                        log_entry["event_data"] = getattr(record, 'event_data', None)
                    
                    if hasattr(record, 'error_type'):
                        log_entry["error_type"] = record.error_type
                        log_entry["error_message"] = getattr(record, 'error_message', '')
                        log_entry["error_context"] = getattr(record, 'error_context', '')
                        log_entry["error_data"] = getattr(record, 'error_data', None)
                        log_entry["traceback"] = getattr(record, 'traceback', '')
                    
                    # Add session ID if present
                    if hasattr(record, 'session_id'):
                        log_entry["session_id"] = record.session_id
                    
                    # Add exception info if present
                    if record.exc_info:
                        log_entry["exception"] = self.format(record)
                    
                    # Read current logs
                    try:
                        with open(self.filename, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                    except (FileNotFoundError, json.JSONDecodeError):
                        data = {"logs": [], "metadata": {"created": datetime.now().isoformat(), "version": "1.0"}}
                    
                    # Add new log entry
                    data["logs"].append(log_entry)
                    data["metadata"]["last_updated"] = datetime.now().isoformat()
                    data["metadata"]["total_logs"] = len(data["logs"])
                    
                    # Keep only last 1000 logs to prevent file from growing too large
                    if len(data["logs"]) > 1000:
                        data["logs"] = data["logs"][-1000:]
                        data["metadata"]["truncated"] = True
                        data["metadata"]["truncated_at"] = datetime.now().isoformat()
                    
                    # Write back to file (real-time update)
                    with open(self.filename, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                        
                except Exception:
                    # Silently fail if logging doesn't work
                    pass
        
        # Add JSON handler to logger
        json_handler = JSONFileHandler(log_file)
        json_handler.setLevel(logging.DEBUG)
        logger.addHandler(json_handler)
        
        # Console handler for development (optional)
        if os.getenv('NUTRISENSE_DEBUG', 'false').lower() == 'true':
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s')
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
    
    except Exception:
        # If any logging setup fails, return disabled logger
        return logging.getLogger('nutrisense_disabled')
    
    return logger

# Initialize logging - will be disabled in production
logger = setup_logging()

def is_production_environment() -> bool:
    """Check if running in production/cloud environment"""
    return bool(os.getenv('STREAMLIT_SHARING') or os.getenv('STREAMLIT_CLOUD') or os.getenv('RAILWAY_ENVIRONMENT'))

def is_logging_enabled() -> bool:
    """Check if logging is enabled (local development only)"""
    return logger.name != 'nutrisense_disabled' and os.path.exists('logs')

def log_event(event_type: str, message: str, data: Optional[Dict] = None):
    """Log application events with structured data - LOCAL ONLY"""
    if not is_logging_enabled():
        return
        
    try:
        # Create structured log message
        log_message = f"EVENT: {event_type} | {message}"
        if data:
            log_message += f" | Data: {json.dumps(data, default=str)}"
        
        # Create extra fields for structured logging
        extra_data = {
            'event_type': event_type,
            'event_message': message,
            'event_data': data,
            'session_id': st.session_state.get('session_id', 'unknown')
        }
        
        # Log with extra data
        logger.info(log_message, extra=extra_data)
    except:
        pass  # Silently fail if logging doesn't work

def log_error(error: Exception, context: str = "", additional_data: Optional[Dict] = None):
    """Log errors with full context and traceback - LOCAL ONLY"""
    if not is_logging_enabled():
        return
        
    try:
        # Create structured error message
        error_message = f"ERROR: {context} | {type(error).__name__}: {str(error)}"
        if additional_data:
            error_message += f" | Data: {json.dumps(additional_data, default=str)}"
        
        # Create extra fields for structured logging
        extra_data = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'error_context': context,
            'error_data': additional_data,
            'session_id': st.session_state.get('session_id', 'unknown'),
            'traceback': traceback.format_exc()
        }
        
        # Log error with extra data
        logger.error(error_message, extra=extra_data)
    except:
        pass  # Silently fail if logging doesn't work

def log_user_action(action: str, details: Optional[Dict] = None):
    """Log user interactions and actions - LOCAL ONLY"""
    if is_logging_enabled():
        log_event('USER_ACTION', action, details)

def log_system_event(event: str, details: Optional[Dict] = None):
    """Log system events and status changes - LOCAL ONLY"""
    if is_logging_enabled():
        log_event('SYSTEM_EVENT', event, details)

def log_ai_interaction(model: str, prompt_type: str, success: bool, response_length: int = 0, error: str = None):
    """Log AI API interactions - LOCAL ONLY"""
    if not is_logging_enabled():
        return
        
    data = {
        'model': model,
        'prompt_type': prompt_type,
        'success': success,
        'response_length': response_length
    }
    
    if error:
        data['error'] = error
    
    log_event('AI_INTERACTION', f"AI call: {prompt_type} with {model}", data)

def log_database_operation(operation: str, table: str, success: bool, error: str = None, record_count: int = None):
    """Log database operations - LOCAL ONLY"""
    if not is_logging_enabled():
        return
        
    data = {
        'operation': operation,
        'table': table,
        'success': success
    }
    
    if error:
        data['error'] = error
    if record_count is not None:
        data['record_count'] = record_count
    
    log_event('DATABASE_OPERATION', f"DB {operation} on {table}", data)

# Initialize session ID for tracking - LOCAL ONLY
def initialize_session():
    """Initialize session ID for tracking - LOCAL ONLY"""
    if 'session_id' not in st.session_state and is_logging_enabled():
        st.session_state.session_id = hashlib.md5(f"{datetime.now().isoformat()}_{os.getpid()}".encode()).hexdigest()[:8]
        log_system_event('SESSION_START', {'session_id': st.session_state.session_id})

# Log application startup - LOCAL ONLY
def log_application_startup():
    """Log application startup - LOCAL ONLY"""
    if is_logging_enabled():
        log_system_event('APPLICATION_START', {
            'version': '1.0.1',
            'python_version': os.sys.version,
            'streamlit_version': st.__version__,
            'updates': ['Fixed Streamlit deprecation warnings for button width parameters'],
            'environment': 'local_development'
        })

class SoilData(BaseModel):
    pH: float
    EC: float
    Moisture: float
    Nitrogen: float
    Phosphorus: float
    Potassium: float
    Microbial: float
    Temperature: float
    
    @field_validator('pH')
    @classmethod
    def pH_range(cls, v):
        try:
            if not 0 <= v <= 14:
                log_error(ValueError(f'pH value {v} out of range 0-14'), 'SOIL_DATA_VALIDATION')
                raise ValueError('pH must be 0-14')
            log_event('VALIDATION_SUCCESS', f'pH validation passed: {v}')
            return v
        except Exception as e:
            log_error(e, 'pH_VALIDATION_ERROR', {'value': v})
            raise

def get_health_score(soil: Dict) -> float:
    """Calculate soil health score with comprehensive error handling and logging"""
    try:
        log_user_action('HEALTH_SCORE_CALCULATION', {'soil_params': list(soil.keys())})
        
        # Validate input data
        required_params = ['pH', 'EC', 'Moisture', 'Nitrogen', 'Phosphorus', 'Potassium']
        missing_params = [param for param in required_params if param not in soil]
        if missing_params:
            log_error(ValueError(f'Missing required parameters: {missing_params}'), 'HEALTH_SCORE_MISSING_PARAMS')
            return 50.0
        
        # Validate data types and ranges
        for param, value in soil.items():
            if not isinstance(value, (int, float)):
                log_error(TypeError(f'Parameter {param} must be numeric, got {type(value)}'), 'HEALTH_SCORE_TYPE_ERROR')
                return 50.0
            if value < 0:
                log_error(ValueError(f'Parameter {param} cannot be negative: {value}'), 'HEALTH_SCORE_NEGATIVE_VALUE')
                return 50.0
        
        # Calculate components with bounds checking
        ph = max(0, min(25, 25 - abs(soil['pH'] - 7.0) * 3.5))
        ec = max(0, min(25, 25 - min(soil['EC'], 4.0) * 6.25))
        moist = 20 if 25 <= soil['Moisture'] <= 40 else max(0, min(20, 20 - abs(soil['Moisture'] - 32.5) * 0.5))
        npk = (min(soil['Nitrogen']/80*10, 10) + 
               min(soil['Phosphorus']/50*10, 10) + 
               min(soil['Potassium']/250*10, 10))
        
        score = min(max(ph + ec + moist + npk, 0), 100)
        
        log_event('HEALTH_SCORE_CALCULATED', f'Health score: {score:.1f}/100', {
            'score': score,
            'components': {'ph': ph, 'ec': ec, 'moisture': moist, 'npk': npk}
        })
        
        return score
    except Exception as e:
        log_error(e, 'HEALTH_SCORE_CALCULATION_ERROR', {'soil_data': soil})
        return 50.0  # Safe fallback

def interpret(param: str, val: float) -> tuple:
    """Interpret soil parameter with logging"""
    try:
        log_event('PARAMETER_INTERPRETATION', f'Interpreting {param}: {val}')
        
        data = {
            'pH': [(0,5.5,"Acidic","🔴"),(5.5,6.5,"Low","🟡"),(6.5,7.5,"Optimal","🟢"),(7.5,8.5,"High","🟡"),(8.5,15,"Alkaline","🔴")],
            'EC': [(0,0.8,"Low","🟢"),(0.8,2,"Moderate","🟡"),(2,4,"High","🟠"),(4,25,"Very High","🔴")],
            'Moisture': [(0,15,"Dry","🔴"),(15,25,"Low","🟡"),(25,40,"Optimal","🟢"),(40,60,"High","🟡"),(60,101,"Wet","🔴")],
            'Nitrogen': [(0,40,"Low","🔴"),(40,80,"Optimal","🟢"),(80,501,"High","🟡")],
            'Phosphorus': [(0,20,"Low","🔴"),(20,50,"Optimal","🟢"),(50,201,"High","🟡")],
            'Potassium': [(0,100,"Low","🔴"),(100,250,"Optimal","🟢"),(250,501,"High","🟡")],
            'Microbial': [(0,3,"Poor","🔴"),(3,7,"Good","🟢"),(7,11,"Excellent","💚")],
            'Temperature': [(0,10,"Cold","🔵"),(10,30,"Optimal","🟢"),(30,51,"Hot","🔴")]
        }
        
        for low, high, status, emoji in data.get(param, []):
            if low <= val < high:
                log_event('PARAMETER_INTERPRETED', f'{param} {val} -> {status}', {
                    'parameter': param,
                    'value': val,
                    'status': status,
                    'range': f'{low}-{high}'
                })
                return status, emoji
        
        log_event('PARAMETER_UNKNOWN', f'{param} {val} -> Unknown', {
            'parameter': param,
            'value': val
        })
        return "Unknown", "⚪"
        
    except Exception as e:
        log_error(e, 'PARAMETER_INTERPRETATION_ERROR', {'parameter': param, 'value': val})
        return "Error", "❌"

@st.cache_resource
def init_db():
    """Initialize database with comprehensive logging"""
    try:
        log_system_event('DATABASE_INIT_START')
        
        # Use absolute path for database directory to avoid CWD issues
        base_dir = os.path.dirname(os.path.abspath(__file__))
        db_dir = os.path.join(base_dir, 'data')
        os.makedirs(db_dir, exist_ok=True)
        db_path = os.path.join(db_dir, 'soil_history.db')
        conn = sqlite3.connect(db_path, check_same_thread=False)
        
        # Create table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS soil_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data_hash TEXT UNIQUE,
                soil_data TEXT,
                timestamp DATETIME,
                summary TEXT,
                location TEXT,
                health_score REAL
            )
        """)
        
        # Add missing columns for existing databases (migration)
        # 1) health_score column
        try:
            conn.execute("ALTER TABLE soil_records ADD COLUMN health_score REAL")
            log_database_operation('ALTER_TABLE', 'soil_records', True, record_count=None)
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e).lower():
                log_system_event('DATABASE_COLUMN_EXISTS', {'column': 'health_score'})
            else:
                log_error(e, 'DATABASE_ALTER_ERROR', {'column': 'health_score'})
        
        # 2) location column (fix for 'no column named location' errors on old DBs)
        try:
            conn.execute("ALTER TABLE soil_records ADD COLUMN location TEXT")
            log_database_operation('ALTER_TABLE', 'soil_records', True, record_count=None)
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e).lower():
                log_system_event('DATABASE_COLUMN_EXISTS', {'column': 'location'})
            else:
                log_error(e, 'DATABASE_ALTER_ERROR', {'column': 'location'})
        
        conn.commit()
        
        # Get record count for logging
        cursor = conn.execute("SELECT COUNT(*) FROM soil_records")
        record_count = cursor.fetchone()[0]
        
        log_database_operation('INIT', 'soil_records', True, record_count=record_count)
        log_system_event('DATABASE_INIT_SUCCESS', {'record_count': record_count})
        
        return conn
        
    except Exception as e:
        log_error(e, 'DATABASE_INIT_ERROR')
        return None

@st.cache_resource
def get_groq_client():
    """Initialize Groq client with logging"""
    try:
        log_system_event('GROQ_CLIENT_INIT_START')
        api_key = None

        # Safely read from Streamlit secrets if available
        try:
            api_key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            # Secrets file may not exist in local/dev environments
            api_key = None

        # Fallback to environment variable
        if not api_key:
            api_key = os.getenv("GROQ_API_KEY")
        
        if not api_key:
            log_system_event('GROQ_API_KEY_MISSING')
            return None
            
        client = Groq(api_key=api_key)
        log_system_event('GROQ_CLIENT_INIT_SUCCESS')
        return client
        
    except Exception as e:
        log_error(e, 'GROQ_CLIENT_INIT_ERROR')
        return None

def build_prompt(soil: Dict, task: str, loc: str = "") -> str:
    """Build AI prompt with logging"""
    try:
        log_user_action('BUILD_PROMPT', {'task': task, 'location': loc, 'has_location': bool(loc)})
        
        base = f"""Soil Data{f' - {loc}' if loc else ''}:
pH: {soil['pH']:.2f}, EC: {soil['EC']:.2f} dS/m, Moisture: {soil['Moisture']:.1f}%
N: {soil['Nitrogen']:.2f}, P: {soil['Phosphorus']:.2f}, K: {soil['Potassium']:.2f} mg/kg
Microbial: {soil['Microbial']:.2f}/10, Temp: {soil['Temperature']:.1f}°C"""

        prompts = {
            "summary": f"{base}\n\nProvide: 1) Overall condition 2) Main concerns 3) Top 3 actions. Keep brief.",
            "crops": f"{base}\n\nSuggest TOP 5 suitable crops with reasons. Include Indian varieties.",
            "fertilizer": f"{base}\n\nProvide: NPK ratio, kg/hectare, timing, organic alternatives.",
            "irrigation": f"{base}\n\nProvide: frequency, water amount, best timing for irrigation."
        }
        
        prompt = prompts.get(task, base)
        log_event('PROMPT_BUILT', f'Built {task} prompt', {'prompt_length': len(prompt)})
        
        return prompt
        
    except Exception as e:
        log_error(e, 'BUILD_PROMPT_ERROR', {'task': task, 'location': loc})
        return f"Error building prompt for {task}"

@st.cache_data(ttl=300)
def call_groq(_hash: str, prompt: str, _task: str) -> str:
    """Call Groq API with comprehensive error handling and logging"""
    try:
        log_ai_interaction('START', st.session_state.get("selected_model", "unknown"), _task, True)
        
        client = get_groq_client()
        if not client:
            log_ai_interaction(st.session_state.get("selected_model", "unknown"), _task, False, error="No client available")
            return "⚠️ Configure GROQ_API_KEY in Streamlit secrets"
        
        model = st.session_state.get("selected_model", "llama-3.3-70b-versatile")
        
        # Validate inputs
        if not prompt or not prompt.strip():
            log_error(ValueError("Empty prompt provided"), 'AI_REQUEST_EMPTY_PROMPT')
            return "⚠️ Error: Empty prompt"
        
        if len(prompt) > 10000:  # Reasonable limit
            log_error(ValueError(f"Prompt too long: {len(prompt)} characters"), 'AI_REQUEST_PROMPT_TOO_LONG')
            return "⚠️ Error: Prompt too long"
        
        log_event('AI_REQUEST_START', f'Calling {model} for {_task}', {
            'model': model,
            'task': _task,
            'prompt_length': len(prompt)
        })
        
        # Add timeout and retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are an agricultural expert. Provide practical advice for Indian farmers in simple language."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=600,
                    timeout=30  # 30 second timeout
                )
                break
            except Exception as retry_error:
                if attempt == max_retries - 1:
                    raise retry_error
                log_event('AI_REQUEST_RETRY', f'Attempt {attempt + 1} failed, retrying', {
                    'error': str(retry_error),
                    'attempt': attempt + 1
                })
                time.sleep(1)  # Brief delay before retry
        
        if not resp or not resp.choices or not resp.choices[0].message:
            log_error(ValueError("Invalid API response structure"), 'AI_REQUEST_INVALID_RESPONSE')
            return "⚠️ Error: Invalid response from AI service"
        
        response_content = resp.choices[0].message.content
        if not response_content or not response_content.strip():
            log_error(ValueError("Empty response from API"), 'AI_REQUEST_EMPTY_RESPONSE')
            return "⚠️ Error: Empty response from AI service"
        
        response_length = len(response_content)
        
        log_ai_interaction(model, _task, True, response_length)
        log_event('AI_REQUEST_SUCCESS', f'Received response from {model}', {
            'model': model,
            'task': _task,
            'response_length': response_length,
            'tokens_used': getattr(resp.usage, 'total_tokens', 'unknown') if hasattr(resp, 'usage') else 'unknown'
        })
        
        return response_content
        
    except Exception as e:
        error_msg = str(e)
        log_error(e, 'AI_REQUEST_ERROR', {
            'model': st.session_state.get("selected_model", "unknown"),
            'task': _task,
            'prompt_length': len(prompt) if 'prompt' in locals() else 0,
            'error_type': type(e).__name__
        })
        log_ai_interaction(st.session_state.get("selected_model", "unknown"), _task, False, error=error_msg)
        
        # Return user-friendly error messages
        if "timeout" in error_msg.lower():
            return "⚠️ Request timed out. Please try again."
        elif "rate limit" in error_msg.lower():
            return "⚠️ Rate limit exceeded. Please wait a moment and try again."
        elif "api key" in error_msg.lower():
            return "⚠️ API key issue. Please check your configuration."
        else:
            return f"⚠️ AI service temporarily unavailable. Please try again later."

def save_record(soil: Dict, summary: str, loc: str = ""):
    """Save soil record with comprehensive logging"""
    try:
        log_user_action('SAVE_RECORD_START', {'location': loc, 'has_summary': bool(summary)})
        
        conn = init_db()
        if conn:
            data_str = json.dumps(soil)
            hash_val = hashlib.md5(data_str.encode()).hexdigest()
            health_score = get_health_score(soil)
            
            log_event('RECORD_PREPARED', 'Record prepared for saving', {
                'hash': hash_val[:8],
                'health_score': health_score,
                'location': loc,
                'data_size': len(data_str)
            })
            
            conn.execute(
                "INSERT OR IGNORE INTO soil_records (data_hash, soil_data, timestamp, summary, location, health_score) VALUES (?,?,?,?,?,?)",
                (hash_val, data_str, datetime.now(), summary, loc, health_score)
            )
            conn.commit()
            
            # Check if record was actually inserted
            cursor = conn.execute("SELECT id FROM soil_records WHERE data_hash = ?", (hash_val,))
            record = cursor.fetchone()
            
            if record:
                log_database_operation('INSERT', 'soil_records', True, record_count=1)
                log_user_action('SAVE_RECORD_SUCCESS', {
                    'record_id': record[0],
                    'hash': hash_val[:8],
                    'health_score': health_score
                })
            else:
                log_database_operation('INSERT', 'soil_records', False, error="Record already exists (duplicate hash)")
                
    except Exception as e:
        log_error(e, 'SAVE_RECORD_ERROR', {'location': loc, 'has_summary': bool(summary)})
        log_database_operation('INSERT', 'soil_records', False, error=str(e))

def load_history() -> pd.DataFrame:
    """Load history with logging"""
    try:
        log_user_action('LOAD_HISTORY_START')
        
        conn = init_db()
        if conn:
            df = pd.read_sql_query("SELECT * FROM soil_records ORDER BY timestamp DESC LIMIT 30", conn)
            
            log_database_operation('SELECT', 'soil_records', True, record_count=len(df))
            log_user_action('LOAD_HISTORY_SUCCESS', {'record_count': len(df)})
            
            return df
            
    except Exception as e:
        log_error(e, 'LOAD_HISTORY_ERROR')
        log_database_operation('SELECT', 'soil_records', False, error=str(e))
        
    return pd.DataFrame()
