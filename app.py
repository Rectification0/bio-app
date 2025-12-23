import streamlit as st

st.set_page_config(
    page_title="NutriSense - Smart Soil Analytics",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

import os
import json
import logging
import traceback
import time
from datetime import datetime
from typing import Dict, Optional, Any
from groq import Groq
import pandas as pd
import plotly.graph_objects as go
import sqlite3
from pydantic import BaseModel, field_validator
import hashlib

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
if 'session_id' not in st.session_state and is_logging_enabled():
    st.session_state.session_id = hashlib.md5(f"{datetime.now().isoformat()}_{os.getpid()}".encode()).hexdigest()[:8]
    log_system_event('SESSION_START', {'session_id': st.session_state.session_id})

# Log application startup - LOCAL ONLY
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
        
        os.makedirs('data', exist_ok=True)
        conn = sqlite3.connect('data/soil_history.db', check_same_thread=False)
        
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
        
        # Add health_score column if it doesn't exist (for existing databases)
        try:
            conn.execute("ALTER TABLE soil_records ADD COLUMN health_score REAL")
            log_database_operation('ALTER_TABLE', 'soil_records', True, record_count=None)
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e).lower():
                log_system_event('DATABASE_COLUMN_EXISTS', {'column': 'health_score'})
            else:
                log_error(e, 'DATABASE_ALTER_ERROR')
            
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
        
        api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
        
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

# Init session
for key in ['soil_data', 'location', 'selected_model']:
    if key not in st.session_state:
        st.session_state[key] = None if key == 'soil_data' else ("" if key == 'location' else "llama-3.3-70b-versatile")

# Enhanced Dark Mode CSS
st.markdown("""
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global Dark Theme */
.stApp {
    background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    color: #e2e8f0;
    font-family: 'Inter', sans-serif;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Main container styling */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* Enhanced Header */
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 3rem 2rem;
    border-radius: 20px;
    color: white;
    text-align: center;
    box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}

.main-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="25" cy="25" r="1" fill="white" opacity="0.1"/><circle cx="75" cy="75" r="1" fill="white" opacity="0.1"/><circle cx="50" cy="10" r="0.5" fill="white" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
    opacity: 0.3;
}

.main-header h1 {
    font-size: 3rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    position: relative;
    z-index: 1;
}

.main-header p {
    font-size: 1.2rem;
    opacity: 0.9;
    position: relative;
    z-index: 1;
}

/* Enhanced Metric Cards */
.metric-card {
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    padding: 1.5rem;
    border-radius: 16px;
    border: 1px solid #334155;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #667eea, #764ba2);
}

.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 48px rgba(102, 126, 234, 0.2);
}

/* Status Cards with Glassmorphism */
.status-good {
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.2);
    backdrop-filter: blur(10px);
    padding: 1rem;
    border-radius: 12px;
    margin: 0.5rem 0;
    box-shadow: 0 4px 16px rgba(16, 185, 129, 0.1);
    transition: all 0.3s ease;
}

.status-warning {
    background: rgba(245, 158, 11, 0.1);
    border: 1px solid rgba(245, 158, 11, 0.2);
    backdrop-filter: blur(10px);
    padding: 1rem;
    border-radius: 12px;
    margin: 0.5rem 0;
    box-shadow: 0 4px 16px rgba(245, 158, 11, 0.1);
    transition: all 0.3s ease;
}

.status-critical {
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.2);
    backdrop-filter: blur(10px);
    padding: 1rem;
    border-radius: 12px;
    margin: 0.5rem 0;
    box-shadow: 0 4px 16px rgba(239, 68, 68, 0.1);
    transition: all 0.3s ease;
}

.status-good:hover, .status-warning:hover, .status-critical:hover {
    transform: translateX(4px);
}

/* Recommendation Boxes */
.rec-box {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(147, 51, 234, 0.1) 100%);
    border: 1px solid rgba(59, 130, 246, 0.2);
    backdrop-filter: blur(10px);
    padding: 1.5rem;
    border-radius: 16px;
    margin: 1rem 0;
    box-shadow: 0 8px 32px rgba(59, 130, 246, 0.1);
    transition: all 0.3s ease;
}

.rec-box:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 48px rgba(59, 130, 246, 0.2);
}

.rec-box h4 {
    color: #60a5fa;
    margin-bottom: 1rem;
    font-weight: 600;
}

/* Sidebar Styling */
.css-1d391kg {
    background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    border-right: 1px solid #334155;
}

/* Tab Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(30, 41, 59, 0.5);
    border-radius: 12px;
    padding: 4px;
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: #94a3b8;
    font-weight: 500;
    transition: all 0.3s ease;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    box-shadow: 0 4px 16px rgba(102, 126, 234, 0.3);
}

/* Button Enhancements */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 1.5rem;
    font-weight: 500;
    transition: all 0.3s ease;
    box-shadow: 0 4px 16px rgba(102, 126, 234, 0.3);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.4);
}

/* Form Styling */
.stForm {
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 2rem;
    backdrop-filter: blur(10px);
}

/* Input Styling */
.stNumberInput > div > div > input {
    background: rgba(51, 65, 85, 0.5);
    border: 1px solid #475569;
    border-radius: 8px;
    color: #e2e8f0;
    transition: all 0.3s ease;
}

.stNumberInput > div > div > input:focus {
    border-color: #667eea;
    box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
}

.stTextInput > div > div > input {
    background: rgba(51, 65, 85, 0.5);
    border: 1px solid #475569;
    border-radius: 8px;
    color: #e2e8f0;
    transition: all 0.3s ease;
}

.stTextInput > div > div > input:focus {
    border-color: #667eea;
    box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
}

/* Selectbox Styling */
.stSelectbox > div > div {
    background: rgba(51, 65, 85, 0.5);
    border: 1px solid #475569;
    border-radius: 8px;
}

/* Expander Styling */
.streamlit-expanderHeader {
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid #334155;
    border-radius: 12px;
    color: #e2e8f0;
    transition: all 0.3s ease;
}

.streamlit-expanderHeader:hover {
    background: rgba(51, 65, 85, 0.5);
    transform: translateX(4px);
}

/* Metric Styling */
.metric-container {
    background: rgba(30, 41, 59, 0.3);
    border-radius: 12px;
    padding: 1rem;
    border: 1px solid #334155;
}

/* Success/Error/Warning Messages */
.stSuccess {
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.2);
    border-radius: 12px;
    backdrop-filter: blur(10px);
}

.stError {
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.2);
    border-radius: 12px;
    backdrop-filter: blur(10px);
}

.stWarning {
    background: rgba(245, 158, 11, 0.1);
    border: 1px solid rgba(245, 158, 11, 0.2);
    border-radius: 12px;
    backdrop-filter: blur(10px);
}

.stInfo {
    background: rgba(59, 130, 246, 0.1);
    border: 1px solid rgba(59, 130, 246, 0.2);
    border-radius: 12px;
    backdrop-filter: blur(10px);
}

/* Loading Spinner */
.stSpinner {
    color: #667eea;
}

/* Divider */
hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, #334155, transparent);
    margin: 2rem 0;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #1e293b;
}

::-webkit-scrollbar-thumb {
    background: #475569;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #667eea;
}

/* Animation for cards */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.metric-card, .status-good, .status-warning, .status-critical, .rec-box {
    animation: fadeInUp 0.6s ease-out;
}

/* Responsive Design */
@media (max-width: 768px) {
    .main-header h1 {
        font-size: 2rem;
    }
    
    .main-header p {
        font-size: 1rem;
    }
    
    .metric-card {
        padding: 1rem;
    }
}

/* Custom Progress Bar */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #667eea, #764ba2);
    border-radius: 4px;
}

/* Enhanced Plotly Charts */
.js-plotly-plot {
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
}
</style>
""", unsafe_allow_html=True)

# Enhanced Header with Dark Theme
environment_indicator = "🌐 PROD" if is_production_environment() else "🔧 DEV"
st.markdown(f'''
<div class="main-header">
    <h1>🌱 NutriSense</h1>
    <p>AI-Powered Precision Agriculture Platform</p>
    <div style="margin-top: 1rem; opacity: 0.8;">
        <span style="margin: 0 1rem;">🧪 Smart Analysis</span>
        <span style="margin: 0 1rem;">🤖 AI Insights</span>
        <span style="margin: 0 1rem;">📊 Real-time Data</span>
        <span style="margin: 0 1rem; font-size: 0.8rem;">{environment_indicator}</span>
    </div>
</div>
''', unsafe_allow_html=True)

# Enhanced Sidebar
with st.sidebar:
    st.markdown("### ⚙️ System Status")
    
    # System status with enhanced styling
    client_status = get_groq_client()
    db_status = init_db()
    
    if client_status:
        st.markdown("""
        <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.2); 
                    padding: 0.5rem; border-radius: 8px; margin: 0.5rem 0;">
            <span style="color: #10b981;">✅ AI Engine: Online</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.2); 
                    padding: 0.5rem; border-radius: 8px; margin: 0.5rem 0;">
            <span style="color: #f59e0b;">⚠️ AI Engine: Configure API</span>
        </div>
        """, unsafe_allow_html=True)
    
    if db_status:
        st.markdown("""
        <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.2); 
                    padding: 0.5rem; border-radius: 8px; margin: 0.5rem 0;">
            <span style="color: #10b981;">✅ Database: Connected</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.2); 
                    padding: 0.5rem; border-radius: 8px; margin: 0.5rem 0;">
            <span style="color: #ef4444;">❌ Database: Error</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 🤖 AI Model Selection")
    models = {
        "🦙 Llama 3.3 70B": "llama-3.3-70b-versatile",
        "⚡ Llama 3.1 8B (Fast)": "llama-3.1-8b-instant", 
        "🔥 Mixtral 8x7B": "mixtral-8x7b-32768",
        "💎 Gemma 2 9B": "gemma2-9b-it"
    }
    selected = st.selectbox("Choose AI Model", list(models.keys()), index=0)
    st.session_state.selected_model = models[selected]
    
    # Model info
    model_info = {
        "🦙 Llama 3.3 70B": "Best quality, slower",
        "⚡ Llama 3.1 8B (Fast)": "Fast responses",
        "🔥 Mixtral 8x7B": "Balanced performance",
        "💎 Gemma 2 9B": "Efficient and accurate"
    }
    st.caption(f"ℹ️ {model_info[selected]}")
    
    st.markdown("---")
    
    # Development mode indicator - LOCAL ONLY
    if is_logging_enabled():
        st.markdown("### 🔧 Development Mode")
        st.info("📝 Logging enabled for local development\n\nUse command line tools:\n- `python log_analyzer.py`\n- `python log_monitor.py`")

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "➕ Input", "📚 Guide"])

with tab1:
    if not st.session_state.soil_data:
        # Welcome screen with enhanced styling
        st.markdown("""
        <div style="text-align: center; padding: 3rem 1rem;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🌱</div>
            <h2 style="color: #60a5fa; margin-bottom: 1rem;">Welcome to NutriSense</h2>
            <p style="font-size: 1.2rem; color: #94a3b8; margin-bottom: 2rem;">
                Get started by entering your soil data in the <strong>Input</strong> tab
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature showcase
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card" style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🧪</div>
                <h4>Chemistry Analysis</h4>
                <p style="color: #94a3b8; font-size: 0.9rem;">pH, EC, NPK levels</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card" style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">💧</div>
                <h4>Physical Properties</h4>
                <p style="color: #94a3b8; font-size: 0.9rem;">Moisture & temperature</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card" style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🦠</div>
                <h4>Biological Activity</h4>
                <p style="color: #94a3b8; font-size: 0.9rem;">Microbial health index</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card" style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🤖</div>
                <h4>AI Insights</h4>
                <p style="color: #94a3b8; font-size: 0.9rem;">Smart crop & fertilizer advice</p>
            </div>
            """, unsafe_allow_html=True)
        
    else:
        soil = st.session_state.soil_data
        loc = st.session_state.location
        
        # Enhanced overview section
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h2 style="color: #60a5fa;">🎯 Soil Analysis Overview</h2>
            {f'<p style="color: #94a3b8; font-size: 1.1rem;">📍 {loc}</p>' if loc else ''}
        </div>
        """, unsafe_allow_html=True)
        
        health = get_health_score(soil)
        
        # Enhanced metrics row
        col1, col2, col3 = st.columns(3)
        
        with col1:
            health_color = "#10b981" if health >= 70 else "#f59e0b" if health >= 50 else "#ef4444"
            health_status = "Excellent" if health >= 70 else "Good" if health >= 50 else "Needs Attention"
            st.markdown(f"""
            <div class="metric-card" style="text-align: center;">
                <div style="font-size: 2.5rem; color: {health_color}; margin-bottom: 0.5rem;">{health:.0f}</div>
                <h4>Health Score</h4>
                <p style="color: {health_color}; font-weight: 500;">{health_status}</p>
                <div style="background: {health_color}; height: 4px; border-radius: 2px; margin-top: 1rem; width: {health}%;"></div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card" style="text-align: center;">
                <div style="font-size: 2.5rem; color: #60a5fa; margin-bottom: 0.5rem;">8</div>
                <h4>Parameters Tracked</h4>
                <p style="color: #10b981; font-weight: 500;">All Systems Active</p>
                <div style="background: #10b981; height: 4px; border-radius: 2px; margin-top: 1rem;"></div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card" style="text-align: center;">
                <div style="font-size: 2.5rem; color: #60a5fa; margin-bottom: 0.5rem;">✓</div>
                <h4>Analysis Status</h4>
                <p style="color: #10b981; font-weight: 500;">Complete</p>
                <div style="background: #10b981; height: 4px; border-radius: 2px; margin-top: 1rem;"></div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 🔬 Parameter Analysis")
            
            params = {
                'pH': (soil['pH'], 'pH'), 'EC': (soil['EC'], 'dS/m'),
                'Moisture': (soil['Moisture'], '%'), 'Nitrogen': (soil['Nitrogen'], 'mg/kg'),
                'Phosphorus': (soil['Phosphorus'], 'mg/kg'), 'Potassium': (soil['Potassium'], 'mg/kg'),
                'Microbial': (soil['Microbial'], 'Index'), 'Temperature': (soil['Temperature'], '°C')
            }
            
            for name, (val, unit) in params.items():
                status, emoji = interpret(name, val)
                css = "status-good" if "🟢" in emoji or "💚" in emoji else ("status-warning" if "🟡" in emoji else "status-critical")
                
                # Enhanced parameter display with progress bars
                progress_val = min(val / {"pH": 14, "EC": 4, "Moisture": 100, "Nitrogen": 100, "Phosphorus": 100, "Potassium": 300, "Microbial": 10, "Temperature": 50}[name], 1.0)
                
                st.markdown(f'''
                <div class="{css}">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <div><strong>{emoji} {name}:</strong> {val:.1f} {unit}</div>
                        <div style="color: #94a3b8; font-size: 0.9rem;">{status}</div>
                    </div>
                    <div style="background: rgba(255,255,255,0.1); height: 4px; border-radius: 2px;">
                        <div style="background: {"#10b981" if "🟢" in emoji or "💚" in emoji else "#f59e0b" if "🟡" in emoji else "#ef4444"}; 
                                    height: 100%; width: {progress_val*100:.0f}%; border-radius: 2px; transition: width 0.5s ease;"></div>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 📊 Health Score")
            
            # Compact gauge chart
            fig1 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=health,
                title={'text': "Overall Health", 'font': {'color': '#e2e8f0', 'size': 14}},
                number={'font': {'color': '#e2e8f0', 'size': 24}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#475569'},
                    'bar': {'color': "#667eea", 'thickness': 0.3},
                    'bgcolor': "rgba(30, 41, 59, 0.5)",
                    'bordercolor': "#334155",
                    'borderwidth': 2,
                    'steps': [
                        {'range': [0, 40], 'color': "rgba(239, 68, 68, 0.3)"},
                        {'range': [40, 70], 'color': "rgba(245, 158, 11, 0.3)"},
                        {'range': [70, 100], 'color': "rgba(16, 185, 129, 0.3)"}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 3},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig1.update_layout(
                height=200,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': '#e2e8f0'},
                margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(fig1)
            
            # Quick stats
            health_status = "Excellent" if health >= 70 else "Good" if health >= 50 else "Needs Attention"
            health_color = "#10b981" if health >= 70 else "#f59e0b" if health >= 50 else "#ef4444"
            
            st.markdown(f"""
            <div style="text-align: center; margin-top: 1rem;">
                <div style="color: {health_color}; font-weight: 600; font-size: 1.1rem;">{health_status}</div>
                <div style="color: #94a3b8; font-size: 0.9rem; margin-top: 0.5rem;">
                    {len([p for p in params.items() if interpret(p[0], p[1][0])[1] in ["🟢", "💚"]])} parameters optimal
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 🤖 AI-Powered Insights")
        
        # Enhanced AI buttons in a more compact layout
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("✨ Health Summary", width='stretch', type="primary"):
                log_user_action('AI_SUMMARY_REQUESTED', {'location': loc, 'health_score': health})
                with st.spinner("🧠 AI is analyzing your soil..."):
                    prompt = build_prompt(soil, "summary", loc)
                    result = call_groq(hashlib.md5(prompt.encode()).hexdigest(), prompt, "summary")
                    st.session_state.summary = result
                    save_record(soil, result, loc)
                    log_user_action('AI_SUMMARY_COMPLETED', {'result_length': len(result)})
        
        with col2:
            if st.button("🌾 Crop Recommendations", width='stretch'):
                log_user_action('AI_CROPS_REQUESTED', {'location': loc, 'health_score': health})
                with st.spinner("🌱 Finding optimal crops..."):
                    prompt = build_prompt(soil, "crops", loc)
                    result = call_groq(hashlib.md5(prompt.encode()).hexdigest(), prompt, "crops")
                    st.session_state.crops = result
                    log_user_action('AI_CROPS_COMPLETED', {'result_length': len(result)})
        
        with col3:
            if st.button("💊 Fertilizer Plan", width='stretch'):
                log_user_action('AI_FERTILIZER_REQUESTED', {'location': loc, 'health_score': health})
                with st.spinner("🧪 Calculating nutrients..."):
                    prompt = build_prompt(soil, "fertilizer", loc)
                    result = call_groq(hashlib.md5(prompt.encode()).hexdigest(), prompt, "fertilizer")
                    st.session_state.fertilizer = result
                    log_user_action('AI_FERTILIZER_COMPLETED', {'result_length': len(result)})
        
        # Display AI recommendations in a better layout
        if 'summary' in st.session_state and st.session_state.summary:
            st.markdown("#### 📋 Soil Health Summary")
            st.info(st.session_state.summary)
        
        # Two-column layout for crops and fertilizer recommendations
        if ('crops' in st.session_state and st.session_state.crops) or ('fertilizer' in st.session_state and st.session_state.fertilizer):
            col1, col2 = st.columns(2)
            
            with col1:
                if 'crops' in st.session_state and st.session_state.crops:
                    st.markdown("#### 🌾 Recommended Crops")
                    st.success(st.session_state.crops)
            
            with col2:
                if 'fertilizer' in st.session_state and st.session_state.fertilizer:
                    st.markdown("#### 💊 Fertilizer Plan")
                    st.info(st.session_state.fertilizer)

with tab2:
    st.markdown("### ➕ Enter Soil Test Results")
    st.markdown("Input your laboratory soil analysis data for AI-powered insights")
    
    # Enhanced location input
    st.markdown("#### 📍 Location Information")
    loc_input = st.text_input(
        "Field/Farm Location (Optional)", 
        value=st.session_state.location, 
        placeholder="e.g., North Field, Farm Block A, GPS coordinates...",
        help="Adding location helps provide more targeted recommendations"
    )
    
    st.markdown("---")
    
    with st.form("soil_form"):
        st.markdown("#### 🧪 Soil Parameters")
        
        col1, col2 = st.columns(2)
        
        # Use sample data if available
        sample = st.session_state.get('sample_data', {})
        
        with col1:
            st.markdown("**🔬 Chemical Properties**")
            
            pH = st.number_input(
                "pH Level", 
                min_value=0.0, max_value=14.0, value=sample.get('pH', 7.0), step=0.1,
                help="Soil acidity/alkalinity. Optimal range: 6.5-7.5"
            )
            
            EC = st.number_input(
                "Electrical Conductivity (dS/m)", 
                min_value=0.0, max_value=20.0, value=sample.get('EC', 2.0), step=0.1,
                help="Soil salinity indicator. <2.0 dS/m is ideal for most crops"
            )
            
            N = st.number_input(
                "Available Nitrogen (mg/kg)", 
                min_value=0.0, max_value=500.0, value=sample.get('Nitrogen', 50.0), step=1.0,
                help="Essential for plant growth. Optimal: 40-80 mg/kg"
            )
            
            P = st.number_input(
                "Available Phosphorus (mg/kg)", 
                min_value=0.0, max_value=200.0, value=sample.get('Phosphorus', 30.0), step=1.0,
                help="Important for root development. Optimal: 20-50 mg/kg"
            )
        
        with col2:
            st.markdown("**🌡️ Physical & Biological Properties**")
            
            Moist = st.number_input(
                "Moisture Content (%)", 
                min_value=0.0, max_value=100.0, value=sample.get('Moisture', 25.0), step=1.0,
                help="Current soil water content. Optimal: 25-40%"
            )
            
            Temp = st.number_input(
                "Soil Temperature (°C)", 
                min_value=0.0, max_value=50.0, value=sample.get('Temperature', 25.0), step=0.5,
                help="Current soil temperature affects microbial activity"
            )
            
            K = st.number_input(
                "Available Potassium (mg/kg)", 
                min_value=0.0, max_value=500.0, value=sample.get('Potassium', 150.0), step=1.0,
                help="Essential for disease resistance. Optimal: 100-250 mg/kg"
            )
            
            Micro = st.number_input(
                "Microbial Activity Index", 
                min_value=0.0, max_value=10.0, value=sample.get('Microbial', 5.0), step=0.1,
                help="Biological activity level (0-10 scale). Higher is better"
            )
        
        st.markdown("---")
        
        # Enhanced submit button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submitted = st.form_submit_button(
                "🔬 Analyze Soil Data", 
                width='stretch', 
                type="primary"
            )
        
        if submitted:
            try:
                log_user_action('SOIL_DATA_FORM_SUBMITTED', {'location': loc_input})
                
                # Validate all inputs are present and numeric
                input_values = {
                    "pH": pH, "EC": EC, "Moisture": Moist, "Nitrogen": N, 
                    "Phosphorus": P, "Potassium": K, "Microbial": Micro, "Temperature": Temp
                }
                
                # Check for None or invalid values
                for param, value in input_values.items():
                    if value is None:
                        raise ValueError(f"{param} cannot be empty")
                    if not isinstance(value, (int, float)):
                        raise ValueError(f"{param} must be a number")
                    if value < 0:
                        raise ValueError(f"{param} cannot be negative")
                
                # Additional range validations
                if not (0 <= pH <= 14):
                    raise ValueError("pH must be between 0 and 14")
                if EC > 20:
                    raise ValueError("EC cannot exceed 20 dS/m")
                if Moist > 100:
                    raise ValueError("Moisture cannot exceed 100%")
                if N > 500:
                    raise ValueError("Nitrogen cannot exceed 500 mg/kg")
                if P > 200:
                    raise ValueError("Phosphorus cannot exceed 200 mg/kg")
                if K > 500:
                    raise ValueError("Potassium cannot exceed 500 mg/kg")
                if Micro > 10:
                    raise ValueError("Microbial index cannot exceed 10")
                if Temp > 50:
                    raise ValueError("Temperature cannot exceed 50°C")
                
                soil_dict = input_values
                
                log_event('SOIL_DATA_VALIDATION_START', 'Validating soil data', soil_dict)
                
                # Validate using Pydantic with additional error context
                try:
                    SoilData(**soil_dict)
                except Exception as pydantic_error:
                    log_error(pydantic_error, 'PYDANTIC_VALIDATION_ERROR', soil_dict)
                    raise ValueError(f"Data validation failed: {str(pydantic_error)}")
                
                log_event('SOIL_DATA_VALIDATION_SUCCESS', 'Soil data validation passed')
                
                # Store data with error handling
                try:
                    st.session_state.soil_data = soil_dict
                    st.session_state.location = loc_input or "Unknown Location"
                    st.session_state.timestamp = datetime.now().isoformat()
                except Exception as storage_error:
                    log_error(storage_error, 'SESSION_STORAGE_ERROR', soil_dict)
                    raise ValueError("Failed to store data in session")
                
                # Clear previous AI results
                cleared_keys = []
                for key in ['summary', 'crops', 'fertilizer']:
                    if key in st.session_state:
                        del st.session_state[key]
                        cleared_keys.append(key)
                
                log_user_action('SOIL_DATA_SAVED', {
                    'location': loc_input,
                    'cleared_ai_results': cleared_keys,
                    'parameters': list(soil_dict.keys())
                })
                
                # Calculate health score with error handling
                try:
                    health_score = get_health_score(soil_dict)
                    if not isinstance(health_score, (int, float)) or health_score < 0 or health_score > 100:
                        log_error(ValueError(f"Invalid health score: {health_score}"), 'INVALID_HEALTH_SCORE')
                        health_score = 50.0  # Safe fallback
                except Exception as health_error:
                    log_error(health_error, 'HEALTH_SCORE_ERROR', soil_dict)
                    health_score = 50.0
                
                st.success("✅ Soil data saved and analyzed successfully!")
                
                # Show immediate analysis results with error handling
                try:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Health Score", f"{health_score:.0f}/100")
                    with col2:
                        ph_status, ph_icon = interpret('pH', soil_dict['pH'])
                        st.metric("pH Status", f"{ph_icon} {ph_status}")
                    with col3:
                        st.metric("Parameters", "8 tracked")
                except Exception as display_error:
                    log_error(display_error, 'METRICS_DISPLAY_ERROR')
                    st.warning("Analysis completed but display metrics failed")
                
                st.info("💡 Go to the **Dashboard** tab to view detailed analysis and AI recommendations.")
                
                # Trigger page refresh to update dashboard immediately
                st.rerun()
                
            except ValueError as e:
                log_error(e, 'SOIL_DATA_VALIDATION_ERROR', {
                    'form_data': soil_dict if 'soil_dict' in locals() else 'not_created',
                    'location': loc_input,
                    'error_type': 'ValueError'
                })
                st.error(f"❌ Validation Error: {str(e)}")
                st.info("💡 Please check that all values are within the specified ranges")
            
            except Exception as e:
                log_error(e, 'SOIL_DATA_FORM_ERROR', {
                    'location': loc_input,
                    'form_data': soil_dict if 'soil_dict' in locals() else 'not_created',
                    'error_type': type(e).__name__
                })
                st.error(f"❌ Unexpected Error: {str(e)}")
                st.info("💡 Please try again or contact support if the problem persists")
                
                # Show debug info in development
                if is_logging_enabled():
                    with st.expander("🔧 Debug Information (Development Only)"):
                        st.code(traceback.format_exc())
    
    # Help section
    st.markdown("---")
    st.markdown("### ❓ Need Help Getting Soil Data?")
    
    with st.expander("🔬 How to Obtain Soil Test Values"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🏢 Professional Lab Testing** (Recommended)
            - Contact local agricultural extension office
            - Use certified soil testing laboratories
            - Most accurate and comprehensive results
            - Usually costs $15-50 per test
            
            **📱 Digital Soil Meters**
            - pH meters, EC meters, moisture sensors
            - Good for regular monitoring
            - Calibrate regularly for accuracy
            - Investment: $50-200
            """)
        
        with col2:
            st.markdown("""
            **🏠 Home Test Kits**
            - Available at garden centers
            - Less accurate but affordable
            - Good for general assessment
            - Cost: $10-30
            
            **📊 Typical Testing Schedule**
            - Spring: Before planting season
            - Fall: After harvest
            - Every 2-3 years: Comprehensive analysis
            - Monthly: Basic pH and moisture
            """)
    
    with st.expander("📋 Parameter Guidelines & Interpretation"):
        st.markdown("""
        | Parameter | Low | Optimal | High | Critical Actions |
        |-----------|-----|---------|------|------------------|
        | **pH** | <5.5 | 6.5-7.5 | >8.5 | Add lime (low) or sulfur (high) |
        | **EC (dS/m)** | <0.4 | 0.4-0.8 | >2.0 | Improve drainage, leaching |
        | **Moisture (%)** | <15 | 25-40 | >60 | Irrigation or drainage needed |
        | **Nitrogen (mg/kg)** | <40 | 40-80 | >120 | Adjust fertilizer application |
        | **Phosphorus (mg/kg)** | <20 | 20-50 | >80 | Monitor for runoff risk |
        | **Potassium (mg/kg)** | <100 | 100-250 | >350 | Balance with other nutrients |
        | **Microbial Index** | <3 | 5-7 | >8 | Add organic matter if low |
        """)
    
    # Sample data button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("📝 Load Sample Data", width='stretch', type="secondary"):
            # Load sample data and refresh form
            sample_data = {
                'pH': 6.8, 'EC': 1.2, 'Moisture': 32.0, 'Temperature': 24.0,
                'Nitrogen': 65.0, 'Phosphorus': 28.0, 'Potassium': 180.0, 'Microbial': 5.8
            }
            st.session_state.sample_data = sample_data
            st.success("✅ Sample data loaded! The form above now contains realistic soil test values.")
            st.rerun()

with tab3:
    st.markdown("### 📚 Soil Science Knowledge Base")
    st.markdown("Comprehensive guide to understanding and optimizing soil health")
    
    # Quick reference cards
    st.markdown("#### 🎯 Quick Reference")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="rec-box">
            <h4>🚨 Critical Ranges</h4>
            <ul>
                <li><strong>pH:</strong> Avoid <5.0 or >9.0</li>
                <li><strong>EC:</strong> Keep <4.0 dS/m</li>
                <li><strong>Moisture:</strong> Prevent <10% or >80%</li>
                <li><strong>NPK:</strong> Balance is key</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="rec-box">
            <h4>✅ Optimal Targets</h4>
            <ul>
                <li><strong>pH:</strong> 6.5-7.5 for most crops</li>
                <li><strong>EC:</strong> 0.4-0.8 dS/m</li>
                <li><strong>Moisture:</strong> 25-40%</li>
                <li><strong>Microbial:</strong> 5-7 index</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Detailed parameter guides
    with st.expander("🎯 pH - Soil Acidity & Alkalinity", expanded=False):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            **Understanding pH:**
            - **Scale:** 0-14 (7 is neutral)
            - **Optimal Range:** 6.5-7.5 for most crops
            - **Impact:** Affects nutrient availability and microbial activity
            
            **pH Levels & Effects:**
            - **<5.5 (Highly Acidic):** Aluminum toxicity, reduced nutrient uptake
            - **5.5-6.5 (Slightly Acidic):** Some crops thrive, others struggle
            - **6.5-7.5 (Optimal):** Maximum nutrient availability
            - **7.5-8.5 (Slightly Alkaline):** Iron and zinc deficiency possible
            - **>8.5 (Highly Alkaline):** Severe nutrient lockout
            
            **Correction Methods:**
            - **Too Low:** Add agricultural lime (CaCO₃)
            - **Too High:** Add sulfur or organic matter
            - **Monitoring:** Test every 6 months during correction
            """)
        
        with col2:
            st.markdown("""
            **🌱 Crop pH Preferences:**
            - **Acidic (5.5-6.5):** Blueberries, potatoes
            - **Neutral (6.5-7.5):** Most vegetables, grains
            - **Alkaline (7.5-8.5):** Asparagus, beets
            
            **⚠️ Warning Signs:**
            - Yellowing leaves (chlorosis)
            - Poor root development
            - Stunted growth
            - Increased pest problems
            """)
    
    with st.expander("⚡ EC - Electrical Conductivity & Salinity"):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            **Understanding EC:**
            - **Measurement:** Electrical conductivity in dS/m
            - **Indicates:** Total dissolved salts in soil
            - **Optimal Range:** 0.4-0.8 dS/m
            
            **Salinity Levels:**
            - **0-0.8 dS/m:** Low salinity - safe for all crops
            - **0.8-2.0 dS/m:** Moderate - some sensitive crops affected
            - **2.0-4.0 dS/m:** High - only tolerant crops survive
            - **>4.0 dS/m:** Very high - severe crop damage
            
            **Management Strategies:**
            - **Leaching:** Apply excess water to flush salts
            - **Drainage:** Improve soil drainage systems
            - **Amendments:** Add gypsum for sodium-rich soils
            - **Crop Selection:** Choose salt-tolerant varieties
            """)
        
        with col2:
            st.markdown("""
            **🧂 Salt-Tolerant Crops:**
            - **High Tolerance:** Barley, sugar beet
            - **Moderate:** Wheat, cotton, tomato
            - **Low Tolerance:** Beans, strawberries
            
            **💧 Leaching Requirements:**
            - **EC 2-4:** 15-30% extra water
            - **EC 4-8:** 30-50% extra water
            - **EC >8:** Professional remediation
            """)
    
    with st.expander("🌿 NPK - Essential Macronutrients"):
        st.markdown("""
        ### Nitrogen (N) - The Growth Engine
        **Functions:** Protein synthesis, chlorophyll production, vegetative growth
        - **Optimal Range:** 40-80 mg/kg
        - **Deficiency Signs:** Yellowing leaves (starting from bottom), stunted growth
        - **Excess Signs:** Dark green foliage, delayed maturity, lodging
        - **Sources:** Urea (46-0-0), Ammonium sulfate (21-0-0), Compost
        
        ### Phosphorus (P) - The Root Builder
        **Functions:** Root development, flowering, fruit formation, energy transfer
        - **Optimal Range:** 20-50 mg/kg
        - **Deficiency Signs:** Purple leaf tinge, poor root growth, delayed maturity
        - **Excess Signs:** Reduced zinc and iron uptake
        - **Sources:** DAP (18-46-0), SSP (0-16-0), Bone meal
        
        ### Potassium (K) - The Protector
        **Functions:** Disease resistance, water regulation, enzyme activation
        - **Optimal Range:** 100-250 mg/kg
        - **Deficiency Signs:** Brown leaf edges, weak stems, poor fruit quality
        - **Excess Signs:** Reduced calcium and magnesium uptake
        - **Sources:** MOP (0-0-60), SOP (0-0-50), Wood ash
        
        ### 🎯 NPK Balance Tips:
        - **Vegetative Growth:** Higher N ratio (3-1-2)
        - **Flowering/Fruiting:** Higher P and K (1-3-2)
        - **Maintenance:** Balanced ratio (1-1-1)
        """)
    
    with st.expander("💧 Moisture Management"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Optimal Moisture Levels:**
            - **Sandy Soils:** 15-25%
            - **Loamy Soils:** 25-35%
            - **Clay Soils:** 35-45%
            
            **Moisture Stress Indicators:**
            - **Too Low (<15%):** Wilting, leaf drop, stunted growth
            - **Too High (>60%):** Root rot, fungal diseases, poor aeration
            
            **Irrigation Guidelines:**
            - **Frequency:** Deep, infrequent watering preferred
            - **Timing:** Early morning or late evening
            - **Amount:** 1-2 inches per week for most crops
            """)
        
        with col2:
            st.markdown("""
            **💡 Water Management Tips:**
            - **Mulching:** Reduces evaporation by 50-70%
            - **Drip Irrigation:** 90% efficiency vs 60% sprinkler
            - **Soil Amendments:** Compost improves water retention
            - **Cover Crops:** Reduce soil moisture loss
            
            **🌡️ Temperature Effects:**
            - **Hot Weather:** Increase watering frequency
            - **Cool Weather:** Reduce watering, improve drainage
            - **Seasonal:** Adjust based on crop growth stage
            """)
    
    with st.expander("🦠 Microbial Activity & Soil Biology"):
        st.markdown("""
        **Understanding Soil Biology:**
        - **Microbial Index:** 0-10 scale measuring biological activity
        - **Optimal Range:** 5-7 for healthy soil ecosystem
        - **Key Players:** Bacteria, fungi, protozoa, nematodes
        
        **Benefits of Active Soil Biology:**
        - **Nutrient Cycling:** Converts organic matter to plant-available nutrients
        - **Disease Suppression:** Beneficial microbes outcompete pathogens
        - **Soil Structure:** Fungal hyphae bind soil particles
        - **Water Retention:** Improved soil aggregation
        
        **Enhancing Microbial Activity:**
        - **Organic Matter:** Add compost, manure, crop residues
        - **Reduce Tillage:** Minimal disturbance preserves fungal networks
        - **Cover Crops:** Provide continuous root exudates
        - **Avoid Chemicals:** Reduce pesticide and synthetic fertilizer use
        - **pH Management:** Maintain optimal pH for microbial growth
        
        **🔬 Biological Indicators:**
        - **High Activity (7-10):** Rich, dark soil with earthworms
        - **Moderate Activity (4-6):** Some organic matter, limited biology
        - **Low Activity (0-3):** Compacted, lifeless soil
        """)
    
    with st.expander("🌡️ Temperature Effects on Soil Health"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Temperature Ranges:**
            - **Cold (<10°C):** Slow microbial activity, reduced nutrient availability
            - **Optimal (15-30°C):** Peak biological activity, good root growth
            - **Hot (>35°C):** Heat stress, increased water demand
            
            **Seasonal Management:**
            - **Spring:** Gradual warming, start fertilization
            - **Summer:** Peak activity, monitor moisture
            - **Fall:** Prepare for dormancy, reduce inputs
            - **Winter:** Minimal activity, plan improvements
            """)
        
        with col2:
            st.markdown("""
            **🌡️ Temperature Tips:**
            - **Mulching:** Moderates soil temperature
            - **Shade Cloth:** Protects from extreme heat
            - **Irrigation:** Cooling effect in hot weather
            - **Timing:** Plant when soil temps are optimal
            
            **📊 Crop Temperature Preferences:**
            - **Cool Season:** 10-20°C (lettuce, peas)
            - **Warm Season:** 20-30°C (tomatoes, peppers)
            - **Hot Season:** 25-35°C (melons, okra)
            """)
    
    # Action plan generator
    st.markdown("---")
    st.markdown("### 🎯 Personalized Action Plan")
    
    if st.session_state.soil_data:
        soil = st.session_state.soil_data
        
        st.markdown("**Based on your current soil data:**")
        
        # Generate recommendations
        recommendations = []
        
        # pH recommendations
        if soil['pH'] < 6.5:
            recommendations.append("🔧 **pH Too Low:** Apply agricultural lime at 1-2 tons/hectare")
        elif soil['pH'] > 7.5:
            recommendations.append("🔧 **pH Too High:** Apply sulfur at 200-500 kg/hectare")
        
        # EC recommendations
        if soil['EC'] > 2.0:
            recommendations.append("💧 **High Salinity:** Implement leaching program with 25% extra irrigation")
        
        # Moisture recommendations
        if soil['Moisture'] < 20:
            recommendations.append("💧 **Low Moisture:** Increase irrigation frequency and add mulch")
        elif soil['Moisture'] > 50:
            recommendations.append("🚰 **High Moisture:** Improve drainage and reduce irrigation")
        
        # NPK recommendations
        if soil['Nitrogen'] < 40:
            recommendations.append("🌿 **Low Nitrogen:** Apply nitrogen fertilizer at 100-150 kg N/hectare")
        if soil['Phosphorus'] < 20:
            recommendations.append("🌿 **Low Phosphorus:** Apply phosphate fertilizer at 50-75 kg P₂O₅/hectare")
        if soil['Potassium'] < 100:
            recommendations.append("🌿 **Low Potassium:** Apply potash fertilizer at 75-100 kg K₂O/hectare")
        
        # Microbial recommendations
        if soil['Microbial'] < 4:
            recommendations.append("🦠 **Low Biology:** Add 2-4 tons compost/hectare and reduce tillage")
        
        if recommendations:
            for rec in recommendations:
                st.markdown(f"- {rec}")
        else:
            st.success("🎉 **Excellent!** Your soil parameters are all within optimal ranges!")
    
    else:
        st.info("💡 Enter your soil data in the **Input** tab to get personalized recommendations!")
    
    # Footer with tips
    st.markdown("---")
    st.markdown("""
    <div style="background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.2); 
                padding: 1.5rem; border-radius: 12px; text-align: center;">
        <h4 style="color: #60a5fa; margin-bottom: 1rem;">💡 Pro Tips for Soil Health</h4>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; text-align: left;">
            <div>
                <strong>🔄 Regular Testing:</strong><br>
                Test soil every 3-6 months during growing season
            </div>
            <div>
                <strong>📊 Keep Records:</strong><br>
                Track changes over time to identify trends
            </div>
            <div>
                <strong>🌱 Gradual Changes:</strong><br>
                Make small adjustments rather than drastic changes
            </div>
            <div>
                <strong>🤝 Seek Advice:</strong><br>
                Consult local extension services for region-specific guidance
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)