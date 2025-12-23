#!/usr/bin/env python3
"""
Real-time JSON Log Viewer for NutriSense
Displays logs from the single JSON file in a readable format
"""

import json
import os
import time
from datetime import datetime
import argparse

class LogViewer:
    def __init__(self, log_file='logs/nutrisense_realtime.json'):
        self.log_file = log_file
        self.last_count = 0
        
    def load_logs(self):
        """Load logs from JSON file"""
        try:
            if not os.path.exists(self.log_file):
                return {"logs": [], "metadata": {}}
            
            with open(self.log_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error reading log file: {e}")
            return {"logs": [], "metadata": {}}
    
    def format_log_entry(self, entry):
        """Format a single log entry for display"""
        timestamp = entry.get('timestamp', 'Unknown')
        level = entry.get('level', 'INFO')
        message = entry.get('message', '')
        
        # Color coding for different log levels
        colors = {
            'DEBUG': '\033[36m',    # Cyan
            'INFO': '\033[32m',     # Green
            'WARNING': '\033[33m',  # Yellow
            'ERROR': '\033[31m',    # Red
            'CRITICAL': '\033[35m'  # Magenta
        }
        reset_color = '\033[0m'
        
        color = colors.get(level, '')
        
        # Format timestamp
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            formatted_time = dt.strftime('%H:%M:%S')
        except:
            formatted_time = timestamp[:8] if len(timestamp) > 8 else timestamp
        
        # Basic format
        formatted = f"{color}[{formatted_time}] {level:8} | {message}{reset_color}"
        
        # Add structured data if present
        if entry.get('event_type'):
            formatted += f"\n    Event: {entry['event_type']}"
            if entry.get('event_data'):
                formatted += f" | Data: {json.dumps(entry['event_data'], default=str)}"
        
        if entry.get('error_type'):
            formatted += f"\n    Error: {entry['error_type']} - {entry.get('error_message', '')}"
            if entry.get('error_context'):
                formatted += f" | Context: {entry['error_context']}"
        
        if entry.get('session_id'):
            formatted += f"\n    Session: {entry['session_id']}"
        
        return formatted
    
    def display_logs(self, filter_level=None, tail=None, follow=False):
        """Display logs with optional filtering"""
        data = self.load_logs()
        logs = data.get('logs', [])
        metadata = data.get('metadata', {})
        
        # Display metadata
        print(f"\n📊 Log File: {self.log_file}")
        print(f"📅 Created: {metadata.get('created', 'Unknown')}")
        print(f"🔄 Last Updated: {metadata.get('last_updated', 'Unknown')}")
        print(f"📈 Total Logs: {metadata.get('total_logs', len(logs))}")
        if metadata.get('truncated'):
            print(f"⚠️  Truncated at: {metadata.get('truncated_at', 'Unknown')}")
        print("-" * 80)
        
        # Filter by level if specified
        if filter_level:
            logs = [log for log in logs if log.get('level') == filter_level.upper()]
        
        # Apply tail if specified
        if tail:
            logs = logs[-tail:]
        
        # Display logs
        for entry in logs:
            print(self.format_log_entry(entry))
            print()
        
        if follow:
            self.last_count = len(data.get('logs', []))
    
    def follow_logs(self, filter_level=None):
        """Follow logs in real-time"""
        print("🔄 Following logs in real-time... (Press Ctrl+C to stop)")
        print("=" * 80)
        
        try:
            # Initial display
            self.display_logs(filter_level=filter_level, follow=True)
            
            while True:
                time.sleep(1)  # Check every second
                data = self.load_logs()
                logs = data.get('logs', [])
                
                # Check for new logs
                if len(logs) > self.last_count:
                    new_logs = logs[self.last_count:]
                    
                    # Filter new logs if needed
                    if filter_level:
                        new_logs = [log for log in new_logs if log.get('level') == filter_level.upper()]
                    
                    # Display new logs
                    for entry in new_logs:
                        print(self.format_log_entry(entry))
                        print()
                    
                    self.last_count = len(logs)
                    
        except KeyboardInterrupt:
            print("\n👋 Stopped following logs.")
    
    def search_logs(self, search_term, case_sensitive=False):
        """Search logs for specific terms"""
        data = self.load_logs()
        logs = data.get('logs', [])
        
        matching_logs = []
        for entry in logs:
            message = entry.get('message', '')
            if not case_sensitive:
                message = message.lower()
                search_term = search_term.lower()
            
            if search_term in message:
                matching_logs.append(entry)
        
        print(f"🔍 Found {len(matching_logs)} logs matching '{search_term}':")
        print("-" * 80)
        
        for entry in matching_logs:
            print(self.format_log_entry(entry))
            print()
    
    def get_stats(self):
        """Display log statistics"""
        data = self.load_logs()
        logs = data.get('logs', [])
        
        if not logs:
            print("📭 No logs found.")
            return
        
        # Count by level
        level_counts = {}
        event_counts = {}
        error_counts = {}
        
        for entry in logs:
            level = entry.get('level', 'UNKNOWN')
            level_counts[level] = level_counts.get(level, 0) + 1
            
            if entry.get('event_type'):
                event_type = entry['event_type']
                event_counts[event_type] = event_counts.get(event_type, 0) + 1
            
            if entry.get('error_type'):
                error_type = entry['error_type']
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        print("📊 Log Statistics:")
        print("-" * 40)
        print("📈 By Level:")
        for level, count in sorted(level_counts.items()):
            print(f"   {level}: {count}")
        
        if event_counts:
            print("\n🎯 By Event Type:")
            for event, count in sorted(event_counts.items()):
                print(f"   {event}: {count}")
        
        if error_counts:
            print("\n🚨 By Error Type:")
            for error, count in sorted(error_counts.items()):
                print(f"   {error}: {count}")

def main():
    parser = argparse.ArgumentParser(description='NutriSense Real-time JSON Log Viewer')
    parser.add_argument('--file', '-f', default='logs/nutrisense_realtime.json', 
                       help='Path to JSON log file')
    parser.add_argument('--level', '-l', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='Filter by log level')
    parser.add_argument('--tail', '-t', type=int, help='Show last N log entries')
    parser.add_argument('--follow', action='store_true', help='Follow logs in real-time')
    parser.add_argument('--search', '-s', help='Search for specific term in logs')
    parser.add_argument('--stats', action='store_true', help='Show log statistics')
    parser.add_argument('--case-sensitive', action='store_true', help='Case sensitive search')
    
    args = parser.parse_args()
    
    viewer = LogViewer(args.file)
    
    if args.stats:
        viewer.get_stats()
    elif args.search:
        viewer.search_logs(args.search, args.case_sensitive)
    elif args.follow:
        viewer.follow_logs(args.level)
    else:
        viewer.display_logs(args.level, args.tail)

if __name__ == "__main__":
    main()