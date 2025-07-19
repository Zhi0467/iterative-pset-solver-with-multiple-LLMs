"""
Memory management for auto-pset: discussion tracking and session learning.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class SessionFailure:
    """Data class for tracking session failures."""
    pdf_path: str
    failure_type: str
    error_message: str
    stage: str
    timestamp: str

@dataclass
class SessionSummary:
    """Data class for session summary and learning."""
    session_id: str
    timestamp: str
    pdfs_processed: int
    pdfs_successful: int
    pdfs_failed: int
    failures: List[SessionFailure]
    patterns_observed: List[str]
    lessons_learned: List[str]
    performance_metrics: Dict[str, float]

class DiscussionTracker:
    """Tracks solver-verifier discussions for each PDF."""
    
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.discussions_dir = self.output_dir / "discussions"
        self.discussions_dir.mkdir(exist_ok=True, parents=True)
    
    def get_discussion_file(self, pdf_path: str) -> Path:
        """Get the discussion file path for a PDF."""
        pdf_name = Path(pdf_path).stem
        return self.discussions_dir / f"{pdf_name}_discussion.md"
    
    def start_discussion(self, pdf_path: str, pdf_analysis: Dict) -> None:
        """Initialize discussion file for a PDF."""
        discussion_file = self.get_discussion_file(pdf_path)
        
        content = f"""# Discussion for {Path(pdf_path).name}

## Session Information
- **Started**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **PDF Path**: {pdf_path}

## PDF Analysis
- **Subject**: {pdf_analysis.get('subject', 'Unknown')}
- **Difficulty**: {pdf_analysis.get('estimated_difficulty', 'Unknown')}
- **Topics**: {', '.join(pdf_analysis.get('topics', []))}
- **Has Math**: {pdf_analysis.get('has_math', False)}
- **Has Code**: {pdf_analysis.get('has_code', False)}
- **Has Figures**: {pdf_analysis.get('has_figures', False)}
- **Web Search Enabled**: {pdf_analysis.get('enable_web_search', False)}

## Solving Progress

"""
        
        with open(discussion_file, 'w') as f:
            f.write(content)
    
    def add_solver_thoughts(self, pdf_path: str, round_num: int, solver_provider: str, thoughts: str) -> None:
        """Add solver's thoughts to the discussion."""
        discussion_file = self.get_discussion_file(pdf_path)
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        content = f"""
### Round {round_num} - Solver ({solver_provider}) - {timestamp}

{thoughts}

"""
        
        with open(discussion_file, 'a') as f:
            f.write(content)
    
    def add_verifier_feedback(self, pdf_path: str, round_num: int, verifier_provider: str, feedback: str, corrections: str = None) -> None:
        """Add verifier's feedback to the discussion."""
        discussion_file = self.get_discussion_file(pdf_path)
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        content = f"""
### Round {round_num} - Verifier ({verifier_provider}) - {timestamp}

**Feedback:**
{feedback}
"""
        
        if corrections:
            content += f"""
**Corrections:**
{corrections}
"""
        
        content += "\n"
        
        with open(discussion_file, 'a') as f:
            f.write(content)
    
    def add_final_outcome(self, pdf_path: str, success: bool, final_thoughts: str = None) -> None:
        """Add final outcome to the discussion."""
        discussion_file = self.get_discussion_file(pdf_path)
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        outcome = "✅ SUCCESS" if success else "❌ FAILED"
        
        content = f"""
## Final Outcome - {timestamp}

**Status**: {outcome}

"""
        
        if final_thoughts:
            content += f"""**Final Thoughts:**
{final_thoughts}

"""
        
        content += "---\n\n"
        
        with open(discussion_file, 'a') as f:
            f.write(content)
    
    def get_previous_context(self, pdf_path: str) -> str:
        """Get previous discussion context for this PDF."""
        discussion_file = self.get_discussion_file(pdf_path)
        
        if discussion_file.exists():
            with open(discussion_file, 'r') as f:
                content = f.read()
            
            # Extract previous solving attempts for context
            if "## Solving Progress" in content:
                solving_section = content.split("## Solving Progress")[1]
                if "## Final Outcome" in solving_section:
                    solving_section = solving_section.split("## Final Outcome")[0]
                
                if solving_section.strip():
                    return f"Previous discussion context:\n{solving_section.strip()}\n\n"
        
        return ""

class SessionLearning:
    """Tracks session-level learning and patterns."""
    
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.memory_dir = self.output_dir / "memory"
        self.memory_dir.mkdir(exist_ok=True, parents=True)
        
        self.session_file = self.memory_dir / "sessions.jsonl"
        self.patterns_file = self.memory_dir / "learned_patterns.md"
        
        self.current_session = None
        self.failures = []
    
    def start_session(self, session_id: str = None) -> str:
        """Start a new session."""
        if not session_id:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_session = SessionSummary(
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            pdfs_processed=0,
            pdfs_successful=0,
            pdfs_failed=0,
            failures=[],
            patterns_observed=[],
            lessons_learned=[],
            performance_metrics={}
        )
        
        self.failures = []
        return session_id
    
    def record_pdf_result(self, pdf_path: str, success: bool, error_message: str = None, stage: str = None) -> None:
        """Record the result of processing a PDF."""
        if not self.current_session:
            return
        
        self.current_session.pdfs_processed += 1
        
        if success:
            self.current_session.pdfs_successful += 1
        else:
            self.current_session.pdfs_failed += 1
            
            if error_message:
                failure = SessionFailure(
                    pdf_path=pdf_path,
                    failure_type=self._categorize_failure(error_message),
                    error_message=error_message,
                    stage=stage or "unknown",
                    timestamp=datetime.now().isoformat()
                )
                self.failures.append(failure)
    
    def _categorize_failure(self, error_message: str) -> str:
        """Categorize failure type based on error message."""
        error_lower = error_message.lower()
        
        if "json" in error_lower or "parse" in error_lower:
            return "json_parsing"
        elif "network" in error_lower or "connection" in error_lower:
            return "network"
        elif "rate limit" in error_lower:
            return "rate_limit"
        elif "api" in error_lower:
            return "api_error"
        elif "file" in error_lower or "pdf" in error_lower:
            return "file_processing"
        else:
            return "unknown"
    
    def analyze_patterns(self) -> List[str]:
        """Analyze patterns from current session failures."""
        patterns = []
        
        if not self.failures:
            return patterns
        
        # Analyze failure types
        failure_types = {}
        for failure in self.failures:
            failure_types[failure.failure_type] = failure_types.get(failure.failure_type, 0) + 1
        
        for failure_type, count in failure_types.items():
            if count > 1:
                patterns.append(f"Multiple {failure_type} failures ({count} occurrences)")
        
        # Analyze stages where failures occur
        stage_failures = {}
        for failure in self.failures:
            stage_failures[failure.stage] = stage_failures.get(failure.stage, 0) + 1
        
        for stage, count in stage_failures.items():
            if count > 1:
                patterns.append(f"Repeated failures at {stage} stage ({count} times)")
        
        return patterns
    
    def generate_lessons(self) -> List[str]:
        """Generate lessons learned from current session."""
        lessons = []
        
        # Success rate analysis
        if self.current_session.pdfs_processed > 0:
            success_rate = self.current_session.pdfs_successful / self.current_session.pdfs_processed
            
            if success_rate < 0.5:
                lessons.append("Low success rate - consider improving error handling or retries")
            elif success_rate > 0.9:
                lessons.append("High success rate - current configuration is working well")
        
        # Failure-specific lessons
        failure_types = [f.failure_type for f in self.failures]
        
        if failure_types.count("json_parsing") > 1:
            lessons.append("JSON parsing issues recurring - need better prompt engineering or response cleaning")
        
        if failure_types.count("rate_limit") > 0:
            lessons.append("Rate limiting encountered - consider adding delays or using different providers")
        
        if failure_types.count("network") > 0:
            lessons.append("Network issues detected - implement better retry mechanisms")
        
        return lessons
    
    def end_session(self, discussions_dir: str = None, summarizer = None) -> None:
        """End current session and save summary."""
        if not self.current_session:
            return
        
        # Add failures to session
        self.current_session.failures = self.failures
        
        # Analyze patterns and generate lessons
        self.current_session.patterns_observed = self.analyze_patterns()
        self.current_session.lessons_learned = self.generate_lessons()
        
        # Generate comprehensive session lessons using summarizer if available
        if summarizer and discussions_dir:
            try:
                session_lessons = summarizer.summarize_session_discussions(discussions_dir, self.current_session)
                # Save comprehensive lessons to a separate file
                lessons_file = self.memory_dir / f"session_{self.current_session.session_id}_lessons.md"
                with open(lessons_file, 'w') as f:
                    f.write(f"# Session Lessons: {self.current_session.session_id}\n\n")
                    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.write(session_lessons)
                print(f"📚 Comprehensive session lessons saved to: {lessons_file}")
            except Exception as e:
                print(f"⚠️ Failed to generate comprehensive lessons: {e}")
        
        # Calculate performance metrics
        if self.current_session.pdfs_processed > 0:
            self.current_session.performance_metrics = {
                "success_rate": self.current_session.pdfs_successful / self.current_session.pdfs_processed,
                "failure_rate": self.current_session.pdfs_failed / self.current_session.pdfs_processed,
                "total_failures": len(self.failures)
            }
        
        # Save to JSONL file
        with open(self.session_file, 'a') as f:
            f.write(json.dumps(asdict(self.current_session)) + '\n')
        
        # Update patterns file
        self._update_patterns_file()
        
        print(f"\n📝 Session Summary Saved:")
        print(f"   Success Rate: {self.current_session.performance_metrics.get('success_rate', 0):.1%}")
        print(f"   Patterns Found: {len(self.current_session.patterns_observed)}")
        print(f"   Lessons Learned: {len(self.current_session.lessons_learned)}")
        
        self.current_session = None
        self.failures = []
    
    def _update_patterns_file(self) -> None:
        """Update the patterns markdown file with latest learnings."""
        all_patterns = []
        all_lessons = []
        
        if self.session_file.exists():
            with open(self.session_file, 'r') as f:
                for line in f:
                    session_data = json.loads(line)
                    all_patterns.extend(session_data.get('patterns_observed', []))
                    all_lessons.extend(session_data.get('lessons_learned', []))
        
        # Deduplicate and sort
        unique_patterns = list(set(all_patterns))
        unique_lessons = list(set(all_lessons))
        
        content = f"""# Auto-PSET Learned Patterns and Lessons

Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Common Failure Patterns

"""
        
        for i, pattern in enumerate(unique_patterns, 1):
            content += f"{i}. {pattern}\n"
        
        content += """
## Lessons Learned

"""
        
        for i, lesson in enumerate(unique_lessons, 1):
            content += f"{i}. {lesson}\n"
        
        content += """
## Recommendations

Based on the patterns above:

- Monitor JSON parsing errors and improve prompt engineering when they occur frequently
- Implement progressive retry delays for rate limiting issues  
- Add network connectivity checks for connection failures
- Consider provider fallbacks for API errors

"""
        
        with open(self.patterns_file, 'w') as f:
            f.write(content)
    
    def get_previous_learnings(self) -> str:
        """Get previous learnings to provide context for current session."""
        if self.patterns_file.exists():
            with open(self.patterns_file, 'r') as f:
                return f.read()
        return ""
    
    def get_latest_session_lessons(self) -> str:
        """Get the most recent comprehensive session lessons."""
        lessons_files = list(self.memory_dir.glob("session_*_lessons.md"))
        if not lessons_files:
            return ""
        
        # Sort by modification time and get the latest
        latest_file = max(lessons_files, key=lambda f: f.stat().st_mtime)
        with open(latest_file, 'r') as f:
            return f.read()
    
    def get_recent_session_lessons(self, count: int = 3) -> str:
        """Get recent session lessons for context."""
        lessons_files = list(self.memory_dir.glob("session_*_lessons.md"))
        if not lessons_files:
            return ""
        
        # Sort by modification time and get the most recent ones
        recent_files = sorted(lessons_files, key=lambda f: f.stat().st_mtime, reverse=True)[:count]
        
        combined_lessons = ""
        for file_path in recent_files:
            with open(file_path, 'r') as f:
                content = f.read()
                combined_lessons += f"\n\n{content}"
        
        return combined_lessons.strip()

class MemoryManager:
    """Main interface for memory management."""
    
    def __init__(self, output_dir: str = "outputs"):
        self.discussion_tracker = DiscussionTracker(output_dir)
        self.session_learning = SessionLearning(output_dir)
    
    def start_session(self, session_id: str = None) -> str:
        """Start a new session with memory tracking."""
        return self.session_learning.start_session(session_id)
    
    def end_session(self) -> None:
        """End current session and save learnings."""
        # Import here to avoid circular import
        from processors.summarizer import OutputSummarizer
        
        # Create summarizer for comprehensive lesson generation
        summarizer = OutputSummarizer()
        
        # Pass discussions directory to session learning
        discussions_dir = str(self.discussion_tracker.discussions_dir)
        self.session_learning.end_session(discussions_dir, summarizer)
    
    def start_pdf_discussion(self, pdf_path: str, pdf_analysis: Dict) -> None:
        """Start discussion tracking for a PDF."""
        self.discussion_tracker.start_discussion(pdf_path, pdf_analysis)
    
    def add_solver_thoughts(self, pdf_path: str, round_num: int, solver_provider: str, thoughts: str) -> None:
        """Add solver thoughts to discussion."""
        self.discussion_tracker.add_solver_thoughts(pdf_path, round_num, solver_provider, thoughts)
    
    def add_verifier_feedback(self, pdf_path: str, round_num: int, verifier_provider: str, feedback: str, corrections: str = None) -> None:
        """Add verifier feedback to discussion."""
        self.discussion_tracker.add_verifier_feedback(pdf_path, round_num, verifier_provider, feedback, corrections)
    
    def record_pdf_result(self, pdf_path: str, success: bool, error_message: str = None, stage: str = None) -> None:
        """Record PDF processing result."""
        self.session_learning.record_pdf_result(pdf_path, success, error_message, stage)
        self.discussion_tracker.add_final_outcome(pdf_path, success, error_message if not success else None)
    
    def get_discussion_context(self, pdf_path: str) -> str:
        """Get previous discussion context for a PDF."""
        return self.discussion_tracker.get_previous_context(pdf_path)
    
    def get_previous_learnings(self) -> str:
        """Get previous session learnings for context."""
        return self.session_learning.get_previous_learnings()
    
    def get_latest_session_lessons(self) -> str:
        """Get the most recent comprehensive session lessons."""
        return self.session_learning.get_latest_session_lessons()
    
    def get_recent_session_lessons(self, count: int = 3) -> str:
        """Get recent comprehensive session lessons for context."""
        return self.session_learning.get_recent_session_lessons(count)