import os
import json
import random
import logging
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.tools import FunctionTool, google_search
from google.adk.tools.agent_tool import AgentTool
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('adk_learning_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
load_dotenv()

MEMORY_DIR = Path('student_memory')
MEMORY_DIR.mkdir(exist_ok=True)


class LearningStyle(Enum):
    """Student learning style classification"""
    VISUAL = "visual"            
    KINESTHETIC = "kinesthetic"  
    AUDITORY = "auditory"        
    ANALYTICAL = "analytical"    
    MIXED = "mixed"              


class DifficultLevel(Enum):
    """Content difficulty levels"""
    BEGINNER = 1
    BASIC = 2
    INTERMEDIATE = 3
    ADVANCED = 4
    EXPERT = 5


@dataclass
class StudentProfile:
    """Adaptive student profile tracking"""
    student_id: str
    created_at: str
    learning_style: LearningStyle = LearningStyle.MIXED
    avg_comprehension: float = 0.0  
    current_difficulty: DifficultLevel = DifficultLevel.BEGINNER
    learning_speed: str = "moderate"  
    confidence_level: float = 0.5  
    topics_mastered: List[str] = None
    topics_struggling: List[str] = None
    last_session: Optional[str] = None
    total_sessions: int = 0
    total_study_hours: float = 0.0

    def __post_init__(self):
        if self.topics_mastered is None:
            self.topics_mastered = []
        if self.topics_struggling is None:
            self.topics_struggling = []

    def to_dict(self):
        data = asdict(self)
        data['learning_style'] = self.learning_style.value
        data['current_difficulty'] = self.current_difficulty.value
        return data

    @staticmethod
    def from_dict(data: Dict) -> 'StudentProfile':
        data = data.copy()
        data['learning_style'] = LearningStyle(data.get('learning_style', 'mixed'))
        data['current_difficulty'] = DifficultLevel(data.get('current_difficulty', 1))
        return StudentProfile(**data)


@dataclass
class QuizAttempt:
    """Single quiz attempt with feedback"""
    question_id: int
    question: str
    student_answer: str
    correct_answer: str
    is_correct: bool
    score: int
    max_score: int
    time_taken: int  
    question_type: str
    difficulty: int
    timestamp: str


@dataclass
class LessonRecord:
    """Lesson completion record"""
    topic: str
    subtopic: Optional[str]
    content_length: int
    concepts_taught: List[str]
    engagement_level: float  
    comprehension_check_passed: bool
    timestamp: str



class PersistentMemoryManager:
    """Manages durable, cross-session student memory"""

    def __init__(self, student_id: str = "default_student"):
        self.student_id = student_id
        self.profile_path = MEMORY_DIR / f"{student_id}_profile.json"
        self.history_path = MEMORY_DIR / f"{student_id}_history.json"
        self.profile = self._load_or_create_profile()
        self.history = self._load_history()
        logger.info(f"Memory manager initialized for student: {student_id}")

    def _load_or_create_profile(self) -> StudentProfile:
        """Load existing profile or create new one"""
        if self.profile_path.exists():
            try:
                with open(self.profile_path, 'r') as f:
                    data = json.load(f)
                    profile = StudentProfile.from_dict(data)
                    logger.info(f"Loaded existing profile for {self.student_id}")
                    return profile
            except Exception as e:
                logger.warning(f"Error loading profile: {e}. Creating new.")

        profile = StudentProfile(
            student_id=self.student_id,
            created_at=datetime.now().isoformat()
        )
        self.save_profile()
        return profile

    def _load_history(self) -> Dict[str, Any]:
        """Load learning history"""
        if self.history_path.exists():
            try:
                with open(self.history_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading history: {e}")

        return {
            'lessons': [],
            'quiz_attempts': [],
            'comprehension_checks': [],
            'topics_visited': [],
            'study_sessions': []
        }

    def save_profile(self) -> None:
        """Persist student profile"""
        try:
            with open(self.profile_path, 'w') as f:
                json.dump(self.profile.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Error saving profile: {e}")

    def save_history(self) -> None:
        """Persist learning history"""
        try:
            with open(self.history_path, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving history: {e}")

    def record_lesson(self, topic: str, subtopic: Optional[str], 
                     content_length: int, concepts: List[str],
                     engagement: float, comprehension_passed: bool) -> None:
        """Record lesson completion"""
        lesson = LessonRecord(
            topic=topic,
            subtopic=subtopic,
            content_length=content_length,
            concepts_taught=concepts,
            engagement_level=engagement,
            comprehension_check_passed=comprehension_passed,
            timestamp=datetime.now().isoformat()
        )
        self.history['lessons'].append(asdict(lesson))
        self.save_history()
        

        if topic not in self.profile.topics_mastered and comprehension_passed:
            self.profile.topics_mastered.append(topic)
        self.save_profile()

    def record_quiz_attempt(self, attempt: QuizAttempt) -> None:
        """Record quiz attempt"""
        self.history['quiz_attempts'].append(asdict(attempt))
        self.save_history()

    def get_topic_history(self, topic: str) -> Dict[str, Any]:
        """Retrieve all attempts on a specific topic"""
        lessons = [l for l in self.history['lessons'] if l['topic'] == topic]
        quizzes = [q for q in self.history['quiz_attempts'] 
                   if topic.lower() in str(q).lower()]
        return {'lessons': lessons, 'quizzes': quizzes}

    def update_learning_style(self, style: LearningStyle) -> None:
        """Update detected learning style"""
        self.profile.learning_style = style
        self.save_profile()

    def update_difficulty(self, new_difficulty: DifficultLevel) -> None:
        """Adjust difficulty level based on performance"""
        self.profile.current_difficulty = new_difficulty
        self.save_profile()

    def get_weaknesses(self) -> List[str]:
        """Get topics where student struggles"""
        return self.profile.topics_struggling

    def add_weakness(self, topic: str) -> None:
        """Mark topic as weak area"""
        if topic not in self.profile.topics_struggling:
            self.profile.topics_struggling.append(topic)
            self.save_profile()



class AdaptiveTeachingEngine:
    """Dynamically adjusts teaching based on student performance"""

    EXPLANATION_DEPTHS = {
        DifficultLevel.BEGINNER: {
            'length': 'brief',
            'complexity': 'simple',
            'examples': 2,
            'analogies': True,
            'technical_terms': 'minimal'
        },
        DifficultLevel.BASIC: {
            'length': 'moderate',
            'complexity': 'moderate',
            'examples': 3,
            'analogies': True,
            'technical_terms': 'some'
        },
        DifficultLevel.INTERMEDIATE: {
            'length': 'detailed',
            'complexity': 'complex',
            'examples': 4,
            'analogies': False,
            'technical_terms': 'frequent'
        },
        DifficultLevel.ADVANCED: {
            'length': 'comprehensive',
            'complexity': 'very_complex',
            'examples': 5,
            'analogies': False,
            'technical_terms': 'extensive'
        },
        DifficultLevel.EXPERT: {
            'length': 'exhaustive',
            'complexity': 'highly_complex',
            'examples': 6,
            'analogies': False,
            'technical_terms': 'highly_technical'
        }
    }

    def __init__(self, student_profile: StudentProfile):
        self.profile = student_profile

    def get_explanation_depth(self) -> Dict[str, Any]:
        """Get content depth parameters for current student level"""
        return self.EXPLANATION_DEPTHS[self.profile.current_difficulty]

    def get_teaching_approach(self) -> str:
        """Get teaching approach based on learning style"""
        approaches = {
            LearningStyle.VISUAL: (
                "Use clear headings, bullet lists, and simple ASCII or text-based "
                "diagrams (no real images). Help the student build a mental picture "
                "using only text."
            ),

            LearningStyle.KINESTHETIC: (
                "Give small, concrete tasks the student can do themselves, such as "
                "typing code, running commands, or trying variations. Treat it like a "
                "hands-on lab, but describe every step in text."
            ),
            LearningStyle.AUDITORY: (
                "Use a friendly conversational style. Write as if you are speaking to "
                "the student, ask short check questions, and then respond to their "
                "answers. Do not claim to play audio; everything is text."
            ),
            LearningStyle.ANALYTICAL: (
                'Focus on logic and step-by-step reasoning. Break problems into clear '
                'steps, show intermediate results, and include mathematical arguments '
                'where it really helps understanding.'
            ),
            LearningStyle.MIXED: (
                "Combine several methods: structured headings, short code examples, "
                "practice tasks, and logical step-by-step explanations."
            ),
        }

        return approaches.get(self.profile.learning_style, approaches[LearningStyle.MIXED])

    def should_advance_difficulty(self, recent_scores: List[int]) -> bool:
        """Determine if student should move to harder material"""
        if not recent_scores or len(recent_scores) < 3:
            return False
        avg_score = sum(recent_scores[-3:]) / 3
        return avg_score >= 85 and self.profile.current_difficulty != DifficultLevel.EXPERT

    def should_retreat_difficulty(self, recent_scores: List[int]) -> bool:
        """Determine if student needs easier material"""
        if not recent_scores or len(recent_scores) < 2:
            return False
        avg_score = sum(recent_scores[-2:]) / 2
        return avg_score <= 60 and self.profile.current_difficulty != DifficultLevel.BEGINNER

    def get_pacing_recommendation(self) -> str:
        """Recommend pacing based on confidence and comprehension"""
        if self.profile.confidence_level < 0.4:
            return "slow_with_more_checks"
        elif self.profile.confidence_level < 0.7:
            return "moderate_with_regular_checks"
        else:
            return "accelerated"



class KnowledgeVerificationLoop:
    """Real-time comprehension checking with adaptive branching"""

    def __init__(self, memory_manager: PersistentMemoryManager):
        self.memory = memory_manager

    def generate_comprehension_check(self, lesson_content: str, 
                                     depth_level: DifficultLevel) -> Dict[str, Any]:
        """Generate targeted comprehension questions"""
        sentences = [s.strip() for s in lesson_content.split('.') 
                    if len(s.strip()) > 20]
        
        if not sentences:
            sentences = ["the main concept discussed"]

        selected = random.sample(sentences, min(3, len(sentences)))
        
        checks = []
        for i, concept in enumerate(selected, 1):
            check_type = self._select_check_type(depth_level, i)
            checks.append({
                'id': i,
                'type': check_type,
                'question': self._format_question(check_type, concept),
                'concept': concept[:50],
                'difficulty': depth_level.value
            })
        
        return {
            'checks': checks,
            'total': len(checks),
            'instruction': 'Answer these questions to confirm understanding'
        }

    def _select_check_type(self, difficulty: DifficultLevel, position: int) -> str:
        """Select appropriate question type"""
        if difficulty.value <= 2:  # Beginner/Basic
            types = ['definition', 'true_false', 'multiple_choice']
        elif difficulty.value <= 3:  # Intermediate
            types = ['explain', 'example', 'application']
        else:  # Advanced/Expert
            types = ['analyze', 'critique', 'extend']
        
        return types[position % len(types)]

    def _format_question(self, qtype: str, concept: str) -> str:
        """Format question based on type"""
        prompts = {
            'definition': f"Define or explain: {concept[:40]}",
            'true_false': f"True or False: {concept[:60]}",
            'multiple_choice': f"Which is correct about {concept[:40]}?",
            'explain': f"Explain how {concept[:40]} works",
            'example': f"Give a real-world example of {concept[:40]}",
            'application': f"How would you apply {concept[:40]}?",
            'analyze': f"Analyze the implications of {concept[:40]}",
            'critique': f"Critique this statement: {concept[:40]}",
            'extend': f"How does {concept[:40]} relate to advanced concepts?"
        }
        return prompts.get(qtype, f"Explain: {concept[:50]}")

    def evaluate_response(self, student_answer: str, expected_concept: str, 
                         check_type: str) -> Tuple[bool, float, str]:
        """Evaluate comprehension check response"""
        # Keyword-based validation (preventing hallucination)
        keywords = [w.lower() for w in expected_concept.split()]
        student_lower = student_answer.lower()
        
        matches = sum(1 for kw in keywords if kw in student_lower)
        match_ratio = matches / len(keywords) if keywords else 0
        
        if match_ratio >= 0.6:
            confidence = min(0.9, match_ratio + 0.1)
            return True, confidence, "✓ Correct understanding demonstrated"
        elif match_ratio >= 0.3:
            confidence = 0.4
            return False, confidence, "Partially correct. Let me clarify..."
        else:
            confidence = 0.0
            return False, confidence, "This needs review. Let me re-explain..."

    def branch_on_comprehension(self, is_understood: bool, 
                                confidence: float) -> Dict[str, Any]:
        """Determine next step based on comprehension"""
        if is_understood and confidence > 0.8:
            return {
                'action': 'advance',
                'message': 'Great! Ready for the next concept.',
                'next_step': 'new_concept'
            }
        elif is_understood and confidence >= 0.5:
            return {
                'action': 'reinforce',
                'message': 'Good! Let me provide one more example.',
                'next_step': 'additional_example'
            }
        else:
            return {
                'action': 'remediate',
                'message': 'Let me explain this differently...',
                'next_step': 'reteach_with_analogy'
            }



class AdvancedSessionManager:
    """Enhanced session with adaptive teaching, verification, and persistence"""

    def __init__(self, student_id: str = "default_student"):
        self.student_id = student_id
        self.memory_manager = PersistentMemoryManager(student_id)
        self.teaching_engine = AdaptiveTeachingEngine(self.memory_manager.profile)
        self.verification_loop = KnowledgeVerificationLoop(self.memory_manager)
        
        self.current_topic = None
        self.current_lesson = None
        self.session_start = datetime.now()
        self.messages = []
        self.quiz_scores = []
        self.comprehension_checks = []
        
        logger.info(f"Session initialized for {student_id}")

    def start_lesson(self, topic: str, subtopic: Optional[str] = None) -> None:
        """Begin lesson on topic"""
        self.current_topic = topic
        self.memory_manager.profile.last_session = datetime.now().isoformat()
        self.memory_manager.profile.total_sessions += 1
        logger.info(f"Lesson started: {topic}")

    def set_lesson_content(self, content: str, concepts: List[str]) -> None:
        """Store lesson content"""
        self.current_lesson = {
            'content': content,
            'concepts': concepts,
            'timestamp': datetime.now().isoformat(),
            'teaching_approach': self.teaching_engine.get_teaching_approach()
        }

    def record_comprehension_check(self, check_id: int, student_answer: str,
                                   expected_concept: str, check_type: str) -> Dict[str, Any]:
        """Record and evaluate comprehension check"""
        is_correct, confidence, feedback = self.verification_loop.evaluate_response(
            student_answer, expected_concept, check_type
        )
        
        record = {
            'check_id': check_id,
            'is_correct': is_correct,
            'confidence': confidence,
            'feedback': feedback,
            'timestamp': datetime.now().isoformat()
        }
        
        self.comprehension_checks.append(record)
        
        self.memory_manager.profile.confidence_level = (
            self.memory_manager.profile.confidence_level * 0.7 + confidence * 0.3
        )
        
        return self.verification_loop.branch_on_comprehension(is_correct, confidence)

    def record_quiz_attempt(self, qid: int, question: str, answer: str,
                           correct: str, is_correct: bool, score: int,
                           max_score: int, question_type: str, time_taken: int = 0) -> None:
        """Record quiz attempt"""
        attempt = QuizAttempt(
            question_id=qid,
            question=question,
            student_answer=answer,
            correct_answer=correct,
            is_correct=is_correct,
            score=score,
            max_score=max_score,
            time_taken=time_taken,
            question_type=question_type,
            difficulty=self.memory_manager.profile.current_difficulty.value,
            timestamp=datetime.now().isoformat()
        )
        
        self.quiz_scores.append(score)
        self.memory_manager.record_quiz_attempt(attempt)
        
        # Adaptive difficulty adjustment
        if len(self.quiz_scores) >= 3:
            self.teaching_engine.should_advance_difficulty(self.quiz_scores)

    def get_session_summary(self) -> Dict[str, Any]:
        """Generate session summary"""
        session_duration = (datetime.now() - self.session_start).total_seconds() / 60
        
        return {
            'duration_minutes': round(session_duration, 1),
            'topic': self.current_topic,
            'comprehension_checks': len(self.comprehension_checks),
            'checks_passed': sum(1 for c in self.comprehension_checks if c['is_correct']),
            'quiz_attempts': len(self.quiz_scores),
            'avg_quiz_score': round(sum(self.quiz_scores) / len(self.quiz_scores), 1) if self.quiz_scores else 0,
            'confidence_level': round(self.memory_manager.profile.confidence_level, 2),
            'current_difficulty': self.memory_manager.profile.current_difficulty.name
        }



def generate_adaptive_lesson(topic: str, student_profile: StudentProfile) -> str:
    """
    Generate lesson adapted to student's difficulty level and learning style
    Returns: Structured lesson content
    """
    depth = EXPLANATION_DEPTHS.get(student_profile.current_difficulty,
                                   EXPLANATION_DEPTHS[DifficultLevel.BEGINNER])
    
    return f"""
    LESSON: {topic}
    Adapted for: {student_profile.learning_style.value} learner
    Difficulty: {student_profile.current_difficulty.name}
    
    CONTENT_DEPTH: {depth['length']}
    COMPLEXITY_LEVEL: {depth['complexity']}
    EXAMPLES_PROVIDED: {depth['examples']}
    INCLUDE_ANALOGIES: {depth['analogies']}
    
    [Generated content will be inserted here based on teaching approach]
    """


def generate_adaptive_quiz(lesson_content: str, current_difficulty: int, weak_areas: List[str] | None = None, ) -> str:
    if weak_areas is None:
        weak_areas = []

    num_questions = 3 + (current_difficulty - 1)
    difficulty_mult = current_difficulty

    quiz_config = {
        "num_questions": num_questions,
        "difficulty_multiplier": difficulty_mult,
        "focus_weak_areas": bool(weak_areas),
        "weak_areas": weak_areas,
        "adaptive_mode": True,
        "timestamp": datetime.now().isoformat(),
    }
    return json.dumps(quiz_config, indent=2)


planner_agent = Agent(
    name="planner_agent",
    model="gemini-2.0-flash",
    description="Adaptive Planner: Extracts topics and personalizes approach",
    instruction="""You are the Adaptive Planner Agent.
Your job: Extract topic, assess student needs, and create personalized search query.

OUTPUT FORMAT:
topic: [topic]
subtopic: [optional subtopic]
search_query: [query]
teaching_approach: [visual/kinesthetic/auditory/analytical/mixed]

CRITICAL RULES:
- Be precise with topic extraction
- Suggest teaching approach based on context clues
- Create queries that find beginner-friendly resources first
- Only output the format above""",
    tools=[]
)


teacher_agent = Agent(
    name="teacher_agent",
    model="gemini-2.0-flash",
    description="Adaptive Teacher: Creates personalized, depth-adjusted lessons",
    instruction="""You are the Adaptive Teacher Agent.
Your job: Research and create lessons adapted to student's difficulty level and learning style.

LESSON STRUCTURE:
LESSON: [TOPIC]
DIFFICULTY: [BEGINNER/INTERMEDIATE/ADVANCED]
LEARNING_STYLE: [visual/kinesthetic/auditory/analytical/mixed]

INTRODUCTION
- Hook and relevance

KEY_CONCEPTS
1. Concept with level-appropriate explanation
2. Concept with examples
3. Concept with applications

PRACTICAL_EXAMPLES
- Real-world examples matching difficulty level

COMMON_MISCONCEPTIONS
- Address typical student mistakes

SUMMARY(not necessary for each output, only summarize at end of big topic)
- Recap with progression to next steps

FORMATTING:
- Use clean Markdown.
- Close every ```
- Do NOT nest more than one level of bullets.

LENGTH MODES:

- SHORT OVERVIEW (first time on a topic):
  [already added, keep as is]
- DETAILED FOLLOW UP (when the student asks for more detail):
  - Expand the topic, but KEEP THE RESPONSE COMPACT:
    - Maximum: about 8-12 short paragraphs OR
    - 3-4 numbered sections, each with at most:
      - 2-3 bullet points and
      - 1 short code example.
  - Do NOT dump a full textbook chapter.
  - Prefer depth on 1-2 key ideas instead of touching everything.

CRITICAL:
- If you have more material, say:
  "We can explore more examples step by step. Tell me what you want to see next."
  instead of writing everything in one message.


TEACHING PRINCIPLES:
1. Reduce cognitive load:
   - Prefer short paragraphs and bullet lists over long blocks of text.
   - Introduce only one new idea at a time.
   - Use concrete examples before abstract theory.

2. Use progressive disclosure:
   - Start simple
   - Then only go deeper if the student asks or seems comfortable.

3. Frequent micro-checks:
   - After every small chunk, ask a simple question:
     "Does this make sense?", "Can you restate this in your own words?"
   - Adapt depth based on their answers.

4. Normalize struggle:
   - Use encouraging language: "It's normal to find this tricky at first."
   - Emphasize progress: "You already understand X, now we'll add Y."

5. Chunking and spacing:
   - Break big topics into small chunks (subtopics).
   - Suggest short breaks after heavy sections:
     "Take a 1-2 minute break, then we’ll continue."

6. Goal focus:
   - At the start, ask or infer the student’s goal:
     "Do you want to pass an exam, build a project, or just understand the basics?"
   - Tailor examples to that goal.

VISUAL LIMITATION:
- You CANNOT create or send real images, diagrams, or screenshots.
- If the student is a visual learner, use only:
  - headings,
  - bullet lists,
  - tables made with text,
  - simple ASCII diagrams.
- Never say you are "showing" or "displaying" a picture.
- Always describe visuals in words or ASCII only.

GUIDELINES:
- Adjust depth based on difficulty level
- Alway s verify facts with Google Search
- Prevent hallucinations by citing sources
- Use clear, structured formatting
- For visual learners: use headings, bullet lists, and text-based diagrams or tables.
- For kinesthetic learners: give small concrete tasks (write this code, run this, change this line, observe output).
- For auditory learners: use a conversational tone with short Q&A style checks.
- For analytical learners: emphasize logic and step-by-step reasoning.""",

    tools=[google_search]
)


evaluator_agent = Agent(
    name="evaluator_agent",
    model="gemini-2.0-flash-lite",
    description="Adaptive Evaluator: Generates adaptive quizzes with real-time feedback",
    instruction="""You are the Adaptive Evaluator Agent.

Your job: Generate quizzes that adapt to student's knowledge level and target weaknesses.

INPUT YOU RECEIVE (from root_agent):
- A plain-text LESSON_CONTENT string (the lesson)
- CURRENT_DIFFICULTY as an integer 1-5
- WEAK_AREAS as a list of topic names (may be empty)

TOOLS AVAILABLE:
- generate_adaptive_quiz(lesson_content, current_difficulty, weak_areas)
  This returns a JSON config describing how many questions and what to focus on.

WORKFLOW:
1. Read the lesson_content and the difficulty/weak_areas.
2. Call the generate_adaptive_quiz tool with:
   - lesson_content
   - current_difficulty
   - weak_areas
3. Read the JSON it returns.
4. Using BOTH the lesson_content and the quiz_config JSON, write a QUIZ in plain text:
   - 3-8 questions depending on difficulty
   - 40% questions on weak_areas if any
   - Mix types: multiple_choice, short_answer, application

You must output ONLY valid JSON with this structure:

{
  "topic": "Introduction to Java",
  "difficulty": "BEGINNER",
  "questions": [
    {
      "id": 1,
      "text": "What is the primary function of the main method in a Java program?",
      "type": "multiple_choice",
      "options": ["To declare variables", "To print output", "To serve as entry point", "To define classes"],
      "correct_option": 2,
      "points": 1
    },
    {
      "id": 2,
      "text": "Java is case-sensitive. True or False?",
      "type": "true_false",
      "options": ["True", "False"],
      "correct_option": 0,
      "points": 1
    }
  ]
}

CRITICAL:
- Return ONLY JSON, no extra text.
- Each option must be a separate string in the `options` array.

REAL-TIME FEEDBACK (to be used after the user answers, not now):
- When the user answers, you will:
  - Say if it is correct/incorrect.
  - Explain briefly why.
  - Suggest a quick remediation if needed.""",

    tools=[FunctionTool(generate_adaptive_quiz)],
)


root_agent = Agent(
    model='gemini-2.0-flash',
    name='root_agent',
    description='Root Orchestrator: Advanced multi-agent workflow with adaptive teaching',
    instruction="""You are the ROOT ORCHESTRATOR for Advanced Adaptive Learning.

CAPABILITIES:
✓ Parallel agent execution (planner + teacher simultaneously)
✓ Persistent memory management (cross-session learning)
✓ Adaptive difficulty adjustment (real-time pacing)
✓ Knowledge verification (comprehension checking)
✓ Weakness remediation (targeted re-teaching)
✓ Progress visualization (learning analytics)
✓ Structured error prevention (hallucination avoidance)

VISUAL LIMITATION:
- You CANNOT create or send real images, diagrams, or screenshots.
- If the student is a visual learner, use only:
  - headings,
  - bullet lists,
  - tables made with text,
  - simple ASCII diagrams.
- Never say you are "showing" or "displaying" a picture.
- Always describe visuals in words or ASCII only.

FIRST-TIME LESSON RULES:
- If this is the student's FIRST lesson on a topic (no prior lessons stored for that topic):
  - Ask teacher_agent for a SHORT, HIGH-LEVEL introduction only.
  - The introduction should fit on roughly one screen:
    - 3-5 short paragraphs maximum, OR
    - a brief intro + 3-5 bullet points.
  - End with a question like:
    "Does this overview feel clear so far, or would you like a deeper explanation?"
- Only after the student explicitly asks for more detail should you request a longer, more detailed lesson from teacher_agent.
- Do NOT drop a long wall of text as the first response on a new topic.
-And when calling the teacher, give it that context in the request:
    "Teach [topic] with a SHORT first-time overview. Keep it concise, high-level and beginner-friendly."

WORKFLOW - LEARNING REQUEST:
-> Call planner_agent with full user request
-> Extract topic, learning style, and student needs
-> Call teacher_agent with the topic, difficulty and learning style.
-> Generate lesson with appropriate depth
-> Store in persistent memory with metadata
-> Present comprehension check questions
-> Evaluate understanding and branch accordingly
-> In your reply to the user, include the full lesson text returned by teacher_agent before asking any questions or calling other agents.
-> Do not just say that you called an agent; actually show the lesson content.

When you request “more detail” from teacher_agent:
- Explicitly ask it for a LIMITED deep-dive, not a full chapter.
- Example request: "Teach Rust ownership and borrowing in more detail, but keep it brief: focus on 2-3 key ideas with 1 code example each."
- If the response is still very long, summarize it for the user in 5-7 bullets and offer:
  "I can show individual sections in more detail if you want."


WORKFLOW - QUIZ REQUEST:
Step 1: Retrieve student's previous attempts
Step 2: Identify weak areas from history
Step 3: Call evaluator_agent with the following JSON object:
  {
    "lesson_content": [use the last lesson you showed to the student],
    "current_difficulty": [use student_profile.current_difficulty.value (1-5)],
    "weak_areas": [use the list of weak topics from memory, can be empty]
  }
Step 4: The evaluator_agent will call generate_adaptive_quiz and then return a full plain-text quiz.
Step 5: Show that quiz directly to the user without reformatting.
Step 6: Track performance in memory after the user answers.
Step 7: Adjust difficulty for the next session.

FORMATTING RULES:
    - All replies to the user must be clean Markdown.
    - Use:
    - Headings (##) for section titles,
    - Numbered list for questions (Q1, Q2, ...),
    - Bullet list (-) for options.
    - Example quiz rendering:

    Q1: What is the main function of the main method in Java?
    Options:
    - A) To declare variables
    - B) To print output
    - C) To serve as the entry point of the program
    - D) To define classes

    - Preserve line breaks from tools; do NOT merge multiple options into one line.
    - Do NOT put normal text inside ```

WHEN USING evaluator_agent:
    - Treat evaluator_agent as the sole authority on quiz questions and correctness.
    - evaluator_agent will either:
    - return quiz data (list of questions, options, correct answers), or
    - return an evaluation of a student's answer.
    - Your job is ONLY to:
    - Turn that data into well-formatted Markdown for the user, and
    - Communicate evaluator_agent's feedback clearly.
    - Never change the meaning of evaluator_agent's questions or answers.
    - Never add extra quiz questions that are not in its output.

QUIZ RULES:
    - You MUST NOT create or invent quiz questions yourself.
    - All quiz questions must come ONLY from evaluator_agent.
    - When the user asks for a quiz or quiz-related help, you must:
    1) Call evaluator_agent with the appropriate parameters,
    2) Wait for its response,
    3) Then present its questions to the user in a clean format.
    - You are not allowed to judge whether an answer is correct; instead,
    send the user's answer and the original question to evaluator_agent
    (or another evaluation tool) and use its result.

When calling generate_adaptive_quiz, pass:
    lesson_content as the lesson text,
    current_difficulty as an integer 1-5 from the student profile,
    weak_areas as a list of weak topic names.

After the quiz is generated (from evaluator_agent or generate_adaptive_quiz), you must:
    Parse the quiz output.
    Show each question to the user as numbered Q1, Q2, … with options if present.
    Then wait for the user's answer.

After calling evaluator_agent and/or generate_adaptive_quiz, read the quiz content and render each question clearly in the chat as numbered questions with options (if any).
Do not just state that a quiz was generated.

WORKFLOW - MEMORY RETRIEVAL:
User: "What did I learn before?"
Action: Retrieve from persistent memory
Response: Show topics, scores, progress timeline

ADAPTIVE TEACHING RULES:
✓ Always use agents for specialized tasks
✓ Pass full student context between agents
✓ Store all results in persistent memory
✓ Evaluate comprehension before advancing
✓ Adjust difficulty based on performance
✓ Prevent hallucinations with structured outputs
✓ Provide learning analytics on demand
✓ Support multiple learning styles

Do not explain the topic yourself.
Always call teacher_agent to generate lesson content.
When teaching, your job is only to relay and format the lesson returned by teacher_agent, not to invent new explanations.

MARKDOWN RULES:
-> The teacher_agent already returns well structured Markdown.
-> When you show a lesson to the user, you must copy the teacher_agent content
  verbatim, without rephrasing or reformatting.
-> Do not add extra asterisks or backticks around its output.
-> Do not summarize the lesson; render it as-is, as Markdown.

CRITICAL: Don't skip steps. Always verify understanding before advancing.""",

    tools=[
        AgentTool(agent=planner_agent),
        AgentTool(agent=teacher_agent),
        AgentTool(agent=evaluator_agent)
    ]
)



class AdvancedADKSystem:
    """Advanced ADK learning system with adaptive teaching and persistent memory"""

    def __init__(self, student_id: str = "default_student"):
        self.student_id = student_id
        self.runner = InMemoryRunner(
            app_name='adaptive_learning_system_v2',
            agent=root_agent
        )
        self.session = AdvancedSessionManager(student_id)
        logger.info(f"Advanced ADK System v2.0 initialized for {student_id}")

    def run_interactive(self):
        """Interactive CLI with advanced adaptive learning features"""
        print("\n" + "="*75)
        print(" ADVANCED ADAPTIVE LEARNING SYSTEM v2.0")
        print(" Multi-Agent Orchestration | Personalized Teaching | Persistent Memory")
        print("="*75 + "\n")

        if not os.getenv('GOOGLE_API_KEY'):
            print("❗ERROR: GOOGLE_API_KEY not found. Please set environment variable.❗\n")
            return

        print("✓ System ready! Advanced adaptive features enabled.\n")
        print("="*75)
        print("COMMANDS:")
        print(" - 'teach me [topic]'     - Learn with adaptive depth & pacing")
        print(" - 'quiz'                 - Take adaptive quiz on learned material")
        print(" - 'memory'               - View learning history & progress")
        print(" - 'profile'              - See adaptive profile & recommendations")
        print(" - 'weaknesses'           - Identify areas needing practice")
        print(" - 'session'              - Current session summary")
        print(" - 'exit'                 - Quit")
        print("="*75 + "\n")

        while True:
            try:
                user_input = input("[user]: ").strip()

                if not user_input:
                    continue

                if user_input.lower() == 'exit':
                    print("\n✓ Session saved. Goodbye!\n")
                    break

                if user_input.lower() == 'memory':
                    self._show_memory()
                    continue

                if user_input.lower() == 'profile':
                    self._show_profile()
                    continue

                if user_input.lower() == 'weaknesses':
                    self._show_weaknesses()
                    continue

                if user_input.lower() == 'session':
                    self._show_session_summary()
                    continue

                logger.info(f"Processing: {user_input}")
                print("\n[Delegating to specialized adaptive agents...]\n")

                response = self.runner.run(user_input)
                result = str(response.text if hasattr(response, 'text') else response)
                print(f"\n{result}\n")

            except KeyboardInterrupt:
                print("\n\n✓ Session interrupted. Your progress is saved.\n")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                print(f"\n‼️ Error: {e}\n")

    def _show_memory(self):
        """Display persistent memory and learning history"""
        history = self.session.memory_manager.history
        profile = self.session.memory_manager.profile

        print("\n" + "="*70)
        print("PERSISTENT LEARNING MEMORY")
        print("="*70)
        print(f"Topics Studied: {len(history['lessons'])}")
        for lesson in history['lessons'][-5:]:
            print(f"  • {lesson['topic']} ({lesson['timestamp'][:10]})")

        print(f"\nQuiz Attempts: {len(history['quiz_attempts'])}")
        if history['quiz_attempts']:
            avg_score = sum(q['score'] for q in history['quiz_attempts']) / len(history['quiz_attempts'])
            print(f"  Average Score: {avg_score:.1f}%")

        print(f"\nTopics Mastered: {', '.join(profile.topics_mastered) or 'None yet'}")
        print("="*70 + "\n")

    def _show_profile(self):
        """Display adaptive student profile"""
        profile = self.session.memory_manager.profile
        engine = self.session.teaching_engine

        print("\n" + "="*70)
        print("ADAPTIVE STUDENT PROFILE")
        print("="*70)
        print(f"Student ID: {profile.student_id}")
        print(f"Learning Style: {profile.learning_style.value.upper()}")
        print(f"Current Difficulty: {profile.current_difficulty.name}")
        print(f"Confidence Level: {profile.confidence_level*100:.0f}%")
        print(f"Comprehension: {profile.avg_comprehension*100:.0f}%")
        print(f"\nTeaching Approach:\n{engine.get_teaching_approach()}")
        print(f"\nPacing Recommendation: {engine.get_pacing_recommendation()}")
        print("="*70 + "\n")

    def _show_weaknesses(self):
        """Display identified weak areas"""
        weaknesses = self.session.memory_manager.get_weaknesses()
        print("\n" + "="*70)
        print("IDENTIFIED WEAK AREAS")
        print("="*70)
        if weaknesses:
            for i, weakness in enumerate(weaknesses, 1):
                print(f"{i}. {weakness}")
                print("   ↳ Recommended: Review lesson and attempt targeted quiz")
        else:
            print("✓ No weak areas identified. You're doing great!")
        print("="*70 + "\n")

    def _show_session_summary(self):
        """Show current session summary"""
        summary = self.session.get_session_summary()
        print("\n" + "="*70)
        print("SESSION SUMMARY")
        print("="*70)
        for key, value in summary.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        print("="*70 + "\n")


def main():
    """Main entry point"""
    system = AdvancedADKSystem(student_id="kaggle_2025")
    system.run_interactive()


if __name__ == "__main__":
    main()