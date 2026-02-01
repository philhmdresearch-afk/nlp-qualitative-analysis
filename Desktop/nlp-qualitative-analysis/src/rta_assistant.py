"""
Reflexive Thematic Analysis (RTA) Assistant
Guides researchers through the six phases of RTA following Braun & Clarke methodology
Human-in-the-loop approach: AI augments, never replaces, analyst judgment
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set, Any
from datetime import datetime
from collections import defaultdict
import json
import warnings

warnings.filterwarnings('ignore')


class RTAAssistant:
    """
    Reflexive Thematic Analysis Assistant following Braun & Clarke (2006, 2019, 2021).
    
    Core Principles:
    - Human judgment drives all decisions
    - AI provides provocations, not ground truth
    - Reflexive memos are primary
    - Audit trail tracks all AI assistance
    - No inter-rater reliability (IRR) - this is RTA, not coding reliability TA
    
    Six Phases:
    1. Familiarisation
    2. Generating initial codes
    3. Searching for themes
    4. Reviewing themes
    5. Defining and naming themes
    6. Producing the report
    """
    
    def __init__(
        self,
        project_name: str,
        researcher_name: str,
        epistemology: str = "contextualist",
        ontology: str = "critical realist"
    ):
        """
        Initialize RTA Assistant.
        
        Parameters:
        -----------
        project_name : str
            Name of the research project
        researcher_name : str
            Name of the primary analyst
        epistemology : str
            Epistemological stance (e.g., "contextualist", "constructionist")
        ontology : str
            Ontological stance (e.g., "critical realist", "relativist")
        """
        self.project_name = project_name
        self.researcher_name = researcher_name
        self.epistemology = epistemology
        self.ontology = ontology
        
        # RTA state
        self.current_phase = 1
        self.reflexive_memos = []
        self.audit_trail = []
        self.codes = {}
        self.themes = {}
        self.positionality_statement = None
        
        # Track AI assistance
        self.ai_assistance_log = []
        
        # Initialize project
        self._log_audit_entry(
            "project_initialized",
            f"RTA project '{project_name}' initialized by {researcher_name}",
            {"epistemology": epistemology, "ontology": ontology}
        )
    
    def _log_audit_entry(
        self,
        action: str,
        description: str,
        details: Optional[Dict] = None
    ):
        """Log entry to audit trail."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'phase': self.current_phase,
            'action': action,
            'description': description,
            'researcher': self.researcher_name,
            'details': details or {}
        }
        self.audit_trail.append(entry)
    
    def _log_ai_assistance(
        self,
        task: str,
        ai_output: Any,
        human_decision: str,
        rationale: str
    ):
        """Log AI assistance and human decision."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'phase': self.current_phase,
            'task': task,
            'ai_provided': str(ai_output)[:200],  # Truncate for storage
            'human_decision': human_decision,
            'rationale': rationale
        }
        self.ai_assistance_log.append(entry)
    
    # ========================================================================
    # PHASE 1: FAMILIARISATION
    # ========================================================================
    
    def phase1_familiarisation(
        self,
        texts: List[str],
        metadata: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Phase 1: Familiarisation with the data.
        
        AI Role: Provide navigation aids and summaries as PROMPTS, not replacements.
        Human Role: Read all data, write reflexive memos, note initial hunches.
        
        Parameters:
        -----------
        texts : List[str]
            Raw data (transcripts, responses, etc.)
        metadata : pd.DataFrame, optional
            Contextual information about data
            
        Returns:
        --------
        Dict with familiarisation aids and guidance
        """
        self.current_phase = 1
        self._log_audit_entry(
            "phase1_started",
            "Familiarisation phase initiated",
            {"n_texts": len(texts)}
        )
        
        # Compute basic statistics
        stats = {
            'n_documents': len(texts),
            'total_words': sum(len(text.split()) for text in texts),
            'avg_length': np.mean([len(text.split()) for text in texts]),
            'min_length': min(len(text.split()) for text in texts),
            'max_length': max(len(text.split()) for text in texts)
        }
        
        # AI-generated navigation aids (NOT replacements for reading)
        navigation_aids = self._generate_navigation_aids(texts)
        
        # Guidance for researcher
        guidance = {
            'primary_task': 'READ ALL DATA YOURSELF - AI summaries are navigation aids only',
            'instructions': [
                '1. Read each transcript/response in full',
                '2. Write reflexive memos as you read',
                '3. Note initial patterns, tensions, surprises',
                '4. Record your emotional/intellectual responses',
                '5. Consider your positionality and how it shapes reading'
            ],
            'memo_prompts': [
                'What strikes me as interesting or surprising?',
                'What patterns am I noticing?',
                'What tensions or contradictions appear?',
                'How is my background shaping what I notice?',
                'What am I NOT seeing? What voices are absent?'
            ],
            'warning': '⚠️ Do NOT rely on AI summaries as substitutes for reading. Your engagement with the data is essential.'
        }
        
        return {
            'phase': 1,
            'phase_name': 'Familiarisation',
            'statistics': stats,
            'navigation_aids': navigation_aids,
            'guidance': guidance,
            'next_step': 'Write reflexive memos before proceeding to Phase 2'
        }
    
    def _generate_navigation_aids(self, texts: List[str]) -> Dict:
        """
        Generate AI-powered navigation aids (NOT summaries to replace reading).
        
        These are PROMPTS to guide reading, not substitutes.
        """
        # Simple keyword extraction for navigation
        from collections import Counter
        import re
        
        # Extract frequent words (simple approach)
        all_words = []
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            all_words.extend([w for w in words if len(w) > 3])
        
        word_freq = Counter(all_words).most_common(20)
        
        # Document length distribution
        lengths = [len(text.split()) for text in texts]
        length_quartiles = {
            'q1': int(np.percentile(lengths, 25)),
            'median': int(np.percentile(lengths, 50)),
            'q3': int(np.percentile(lengths, 75))
        }
        
        return {
            'frequent_terms': [word for word, _ in word_freq],
            'length_distribution': length_quartiles,
            'note': 'These are navigation aids to guide your reading, NOT summaries',
            'recommendation': 'Use these to identify potentially rich documents to read first'
        }
    
    def add_reflexive_memo(
        self,
        memo_text: str,
        memo_type: str = "general",
        related_data: Optional[List[int]] = None
    ) -> Dict:
        """
        Add a reflexive memo (PRIMARY ANALYTIC TOOL in RTA).
        
        Parameters:
        -----------
        memo_text : str
            The memo content
        memo_type : str
            Type: "general", "coding", "theme", "methodological", "reflexive"
        related_data : List[int], optional
            Indices of related documents
            
        Returns:
        --------
        Dict with memo details
        """
        memo = {
            'id': len(self.reflexive_memos),
            'timestamp': datetime.now().isoformat(),
            'phase': self.current_phase,
            'type': memo_type,
            'text': memo_text,
            'researcher': self.researcher_name,
            'related_data': related_data or []
        }
        
        self.reflexive_memos.append(memo)
        
        self._log_audit_entry(
            "memo_added",
            f"Reflexive memo added: {memo_type}",
            {"memo_id": memo['id']}
        )
        
        return memo
    
    def set_positionality_statement(self, statement: str):
        """
        Set researcher positionality statement (REQUIRED for RTA).
        
        Parameters:
        -----------
        statement : str
            Statement of researcher's position, background, assumptions
        """
        self.positionality_statement = {
            'text': statement,
            'researcher': self.researcher_name,
            'timestamp': datetime.now().isoformat()
        }
        
        self._log_audit_entry(
            "positionality_set",
            "Positionality statement recorded",
            {}
        )
    
    # ========================================================================
    # PHASE 2: GENERATING INITIAL CODES
    # ========================================================================
    
    def phase2_initial_coding(
        self,
        texts: List[str],
        request_ai_suggestions: bool = False
    ) -> Dict:
        """
        Phase 2: Generating initial codes.
        
        AI Role: Suggest CANDIDATE codes to widen aperture (if requested).
        Human Role: Code the data yourself, memo where you agree/disagree with AI.
        
        Parameters:
        -----------
        texts : List[str]
            Data to code
        request_ai_suggestions : bool
            Whether to request AI candidate codes (optional)
            
        Returns:
        --------
        Dict with coding guidance and optional AI suggestions
        """
        self.current_phase = 2
        self._log_audit_entry(
            "phase2_started",
            "Initial coding phase initiated",
            {"ai_suggestions_requested": request_ai_suggestions}
        )
        
        guidance = {
            'primary_task': 'CODE THE DATA YOURSELF - AI suggestions are prompts only',
            'instructions': [
                '1. Code systematically across the entire dataset',
                '2. Code for as many patterns as possible',
                '3. Keep codes close to the data (semantic level initially)',
                '4. Write memos about coding decisions',
                '5. Note where you agree/disagree with any AI suggestions'
            ],
            'coding_principles': [
                'Codes are labels for features of the data relevant to your question',
                'One segment can have multiple codes',
                'Codes can overlap',
                'Stay open to unexpected patterns',
                'Your interpretation matters - there is no "correct" code'
            ],
            'memo_prompts': [
                'Why did I code this segment this way?',
                'What alternative codes did I consider?',
                'How does my positionality shape this coding?',
                'What patterns am I starting to see?'
            ],
            'warning': '⚠️ AI codes are PROVOCATIONS, not ground truth. You decide what codes fit.'
        }
        
        result = {
            'phase': 2,
            'phase_name': 'Generating Initial Codes',
            'guidance': guidance
        }
        
        # Optional: AI candidate codes (if requested)
        if request_ai_suggestions:
            ai_suggestions = self._generate_candidate_codes(texts)
            result['ai_suggestions'] = ai_suggestions
            result['ai_note'] = 'These are CANDIDATE codes to widen your thinking. Re-code yourself and memo your decisions.'
        
        return result
    
    def _generate_candidate_codes(self, texts: List[str], n_suggestions: int = 15) -> Dict:
        """
        Generate AI candidate codes as PROMPTS (not ground truth).
        
        Uses simple keyword extraction and clustering as starting points.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import KMeans
        
        # Vectorize
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        try:
            X = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Simple clustering to suggest code groups
            n_clusters = min(5, len(texts))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            kmeans.fit(X)
            
            # Get top terms per cluster as candidate codes
            candidate_codes = []
            for i in range(n_clusters):
                center = kmeans.cluster_centers_[i]
                top_indices = center.argsort()[-3:][::-1]
                terms = [feature_names[idx] for idx in top_indices]
                candidate_codes.append({
                    'suggested_code': f"code_cluster_{i}",
                    'related_terms': terms,
                    'note': 'This is a SUGGESTION - you decide if it fits your data'
                })
            
            return {
                'candidate_codes': candidate_codes[:n_suggestions],
                'method': 'TF-IDF + K-means clustering',
                'warning': 'These are STARTING POINTS. Your codes should emerge from YOUR engagement with the data.',
                'instruction': 'Review these, then CODE THE DATA YOURSELF. Memo where you agree/disagree.'
            }
        except Exception as e:
            return {
                'error': str(e),
                'note': 'Could not generate suggestions. Code the data yourself.'
            }
    
    def add_code(
        self,
        code_name: str,
        description: str,
        example_segments: List[str],
        ai_suggested: bool = False,
        agreement_with_ai: Optional[str] = None
    ) -> Dict:
        """
        Add a code created by the researcher.
        
        Parameters:
        -----------
        code_name : str
            Name of the code
        description : str
            What this code captures
        example_segments : List[str]
            Example data segments for this code
        ai_suggested : bool
            Whether this code was suggested by AI
        agreement_with_ai : str, optional
            If AI-suggested, note agreement/disagreement
            
        Returns:
        --------
        Dict with code details
        """
        code = {
            'id': len(self.codes),
            'name': code_name,
            'description': description,
            'examples': example_segments,
            'created_by': self.researcher_name,
            'timestamp': datetime.now().isoformat(),
            'phase': self.current_phase,
            'ai_suggested': ai_suggested,
            'agreement_with_ai': agreement_with_ai
        }
        
        self.codes[code_name] = code
        
        if ai_suggested and agreement_with_ai:
            self._log_ai_assistance(
                "code_suggestion",
                code_name,
                f"Code '{code_name}' created",
                agreement_with_ai
            )
        
        self._log_audit_entry(
            "code_added",
            f"Code '{code_name}' added",
            {"ai_suggested": ai_suggested}
        )
        
        return code
    
    # ========================================================================
    # PHASE 3: SEARCHING FOR THEMES
    # ========================================================================
    
    def phase3_searching_themes(
        self,
        coded_data: Dict[str, List[str]],
        use_ai_clustering: bool = False
    ) -> Dict:
        """
        Phase 3: Searching for themes.
        
        AI Role: Propose provisional groupings via clustering (if requested).
        Human Role: Articulate the central organising concept that makes it a THEME.
        
        Parameters:
        -----------
        coded_data : Dict[str, List[str]]
            Mapping of code names to coded segments
        use_ai_clustering : bool
            Whether to use AI clustering for provisional groupings
            
        Returns:
        --------
        Dict with theme search guidance and optional AI groupings
        """
        self.current_phase = 3
        self._log_audit_entry(
            "phase3_started",
            "Theme searching phase initiated",
            {"n_codes": len(coded_data), "ai_clustering": use_ai_clustering}
        )
        
        guidance = {
            'primary_task': 'YOU articulate the central organising concept for each theme',
            'instructions': [
                '1. Look for patterns across codes',
                '2. Group codes into potential themes',
                '3. Define the CENTRAL ORGANISING CONCEPT for each theme',
                '4. A theme is NOT just a topic - it captures something meaningful',
                '5. Use visual maps, tables, or mind maps to explore relationships'
            ],
            'theme_criteria': [
                'A theme captures something important about the data in relation to your research question',
                'A theme has a central organising concept (not just a topic label)',
                'Themes should be coherent and internally consistent',
                'Themes should be distinct from each other',
                'Themes tell a story about the data'
            ],
            'memo_prompts': [
                'What is the central idea that unifies these codes?',
                'What story does this theme tell?',
                'How does this theme relate to my research question?',
                'What am I calling this theme and why?'
            ],
            'warning': '⚠️ AI can suggest groupings, but YOU define what makes it a theme.'
        }
        
        result = {
            'phase': 3,
            'phase_name': 'Searching for Themes',
            'guidance': guidance,
            'n_codes_to_organize': len(coded_data)
        }
        
        # Optional: AI clustering for provisional groupings
        if use_ai_clustering and coded_data:
            ai_groupings = self._suggest_code_groupings(coded_data)
            result['ai_groupings'] = ai_groupings
            result['ai_note'] = 'These are PROVISIONAL groupings. YOU must define the central organising concept.'
        
        return result
    
    def _suggest_code_groupings(self, coded_data: Dict[str, List[str]]) -> Dict:
        """
        Use clustering to suggest provisional code groupings.
        Researcher must define the central organising concept.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import AgglomerativeClustering
        from scipy.cluster.hierarchy import dendrogram, linkage
        
        code_names = list(coded_data.keys())
        
        if len(code_names) < 3:
            return {
                'note': 'Too few codes for clustering. Group codes manually based on conceptual relationships.'
            }
        
        # Create code descriptions from examples
        code_texts = [' '.join(coded_data[code][:5]) for code in code_names]
        
        try:
            # Vectorize
            vectorizer = TfidfVectorizer(max_features=50)
            X = vectorizer.fit_transform(code_texts)
            
            # Hierarchical clustering
            n_clusters = min(5, len(code_names) // 2)
            clustering = AgglomerativeClustering(n_clusters=n_clusters)
            # Convert to dense array if sparse - type: ignore for scipy sparse matrices
            X_dense = X.toarray() if hasattr(X, 'toarray') else X  # type: ignore
            labels = clustering.fit_predict(X_dense)
            
            # Group codes by cluster
            groupings = defaultdict(list)
            for code, label in zip(code_names, labels):
                groupings[f"provisional_theme_{label}"].append(code)
            
            return {
                'provisional_groupings': dict(groupings),
                'method': 'Hierarchical clustering on code content',
                'warning': 'These are PROVISIONAL. You must define the central organising concept for each theme.',
                'instruction': 'Review these groupings, then articulate what unifies each group conceptually.'
            }
        except Exception as e:
            return {
                'error': str(e),
                'note': 'Could not generate groupings. Group codes manually based on meaning.'
            }
    
    def add_theme(
        self,
        theme_name: str,
        central_concept: str,
        included_codes: List[str],
        description: str,
        ai_suggested_grouping: bool = False,
        conceptual_rationale: Optional[str] = None
    ) -> Dict:
        """
        Add a theme defined by the researcher.
        
        Parameters:
        -----------
        theme_name : str
            Name of the theme
        central_concept : str
            The central organising concept (REQUIRED)
        included_codes : List[str]
            Codes included in this theme
        description : str
            What this theme captures
        ai_suggested_grouping : bool
            Whether AI suggested this code grouping
        conceptual_rationale : str, optional
            Why these codes form a coherent theme
            
        Returns:
        --------
        Dict with theme details
        """
        theme = {
            'id': len(self.themes),
            'name': theme_name,
            'central_concept': central_concept,
            'description': description,
            'codes': included_codes,
            'created_by': self.researcher_name,
            'timestamp': datetime.now().isoformat(),
            'phase': self.current_phase,
            'ai_suggested_grouping': ai_suggested_grouping,
            'conceptual_rationale': conceptual_rationale
        }
        
        self.themes[theme_name] = theme
        
        if ai_suggested_grouping and conceptual_rationale:
            self._log_ai_assistance(
                "theme_grouping",
                f"Codes: {included_codes}",
                f"Theme '{theme_name}' created with concept: {central_concept}",
                conceptual_rationale
            )
        
        self._log_audit_entry(
            "theme_added",
            f"Theme '{theme_name}' added",
            {"n_codes": len(included_codes), "ai_suggested": ai_suggested_grouping}
        )
        
        return theme
    
    # ========================================================================
    # PHASE 4: REVIEWING THEMES
    # ========================================================================
    
    def phase4_reviewing_themes(
        self,
        themes: Optional[Dict] = None,
        request_ai_stress_test: bool = False
    ) -> Dict:
        """
        Phase 4: Reviewing themes.
        
        AI Role: Stress-test themes by finding disconfirming cases, boundary examples.
        Human Role: Decide what holds, refine or collapse themes.
        
        Parameters:
        -----------
        themes : Dict, optional
            Themes to review (uses self.themes if not provided)
        request_ai_stress_test : bool
            Whether to request AI stress-testing
            
        Returns:
        --------
        Dict with review guidance and optional AI stress-tests
        """
        self.current_phase = 4
        themes = themes or self.themes
        
        self._log_audit_entry(
            "phase4_started",
            "Theme reviewing phase initiated",
            {"n_themes": len(themes), "ai_stress_test": request_ai_stress_test}
        )
        
        guidance = {
            'primary_task': 'YOU decide what themes hold and how to refine them',
            'instructions': [
                '1. Review each theme for internal coherence',
                '2. Check themes are distinct from each other',
                '3. Look for disconfirming cases',
                '4. Consider boundary examples',
                '5. Refine, split, merge, or discard themes as needed'
            ],
            'quality_checks': [
                'Does this theme have a clear central concept?',
                'Are all codes/data in this theme coherent?',
                'Is this theme distinct from others?',
                'Does this theme tell a meaningful story?',
                'Are there disconfirming cases I need to account for?'
            ],
            'memo_prompts': [
                'What makes this theme coherent?',
                'Where are the boundaries of this theme?',
                'What data doesn\'t quite fit?',
                'Should I split, merge, or refine this theme?'
            ],
            'warning': '⚠️ AI can find edge cases, but YOU decide if themes hold.'
        }
        
        result = {
            'phase': 4,
            'phase_name': 'Reviewing Themes',
            'guidance': guidance,
            'n_themes_to_review': len(themes)
        }
        
        # Optional: AI stress-testing
        if request_ai_stress_test and themes:
            stress_tests = self._stress_test_themes(themes)
            result['ai_stress_tests'] = stress_tests
            result['ai_note'] = 'These are CHALLENGES to your themes. YOU decide what holds.'
        
        return result
    
    def _stress_test_themes(self, themes: Dict) -> Dict:
        """
        AI stress-tests themes by finding potential issues.
        Researcher decides what holds.
        """
        stress_tests = {}
        
        for theme_name, theme_data in themes.items():
            tests = {
                'theme': theme_name,
                'central_concept': theme_data.get('central_concept', 'Not specified'),
                'challenges': []
            }
            
            # Check for potential issues
            n_codes = len(theme_data.get('codes', []))
            
            if n_codes < 2:
                tests['challenges'].append({
                    'type': 'thin_theme',
                    'description': 'Theme has very few codes. Is this substantial enough?',
                    'question': 'Should this be merged with another theme or developed further?'
                })
            
            if n_codes > 10:
                tests['challenges'].append({
                    'type': 'broad_theme',
                    'description': 'Theme has many codes. Is it too broad?',
                    'question': 'Should this be split into sub-themes?'
                })
            
            # Check for vague central concept
            concept = theme_data.get('central_concept', '')
            if len(concept.split()) < 5:
                tests['challenges'].append({
                    'type': 'vague_concept',
                    'description': 'Central concept is brief. Is it fully articulated?',
                    'question': 'Can you elaborate on what unifies this theme?'
                })
            
            stress_tests[theme_name] = tests
        
        return {
            'stress_tests': stress_tests,
            'instruction': 'Review these challenges. YOU decide if your themes hold or need refinement.',
            'note': 'These are provocations, not verdicts. Your judgment prevails.'
        }
    
    def refine_theme(
        self,
        theme_name: str,
        action: str,
        rationale: str,
        new_definition: Optional[Dict] = None
    ) -> Dict:
        """
        Refine a theme based on review.
        
        Parameters:
        -----------
        theme_name : str
            Theme to refine
        action : str
            Action taken: "refined", "split", "merged", "discarded"
        rationale : str
            Why this action was taken
        new_definition : Dict, optional
            New theme definition if refined/split
            
        Returns:
        --------
        Dict with refinement details
        """
        refinement = {
            'theme': theme_name,
            'action': action,
            'rationale': rationale,
            'researcher': self.researcher_name,
            'timestamp': datetime.now().isoformat(),
            'phase': self.current_phase
        }
        
        if new_definition:
            refinement['new_definition'] = new_definition
            if theme_name in self.themes:
                self.themes[theme_name].update(new_definition)
        
        self._log_audit_entry(
            "theme_refined",
            f"Theme '{theme_name}' {action}",
            {"rationale": rationale}
        )
        
        return refinement
    
    # ========================================================================
    # PHASE 5: DEFINING AND NAMING THEMES
    # ========================================================================
    
    def phase5_defining_naming(
        self,
        themes: Optional[Dict] = None,
        request_ai_alternatives: bool = False
    ) -> Dict:
        """
        Phase 5: Defining and naming themes.
        
        AI Role: Generate alternative names/definitions (if requested).
        Human Role: Keep the analytical narrative yours.
        
        Parameters:
        -----------
        themes : Dict, optional
            Themes to define (uses self.themes if not provided)
        request_ai_alternatives : bool
            Whether to request AI alternative names
            
        Returns:
        --------
        Dict with definition guidance and optional AI alternatives
        """
        self.current_phase = 5
        themes = themes or self.themes
        
        self._log_audit_entry(
            "phase5_started",
            "Defining and naming phase initiated",
            {"n_themes": len(themes), "ai_alternatives": request_ai_alternatives}
        )
        
        guidance = {
            'primary_task': 'YOU write the analytical narrative for each theme',
            'instructions': [
                '1. Write a detailed definition for each theme',
                '2. Explain what the theme is about and why it matters',
                '3. Choose a concise, punchy name',
                '4. Ensure names are informative and capture the essence',
                '5. Write the story of each theme'
            ],
            'definition_components': [
                'What is this theme about?',
                'What is the central organising concept?',
                'How does it relate to the research question?',
                'What story does it tell?',
                'What are the nuances and complexities?'
            ],
            'naming_tips': [
                'Names should be concise but informative',
                'Avoid generic labels',
                'Capture the essence of the theme',
                'Consider using participants\' language where appropriate'
            ],
            'warning': '⚠️ AI can suggest alternatives, but the analytical narrative is YOURS.'
        }
        
        result = {
            'phase': 5,
            'phase_name': 'Defining and Naming Themes',
            'guidance': guidance,
            'n_themes': len(themes)
        }
        
        # Optional: AI alternative names
        if request_ai_alternatives and themes:
            alternatives = self._suggest_alternative_names(themes)
            result['ai_alternatives'] = alternatives
            result['ai_note'] = 'These are ALTERNATIVE suggestions. Choose what best captures YOUR analysis.'
        
        return result
    
    def _suggest_alternative_names(self, themes: Dict) -> Dict:
        """
        Suggest alternative theme names.
        Researcher chooses what fits their analytical narrative.
        """
        alternatives = {}
        
        for theme_name, theme_data in themes.items():
            concept = theme_data.get('central_concept', '')
            codes = theme_data.get('codes', [])
            
            # Generate simple alternatives based on concept and codes
            suggestions = [
                f"{theme_name} (original)",
                f"Theme of {concept.split()[0] if concept else 'concept'}",
                f"{codes[0].replace('_', ' ').title()} and related" if codes else "Theme"
            ]
            
            alternatives[theme_name] = {
                'current_name': theme_name,
                'suggestions': suggestions[:3],
                'note': 'These are STARTING POINTS. Your name should capture YOUR analytical narrative.'
            }
        
        return alternatives
    
    def finalize_theme_definition(
        self,
        theme_name: str,
        final_name: str,
        definition: str,
        narrative: str,
        exemplar_quotes: List[str]
    ) -> Dict:
        """
        Finalize theme definition and narrative.
        
        Parameters:
        -----------
        theme_name : str
            Original theme name
        final_name : str
            Final chosen name
        definition : str
            Complete theme definition
        narrative : str
            Analytical narrative for this theme
        exemplar_quotes : List[str]
            Key quotes that illustrate the theme
            
        Returns:
        --------
        Dict with finalized theme
        """
        if theme_name in self.themes:
            self.themes[theme_name].update({
                'final_name': final_name,
                'definition': definition,
                'narrative': narrative,
                'exemplar_quotes': exemplar_quotes,
                'finalized_by': self.researcher_name,
                'finalized_at': datetime.now().isoformat()
            })
        
        self._log_audit_entry(
            "theme_finalized",
            f"Theme '{theme_name}' finalized as '{final_name}'",
            {}
        )
        
        return self.themes[theme_name]
    
    # ========================================================================
    # PHASE 6: PRODUCING THE REPORT
    # ========================================================================
    
    def phase6_producing_report(
        self,
        themes: Optional[Dict] = None,
        use_ai_structure_help: bool = False
    ) -> Dict:
        """
        Phase 6: Producing the report.
        
        AI Role: Help with structure and clarity (if requested).
        Human Role: Select extracts, write the analytic story, attend to theory.
        
        Parameters:
        -----------
        themes : Dict, optional
            Finalized themes (uses self.themes if not provided)
        use_ai_structure_help : bool
            Whether to request AI structural suggestions
            
        Returns:
        --------
        Dict with report guidance and optional AI structure help
        """
        self.current_phase = 6
        themes = themes or self.themes
        
        self._log_audit_entry(
            "phase6_started",
            "Report production phase initiated",
            {"n_themes": len(themes), "ai_structure_help": use_ai_structure_help}
        )
        
        guidance = {
            'primary_task': 'YOU write the analytic story',
            'instructions': [
                '1. Write a compelling narrative that weaves themes together',
                '2. Select vivid, illustrative extracts',
                '3. Provide analytic commentary on each extract',
                '4. Connect findings to theory and literature',
                '5. Address your research question'
            ],
            'report_structure': [
                'Introduction: Research question, context, approach',
                'Methods: RTA process, epistemology, positionality, AI use',
                'Findings: Theme-by-theme with extracts and analysis',
                'Discussion: Theoretical implications, limitations',
                'Conclusion: Key contributions'
            ],
            'extract_selection': [
                'Choose vivid, illustrative quotes',
                'Provide analytic commentary, not just description',
                'Show the complexity and nuance',
                'Balance across themes'
            ],
            'transparency_requirements': [
                'Declare RTA approach and epistemology',
                'State your positionality',
                'Describe AI assistance (what, how, human decisions)',
                'Provide audit trail summary'
            ],
            'warning': '⚠️ AI can help with structure, but the analytic story is YOURS.'
        }
        
        result = {
            'phase': 6,
            'phase_name': 'Producing the Report',
            'guidance': guidance,
            'n_themes_to_report': len(themes)
        }
        
        # Optional: AI structure suggestions
        if use_ai_structure_help:
            structure = self._suggest_report_structure(themes)
            result['ai_structure'] = structure
            result['ai_note'] = 'This is a SUGGESTED structure. Adapt to fit YOUR analytic story.'
        
        return result
    
    def _suggest_report_structure(self, themes: Dict) -> Dict:
        """
        Suggest report structure.
        Researcher adapts to their analytic narrative.
        """
        structure = {
            'suggested_outline': [
                '1. Introduction',
                '   - Research question and context',
                '   - Why RTA is appropriate',
                '2. Methods',
                '   - RTA process (6 phases)',
                '   - Epistemology: ' + self.epistemology,
                '   - Ontology: ' + self.ontology,
                '   - Positionality statement',
                '   - AI assistance disclosure',
                '3. Findings',
            ],
            'theme_sections': []
        }
        
        for i, (theme_name, theme_data) in enumerate(themes.items(), 1):
            final_name = theme_data.get('final_name', theme_name)
            structure['theme_sections'].append(
                f"   3.{i}. Theme: {final_name}"
            )
        
        structure['suggested_outline'].extend([
            '4. Discussion',
            '   - Theoretical implications',
            '   - Connections to literature',
            '   - Limitations',
            '5. Conclusion',
            '   - Key contributions',
            '   - Future directions'
        ])
        
        return {
            'structure': structure,
            'note': 'This is a STARTING POINT. Adapt to fit your analytic narrative and disciplinary conventions.'
        }
    
    # ========================================================================
    # REPORTING AND EXPORT
    # ========================================================================
    
    def generate_methods_section(self) -> str:
        """
        Generate methods section with required transparency about AI use.
        
        Returns:
        --------
        str: Methods section text
        """
        methods = f"""
## Methods

### Analytic Approach
This study employed Reflexive Thematic Analysis (RTA) following Braun and Clarke (2006, 2019, 2021). RTA is a method for identifying, analyzing, and reporting patterns (themes) within data, with the researcher's subjectivity and reflexivity central to the analytic process.

### Epistemological and Ontological Position
- **Epistemology**: {self.epistemology}
- **Ontology**: {self.ontology}

### Positionality Statement
{self.positionality_statement['text'] if self.positionality_statement else '[Positionality statement to be added]'}

### RTA Process
The analysis followed the six phases of RTA:

1. **Familiarisation**: All data was read multiple times with reflexive memos written throughout.
2. **Initial Coding**: Data was coded systematically, with codes staying close to the data.
3. **Searching for Themes**: Codes were grouped into provisional themes based on central organising concepts.
4. **Reviewing Themes**: Themes were reviewed for coherence and distinctiveness.
5. **Defining and Naming**: Each theme was defined with a clear analytical narrative.
6. **Producing the Report**: Findings were written up with vivid extracts and analytic commentary.

### AI Assistance Disclosure
AI tools were used to **augment, not replace**, human analysis:

**AI Tasks:**
{self._summarize_ai_assistance()}

**Human Decisions:**
All analytic decisions were made by the researcher. AI outputs were treated as provocations to widen thinking, not as ground truth. Reflexive memos documented where AI suggestions were accepted, modified, or rejected, and why.

**Audit Trail:**
A complete audit trail was maintained documenting all AI assistance and human decisions (available upon request).

### Quality Criteria
Quality was assessed using criteria appropriate for RTA:
- Reflexive memos throughout the process
- Theoretical coherence of themes
- Rich, vivid extracts
- Attention to disconfirming cases
- Clear analytical narrative
- Transparent reporting of process

Note: Inter-rater reliability was not used, as RTA prioritizes researcher subjectivity and reflexivity over reliability metrics.
"""
        return methods.strip()
    
    def _summarize_ai_assistance(self) -> str:
        """Summarize AI assistance for methods section."""
        if not self.ai_assistance_log:
            return "- No AI assistance was used in this analysis."
        
        tasks = set(entry['task'] for entry in self.ai_assistance_log)
        summary = []
        for task in tasks:
            count = sum(1 for e in self.ai_assistance_log if e['task'] == task)
            summary.append(f"- {task.replace('_', ' ').title()}: {count} instances")
        
        return '\n'.join(summary)
    
    def export_audit_trail(self, format: str = 'json') -> str:
        """
        Export complete audit trail.
        
        Parameters:
        -----------
        format : str
            Export format: 'json', 'markdown', or 'csv'
            
        Returns:
        --------
        str: Formatted audit trail
        """
        if format == 'json':
            return json.dumps({
                'project': self.project_name,
                'researcher': self.researcher_name,
                'epistemology': self.epistemology,
                'ontology': self.ontology,
                'audit_trail': self.audit_trail,
                'ai_assistance_log': self.ai_assistance_log,
                'reflexive_memos': self.reflexive_memos
            }, indent=2)
        
        elif format == 'markdown':
            md = f"# Audit Trail: {self.project_name}\n\n"
            md += f"**Researcher:** {self.researcher_name}\n"
            md += f"**Epistemology:** {self.epistemology}\n"
            md += f"**Ontology:** {self.ontology}\n\n"
            
            md += "## Audit Trail\n\n"
            for entry in self.audit_trail:
                md += f"- **{entry['timestamp']}** (Phase {entry['phase']}): {entry['description']}\n"
            
            md += "\n## AI Assistance Log\n\n"
            for entry in self.ai_assistance_log:
                md += f"- **{entry['timestamp']}** (Phase {entry['phase']}): {entry['task']}\n"
                md += f"  - Human Decision: {entry['human_decision']}\n"
                md += f"  - Rationale: {entry['rationale']}\n\n"
            
            return md
        
        else:  # csv
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=['timestamp', 'phase', 'action', 'description'])
            writer.writeheader()
            writer.writerows(self.audit_trail)
            
            return output.getvalue()
    
    def get_project_summary(self) -> Dict:
        """
        Get summary of RTA project.
        
        Returns:
        --------
        Dict with project summary
        """
        return {
            'project_name': self.project_name,
            'researcher': self.researcher_name,
            'epistemology': self.epistemology,
            'ontology': self.ontology,
            'current_phase': self.current_phase,
            'n_codes': len(self.codes),
            'n_themes': len(self.themes),
            'n_memos': len(self.reflexive_memos),
            'n_audit_entries': len(self.audit_trail),
            'n_ai_assistance_instances': len(self.ai_assistance_log),
            'positionality_set': self.positionality_statement is not None
        }

# Made with Bob
