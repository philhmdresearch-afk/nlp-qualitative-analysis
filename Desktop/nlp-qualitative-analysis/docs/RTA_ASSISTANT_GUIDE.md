# Reflexive Thematic Analysis (RTA) Assistant Guide

## Overview

The RTA Assistant is a **human-in-the-loop** tool that guides researchers through the six phases of Reflexive Thematic Analysis following Braun & Clarke (2006, 2019, 2021) methodology. 

**Core Principle:** AI augments, never replaces, analyst judgment.

---

## 🎯 Key Principles

### What RTA Assistant Does
✅ Provides provocations to widen your thinking  
✅ Offers navigation aids and structural suggestions  
✅ Stress-tests your analytic decisions  
✅ Maintains complete audit trail  
✅ Tracks reflexive memos (your primary analytic tool)  
✅ Generates transparent methods sections  

### What RTA Assistant Does NOT Do
❌ Make analytic decisions for you  
❌ Replace reading and engaging with data  
❌ Provide "correct" codes or themes  
❌ Calculate inter-rater reliability (not appropriate for RTA)  
❌ Automate the analytic process  

---

## 📚 The Six Phases

### Phase 1: Familiarisation

**Your Task:** Read ALL data yourself, write reflexive memos

**AI Role:** Provide navigation aids (NOT summaries to replace reading)

```python
from src.rta_assistant import RTAAssistant

# Initialize project
rta = RTAAssistant(
    project_name="User Experience Study 2026",
    researcher_name="Dr. Jane Smith",
    epistemology="contextualist",
    ontology="critical realist"
)

# Set positionality (REQUIRED)
rta.set_positionality_statement("""
I am a UX researcher with 10 years of experience in tech. 
My background in HCI shapes how I notice usability patterns.
I approach this data as someone who values user agency and accessibility.
""")

# Phase 1: Get familiarisation guidance
phase1 = rta.phase1_familiarisation(texts, metadata)

print(phase1['guidance']['instructions'])
# 1. Read each transcript/response in full
# 2. Write reflexive memos as you read
# 3. Note initial patterns, tensions, surprises
# ...

# Write reflexive memos as you read
rta.add_reflexive_memo(
    memo_text="Struck by how often participants mention 'trust' - but what KIND of trust?",
    memo_type="general"
)

rta.add_reflexive_memo(
    memo_text="My tech background makes me notice technical issues more than emotional responses. Need to stay open to affective dimensions.",
    memo_type="reflexive"
)
```

**⚠️ Warning:** Do NOT rely on AI summaries as substitutes for reading. Your engagement with the data is essential.

---

### Phase 2: Generating Initial Codes

**Your Task:** Code the data yourself

**AI Role:** Suggest candidate codes to widen aperture (optional)

```python
# Phase 2: Get coding guidance
phase2 = rta.phase2_initial_coding(
    texts,
    request_ai_suggestions=True  # Optional
)

# If you requested AI suggestions, review them
if 'ai_suggestions' in phase2:
    print("AI suggested these CANDIDATE codes:")
    for suggestion in phase2['ai_suggestions']['candidate_codes']:
        print(f"- {suggestion['suggested_code']}: {suggestion['related_terms']}")
    print("\n⚠️ These are PROVOCATIONS. You decide what codes fit.")

# Code the data yourself
rta.add_code(
    code_name="trust_in_automation",
    description="Participants express concerns about trusting automated systems",
    example_segments=[
        "I'm not sure I trust the AI to make decisions for me",
        "How do I know the automation won't make mistakes?"
    ],
    ai_suggested=False  # You created this code
)

# If you used an AI suggestion, document your decision
rta.add_code(
    code_name="control_vs_convenience",
    description="Tension between wanting control and wanting convenience",
    example_segments=["..."],
    ai_suggested=True,
    agreement_with_ai="Agreed with AI suggestion but refined the description to capture the TENSION, not just the topics"
)

# Write memos about coding decisions
rta.add_reflexive_memo(
    memo_text="Decided to code 'trust_in_automation' separately from 'trust_in_company' - they feel qualitatively different",
    memo_type="coding"
)
```

**⚠️ Warning:** AI codes are PROVOCATIONS, not ground truth. You decide what codes fit.

---

### Phase 3: Searching for Themes

**Your Task:** Articulate the central organising concept for each theme

**AI Role:** Propose provisional groupings (optional)

```python
# Prepare coded data
coded_data = {
    'trust_in_automation': ["segment1", "segment2", ...],
    'control_vs_convenience': ["segment3", "segment4", ...],
    # ... more codes
}

# Phase 3: Get theme searching guidance
phase3 = rta.phase3_searching_themes(
    coded_data,
    use_ai_clustering=True  # Optional
)

# If you requested AI clustering, review provisional groupings
if 'ai_groupings' in phase3:
    print("AI suggests these PROVISIONAL groupings:")
    for theme, codes in phase3['ai_groupings']['provisional_groupings'].items():
        print(f"{theme}: {codes}")
    print("\n⚠️ YOU must define the central organising concept.")

# Create themes with YOUR central organising concepts
rta.add_theme(
    theme_name="automation_anxiety",
    central_concept="Participants experience anxiety about ceding control to automated systems, rooted in uncertainty about system reliability and accountability",
    included_codes=['trust_in_automation', 'fear_of_errors', 'accountability_concerns'],
    description="This theme captures the emotional and cognitive tension participants feel when considering automation",
    ai_suggested_grouping=True,
    conceptual_rationale="AI grouped these codes together, but I defined the CENTRAL CONCEPT as 'anxiety' rather than just 'trust issues' - the affective dimension is key"
)

# Write theme memos
rta.add_reflexive_memo(
    memo_text="'Automation anxiety' feels like the right concept - it's not just distrust, it's a deeper anxiety about loss of agency",
    memo_type="theme"
)
```

**⚠️ Warning:** AI can suggest groupings, but YOU define what makes it a theme.

---

### Phase 4: Reviewing Themes

**Your Task:** Decide what themes hold and how to refine them

**AI Role:** Stress-test themes by finding potential issues (optional)

```python
# Phase 4: Get theme review guidance
phase4 = rta.phase4_reviewing_themes(
    request_ai_stress_test=True  # Optional
)

# If you requested stress-testing, review challenges
if 'ai_stress_tests' in phase4:
    for theme_name, tests in phase4['ai_stress_tests']['stress_tests'].items():
        print(f"\nTheme: {theme_name}")
        for challenge in tests['challenges']:
            print(f"  - {challenge['type']}: {challenge['description']}")
            print(f"    Question: {challenge['question']}")

# Review and refine themes
rta.refine_theme(
    theme_name="automation_anxiety",
    action="refined",
    rationale="AI flagged this as potentially too broad. Reviewed and decided it's coherent - the breadth reflects the complexity of the phenomenon",
    new_definition={
        'central_concept': "Participants experience multifaceted anxiety about ceding control to automated systems, encompassing concerns about reliability, accountability, and loss of agency"
    }
)

# Write reflexive memo about refinement
rta.add_reflexive_memo(
    memo_text="Considered splitting 'automation_anxiety' but decided against it - the anxiety is precisely about the INTERCONNECTION of these concerns",
    memo_type="theme"
)
```

**⚠️ Warning:** AI can find edge cases, but YOU decide if themes hold.

---

### Phase 5: Defining and Naming Themes

**Your Task:** Write the analytical narrative for each theme

**AI Role:** Generate alternative names (optional)

```python
# Phase 5: Get defining/naming guidance
phase5 = rta.phase5_defining_naming(
    request_ai_alternatives=True  # Optional
)

# If you requested alternatives, review them
if 'ai_alternatives' in phase5:
    for theme, alts in phase5['ai_alternatives'].items():
        print(f"\nTheme: {theme}")
        print(f"AI suggestions: {alts['suggestions']}")
        print("⚠️ Choose what best captures YOUR analysis")

# Finalize theme with YOUR analytical narrative
rta.finalize_theme_definition(
    theme_name="automation_anxiety",
    final_name="Automation Anxiety: The Burden of Invisible Accountability",
    definition="""
    This theme captures participants' multifaceted anxiety about ceding control 
    to automated systems. The anxiety stems from three interconnected concerns: 
    uncertainty about system reliability, ambiguity about accountability when 
    things go wrong, and a deeper existential worry about loss of human agency.
    """,
    narrative="""
    Participants consistently expressed anxiety when discussing automation, but 
    this was not simple technophobia. Rather, it reflected sophisticated concerns 
    about accountability structures. As one participant noted, "If the AI makes 
    a mistake, who's responsible?" This question reveals the theme's core: 
    automation creates an accountability vacuum that participants find deeply 
    unsettling...
    """,
    exemplar_quotes=[
        "I'm not sure I trust the AI to make decisions for me",
        "If the AI makes a mistake, who's responsible?",
        "It feels like giving up control without knowing what I'm getting in return"
    ]
)
```

**⚠️ Warning:** AI can suggest alternatives, but the analytical narrative is YOURS.

---

### Phase 6: Producing the Report

**Your Task:** Write the analytic story

**AI Role:** Help with structure (optional)

```python
# Phase 6: Get report guidance
phase6 = rta.phase6_producing_report(
    use_ai_structure_help=True  # Optional
)

# If you requested structure help, review it
if 'ai_structure' in phase6:
    print("AI suggested structure:")
    for item in phase6['ai_structure']['structure']['suggested_outline']:
        print(item)
    print("\n⚠️ Adapt to fit YOUR analytic narrative")

# Generate methods section with AI transparency
methods_section = rta.generate_methods_section()
print(methods_section)
# Includes:
# - RTA approach declaration
# - Epistemology and ontology
# - Positionality statement
# - Six-phase process description
# - AI assistance disclosure
# - Quality criteria (NOT inter-rater reliability)

# Export audit trail for transparency
audit_trail_json = rta.export_audit_trail(format='json')
audit_trail_md = rta.export_audit_trail(format='markdown')

# Get project summary
summary = rta.get_project_summary()
print(f"Project: {summary['project_name']}")
print(f"Codes: {summary['n_codes']}")
print(f"Themes: {summary['n_themes']}")
print(f"Memos: {summary['n_memos']}")
print(f"AI assistance instances: {summary['n_ai_assistance_instances']}")
```

**⚠️ Warning:** AI can help with structure, but the analytic story is YOURS.

---

## 📝 Reflexive Memos: Your Primary Tool

Reflexive memos are the HEART of RTA. Write them constantly:

```python
# General observations
rta.add_reflexive_memo(
    memo_text="Noticing a pattern around 'control' - appears in multiple contexts",
    memo_type="general"
)

# Coding decisions
rta.add_reflexive_memo(
    memo_text="Decided to keep 'trust_in_automation' and 'trust_in_company' separate - they feel qualitatively different",
    memo_type="coding"
)

# Theme development
rta.add_reflexive_memo(
    memo_text="'Automation anxiety' captures something the AI clustering missed - the EMOTIONAL dimension",
    memo_type="theme"
)

# Methodological reflections
rta.add_reflexive_memo(
    memo_text="Using AI suggestions helped me see patterns I might have missed, but I'm being careful not to let them constrain my thinking",
    memo_type="methodological"
)

# Reflexivity about positionality
rta.add_reflexive_memo(
    memo_text="My tech background makes me notice technical issues more readily - need to stay open to social/emotional dimensions",
    memo_type="reflexive"
)
```

---

## 🔍 Audit Trail & Transparency

The RTA Assistant maintains a complete audit trail:

```python
# View audit trail
for entry in rta.audit_trail:
    print(f"{entry['timestamp']} (Phase {entry['phase']}): {entry['description']}")

# View AI assistance log
for entry in rta.ai_assistance_log:
    print(f"\nTask: {entry['task']}")
    print(f"Human Decision: {entry['human_decision']}")
    print(f"Rationale: {entry['rationale']}")

# Export for transparency
audit_json = rta.export_audit_trail(format='json')
# Include in supplementary materials
```

---

## ✅ Quality Criteria for RTA

RTA uses DIFFERENT quality criteria than coding-reliability TA:

### ✅ Appropriate for RTA:
- **Reflexive memos** throughout the process
- **Theoretical coherence** of themes
- **Rich, vivid extracts** with analytic commentary
- **Attention to disconfirming cases**
- **Clear analytical narrative**
- **Transparent reporting** of process and AI use

### ❌ NOT Appropriate for RTA:
- Inter-rater reliability (IRR)
- Cohen's kappa
- Model-human "match rates"
- Percentage agreement

**Why?** RTA prioritizes researcher subjectivity and reflexivity over reliability metrics.

---

## 📊 Methods Section Template

The RTA Assistant auto-generates a methods section:

```python
methods = rta.generate_methods_section()
```

This includes:
1. **RTA approach declaration** (Braun & Clarke)
2. **Epistemology and ontology**
3. **Positionality statement**
4. **Six-phase process description**
5. **AI assistance disclosure** (what, how, human decisions)
6. **Quality criteria** (appropriate for RTA)
7. **Audit trail availability**

---

## ⚠️ Common Pitfalls to Avoid

### 1. Confusing Topics with Themes
❌ **Topic:** "Automation"  
✅ **Theme:** "Automation Anxiety: The Burden of Invisible Accountability"

**Fix:** Articulate the central organising concept, not just the topic.

### 2. Treating AI Suggestions as Ground Truth
❌ "The AI identified 5 themes"  
✅ "AI clustering suggested provisional groupings, which I reviewed and refined based on my interpretation"

**Fix:** AI provides provocations, YOU make decisions.

### 3. Using Reliability Metrics
❌ "Inter-rater reliability was 0.85"  
✅ "Quality was assessed through reflexive memos, theoretical coherence, and rich extracts"

**Fix:** Use quality criteria appropriate for RTA.

### 4. Opaque AI Use
❌ "AI was used to assist analysis"  
✅ "AI suggested candidate codes in Phase 2 (15 instances), which I reviewed and modified. See audit trail for details."

**Fix:** Be transparent about exactly how AI was used.

### 5. Letting AI Constrain Thinking
❌ Accepting AI groupings without question  
✅ Using AI groupings as provocations, then developing YOUR conceptual framework

**Fix:** Stay open to patterns AI might miss (especially minority voices, theoretical tensions).

---

## 🎓 Theoretical Foundations

### Braun & Clarke's RTA
- **Reflexive TA** (not coding reliability TA)
- Researcher subjectivity is a resource, not a problem
- Themes are actively created, not passively discovered
- Quality comes from reflexivity and coherence, not reliability

### Human-in-the-Loop AI
- AI augments, never replaces, human judgment
- Researcher maintains analytic agency
- Transparent documentation of AI assistance
- Audit trail tracks all decisions

---

## 📚 Further Reading

### Essential RTA Resources
- Braun, V., & Clarke, V. (2006). Using thematic analysis in psychology. *Qualitative Research in Psychology*, 3(2), 77-101.
- Braun, V., & Clarke, V. (2019). Reflecting on reflexive thematic analysis. *Qualitative Research in Sport, Exercise and Health*, 11(4), 589-597.
- Braun, V., & Clarke, V. (2021). One size fits all? What counts as quality practice in (reflexive) thematic analysis? *Qualitative Research in Psychology*, 18(3), 328-352.

### AI-Assisted Qualitative Analysis
- Contemporary studies emphasize human-in-the-loop approaches
- AI as augmentation, not automation
- Transparency and reflexivity are essential
- Audit trails maintain methodological rigor

---

## 💡 Example Workflow

```python
# 1. Initialize project
rta = RTAAssistant(
    project_name="UX Study 2026",
    researcher_name="Dr. Smith",
    epistemology="contextualist",
    ontology="critical realist"
)

# 2. Set positionality
rta.set_positionality_statement("...")

# 3. Phase 1: Familiarisation
phase1 = rta.phase1_familiarisation(texts)
# READ ALL DATA, write memos

# 4. Phase 2: Coding
phase2 = rta.phase2_initial_coding(texts, request_ai_suggestions=True)
# Review AI suggestions, CODE YOURSELF, write memos

# 5. Phase 3: Themes
phase3 = rta.phase3_searching_themes(coded_data, use_ai_clustering=True)
# Review AI groupings, DEFINE CENTRAL CONCEPTS, write memos

# 6. Phase 4: Review
phase4 = rta.phase4_reviewing_themes(request_ai_stress_test=True)
# Review AI challenges, REFINE THEMES, write memos

# 7. Phase 5: Define
phase5 = rta.phase5_defining_naming(request_ai_alternatives=True)
# Review AI alternatives, WRITE ANALYTICAL NARRATIVE

# 8. Phase 6: Report
phase6 = rta.phase6_producing_report(use_ai_structure_help=True)
methods = rta.generate_methods_section()
audit = rta.export_audit_trail(format='markdown')
```

---

## 🤝 Support

For questions about RTA methodology:
- Consult Braun & Clarke's original papers
- Review the RTARG (Reflexive TA Research Group) resources

For questions about the RTA Assistant:
- Check this guide
- Review the audit trail for transparency
- Examine example notebooks (coming soon)

---

**Remember:** The RTA Assistant is a tool to support YOUR analysis, not to do it for you. Your engagement, reflexivity, and interpretation are what make the analysis rigorous and meaningful.

**Built with ❤️ for qualitative researchers who value both rigor and reflexivity**