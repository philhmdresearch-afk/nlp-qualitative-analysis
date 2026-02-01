"""
Quick test of the RTA Assistant
Demonstrates the six phases of Reflexive Thematic Analysis
"""

from src.rta_assistant import RTAAssistant
import json

# Sample data for testing
sample_texts = [
    "I love the automation features, they save me so much time",
    "I'm worried about trusting the AI to make important decisions",
    "The interface is intuitive but I wish I had more control",
    "Automation is great but what happens when it makes a mistake?",
    "I appreciate the convenience but feel like I'm losing agency",
    "The system works well but I don't understand how it makes decisions",
    "I trust the technology but not sure about accountability",
    "It's efficient but I worry about becoming too dependent on it"
]

print("=" * 70)
print("RTA ASSISTANT DEMO")
print("=" * 70)

# Initialize RTA project
print("\n1. Initializing RTA Project...")
rta = RTAAssistant(
    project_name="Automation Trust Study Demo",
    researcher_name="Demo Researcher",
    epistemology="contextualist",
    ontology="critical realist"
)

# Set positionality
print("2. Setting Positionality Statement...")
rta.set_positionality_statement("""
I am a UX researcher with experience in AI systems. My background in 
human-computer interaction shapes how I notice patterns around trust 
and control. I approach this data valuing user agency and transparency.
""")

# Phase 1: Familiarisation
print("\n" + "=" * 70)
print("PHASE 1: FAMILIARISATION")
print("=" * 70)
phase1 = rta.phase1_familiarisation(sample_texts)
print(f"\nDataset: {phase1['statistics']['n_documents']} documents")
print(f"Total words: {phase1['statistics']['total_words']}")
print("\nGuidance:")
for instruction in phase1['guidance']['instructions'][:3]:
    print(f"  • {instruction}")

# Add reflexive memo
print("\n📝 Adding reflexive memo...")
rta.add_reflexive_memo(
    memo_text="Initial reading reveals tension between appreciation for automation and anxiety about control/trust",
    memo_type="general"
)
print("✓ Memo added")

# Phase 2: Initial Coding
print("\n" + "=" * 70)
print("PHASE 2: GENERATING INITIAL CODES")
print("=" * 70)
phase2 = rta.phase2_initial_coding(sample_texts, request_ai_suggestions=True)
print("\nAI Candidate Codes (as provocations):")
if 'ai_suggestions' in phase2:
    for i, code in enumerate(phase2['ai_suggestions']['candidate_codes'][:3], 1):
        print(f"  {i}. {code['suggested_code']}: {code['related_terms']}")

# Add codes (researcher's decision)
print("\n📝 Adding codes (researcher-created)...")
rta.add_code(
    code_name="automation_appreciation",
    description="Participants value automation for efficiency and time-saving",
    example_segments=["I love the automation features", "It's efficient"],
    ai_suggested=False
)
rta.add_code(
    code_name="trust_anxiety",
    description="Concerns about trusting AI for important decisions",
    example_segments=["I'm worried about trusting the AI", "I don't understand how it makes decisions"],
    ai_suggested=False
)
rta.add_code(
    code_name="control_loss",
    description="Feeling of losing control or agency",
    example_segments=["I wish I had more control", "I'm losing agency"],
    ai_suggested=False
)
print("✓ 3 codes added")

# Phase 3: Searching for Themes
print("\n" + "=" * 70)
print("PHASE 3: SEARCHING FOR THEMES")
print("=" * 70)
coded_data = {
    'automation_appreciation': ["segment1", "segment2"],
    'trust_anxiety': ["segment3", "segment4"],
    'control_loss': ["segment5", "segment6"]
}
phase3 = rta.phase3_searching_themes(coded_data, use_ai_clustering=True)
print(f"\nOrganizing {phase3['n_codes_to_organize']} codes into themes...")

# Add theme (researcher defines central concept)
print("\n📝 Creating theme with central organizing concept...")
rta.add_theme(
    theme_name="automation_ambivalence",
    central_concept="Participants experience ambivalence about automation: appreciating efficiency while anxious about loss of control and trust",
    included_codes=['automation_appreciation', 'trust_anxiety', 'control_loss'],
    description="This theme captures the tension between valuing automation and worrying about its implications",
    ai_suggested_grouping=False,
    conceptual_rationale="These codes cluster together because they represent two sides of the same phenomenon - the ambivalent relationship with automation"
)
print("✓ Theme 'automation_ambivalence' created")

# Phase 4: Reviewing Themes
print("\n" + "=" * 70)
print("PHASE 4: REVIEWING THEMES")
print("=" * 70)
phase4 = rta.phase4_reviewing_themes(request_ai_stress_test=True)
print("\nReviewing theme coherence and distinctiveness...")
if 'ai_stress_tests' in phase4:
    print("\nAI Stress Test Results:")
    for theme_name, tests in phase4['ai_stress_tests']['stress_tests'].items():
        print(f"  Theme: {theme_name}")
        if tests['challenges']:
            for challenge in tests['challenges']:
                print(f"    • {challenge['type']}: {challenge['description']}")

# Phase 5: Defining and Naming
print("\n" + "=" * 70)
print("PHASE 5: DEFINING AND NAMING THEMES")
print("=" * 70)
phase5 = rta.phase5_defining_naming(request_ai_alternatives=True)
print("\nFinalizing theme definition and narrative...")

rta.finalize_theme_definition(
    theme_name="automation_ambivalence",
    final_name="Automation Ambivalence: The Trust-Control Paradox",
    definition="This theme captures participants' ambivalent relationship with automation, characterized by simultaneous appreciation for efficiency and anxiety about loss of control and trust.",
    narrative="Participants consistently expressed a paradoxical relationship with automation. While they valued the time-saving and efficiency benefits, this appreciation was tempered by concerns about trust, control, and accountability...",
    exemplar_quotes=[
        "I love the automation features, they save me so much time",
        "I'm worried about trusting the AI to make important decisions",
        "I appreciate the convenience but feel like I'm losing agency"
    ]
)
print("✓ Theme finalized")

# Phase 6: Producing the Report
print("\n" + "=" * 70)
print("PHASE 6: PRODUCING THE REPORT")
print("=" * 70)
phase6 = rta.phase6_producing_report(use_ai_structure_help=True)
print("\nGenerating methods section with AI transparency...")

# Generate methods section
methods = rta.generate_methods_section()
print("\n" + "-" * 70)
print("METHODS SECTION (excerpt):")
print("-" * 70)
print(methods[:500] + "...\n")

# Get project summary
print("\n" + "=" * 70)
print("PROJECT SUMMARY")
print("=" * 70)
summary = rta.get_project_summary()
print(f"Project: {summary['project_name']}")
print(f"Researcher: {summary['researcher']}")
print(f"Current Phase: {summary['current_phase']}")
print(f"Codes Created: {summary['n_codes']}")
print(f"Themes Created: {summary['n_themes']}")
print(f"Reflexive Memos: {summary['n_memos']}")
print(f"Audit Trail Entries: {summary['n_audit_entries']}")
print(f"AI Assistance Instances: {summary['n_ai_assistance_instances']}")
print(f"Positionality Set: {summary['positionality_set']}")

# Export audit trail
print("\n" + "=" * 70)
print("AUDIT TRAIL")
print("=" * 70)
print("\nExporting audit trail...")
audit_md = rta.export_audit_trail(format='markdown')
print("\nAudit trail saved (first 500 chars):")
print(audit_md[:500] + "...")

print("\n" + "=" * 70)
print("✓ RTA ASSISTANT DEMO COMPLETE")
print("=" * 70)
print("\nKey Features Demonstrated:")
print("  ✓ Human-in-the-loop guidance through 6 phases")
print("  ✓ Reflexive memo system")
print("  ✓ AI suggestions as provocations (not ground truth)")
print("  ✓ Complete audit trail")
print("  ✓ Transparent methods section generation")
print("  ✓ Researcher maintains analytic agency")
print("\nSee docs/RTA_ASSISTANT_GUIDE.md for full documentation")
print("=" * 70)

# Made with Bob
