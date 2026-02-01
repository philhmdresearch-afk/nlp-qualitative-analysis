"""
Interactive RTA Assistant
Try the Reflexive Thematic Analysis Assistant with your own data
"""

from src.rta_assistant import RTAAssistant
import sys

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(text.center(70))
    print("=" * 70)

def print_section(text):
    """Print a section header."""
    print("\n" + "-" * 70)
    print(text)
    print("-" * 70)

def get_input(prompt, default=None):
    """Get user input with optional default."""
    if default:
        user_input = input(f"{prompt} [{default}]: ").strip()
        return user_input if user_input else default
    return input(f"{prompt}: ").strip()

def main():
    print_header("INTERACTIVE RTA ASSISTANT")
    print("\nWelcome! This tool will guide you through Reflexive Thematic Analysis.")
    print("You can use sample data or enter your own.")
    
    # Choose data source
    print("\n" + "=" * 70)
    print("DATA SOURCE")
    print("=" * 70)
    print("\n1. Use sample data (automation trust study)")
    print("2. Enter your own data")
    
    choice = get_input("\nChoose option (1 or 2)", "1")
    
    if choice == "1":
        # Use sample data
        texts = [
            "I love the automation features, they save me so much time",
            "I'm worried about trusting the AI to make important decisions",
            "The interface is intuitive but I wish I had more control",
            "Automation is great but what happens when it makes a mistake?",
            "I appreciate the convenience but feel like I'm losing agency",
            "The system works well but I don't understand how it makes decisions",
            "I trust the technology but not sure about accountability",
            "It's efficient but I worry about becoming too dependent on it"
        ]
        print(f"\n✓ Loaded {len(texts)} sample responses")
    else:
        # Get user data
        print("\nEnter your data (one response per line, empty line to finish):")
        texts = []
        while True:
            line = input(f"Response {len(texts) + 1}: ").strip()
            if not line:
                break
            texts.append(line)
        
        if not texts:
            print("No data entered. Using sample data instead.")
            texts = [
                "I love the automation features, they save me so much time",
                "I'm worried about trusting the AI to make important decisions"
            ]
    
    # Initialize project
    print_header("PROJECT SETUP")
    
    project_name = get_input("Project name", "My RTA Project")
    researcher_name = get_input("Your name", "Researcher")
    epistemology = get_input("Epistemology (contextualist/constructionist/etc)", "contextualist")
    ontology = get_input("Ontology (critical realist/relativist/etc)", "critical realist")
    
    print("\n📝 Initializing RTA project...")
    rta = RTAAssistant(
        project_name=project_name,
        researcher_name=researcher_name,
        epistemology=epistemology,
        ontology=ontology
    )
    print("✓ Project initialized")
    
    # Positionality
    print_section("POSITIONALITY STATEMENT")
    print("\nYour positionality statement is REQUIRED for RTA.")
    print("Describe your background, assumptions, and how they shape your analysis.")
    print("\nExample: 'I am a UX researcher with 10 years in tech. My HCI background")
    print("shapes how I notice usability patterns. I value user agency.'")
    
    use_example = get_input("\nUse example positionality? (y/n)", "y")
    
    if use_example.lower() == 'y':
        positionality = """I am a UX researcher with experience in technology systems. 
My background in human-computer interaction shapes how I notice patterns around 
usability and user experience. I approach this data valuing user agency and transparency."""
    else:
        print("\nEnter your positionality statement (press Enter twice when done):")
        lines = []
        while True:
            line = input()
            if not line and lines:
                break
            lines.append(line)
        positionality = "\n".join(lines)
    
    rta.set_positionality_statement(positionality)
    print("✓ Positionality statement recorded")
    
    # Phase 1: Familiarisation
    print_header("PHASE 1: FAMILIARISATION")
    
    phase1 = rta.phase1_familiarisation(texts)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  • Documents: {phase1['statistics']['n_documents']}")
    print(f"  • Total words: {phase1['statistics']['total_words']}")
    print(f"  • Average length: {phase1['statistics']['avg_length']:.1f} words")
    
    print("\n📖 Your Task:")
    for instruction in phase1['guidance']['instructions']:
        print(f"  • {instruction}")
    
    print("\n⚠️  " + phase1['guidance']['warning'])
    
    # Add memo
    print_section("REFLEXIVE MEMO")
    print("\nReflexive memos are your PRIMARY analytic tool in RTA.")
    print("Write a memo about your initial observations.")
    
    add_memo = get_input("\nAdd a reflexive memo? (y/n)", "y")
    if add_memo.lower() == 'y':
        memo = get_input("Your memo", "Initial patterns emerging around trust and control")
        rta.add_reflexive_memo(memo, memo_type="general")
        print("✓ Memo added")
    
    # Phase 2: Initial Coding
    print_header("PHASE 2: GENERATING INITIAL CODES")
    
    request_ai = get_input("\nRequest AI candidate codes? (y/n)", "y")
    phase2 = rta.phase2_initial_coding(texts, request_ai_suggestions=(request_ai.lower() == 'y'))
    
    if 'ai_suggestions' in phase2:
        print("\n🤖 AI Candidate Codes (as PROVOCATIONS, not ground truth):")
        for i, code in enumerate(phase2['ai_suggestions']['candidate_codes'][:5], 1):
            print(f"  {i}. {code['suggested_code']}")
            print(f"     Related terms: {code['related_terms']}")
        print("\n⚠️  " + phase2['ai_note'])
    
    print("\n📝 Now YOU code the data.")
    print("Create codes that capture meaningful patterns.")
    
    add_codes = get_input("\nAdd codes now? (y/n)", "y")
    if add_codes.lower() == 'y':
        while True:
            code_name = get_input("\nCode name (or press Enter to finish)", "")
            if not code_name:
                break
            
            description = get_input("Code description", "Captures a pattern in the data")
            
            rta.add_code(
                code_name=code_name,
                description=description,
                example_segments=["example1", "example2"],
                ai_suggested=False
            )
            print(f"✓ Code '{code_name}' added")
    
    # Phase 3: Searching for Themes
    print_header("PHASE 3: SEARCHING FOR THEMES")
    
    if len(rta.codes) > 0:
        print(f"\n📊 You have {len(rta.codes)} codes to organize into themes.")
        print("\nYour task: Define the CENTRAL ORGANIZING CONCEPT for each theme.")
        print("A theme is NOT just a topic - it captures something meaningful.")
        
        add_theme = get_input("\nCreate a theme? (y/n)", "y")
        if add_theme.lower() == 'y':
            theme_name = get_input("Theme name", "my_theme")
            central_concept = get_input("Central organizing concept", "The core idea that unifies this theme")
            
            rta.add_theme(
                theme_name=theme_name,
                central_concept=central_concept,
                included_codes=list(rta.codes.keys()),
                description="Theme description",
                conceptual_rationale="Why these codes form a coherent theme"
            )
            print(f"✓ Theme '{theme_name}' created")
    else:
        print("\n⚠️  No codes created yet. Skipping theme creation.")
    
    # Generate report
    print_header("GENERATING REPORT")
    
    print("\n📄 Generating methods section with AI transparency...")
    methods = rta.generate_methods_section()
    
    print("\n" + "=" * 70)
    print("METHODS SECTION (excerpt)")
    print("=" * 70)
    print(methods[:800] + "...\n")
    
    # Project summary
    print_header("PROJECT SUMMARY")
    
    summary = rta.get_project_summary()
    print(f"\n📊 Project: {summary['project_name']}")
    print(f"👤 Researcher: {summary['researcher']}")
    print(f"📍 Current Phase: {summary['current_phase']}")
    print(f"🏷️  Codes: {summary['n_codes']}")
    print(f"🎯 Themes: {summary['n_themes']}")
    print(f"📝 Memos: {summary['n_memos']}")
    print(f"📋 Audit Entries: {summary['n_audit_entries']}")
    print(f"🤖 AI Assistance: {summary['n_ai_assistance_instances']} instances")
    
    # Export options
    print_header("EXPORT OPTIONS")
    
    print("\n1. Export audit trail (JSON)")
    print("2. Export audit trail (Markdown)")
    print("3. View full methods section")
    print("4. Exit")
    
    export_choice = get_input("\nChoose option (1-4)", "4")
    
    if export_choice == "1":
        audit = rta.export_audit_trail(format='json')
        filename = f"{project_name.replace(' ', '_')}_audit.json"
        with open(filename, 'w') as f:
            f.write(audit)
        print(f"\n✓ Audit trail saved to {filename}")
    
    elif export_choice == "2":
        audit = rta.export_audit_trail(format='markdown')
        filename = f"{project_name.replace(' ', '_')}_audit.md"
        with open(filename, 'w') as f:
            f.write(audit)
        print(f"\n✓ Audit trail saved to {filename}")
    
    elif export_choice == "3":
        print("\n" + "=" * 70)
        print("FULL METHODS SECTION")
        print("=" * 70)
        print(methods)
    
    print_header("✓ RTA SESSION COMPLETE")
    
    print("\n📚 Next Steps:")
    print("  • Continue through all 6 phases with your real data")
    print("  • Write reflexive memos throughout")
    print("  • Use AI suggestions as provocations, not ground truth")
    print("  • Maintain your analytic agency")
    print("\n📖 See docs/RTA_ASSISTANT_GUIDE.md for complete documentation")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✓ Session interrupted. Your work is saved in the RTA object.")
        sys.exit(0)

# Made with Bob
