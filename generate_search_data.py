from pathlib import Path
import pandas as pd

# Directory to store mock files
base_dir = Path("./metanavit_test_files")
base_dir.mkdir(parents=True, exist_ok=True)

# File definitions: (filename, content, type)
files = [
    ("climate_adaptation_2023.pdf", "Research on innovative strategies for climate resilience and sustainable development in vulnerable communities.", "pdf"),
    ("budget_proposal_pathways.docx", "Analysis of the decision-making pathways involved in preparing the 2022 budget proposal.", "docx"),
    ("population_spike_datasets.csv", "Dataset showing resource allocation models under sudden population changes in urban areas.", "csv"),
    ("renewable_energy_advancements_impact.docx", "Overview of recent renewable tech innovations and their sustainability metrics.", "docx"),
    ("autonomous_ethics_archive.pdf", "Documents on the ethical implications of autonomous systems and AI decision-making.", "pdf"),
    ("biodiversity_conservation_strategies.docx", "Studies on habitat restoration, species protection, and climate change adaptation methods.", "docx"),
    ("annual_report_2022_financials.csv", "Financial analysis, sustainability metrics, and future objectives for the fiscal year 2022.", "csv"),
    
    # Distractor files
    ("recipe_booklet.pdf", "A collection of traditional recipes with nutritional information.", "pdf"),
    ("grocery_inventory_list.csv", "Weekly inventory list for grocery items in a local supermarket.", "csv"),
    ("employee_birthdays_2024.docx", "Internal document tracking upcoming staff birthdays.", "docx")
]

# Save mock files
for filename, content, filetype in files:
    file_path = base_dir / filename
    if filetype == "pdf" or filetype == "docx":
        # Save as simple text file with appropriate extension
        file_path.write_text(content)
    elif filetype == "csv":
        # Create a very simple CSV file
        df = pd.DataFrame({"Description": [content], "Year": [2023]})
        df.to_csv(file_path, index=False)

import ace_tools as tools; tools.display_dataframe_to_user(name="MetaNaviT Mock File Index", dataframe=pd.DataFrame(files, columns=["Filename", "Description", "Type"]))
