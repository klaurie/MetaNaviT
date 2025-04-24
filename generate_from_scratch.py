import os
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import ContextConstructionConfig

file_paths = []
for root, dirs, files in os.walk("data"):
    for file in files:
        if file.endswith(".pdf") or file.endswith(".docx") or file.endswith(".doc") or file.endswith(".txt"):  
            full_path = os.path.join(root, file)
            print(f"Adding file: {full_path}")
            file_paths.append(full_path)

synthesizer = Synthesizer()
synthesizer.generate_goldens_from_docs(
    document_paths=file_paths,
    include_expected_output=True,
    max_goldens_per_context=1,
    context_construction_config=ContextConstructionConfig(
        critic_model="gpt-4o-mini"
    )
)

print(synthesizer.synthetic_goldens)