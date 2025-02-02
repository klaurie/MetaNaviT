from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.integrations import trace_llama_index
from deepeval.auto_evaluate import auto_evaluate

# Test cases with expected behaviors
TEST_CASES = [
    {
        "query": "What standards for letters exist?",
        "expected_answer": "There are several ISO standards that specify requirements for letters",
        "context": "ISO standards define various letter formats including business letters",
        "threshold": 0.7
    },
    {
        "query": "Explain document indexing",
        "expected_answer": "Document indexing is the process of analyzing documents and creating searchable entries",
        "context": "Document indexing involves creating metadata and searchable content from documents",
        "threshold": 0.7
    }
]

trace_llama_index(auto_eval=True)


# define your chatbot
def target_model(input: str):
    pass

auto_evaluate(target_model, metrics=[AnswerRelevancyMetric(), FaithfulnessMetric()])