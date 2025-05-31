from llm_model import load_gemini_model
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List
from langchain_community.retrievers import ArxivRetriever


class ResearchPaper(BaseModel):
    titles: List[str] = Field(description="List of research paper titles")


gemini_model = load_gemini_model()


def get_research_papers(user_query: str):
    """
    Retrieves a list of research paper titles based on a user query.
    """
    parser = PydanticOutputParser(pydantic_object=ResearchPaper)

    prompt_template = PromptTemplate(
        template=(
            "You are an expert research assistant.\n"
            "Based on the user query below, identify exactly three relevant research paper titles.\n"
            "Return the output strictly as a JSON object matching this format:\n"
            "{format_instruction}\n\n"
            "User Query: {user_query}"
        ),
        input_variables=["user_query"],
        partial_variables={"format_instruction": parser.get_format_instructions()}
    )

    prompt = prompt_template.format(user_query=user_query)

    # Call model
    result = gemini_model.invoke(prompt)  # DEBUG LINE

    # Parse response
    final_result = parser.parse(result.content)
    return final_result

def format_docs_for_arxiv(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_research_papers_from_arxiv(user_query: List[str]):
    """
    Retrieves a list of research paper titles based on a user query from arXiv.
    """
    formatted_docs=[]
    retriever = ArxivRetriever(
            load_max_docs=1,
            get_ful_documents=True,
            # load_max_docs=200,
    )
    for paper in user_query:
        docs = retriever.invoke(paper)
        formatted_doc = format_docs_for_arxiv(docs)
        formatted_docs.append(formatted_doc)
    return formatted_docs

