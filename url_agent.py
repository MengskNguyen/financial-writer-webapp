from langchain.document_loaders.url import UnstructuredURLLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
import dotenv

dotenv.load_dotenv()

map_prompt = """Below is a section of a financial website
    Write a concise summary.
    {text}
    % CONCISE SUMMARY:
"""

combine_prompt = """
    Your goal is to write a financial article.
    
    A good financial article is easy to understand.
    Be sure to write an article contains:
    a title with no more than {title_character_count} characters, description with no more than {desc_character_count} characters and body form 300 words to {body_word_count} words
    
    % INFORMATION:
    {text}
    
    % YOUR RESPONSE:
"""


def urls_summarize_agent(urls: list, title_character_count: int, desc_character_count: int, body_word_count: int):
    loader = UnstructuredURLLoader(urls=urls)
    doc = loader.load()
    if len(doc) != 0:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60)
        doc_split = text_splitter.split_documents(doc)
        map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text", "topic"])
        print("map_prompt_template: ", map_prompt_template.get_prompts())
        combine_prompt_template = PromptTemplate(template=combine_prompt,
                                                 input_variables=["title_character_count", "desc_character_count"
                                                     , "body_word_count", "text"])

        print("combine_prompt_template: ", combine_prompt_template.get_prompts())

        llm = OpenAI(temperature=.7)

        chain = load_summarize_chain(
            llm=llm,
            chain_type="map_reduce",
            map_prompt=map_prompt_template,
            combine_prompt=combine_prompt_template,
            verbose=True
        )

        output = chain({
            "input_documents": doc_split,
            "title_character_count": title_character_count,
            "desc_character_count": desc_character_count,
            "body_word_count": body_word_count
        })

        return output

    else:
        print("Doc is empty")


output = urls_summarize_agent(urls=['https://finance.yahoo.com/news/broadening-us-market-rally-gets-224650088.html'],
                              title_character_count=60, desc_character_count=160, body_word_count=500)

print(output['output_text'])
