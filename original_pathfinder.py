import json
import os
from datetime import datetime
from string import punctuation
from typing import List, Literal

# import anthropic
import cohere
import gradio as gr
import instructor
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import requests
import spacy
from datasets import load_dataset
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI as openai_llm
from langchain_openai import OpenAIEmbeddings
from nltk.corpus import stopwords
from openai import OpenAI, moderations
from pydantic import BaseModel, Field

from prompts import *

openai_key = os.environ["openai_key"]
cohere_key = os.environ["cohere_key"]
os.environ["OPENAI_API_KEY"] = os.environ["openai_key"]


def load_nlp():
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("textrank")
    try:
        stopwords.words("english")
    except:
        nltk.download("stopwords")
        stopwords.words("english")
    return nlp


gen_llm = openai_llm(temperature=0, model_name="gpt-4o-mini", openai_api_key=openai_key)
consensus_client = instructor.patch(OpenAI(api_key=openai_key))
embed_client = OpenAI(api_key=openai_key)
embed_model = "text-embedding-3-small"
embeddings = OpenAIEmbeddings(model=embed_model, api_key=openai_key)
nlp = load_nlp()


def check_mod(query):
    mod_report = moderations.create(input=query)
    for i in mod_report.results[0].categories:
        if i[1] == True:
            return True
    return False


def get_keywords(text, nlp=nlp):
    result = []
    pos_tag = ["PROPN", "ADJ", "NOUN"]
    doc = nlp(text.lower())
    for token in doc:
        if token.text in nlp.Defaults.stop_words or token.text in punctuation:
            continue
        if token.pos_ in pos_tag:
            result.append(token.text)
    return result


def load_arxiv_corpus():
    # arxiv_corpus = load_from_disk('data/')
    # arxiv_corpus.load_faiss_index('embed', 'data/astrophindex.faiss')

    # keeping it up to date with the dataset
    arxiv_corpus = load_dataset("kiyer/pathfinder_arxiv_data", split="train")
    arxiv_corpus.add_faiss_index(column="embed")
    print("loading arxiv corpus from disk")
    return arxiv_corpus


class RetrievalSystem:

    def __init__(self):

        self.dataset = arxiv_corpus
        self.client = OpenAI(api_key=openai_key)
        self.embed_model = "text-embedding-3-small"
        self.generation_client = openai_llm(
            temperature=0, model_name="gpt-4o-mini", openai_api_key=openai_key
        )
        self.hyde_client = openai_llm(
            temperature=0.5, model_name="gpt-4o-mini", openai_api_key=openai_key
        )
        self.cohere_client = cohere.Client(cohere_key)

    def make_embedding(self, text):
        str_embed = (
            self.client.embeddings.create(input=[text], model=self.embed_model)
            .data[0]
            .embedding
        )
        return str_embed

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        embeddings = self.client.embeddings.create(
            input=texts, model=self.embed_model
        ).data
        return [
            np.array(embedding.embedding, dtype=np.float32) for embedding in embeddings
        ]

    def get_query_embedding(self, query):
        return self.make_embedding(query)

    def calc_faiss(self, query_embedding, top_k=100):
        # xq = query_embedding.reshape(-1,1).T.astype('float32')
        # D, I = self.index.search(xq, top_k)
        # return I[0], D[0]
        tmp = self.dataset.search("embed", query_embedding, k=top_k)
        return [tmp.indices, tmp.scores, self.dataset[tmp.indices]]

    def rank_and_filter(
        self, query, query_embedding, top_k=10, top_k_internal=1000, return_scores=False
    ):

        if "Keywords" in self.toggles:
            self.weight_keywords = True
        else:
            self.weight_keywords = False

        if "Time" in self.toggles:
            self.weight_date = True
        else:
            self.weight_date = False

        if "Citations" in self.toggles:
            self.weight_citation = True
        else:
            self.weight_citation = False

        topk_indices, similarities, small_corpus = self.calc_faiss(
            np.array(query_embedding), top_k=top_k_internal
        )
        similarities = (
            1 / similarities
        )  # converting from a distance (less is better) to a similarity (more is better)

        if self.weight_keywords == True:

            query_kws = get_keywords(query)
            input_kws = self.query_input_keywords
            query_kws = query_kws + input_kws
            self.query_kws = query_kws
            sub_kws = [small_corpus["keywords"][i] for i in range(top_k_internal)]
            kw_weight = np.zeros((len(topk_indices),)) + 0.1

            for k in query_kws:
                for i in range(len(topk_indices)):
                    for j in range(len(sub_kws[i])):
                        if k.lower() in sub_kws[i][j].lower():
                            kw_weight[i] = kw_weight[i] + 0.1
                            # print(i, k, sub_kws[i][j])

            # kw_weight = kw_weight**0.36 / np.amax(kw_weight**0.36)
            kw_weight = kw_weight / np.amax(kw_weight)
        else:
            kw_weight = np.ones((len(topk_indices),))

        if self.weight_date == True:
            sub_dates = [small_corpus["date"][i] for i in range(top_k_internal)]
            date = datetime.now().date()
            date_diff = np.array([((date - i).days / 365.0) for i in sub_dates])
            # age_weight = (1 + np.exp(date_diff/2.1))**(-1) + 0.5
            age_weight = (1 + np.exp(date_diff / 0.7)) ** (-1)
            age_weight = age_weight / np.amax(age_weight)
        else:
            age_weight = np.ones((len(topk_indices),))

        if self.weight_citation == True:
            # st.write('weighting by citations')
            sub_cites = np.array(
                [small_corpus["cites"][i] for i in range(top_k_internal)]
            )
            temp = sub_cites.copy()
            temp[sub_cites > 300] = 300.0
            cite_weight = (1 + np.exp((300 - temp) / 42.0)) ** (-1.0)
            cite_weight = cite_weight / np.amax(cite_weight)
        else:
            cite_weight = np.ones((len(topk_indices),))

        similarities = similarities * (kw_weight) * (age_weight) * (cite_weight)

        filtered_results = [
            [topk_indices[i], similarities[i]] for i in range(len(similarities))
        ]
        top_results = sorted(filtered_results, key=lambda x: x[1], reverse=True)[:top_k]

        top_scores = [doc[1] for doc in top_results]
        top_indices = [doc[0] for doc in top_results]
        small_df = self.dataset[top_indices]

        if return_scores:
            return {doc[0]: doc[1] for doc in top_results}, small_df

        # Only keep the document IDs
        top_results = [doc[0] for doc in top_results]
        return top_results, small_df

    def generate_doc(self, query: str):
        prompt = """You are an expert astronomer. Given a scientific query, generate the abstract of an expert-level research paper
                            that answers the question. Stick to a maximum length of {} tokens and return just the text of the abstract and conclusion.
                            Do not include labels for any section. Use research-specific jargon.""".format(
            self.max_doclen
        )

        messages = [
            (
                "system",
                prompt,
            ),
            ("human", query),
        ]
        return self.hyde_client.invoke(messages).content

    def generate_docs(self, query: str):
        docs = []
        for i in range(self.generate_n):
            docs.append(self.generate_doc(query))
        return docs

    def embed_docs(self, docs: List[str]):
        return self.embed_batch(docs)

    def retrieve(
        self,
        query,
        top_k,
        return_scores=False,
        embed_query=True,
        max_doclen=250,
        generate_n=1,
        temperature=0.5,
        rerank_top_k=250,
    ):

        if max_doclen * generate_n > 8191:
            raise ValueError("Too many tokens. Please reduce max_doclen or generate_n.")

        query_embedding = self.get_query_embedding(query)

        if self.hyde == True:
            self.max_doclen = max_doclen
            self.generate_n = generate_n
            self.hyde_client.temperature = temperature
            self.embed_query = embed_query
            docs = self.generate_docs(query)
            # st.expander('Abstract generated with hyde', expanded=False).write(docs)
            doc_embeddings = self.embed_docs(docs)
            if self.embed_query:
                query_emb = self.embed_docs([query])[0]
                doc_embeddings.append(query_emb)
            query_embedding = np.mean(np.array(doc_embeddings), axis=0)

        if self.rerank == True:
            top_results, small_df = self.rank_and_filter(
                query, query_embedding, rerank_top_k, return_scores=False
            )
            # try:
            docs_for_rerank = [small_df["abstract"][i] for i in range(rerank_top_k)]
            if len(docs_for_rerank) == 0:
                return []
            reranked_results = self.cohere_client.rerank(
                query=query,
                documents=docs_for_rerank,
                model="rerank-english-v3.0",
                top_n=top_k,
            )
            final_results = []
            for result in reranked_results.results:
                doc_id = top_results[result.index]
                doc_text = docs_for_rerank[result.index]
                score = float(result.relevance_score)
                final_results.append([doc_id, "", score])
            final_indices = [doc[0] for doc in final_results]
            if return_scores:
                return {result[0]: result[2] for result in final_results}, self.dataset[
                    final_indices
                ]
            return [doc[0] for doc in final_results], self.dataset[final_indices]
            # except:
            # print('heavy load, please wait 10s and try again.')
        else:
            top_results, small_df = self.rank_and_filter(
                query, query_embedding, top_k, return_scores=return_scores
            )

        return top_results, small_df

    def return_formatted_df(self, top_results, small_df):

        df = pd.DataFrame(small_df)
        df = df.drop(columns=["umap_x", "umap_y", "cite_bibcodes", "ref_bibcodes"])
        links = [
            "[" + i + "](https://ui.adsabs.harvard.edu/abs/" + i + "/abstract)"
            for i in small_df["bibcode"]
        ]

        # st.write(top_results[0:10])
        scores = [top_results[i] for i in top_results]
        indices = [i for i in top_results]
        df.insert(1, "ADS Link", links, True)
        df.insert(2, "Relevance", scores, True)
        df.insert(3, "indices", indices, True)
        df = df[
            [
                "ADS Link",
                "Relevance",
                "date",
                "cites",
                "title",
                "authors",
                "abstract",
                "keywords",
                "ads_id",
                "indices",
                "embed",
            ]
        ]
        df.index += 1
        return df


arxiv_corpus = load_arxiv_corpus()
ec = RetrievalSystem()
print("loaded retrieval system")


def Library(papers_df):
    op_docs = ""
    for i in range(len(papers_df)):
        op_docs = (
            op_docs
            + "Paper %.0f:" % (i + 1)
            + papers_df["title"][i + 1]
            + "\n"
            + papers_df["abstract"][i + 1]
            + "\n\n"
        )

    return op_docs


def run_rag_qa(query, papers_df, question_type):

    loaders = []

    documents = []

    for i, row in papers_df.iterrows():
        content = f"Paper {i+1}: {row['title']}\n{row['abstract']}\n\n"
        metadata = {"source": row["ads_id"]}
        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=150, chunk_overlap=50, add_start_index=True
    )
    splits = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(
        documents=splits, embedding=embeddings, collection_name="retdoc4"
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 6}
    )

    if question_type == "Bibliometric":
        template = bibliometric_prompt
    elif question_type == "Single-paper":
        template = single_paper_prompt
    elif question_type == "Broad but nuanced":
        template = deep_knowledge_prompt
    else:
        template = regular_prompt
    prompt = PromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | gen_llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)
    rag_answer = rag_chain_with_source.invoke(
        query,
    )
    vectorstore.delete_collection()

    # except:
    #     st.subheader('heavy load! please wait 10 seconds and try again.')

    return rag_answer


def guess_question_type(query: str):

    gen_client = openai_llm(
        temperature=0, model_name="gpt-4o-mini", openai_api_key=openai_key
    )
    messages = [
        (
            "system",
            question_categorization_prompt,
        ),
        ("human", query),
    ]
    return gen_client.invoke(messages).content


def log_to_gist(strings):
    # Adding query logs to prevent and account for possible malicious use.
    # Logs will be deleted periodically if not needed.
    github_token = os.environ["github_token"]
    gist_id = os.environ["gist_id"]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content = f"\n{timestamp}: {' '.join(strings)}\n"
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json",
    }
    response = requests.get(f"https://api.github.com/gists/{gist_id}", headers=headers)
    if response.status_code == 200:
        existing_content = response.json()["files"]["log.txt"]["content"]
        content = existing_content + content
    data = {
        "description": "Logged Strings",
        "public": False,
        "files": {"log.txt": {"content": content}},
    }
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json",
    }
    response = requests.patch(
        f"https://api.github.com/gists/{gist_id}",
        headers=headers,
        data=json.dumps(data),
    )  # Update existing gist
    return


class OverallConsensusEvaluation(BaseModel):
    rewritten_statement: str = Field(
        ...,
        description="The query rewritten as a statement if it was initially a question",
    )
    consensus: Literal[
        "Strong Agreement Between Abstracts and Query",
        "Moderate Agreement Between Abstracts and Query",
        "Weak Agreement Between Abstracts and Query",
        "No Clear Agreement/Disagreement Between Abstracts and Query",
        "Weak Disagreement Between Abstracts and Query",
        "Moderate Disagreement Between Abstracts and Query",
        "Strong Disagreement Between Abstracts and Query",
    ] = Field(
        ...,
        description="The overall level of consensus between the rewritten statement and the abstracts",
    )
    explanation: str = Field(
        ...,
        description="A detailed explanation of the consensus evaluation (maximum six sentences)",
    )
    relevance_score: float = Field(
        ...,
        description="A score from 0 to 1 indicating how relevant the abstracts are to the query overall",
        ge=0,
        le=1,
    )


def evaluate_overall_consensus(
    query: str, abstracts: List[str]
) -> OverallConsensusEvaluation:
    prompt = f"""
    Query: {query}
    You will be provided with {len(abstracts)} scientific abstracts. Your task is to do the following:
    1. If the provided query is a question, rewrite it as a statement. This statement does not have to be true. Output this as 'Rewritten Statement:'.
    2. Evaluate the overall consensus between the rewritten statement and the abstracts using one of the following levels:
        - Strong Agreement Between Abstracts and Query
        - Moderate Agreement Between Abstracts and Query
        - Weak Agreement Between Abstracts and Query
        - No Clear Agreement/Disagreement Between Abstracts and Query
        - Weak Disagreement Between Abstracts and Query
        - Moderate Disagreement Between Abstracts and Query
        - Strong Disagreement Between Abstracts and Query
    Output this as 'Consensus:'
    3. Provide a detailed explanation of your consensus evaluation in maximum six sentences. Output this as 'Explanation:'
    4. Assign a relevance score as a float between 0 to 1, where:
        - 1.0: Perfect match in content and quality
        - 0.8-0.9: Excellent, with minor differences
        - 0.6-0.7: Good, captures main points but misses some details
        - 0.4-0.5: Fair, partially relevant but significant gaps
        - 0.2-0.3: Poor, major inaccuracies or omissions
        - 0.0-0.1: Completely irrelevant or incorrect
    Output this as 'Relevance Score:'
    Here are the abstracts:
    {' '.join([f"Abstract {i+1}: {abstract}" for i, abstract in enumerate(abstracts)])}
    Provide your evaluation in the structured format described above.
    """

    response = consensus_client.chat.completions.create(
        model="gpt-4o-mini",  # used to be "gpt-4",
        response_model=OverallConsensusEvaluation,
        messages=[
            {
                "role": "system",
                "content": """You are an assistant with expertise in astrophysics for question-answering tasks.
            Evaluate the overall consensus of the retrieved scientific abstracts in relation to a given query.
            If you don't know the answer, just say that you don't know.
            Use six sentences maximum and keep the answer concise.""",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    return response


def calc_outlier_flag(papers_df, top_k, cutoff_adjust=0.1):

    cut_dist = np.load("pfdr_arxiv_cutoff_distances.npy") - cutoff_adjust
    pts = np.array(papers_df["embed"].tolist())
    centroid = np.mean(pts, 0)
    dists = np.sqrt(np.sum((pts - centroid) ** 2, 1))
    outlier_flag = dists > cut_dist[top_k - 1]

    return outlier_flag


def make_embedding_plot(papers_df, top_k, consensus_answer, arxiv_corpus=arxiv_corpus):

    plt_indices = np.array(papers_df["indices"].tolist())

    xax = np.array(arxiv_corpus["umap_x"])
    yax = np.array(arxiv_corpus["umap_y"])

    outlier_flag = calc_outlier_flag(papers_df, top_k, cutoff_adjust=0.25)
    alphas = np.ones((len(plt_indices),)) * 0.9
    alphas[outlier_flag] = 0.5

    fig = plt.figure(figsize=(9 * 1.8, 12 * 1.8))
    plt.scatter(xax, yax, s=1, alpha=0.01, c="k")

    clkws = np.load("kw_tags.npz")
    all_x, all_y, all_topics, repeat_flag = (
        clkws["all_x"],
        clkws["all_y"],
        clkws["all_topics"],
        clkws["repeat_flag"],
    )
    for i in range(len(all_topics)):
        if repeat_flag[i] == False:
            plt.text(
                all_x[i],
                all_y[i],
                all_topics[i],
                fontsize=9,
                ha="center",
                va="center",
                bbox=dict(
                    facecolor="white",
                    edgecolor="black",
                    boxstyle="round,pad=0.3",
                    alpha=0.81,
                ),
            )
    plt.scatter(
        xax[plt_indices],
        yax[plt_indices],
        s=300 * alphas**2,
        alpha=alphas,
        c="w",
        zorder=1000,
    )
    plt.scatter(
        xax[plt_indices],
        yax[plt_indices],
        s=100 * alphas**2,
        alpha=alphas,
        c="dodgerblue",
        zorder=1001,
    )
    # plt.scatter(xax[plt_indices][outlier_flag], yax[plt_indices][outlier_flag], s=100, alpha=1., c='firebrick')
    plt.axis([0, 20, -4.2, 18])
    plt.axis("off")
    return fig


def run_pathfinder(
    query,
    top_k,
    extra_keywords,
    toggles,
    prompt_type,
    rag_type,
    ec=ec,
    progress=gr.Progress(),
):

    yield None, None, None, None, None

    search_text_list = [
        "rooting around in the paper pile...",
        "looking for clarity...",
        "scanning the event horizon...",
        "peering into the abyss...",
        "potatoes power this ongoing search...",
    ]
    gen_text_list = [
        "making the LLM talk to the papers...",
        "invoking arcane rituals...",
        "gone to library, please wait...",
        "is there really an answer to this...",
    ]

    log_to_gist(["[mod flag: " + str(check_mod(query)) + "]", query])
    if check_mod(query) == False:

        input_keywords = (
            [kw.strip() for kw in extra_keywords.split(",")] if extra_keywords else []
        )
        query_keywords = get_keywords(query)
        ec.query_input_keywords = input_keywords + query_keywords
        ec.toggles = toggles
        if rag_type == "Semantic Search":
            ec.hyde = False
            ec.rerank = False
        elif rag_type == "Semantic + HyDE":
            ec.hyde = True
            ec.rerank = False
        elif rag_type == "Semantic + CoHERE":
            ec.hyde = False
            ec.rerank = True
        elif rag_type == "Semantic + HyDE + CoHERE":
            ec.hyde = True
            ec.rerank = True

        progress(0.2, desc=search_text_list[np.random.choice(len(search_text_list))])
        rs, small_df = ec.retrieve(query, top_k=top_k, return_scores=True)
        formatted_df = ec.return_formatted_df(rs, small_df)
        yield formatted_df, None, None, None, None

        progress(0.4, desc=gen_text_list[np.random.choice(len(gen_text_list))])
        rag_answer = run_rag_qa(query, formatted_df, prompt_type)
        yield formatted_df, rag_answer["answer"], None, None, None

        progress(0.6, desc="Generating consensus")
        consensus_answer = evaluate_overall_consensus(
            query, [formatted_df["abstract"][i + 1] for i in range(len(formatted_df))]
        )
        consensus = (
            "## Consensus \n"
            + consensus_answer.consensus
            + "\n\n"
            + consensus_answer.explanation
            + "\n\n > Relevance of retrieved papers to answer: %.1f"
            % consensus_answer.relevance_score
        )
        yield formatted_df, rag_answer["answer"], consensus, None, None

        progress(0.8, desc="Analyzing question type")
        question_type_gen = guess_question_type(query)
        if "<categorization>" in question_type_gen:
            question_type_gen = question_type_gen.split("<categorization>")[1]
        if "</categorization>" in question_type_gen:
            question_type_gen = question_type_gen.split("</categorization>")[0]
        question_type_gen = question_type_gen.replace("\n", "  \n")
        qn_type = question_type_gen
        yield formatted_df, rag_answer["answer"], consensus, qn_type, None

        progress(1.0, desc="Visualizing embeddings")
        fig = make_embedding_plot(formatted_df, top_k, consensus_answer)

        yield formatted_df, rag_answer["answer"], consensus, qn_type, fig


def create_interface():
    custom_css = """
    #custom-slider-* {
        background-color: #ffffff;
    }
    """

    with gr.Blocks(css=custom_css) as demo:

        with gr.Tabs():
            # with gr.Tab("What is Pathfinder?"):
            #     gr.Markdown(pathfinder_text)
            with gr.Tab("pathfinder"):
                with gr.Accordion("What is Pathfinder? / How do I use it?", open=False):
                    gr.Markdown(pathfinder_text)

                with gr.Row():
                    query = gr.Textbox(label="Ask me anything")
                with gr.Row():
                    with gr.Column(scale=1, min_width=300):
                        top_k = gr.Slider(
                            1,
                            30,
                            step=1,
                            value=10,
                            label="top-k",
                            info="Number of papers to retrieve",
                        )
                        keywords = gr.Textbox(
                            label="Optional Keywords (comma-separated)", value=""
                        )
                        toggles = gr.CheckboxGroup(
                            ["Keywords", "Time", "Citations"],
                            label="Weight by",
                            info="weighting retrieved papers",
                            value=["Keywords"],
                        )
                        prompt_type = gr.Radio(
                            choices=[
                                "Single-paper",
                                "Multi-paper",
                                "Bibliometric",
                                "Broad but nuanced",
                            ],
                            label="Prompt Specialization",
                            value="Multi-paper",
                        )
                        rag_type = gr.Radio(
                            choices=[
                                "Semantic Search",
                                "Semantic + HyDE",
                                "Semantic + CoHERE",
                                "Semantic + HyDE + CoHERE",
                            ],
                            label="RAG Method",
                            value="Semantic + HyDE + CoHERE",
                        )
                    with gr.Column(scale=2, min_width=300):
                        img1 = gr.Image("local_files/pathfinder_logo.png")
                        btn = gr.Button("Run pfdr!")
                        # search_results_state = gr.State([])
                        ret_papers = gr.Dataframe(
                            label="top-k retrieved papers", datatype="markdown"
                        )
                        search_results_state = gr.Markdown(label="Generated Answer")
                        qntype = gr.Markdown(label="Question type suggestion")
                        conc = gr.Markdown(label="Consensus")
                        plot = gr.Plot(label="top-k in embedding space")

                        inputs = [
                            query,
                            top_k,
                            keywords,
                            toggles,
                            prompt_type,
                            rag_type,
                        ]
                        outputs = [ret_papers, search_results_state, qntype, conc, plot]
                        btn.click(fn=run_pathfinder, inputs=inputs, outputs=outputs)

    return demo


if __name__ == "__main__":

    pathfinder = create_interface()
    pathfinder.launch()
