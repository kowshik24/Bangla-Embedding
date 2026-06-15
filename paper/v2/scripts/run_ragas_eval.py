#!/usr/bin/env python3
"""Run the RAGAS evaluation described in the v2 manuscript.

This script is the reproducible version of the notebook experiment. It requires
langchain, langchain-openai, langchain-community, faiss-cpu, ragas, datasets, and
sentence-transformers. It does not hard-code API keys; set OPENAI_API_KEY.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


PROMPT_TEMPLATE = """আপনি একজন প্রশ্ন-উত্তর সহায়ক। প্রদত্ত তথ্যের উপর ভিত্তি করে প্রশ্নের উত্তর দিন।
যদি উত্তরটি তথ্যের মধ্যে না থাকে, তবে বলুন যে আপনি জানেন না। উত্তরটি সংক্ষিপ্ত রাখুন।
প্রশ্ন: {question}
তথ্য: {context}
উত্তর:
"""


DEFAULT_QUESTIONS = [
    "ঔপন্যাসিক ও গল্পকার জহির রায়হান পরিচালিত প্রথম চলচ্চিত্রের নাম কী ?",
    "বিশ্বের প্রথম চলচ্চিত্রের পরিচালক কে ছিলেন ?",
    "গঠন ও প্রচলন নীতির ভিত্তি তে কম্পিউটারকে কয় ভাগে ভাগ করা হয় ?",
    "দেবশ্রী রায়ের ডাক নাম কি ?",
    "প্রিন্স দ্বারকানাথ ঠাকুরের বাবার নাম কী ?",
    "ভারতের ছত্তীসগঢ় রাজ্যের রাজধানী কোথায় ?",
    "তিব্বতের শান্তিপূর্ণ মুক্তির পদক্ষেপের চুক্তি কত সালে স্বাক্ষরিত হয় ?",
    "ত্রিপিটকের মোট কয়টি পিটক আছে ?",
    "অভ্র কিবোর্ডটি কবে প্রথম তৈরি হয় ?",
    "বিশ্বের প্রথম কম্পিউটার কে তৈরি করেন ?",
]

DEFAULT_REFERENCES = [
    "কখনো আসেনি",
    "লুমিয়ের ভ্রাতৃদ্বয়",
    "তিন",
    "চুমকি",
    "রামলোচনে",
    "রায়পুর",
    "১৯৫১ খ্রিষ্টাব্দে",
    "তিন",
    "২০০৩ সালের ২৬শে মার্চ",
    "চার্লস ব্যাবেজ",
]


def mean_ci95(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return mean, 1.96 * math.sqrt(var) / math.sqrt(len(values))


def write_aggregate(rows: list[dict[str, object]], output: Path) -> None:
    metrics = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
    by_model: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_model[str(row["model"])].append(row)
    aggregate_rows = []
    for model, model_rows in sorted(by_model.items()):
        out: dict[str, object] = {"model": model, "n": len(model_rows)}
        for metric in metrics:
            vals = [float(row[metric]) for row in model_rows]
            mean, ci = mean_ci95(vals)
            out[f"{metric}_mean"] = f"{mean:.4f}"
            out[f"{metric}_ci95"] = f"{ci:.4f}"
        aggregate_rows.append(out)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregate_rows[0].keys()))
        writer.writeheader()
        writer.writerows(aggregate_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", nargs=2, metavar=("NAME", "HF_OR_PATH"), required=True)
    parser.add_argument("--output", default=Path("paper/v2/results/ragas_full_results.csv"), type=Path)
    parser.add_argument("--aggregate-output", default=Path("paper/v2/results/ragas_aggregate.csv"), type=Path)
    parser.add_argument("--knowledge-dataset", default="sajid-hossain/wikidac-bengali")
    parser.add_argument("--knowledge-split", default="train[:500]")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--generator", default="gpt-3.5-turbo")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    from datasets import Dataset, load_dataset
    from langchain.docstore.document import Document
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema.output_parser import StrOutputParser
    from langchain.schema.runnable import RunnablePassthrough
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_openai import ChatOpenAI
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    source = load_dataset(args.knowledge_dataset, split=args.knowledge_split)
    docs = [Document(page_content=text) for text in source["text"]]
    splitter = RecursiveCharacterTextSplitter(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    chunks = splitter.split_documents(docs)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = ChatOpenAI(model_name=args.generator, temperature=args.temperature)

    all_rows = []
    for model_name, model_path in args.model:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},
        )
        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": args.top_k})
        chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
        answers = []
        contexts = []
        for question in DEFAULT_QUESTIONS:
            answers.append(chain.invoke(question))
            contexts.append([doc.page_content for doc in retriever.get_relevant_documents(question)])
        dataset = Dataset.from_dict(
            {
                "question": DEFAULT_QUESTIONS,
                "answer": answers,
                "contexts": contexts,
                "ground_truths": [[ref] for ref in DEFAULT_REFERENCES],
            }
        )
        result = evaluate(
            dataset=dataset,
            metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
            raise_exceptions=False,
        )
        for row in result.to_pandas().to_dict(orient="records"):
            row["model"] = model_name
            all_rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    write_aggregate(all_rows, args.aggregate_output)
    print(f"Wrote {args.output} and {args.aggregate_output}")


if __name__ == "__main__":
    main()
