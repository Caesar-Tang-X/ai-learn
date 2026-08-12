"""
命令行入口：入库与问答。

用法：
    python cli.py ingest <文件路径> [--source 来源]
    python cli.py ask "<你的问题>"
"""
import argparse

from agents import ask as agent_ask
from core.vectorstore import VectorStore
from core.pipeline import ingest_file


def main() -> None:
    parser = argparse.ArgumentParser(description="私有 RAG 系统命令行")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="重置向量表（清空 documents 并重建）")
    p_init.add_argument("--yes", action="store_true", help="确认清空并重建，不加则只提示不执行")


    p_ingest = sub.add_parser("ingest", help="将文档入库")
    p_ingest.add_argument("path", help="文档路径（.txt/.md/.pdf）")
    p_ingest.add_argument("--source", default=None, help="来源标记")

    p_ask = sub.add_parser("ask", help="基于知识库问答")
    p_ask.add_argument("query", help="用户问题")
    p_ask.add_argument("--rerank-top-n", type=int, default=None, help="重排后返回数量")

    args = parser.parse_args()

    if args.cmd == "init":
        if not args.yes:
            print("⚠️ 此操作会清空 documents 表全部数据。确认请加 --yes")
        else:
            VectorStore().init()
            print("向量表已重置（全部文档已清空并重建）")
    elif args.cmd == "ingest":
        n = ingest_file(args.path, source=args.source)
        print(f"已入库文本块数：{n}")
    elif args.cmd == "ask":
        answer = agent_ask(args.query, rerank_top_n=args.rerank_top_n)
        print(answer)


if __name__ == "__main__":
    main()
