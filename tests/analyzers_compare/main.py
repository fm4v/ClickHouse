"""
1. parse fuzzer output
2. run queries one-by-one with allow_experimental_analyzer=0/1 and store result somehow
3. compare results:
    a. exceptions
    b. if no exception: compare result with alphanumerically sorted


timeout -s TERM --preserve-status 30m
clickhouse-client --max_memory_usage_in_client=1000000000 --receive_timeout=10 --receive_data_timeout_ms=10000
--stacktrace --query-fuzzer-runs=1000 --create-query-fuzzer-runs=50 --queries-file
"""

import argparse
import os
import time
from collections import OrderedDict

import pandas as pd
from tqdm import tqdm

from helpers.client import Client, QueryRuntimeException

pd.options.display.max_rows = 1000
pd.options.display.max_columns = 30
pd.options.display.max_colwidth = None


def parse_queries(queries_file: str) -> list[str]:
    with open(queries_file, "r", encoding="utf-8", errors="ignore") as f:
        queries = f.read().split("\n\n\n")

    # SHOW TABLES is useless
    queries = filter(
        lambda x: x != "SHOW TABLES" and x != "", map(lambda q: q.strip(), queries)
    )

    # remove dups
    return list(OrderedDict.fromkeys(queries))


class TimeoutError(Exception):
    pass


def timeout(func, timeout_duration=1):
    import signal

    def handler(signum, frame):
        raise TimeoutError()

    # set the timeout handler
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout_duration)
    try:
        result = func()
    except TimeoutError as exc:
        raise
    finally:
        signal.alarm(0)

    return result


def run_ch_queries(ch_client, queries: list[str], analyzer: int = 0):
    # set_analyzer = f"SET allow_experimental_analyzer={analyzer}"
    ch = Client(host="localhost", command=ch_client)

    settings = {
        "allow_experimental_analyzer": analyzer,
        "allow_suspicious_low_cardinality_types": 1,
        "max_execution_time": 10,
        "receive_timeout": 10,
        "max_memory_usage_in_client": 1000000000,
        "receive_data_timeout_ms": 10000,
    }

    results = []

    for query in tqdm(
        queries,
        desc=f"Progress (analyzer={analyzer})",
        total=len(queries),
        unit="queries",
    ):
        # with open("current_query.txt", mode="w") as f:
        #     f.write(query + "\n")
        #     f.flush()

        response_dict = {
            "query": query,
            "result": None,
            "client_returncode": None,
            "server_returncode": None,
            "stderr": None,
            "short_error": None,
            "exec_time": None,
        }
        start_time = time.perf_counter()
        try:
            # sometime queries hangs and CH timeouts doesn't stop it
            result = timeout(
                lambda: ch.query(query, settings=settings), timeout_duration=10
            )
            response_dict["result"] = result
        except QueryRuntimeException as e:
            response_dict["stderr"] = e.stderr
            response_dict["client_returncode"] = int(e.returncode)

            err_split = e.stderr.strip().split("\n")
            first_line = err_split[0]
            if err_split[0].startswith("Received exception from server"):
                first_line = err_split[1]

            short_error = first_line.split(
                "Stack trace (when copying this message, always include the lines below)",
                1,
            )[0].strip()
            response_dict["short_error"] = short_error

            # if int(e.returncode) == 199:
            #     match = re.search(r"Code: (\d+)\.", short_error)
            #     if match:
            #         code = match.group(1)
            #         response_dict["server_returncode"] = int(code)
        except TimeoutError:
            response_dict["short_error"] = "TimeoutError"
        finally:
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            response_dict["exec_time"] = execution_time

        results.append(response_dict)

    return results


def compare(r0, r1, save_to):
    df0 = pd.DataFrame(r0)
    df1 = pd.DataFrame(r1)

    df = df0.merge(df1, left_index=True, right_index=True, suffixes=("", "_new"))
    df = df.drop("query_new", axis=1)
    df.to_parquet(save_to)
    print(f"Result is saved to: {save_to}")
    return df


def queries_stats(queries, result):
    """
    total queries
    unique queries
    SELECT queries
    DROP queries
    CREATE TABLE queries
    CREATE DATABASE queries
    ALTER queries
    other DDL?

    top exceptions (group by):

    result:
    exceptions left, exceptions right
    exceptions mismatches by code
    exceptions mismatches by error

    result mismatches after ordering
    result mismatches before ordering
    """


def prepare_html(cmp_result, stats):
    pass


def drop_default_tables(ch_client):
    ch = Client(host="localhost", command=ch_client)
    res = ch.query("SELECT name FROM system.tables WHERE database='default'")

    for table in res.splitlines():
        ch.query(f"DROP TABLE IF EXISTS {table}")


def main(args: argparse.Namespace):

    for file in args.queries_files:
        filename = os.path.basename(file)
        print(f"Read queries from: {file}")
        queries = parse_queries(file)

        print("Total queries:", len(queries))

        # cleanup
        drop_default_tables(args.ch_client)

        print("Execute queries with old analyzer")
        r0 = run_ch_queries(args.ch_client, queries, analyzer=0)

        # cleanup
        drop_default_tables(args.ch_client)

        print("Execute queries with new analyzer")
        r1 = run_ch_queries(args.ch_client, queries, analyzer=1)

        result = compare(r0, r1, save_to=filename + "cmp.parquet")

        stats = queries_stats(queries, result)
        prepare_html(result, stats)

    breakpoint()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare execution result of fuzzer-generated queries with old and new analyzer"
    )

    parser.add_argument(
        "queries_files",
        nargs="+",
        type=str,
        help="The path to the file with queries generated by fuzzer",
    )
    parser.add_argument(
        "ch_client",
        default="/usr/bin/clickhouse-client",
        type=str,
        help="The path to the clickhouse-client executable",
    )

    args = parser.parse_args()

    main(args)
