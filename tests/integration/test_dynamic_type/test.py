import datetime
import os
import random
import shutil
import time
import re
from decimal import Decimal
from ipaddress import IPv6Address, IPv4Address
from typing import Optional

import numpy as np
import pytest
import threading

from dateutil.relativedelta import relativedelta
from jinja2 import Template, Environment
from datetime import datetime, timedelta

from helpers.cluster import ClickHouseCluster
from helpers.test_tools import assert_eq_with_retry, assert_logs_contain
from helpers.network import PartitionManager
from test_refreshable_mat_view.schedule_model import get_next_refresh_time

test_recover_staled_replica_run = 1

cluster = ClickHouseCluster(__file__)

node1_1 = cluster.add_instance(
    "node1_1",
    main_configs=["configs/remote_servers.xml"],
    user_configs=["configs/settings.xml"],
    with_zookeeper=True,
    stay_alive=True,
    macros={"shard": 1, "replica": 1},
)
node1_2 = cluster.add_instance(
    "node1_2",
    main_configs=["configs/remote_servers.xml"],
    user_configs=["configs/settings.xml"],
    with_zookeeper=True,
    stay_alive=True,
    macros={"shard": 1, "replica": 2},
)
node2_1 = cluster.add_instance(
    "node2_1",
    main_configs=["configs/remote_servers.xml"],
    user_configs=["configs/settings.xml"],
    with_zookeeper=True,
    stay_alive=True,
    macros={"shard": 2, "replica": 1},
)

node2_2 = cluster.add_instance(
    "node2_2",
    main_configs=["configs/remote_servers.xml"],
    user_configs=["configs/settings.xml"],
    with_zookeeper=True,
    stay_alive=True,
    macros={"shard": 2, "replica": 2},
)

# all_nodes = [
#     main_node,
#     dummy_node,
#     competing_node,
# ]

uuid_regex = re.compile("[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


@pytest.fixture(scope="module", autouse=True)
def started_cluster():
    try:
        cluster.start()
        yield cluster

    finally:
        cluster.shutdown()


@pytest.fixture(scope="module", autouse=True)
def setup_tables(started_cluster):
    print(node1_1.query("SELECT version()"))

    node1_1.query(f"CREATE DATABASE test_db ON CLUSTER test_cluster ENGINE = Atomic")

    node1_1.query(
        f"CREATE TABLE src1 ON CLUSTER test_cluster (a DateTime, b UInt64) ENGINE = Memory"
    )

    node1_1.query(
        f"CREATE TABLE src2 ON CLUSTER test_cluster (a DateTime, b UInt64) ENGINE = Memory"
    )

    node1_1.query(
        f"CREATE TABLE tgt1 ON CLUSTER test_cluster (a DateTime, b UInt64) ENGINE = Memory"
    )

    node1_1.query(
        f"CREATE TABLE tgt2 ON CLUSTER test_cluster (a DateTime, b UInt64) ENGINE = Memory"
    )

    # node1_1.query(
    #     f"CREATE MATERIALIZED VIEW dummy_rmv ON CLUSTER test_cluster "
    #     f"REFRESH EVERY 10 HOUR engine Memory AS select number as x from numbers(10)"
    # )


"""
IDEAS:

- FROM MV
- Null engine
- Aggregate function
- Simple agg func
- All types
- Types overflow
- functions over it (cityHash64...)
- alter

- alias from this column
- meterialize
- default

- in mergetree order by

- distributed table
- mergetree table

- filter by it
- order by it
- group by it


max_types=0
max_types=1
max_types=9999999999999999
max_types=Null

max_types='9'

"""


@pytest.mark.parametrize(
    "max_types",
    ["255"],
)
def test_basic_correct(started_cluster, request, max_types):
    def teardown():
        node1_1.query(f"""DROP TABLE test""")

    request.addfinalizer(teardown)

    node1_1.query(
        f"""CREATE TABLE test (d Dynamic(max_types={max_types})) ENGINE = Memory"""  # ORDER BY d PARTITION BY d;"""
    )

    tup = []
    cur = []
    for i in range(240):
        cur.append(i)
        tup.append(str(tuple(cur)))

    q = f"INSERT INTO test VALUES (NULL), (42), ('Hello, World!'), ([1, 2, 3]), ((1,2,3,4)), ({'),('.join(tup)})"
    node1_1.query(q)

    r = node1_1.query("SELECT d, dynamicType(d) FROM test order by d;", parse=True)
    print(r)

    # r = node1_1.query("SELECT cityHash64(), dynamicType(d) FROM test;")


@pytest.mark.parametrize(
    "max_types",
    [
        "-1",
        "-9999999999999999999",
        "Null",
        "NONE",
        "NULL",
        "",
        "-",
        "%",
        "+",
        "=",
        "0",
        "9999999999999999999",
        "``",
        "`" ":",
        "'",
        "''",
        '"',
        '""',
        "000000000000000000000000000000",
    ],
)
def test_basic_incorrect(request, started_cluster, max_types):
    def teardown():
        node1_1.query(f"""DROP TABLE IF EXISTS test""")

    request.addfinalizer(teardown)

    with pytest.raises(Exception):
        node1_1.query(
            f"""CREATE TABLE test (d Dynamic(max_types={max_types})) ENGINE = MergeTree ORDER BY d PARTITION BY d"""
        )

    assert (
        len(node1_1.query("SELECT * FROM system.tables where name='test'", parse=True))
        == 0
    )
    # node1_1.query(
    #     "INSERT INTO test VALUES (NULL), (42), ('Hello, World!'), ([1, 2, 3])",
    #     parse=True,
    # )
    #
    # r = node1_1.query("SELECT d, dynamicType(d) FROM test;", parse=True)
    # print(r)


def test_distributed(started_cluster, request):
    def teardown():
        node1_1.query(f"""DROP TABLE IF EXISTS test ON CLUSTER test_cluster""")
        node1_1.query(f"""DROP TABLE IF EXISTS dist_test ON CLUSTER test_cluster""")

    request.addfinalizer(teardown)

    node1_1.query(
        f"""CREATE TABLE test ON CLUSTER test_cluster (d Dynamic)
        ENGINE = ReplicatedMergeTree('/clickhouse/tables/test1/{{shard}}/test_table', '{{replica}}') ORDER BY (d);"""
    )
    node1_1.query(
        f"""CREATE TABLE dist_test ON CLUSTER test_cluster as test ENGINE=Distributed(test_cluster, default, test, 1)"""
    )

    node1_1.query(
        "INSERT INTO dist_test VALUES (NULL), (42), ('Hello, World!'), ([1, 2, 3])"
    )

    node1_1.query(
        "INSERT INTO test VALUES (NULL), (421), ('Hello, World!!'), ([1, 2, 3, 4])"
    )

    node2_2.query(
        "INSERT INTO test VALUES ((NULL, NULL)), (423), ('Hello, World!!!'), ([1, 2, 3, 5])"
    )

    r = node1_1.query("SELECT d, dynamicType(d) FROM dist_test;", parse=True)
    print(r)

    r = node1_1.query("SELECT d, dynamicType(d) FROM test;", parse=True)
    print(r)

    r = node1_2.query("SELECT d, dynamicType(d) FROM test;", parse=True)
    print(r)

    r = node2_2.query("SELECT d, dynamicType(d) FROM test;", parse=True)
    print(r)


def test_aggr():
    r = node1_1.query(
        """
        SELECT finalizeAggregation(dynamic_state), uniqMerge(dynamic_state) FROM (
            SELECT CAST(state, 'Dynamic') dynamic_state FROM (
                SELECT uniqState(number) as state FROM numbers(1000)
                )
            );
        """,
        parse=True,
    )
    print(r)


def test_alters(request, started_cluster):
    def teardown():
        node1_1.query(f"""DROP TABLE test""")

    request.addfinalizer(teardown)

    node1_1.query(
        f"""CREATE TABLE test (a UInt8, d String) ENGINE = MergeTree ORDER BY a"""
    )

    node1_1.query(
        "INSERT INTO test VALUES (1, 'Hello'), (3,'42'), (10,'Hello, World!'), (20,'[1, 2, 3]')",
    )

    node1_1.query(f"""ALTER TABLE test ALTER COLUMN d TYPE Dynamic""")

    r = node1_1.query("SELECT a, d, dynamicType(d) FROM test;", parse=True)
    print(r)

    node1_1.query(f"""ALTER TABLE test ALTER COLUMN d TYPE String""")

    r = node1_1.query("SELECT a, d FROM test;", parse=True)
    print(r)


def test_alters_rename(request, started_cluster):
    def teardown():
        node1_1.query(f"""DROP TABLE test""")

    request.addfinalizer(teardown)

    node1_1.query(
        f"""CREATE TABLE test (a UInt8, d Dynamic, c Dynamic MATERIALIZEd tuple(d))
        ENGINE = MergeTree ORDER BY a"""
    )

    node1_1.query(
        "INSERT INTO test VALUES (10000, NULL), (1, 'Hello'), (3,'42'), (10,'Hello, World!'), (20,'[1, 2, 3]')",
    )

    # node1_1.query(f"""ALTER TABLE test RENAME COLUMN d TO XXX""")
    #
    # r = node1_1.query("SELECT a, XXX, dynamicType(XXX) FROM test;", parse=True)
    # print(r)
    # node1_1.query(f"""ALTER TABLE test RENAME COLUMN XXX TO d""")
    #
    # r = node1_1.query("SELECT a, d, dynamicType(d) FROM test;", parse=True)
    # print(r)
    #
    # node1_1.query(f"""ALTER TABLE test CLEAR COLUMN d """)

    r = node1_1.query("SELECT * FROM test;", parse=True)

    node1_1.query(f"""DETACH TABLE test""")
    node1_1.query(f"""ATTACH TABLE test""")
    r = node1_1.query("SELECT *, c FROM test;", parse=True)
    print(r)
    r = node1_1.query("OPTIMIZE TABLE test")
    r = node1_1.query("SELECT *, c FROM test;", parse=True)
    print(r)

    r = node1_1.query("OPTIMIZE TABLE test FINAL")
    r = node1_1.query("SELECT *, c FROM test;", parse=True)
    print(r)
    node1_1.query(
        "INSERT INTO test VALUES (10000, NULL), (1, 'Hello'), (3,'42'), (3,42), (10,'Hello, World!'), (20,'[1, 2, 3]')",
    )
    r = node1_1.query("SELECT *, c FROM test;", parse=True)
    print(r)
    r = node1_1.query("OPTIMIZE TABLE test")
    r = node1_1.query("SELECT *, c FROM test;", parse=True)
    print(r)

    r = node1_1.query("OPTIMIZE TABLE test FINAL")
    r = node1_1.query("SELECT *, c FROM test;", parse=True)

    r = node1_1.query("SELECT *, c FROM test ORDER BY d ASC;", parse=True)
    print(r)

    r = node1_1.query("SELECT *, c FROM test ORDER BY d DESC;", parse=True)
    print(r)

    print("-----")
    r = node1_1.query("SELECT *, c FROM test WHERE d is NULL;", parse=True)
    print(r)

    print("-----")
    r = node1_1.query(
        "SELECT *, c FROM test WHERE d is NULL and d =='Hello';", parse=True
    )
    print(r)

    print("-----")
    r = node1_1.query(
        "SELECT *, c FROM test WHERE d is NULL or d =='Hello';", parse=True
    )
    print(r)

    print("-----")
    r = node1_1.query(
        "SELECT *, c FROM test WHERE d is NULL or toTypeName(d) =='Array()';",
        parse=True,
    )
    print(r)

    print("-----")
    r = node1_1.query(
        "SELECT a,d, dynamicType(d), c, dynamicType(c) FROM test",
        parse=True,
    )
    print(r)
    r = node1_1.query(
        "SELECT *, c FROM test WHERE d ==42 and c = ('42')",
        parse=True,
    )
    print(r)


def test_default(request, started_cluster):
    def teardown():
        node1_1.query(f"""DROP TABLE test""")

    request.addfinalizer(teardown)

    node1_1.query(
        f"""CREATE TABLE test (a UInt8, d Dynamic, c Dynamic DEFAULT 1337
        )
        ENGINE = MergeTree ORDER BY a"""
    )

    node1_1.query(
        "INSERT INTO test (a, d) VALUES (10000, NULL), (1, 'Hello'), (3,'42'), (10,'Hello, World!'), (20,'[1, 2, 3]')",
    )

    node1_1.query(
        "INSERT INTO test (a, d, c) VALUES (10000, NULL, '2'), (1, 'Hello', 3), (3,'42', [1,2,3]), (10,'Hello, World!',(2,333)), (20,'[1, 2, 3]', (1,2,3,2,2,1))",
    )
    # node1_1.query(f"""ALTER TABLE test RENAME COLUMN d TO XXX""")
    #
    # r = node1_1.query("SELECT a, XXX, dynamicType(XXX) FROM test;", parse=True)
    # print(r)
    # node1_1.query(f"""ALTER TABLE test RENAME COLUMN XXX TO d""")
    #
    # r = node1_1.query("SELECT a, d, dynamicType(d) FROM test;", parse=True)
    # print(r)
    #
    # node1_1.query(f"""ALTER TABLE test CLEAR COLUMN d """)

    r = node1_1.query("SELECT * FROM test;", parse=True)

    node1_1.query(f"""DETACH TABLE test""")
    node1_1.query(f"""ATTACH TABLE test""")
    r = node1_1.query("SELECT * FROM test;", parse=True)
    print(r)
    r = node1_1.query("OPTIMIZE TABLE test")
    r = node1_1.query("SELECT * FROM test;", parse=True)
    print(r)

    node1_1.query(f"""ALTER TABLE test CLEAR COLUMN c""")
    r = node1_1.query("SELECT * FROM test;", parse=True)
    print(r)


def test_simple(request, started_cluster):
    def teardown():
        node1_1.query(f"""DROP TABLE test""")

    request.addfinalizer(teardown)

    node1_1.query(
        f"""CREATE TABLE test (d Dynamic)
           ENGINE = Memory"""
    )

    node1_1.query(
        "INSERT INTO test VALUES (42)",
    )
    r = node1_1.query(
        "SELECT d, dynamicType(d) FROM test",
        parse=True,
    )
    print(r)

    r = node1_1.query(
        "SELECT * FROM test WHERE d == 42",
        parse=True,
    )
    print(r)


def test_simple2(request, started_cluster):
    def teardown():
        node1_1.query(f"""DROP TABLE test""")

    request.addfinalizer(teardown)

    node1_1.query(
        f"""CREATE TABLE test (d Dynamic(max_types=255))
           ENGINE = Memory"""
    )

    vals = [
        -128,
        127,
        0,
        255,
        32768,
        32767,
        65535,
        2147483648,
        2147483647,
        4294967295,
        -9223372036854775808,
        9223372036854775807,
        18446744073709551615,
        -9007199254740991,
        9007199254740991,
        18446744073709551615,
    ]

    # l = ",".join([f"({v})" for v in vals])
    #
    # node1_1.query(
    #     f"INSERT INTO test VALUES {l}",
    # )

    """
                               d dynamicType(d)
    0                   -128          Int64
    1                    127          Int64
    2                      0          Int64
    3                    255          Int64
    4                  32768          Int64
    5                  32767          Int64
    6                  65535          Int64
    7             2147483648          Int64
    8             2147483647          Int64
    9             4294967295          Int64
    10  -9223372036854775808          Int64
    11   9223372036854775807          Int64
    12  18446744073709551615         UInt64
    13     -9007199254740991          Int64
    14      9007199254740991          Int64
    15  18446744073709551615         UInt64
    FAILED
    """

    for v in vals:
        node1_1.query(
            f"INSERT INTO test VALUES ({v})",
        )

    r = node1_1.query(
        "SELECT d, dynamicType(d) FROM test",
        parse=True,
    )
    print(r)

    r = node1_1.query(
        "SELECT * FROM test WHERE d == 42",
        parse=True,
    )
    print(r)


from hypothesis import example, given, strategies as st, settings, assume

all_strategies = st.one_of(
    st.integers(),
    # st.none(),
    # st.binary(),
    st.just("NULL"),
    st.booleans(),
    # st.decimals(),
    st.lists(
        st.one_of(
            st.floats(),
            st.text(),
            st.none(),
            st.lists(st.one_of(st.integers(), st.text(), st.none())),
        )
    ),
    # st.times(),
    # st.ip_addresses(),
    st.dates(),
    st.floats(),
    # st.uuids(),
    # st.characters(),
)


@given(
    st.one_of(
        # st.text(),
        st.integers(),
        # st.none(),
        # st.binary(),
        st.just("NULL"),
        st.booleans(),
        # st.decimals(),
        st.lists(
            st.one_of(
                st.floats(),
                st.text(),
                st.lists(st.one_of(st.integers(), st.text(), st.none())),
            )
        ),
        st.tuples(),
        # st.times(),
        # st.ip_addresses(),
        st.dates(),
        st.floats(),
        # st.uuids(),
        # st.characters(),
        st.dictionaries(
            keys=st.one_of(st.integers(), st.text(), st.floats()),
            values=st.one_of(st.integers(), st.text(), st.floats()),
        ),
    )
)
@settings(deadline=None, max_examples=5000)
def test_hypo(s):
    assume(s != "'")
    assume(s is not None)
    assume(s != IPv6Address("::1"))
    assume(s != IPv4Address("0.0.0.0"))
    assume(s is not Decimal("sNaN"))
    assume(s != ["'"])

    node1_1.query(
        f"""CREATE TABLE IF NOT EXISTS test (d Dynamic(max_types=255))
           ENGINE = Memory"""
    )

    node1_1.query(
        f"INSERT INTO test VALUES ({s})",
    )

    print(s)
    r = node1_1.query(
        "SELECT d, dynamicType(d) typ FROM test",
        parse=True,
    )
    print(r)

    pass
