from __future__ import annotations

import contextlib
import dataclasses
import random
import re
import string
import subprocess
import textwrap
import time
from contextlib import contextmanager
from io import StringIO
from typing import Optional

import hypothesis.strategies as st
import pandas as pd
import pytest
from hypothesis import settings, assume, Phase
from hypothesis.stateful import (
    RuleBasedStateMachine,
    rule,
    Bundle,
    initialize,
    run_state_machine_as_test,
    consumes,
    precondition,
    multiple,
)

from helpers.cluster import ClickHouseCluster
from helpers.test_tools import TSV

cluster = ClickHouseCluster(__file__)
node = cluster.add_instance("node")


@pytest.fixture(scope="module")
def started_cluster():
    try:
        cluster.start()
        yield cluster
    finally:
        cluster.shutdown()


# @pytest.fixture(scope="session", autouse=True)
# def init_ch():
#     # https://hub.docker.com/r/clickhouse/clickhouse-server/
#     run_cmd(
#         "docker run -d --name ch-test --network=host --ulimit nofile=262144:262144 "
#         "clickhouse/clickhouse-server"
#     )
#     ChClient().wait_ch_ready()
#     yield
#     run_cmd("docker rm -f ch-test")


# def test_file_path_escaping(started_cluster):
#     node.query(
#         "CREATE DATABASE IF NOT EXISTS test ENGINE = Ordinary",
#         settings={"allow_deprecated_database_ordinary": 1},
#     )
#     # node.query(
#     #     """
#     #     CREATE TABLE test.`T.a_b,l-e!` (`~Id` UInt32)
#     #     ENGINE = MergeTree() PARTITION BY `~Id` ORDER BY `~Id` SETTINGS min_bytes_for_wide_part = 0, replace_long_file_name_to_hash = 1, max_file_name_length=1;
#     #     """
#     # )
#     # node.query("""INSERT INTO test.`T.a_b,l-e!` VALUES (1);""")
#     # node.query("""ALTER TABLE test.`T.a_b,l-e!` FREEZE;""")
#
#     # node.exec_in_container(
#     #     [
#     #         "bash",
#     #         "-c",
#     #         "ls -alht /var/lib/clickhouse/data/test/*",
#     #     ]
#     # )
#     #
#     # # node.exec_in_container(
#     # #     [
#     # #         "bash",
#     # #         "-c",
#     # #         "test -f /var/lib/clickhouse/data/test/T%2Ea_b%2Cl%2De%21/1_1_1_0/%7EId.bin",
#     # #     ]
#     # # )
#     # node.exec_in_container(
#     #     [
#     #         "bash",
#     #         "-c",
#     #         "test -f /var/lib/clickhouse/shadow/1/data/test/*",
#     #     ]
#     # )
#     #
#     # # node.exec_in_container(
#     # #     [
#     # #         "bash",
#     # #         "-c",
#     # #         "test -f /var/lib/clickhouse/shadow/1/data/test/T%2Ea_b%2Cl%2De%21/1_1_1_0/%7EId.bin",
#     # #     ]
#     # # )
#
#     node.query("CREATE DATABASE IF NOT EXISTS `test 2` ENGINE = Atomic")
#     node.query(
#         """
#         CREATE TABLE `test 2`.`T.a_b,l-e!` UUID '12345678-1000-4000-8000-000000000001' (`~Id` UInt32, x UInt64,
#         INDEX idx_value x TYPE minmax GRANULARITY 8192)
#         ENGINE = MergeTree() PARTITION BY `~Id` ORDER BY `~Id` SETTINGS min_bytes_for_wide_part = 0, replace_long_file_name_to_hash = 1, max_file_name_length=3;
#         """
#     )
#     node.query(
#         """INSERT INTO `test 2`.`T.a_b,l-e!` VALUES (1, 10),(2, 20),(3, 30),(4, 40),(5, 50),(6, 60),(7, 70),(8, 01),;"""
#     )
#     node.query(
#         """INSERT INTO `test 2`.`T.a_b,l-e!` VALUES (10, 10),(20, 20),(30, 30),(40, 40),(50, 50),(60, 60),(70, 70),(80, 01),;"""
#     )
#     # node.query("""ALTER TABLE `test 2`.`T.a_b,l-e!` FREEZE;""")
#
#     node.exec_in_container(
#         [
#             "bash",
#             "-c",
#             "ls -R /var/lib/clickhouse/store/**",
#         ]
#     )
#     # Check symlink
#     node.exec_in_container(["bash", "-c", "ls -R -laht /var/lib/clickhouse/data/**/**"])
#     node.exec_in_container(
#         [
#             "bash",
#             "-c",
#             "ls -R -laht /var/lib/clickhouse/data/**/**",
#         ]
#     )
#     # node.exec_in_container(
#     #     [
#     #         "bash",
#     #         "-c",
#     #         "ls -R -laht /var/lib/clickhouse/shadow/**/**",
#     #     ]
#     # )
#
#     r = node.query("""SELECT * FROM `test 2`.`T.a_b,l-e!` where `~Id`<10""")
#     print(r)


class CommandError(Exception):
    def __init__(self, cmd, return_code, stderr):
        self.cmd = cmd
        self.return_code = return_code
        self.stderr = stderr
        super().__init__(
            f"Command [{cmd}] failed with return code {return_code}:\n {stderr}"
        )


class ChError(Exception):
    def __init__(self, sql, error_message):
        self.sql = sql
        self.error_message = error_message
        self.error_code = 0

        pattern = re.compile(r"Code:\s+(\d+)\.")
        match = pattern.search(error_message)

        if match:
            self.error_code = int(match.group(1))

        super().__init__(
            f"CH SQL [{sql}] failed with error code {self.error_code}:\n {error_message}"
        )


def run_cmd(bash_cmd):
    result = subprocess.run(
        bash_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if result.returncode != 0:
        raise CommandError(bash_cmd, result.returncode, result.stderr.strip())

    return result.stdout.strip()


class Password:
    pass


@dataclasses.dataclass
class PlainPassword(Password):
    password: str


@dataclasses.dataclass
class NoPassword(Password):
    password = None


@dataclasses.dataclass
class NotIdentifiedPassword(Password):
    password = None


@dataclasses.dataclass
class User:
    name: str
    password: Password = None

    id: str | None = None

    def update(self, alter_user: User):
        if alter_user.name:
            self.name = alter_user.name

        if alter_user.password:
            self.password = alter_user.password

    def __hash__(self):
        return hash(self.name)


class ChClient:
    def __init__(self, user: User = None):
        self.user: User = User(name="default", password=PlainPassword(password=""))
        if user:
            self.user = user

    def exec(self, cmd, parse=False):
        psw_str = ""

        if isinstance(self.user.password, NotIdentifiedPassword):
            psw_str = ""
        if isinstance(self.user.password, NoPassword):
            psw_str = '--password ""'
        if isinstance(self.user.password, PlainPassword):
            psw_str = f'--password "{self.user.password.password}"'

        if parse:
            cmd += " FORMAT TabSeparatedWithNames"

        print(cmd)
        try:
            result = run_cmd(
                f"clickhouse-client " f"-u {self.user.name} {psw_str} " f'-q "{cmd}"'
            )
        except CommandError as e:
            raise ChError(cmd, e.stderr)

        if parse:
            return pd.read_csv(
                StringIO(result), sep="\t", keep_default_na=False, na_values=[]
            )
        else:
            return result.strip()

    def wait_ch_ready(self, timeout=10):
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                if self.exec("SELECT 1") == "1":
                    print("Connection status changed.")
                    return True
            except Exception:
                continue
            time.sleep(0.3)
        print("Timeout reached without connection status change.")
        raise Exception()

    def try_login(self):
        self.exec("SELECT 1")

    def create_user(self, user: User):
        ddl = "CREATE USER {username} {password_ddl}"

        password_ddl = ""
        if isinstance(user.password, NotIdentifiedPassword):
            password_ddl = "NOT IDENTIFIED"
        elif isinstance(user.password, NoPassword):
            password_ddl = "IDENTIFIED WITH no_password"
        elif isinstance(user.password, PlainPassword):
            password_ddl = "IDENTIFIED WITH plaintext_password BY '{}'".format(
                user.password.password
            )

        sql = ddl.format(username=user.name, password_ddl=password_ddl)
        self.exec(sql)

    def alter_user(self, user: User, alter_user: User):
        ddl = "ALTER USER {username} {rename_user_ddl} {password_ddl}"

        rename_user_ddl = ""
        if alter_user.name is not None and user.name != alter_user.name:
            rename_user_ddl = f"RENAME TO {alter_user.name}"

        password_ddl = ""
        if alter_user.password:
            if isinstance(alter_user.password, NotIdentifiedPassword):
                password_ddl = "NOT IDENTIFIED"
            elif isinstance(alter_user.password, NoPassword):
                password_ddl = "IDENTIFIED WITH no_password"
            elif isinstance(alter_user.password, PlainPassword):
                password_ddl = "IDENTIFIED WITH plaintext_password BY '{}'".format(
                    alter_user.password.password
                )

        sql = ddl.format(
            username=user.name,
            password_ddl=password_ddl,
            rename_user_ddl=rename_user_ddl,
        )

        self.exec(sql)

    def drop(self, user):
        self.exec(f"DROP USER {user.name}")

    def get_users(self):
        """
        name:                 v
        id:                   f94e0f37-ecd4-aea5-bac3-2f6bed7e5ba7
        storage:              local_directory
        auth_type:            no_password
        auth_params:          {}
        host_ip:              ['::/0']
        host_names:           []
        host_names_regexp:    []
        host_names_like:      []
        default_roles_all:    1
        default_roles_list:   []
        default_roles_except: []
        grantees_any:         1
        grantees_list:        []
        grantees_except:      []
        default_database:
        :return:
        """
        return self.exec(
            "SELECT * FROM system.users WHERE name != 'default'", parse=True
        )

    def delete_all_users(self):
        users = self.exec(
            "SELECT * FROM system.users WHERE name != 'default'", parse=True
        )

        if "nan" in users["name"]:
            users.to_csv("users.csv")
            breakpoint()

        for user_name in users["name"]:
            self.exec(f"DROP USER {user_name}")


@st.composite
def unique_name_strategy(
    draw, name_strategy=st.text(string.ascii_letters, min_size=1, max_size=2)
):
    used_names = draw(st.shared(st.builds(set), key="used names"))
    name = draw(name_strategy.filter(lambda x: x not in used_names))
    used_names.add(name)
    return name


user_password_strategy = st.one_of(
    st.builds(NoPassword),
    st.builds(
        PlainPassword,
        password=st.text(alphabet=string.ascii_letters, min_size=1, max_size=10),
    ),
)

# create user strategy
new_user = st.builds(User, name=unique_name_strategy(), password=user_password_strategy)

alter_user = st.builds(
    User,
    name=st.one_of(st.none(), unique_name_strategy()),
    password=st.one_of(st.none(), user_password_strategy),
)


class CHUserTest(RuleBasedStateMachine):
    created_user = Bundle("created_user")
    deleted_user = Bundle("deleted_user")

    @initialize()
    def init(self):
        ChClient().delete_all_users()

    @rule(
        target=created_user,
        user=st.one_of(new_user, consumes(deleted_user)),
    )
    def create_user(self, user):
        ChClient().create_user(user)
        ChClient(user).try_login()
        return user

    @rule(target=created_user, user=consumes(created_user), alter_user=alter_user)
    def update_user(self, user: User, alter_user: User):
        ChClient().alter_user(user, alter_user)
        user.update(alter_user)
        ChClient(user).try_login()
        return user

    @rule(target=deleted_user, user=consumes(created_user))
    def drop_user(self, user):
        ChClient().drop(user)
        with pytest.raises(ChError) as e:
            ChClient(user).try_login()
            assert e.error_code == 516
        return user

    @rule(user=st.one_of(created_user))
    def cant_create_existed_user(self, user):
        with pytest.raises(ChError) as e:
            ChClient().create_user(user)
            assert e.error_code == 493

    @rule(user=st.one_of(new_user, deleted_user), alter_user=alter_user)
    def cant_alter_non_existent_user(self, user, alter_user):
        with pytest.raises(ChError) as e:
            ChClient().alter_user(user, alter_user)
            assert e.error_code == 192

    @rule(user=st.one_of(new_user, deleted_user))
    def cant_drop_not_existing_user(self, user):
        with pytest.raises(ChError) as e:
            ChClient().drop(user)
            assert e.error_code == 192

    @rule(user=st.one_of(new_user, deleted_user))
    def cant_login_non_existent_user(self, user):
        with pytest.raises(ChError) as e:
            ChClient(user).try_login()
            assert e.error_code == 516


"""
- create mergetree in init
  - random columns
  - random partitioning with/without
  - random order by
  - indexes
  - projections
  - sample by

- rule upload random data
- rule add column
- rule alter column
- rule delete column
- rule column
    ADD COLUMN
    DROP COLUMN
    MODIFY COLUMN
    RENAME COLUMN
    CLEAR COLUMN
    COMMENT COLUMN

- rule optimize table
- rule optimize table final
- rule detach part
- rule attach part
- rule change settings
    replace_long_file_name_to_hash
    max_file_name_length
    min_bytes_for_full_part_storage
    min_bytes_for_wide_part=0

- rule rename table
- ttl

- rule projections
    ADD PROJECTION
    DROP PROJECTION
    MATERIALIZE PROJECTION
    CLEAR PROJECTION

-rule partition
    ATTACH PARTITION
    DETACH PARTITION
    DROP PARTITION
    MOVE PARTITION
    REPLACE PARTITION

- rule indexes
    ADD INDEX
    DROP INDEX
    MATERIALIZE INDEX
    CLEAR INDEX

"""


def q(*args, **kwargs):
    kwargs["database"] = "test"
    query = textwrap.dedent(args[0]).strip()
    return node.query(query, *args, **kwargs)


types = [
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "Int128",
    "Int256",
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
    "UInt128",
    "UInt256",
    "Float32",
    "Float64"
    # 'Decimal(P, S)',
    # 'Decimal32(S)', 'Decimal64(S)', 'Decimal128(S)', 'Decimal256(S)',
    "Bool",
    "String",
    # 'FixedString(N),
    "UUID",
    "Date",
    "Date32",
    "DateTime",
    # 'DateTime64(precision)'
    "Enum8",
    "Enum16",
    # Array(T)
    # Tuple(T1, T2, ...)
    # Nullable(T)
    "IPv4",
    "IPv6",
    # LowCardinality(T)
    # 'Map(key, value)'
    # Nested(Name1 Type1, Name2 Type2, ...)
    # SimpleAggregateFunction(name, types_of_arguments...)
    # AggregateFunction(name, types_of_arguments...)
    # 'Point',
    # 'Ring',
    # 'Polygon'
    # 'MultiPolygon'
    # 'Interval',
    # 'Dynamic'
]
projections = []


int_types = st.sampled_from(
    ["Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64"]
)
float_types = st.sampled_from(["Float32", "Float64"])


@st.composite
def st_decimal(draw):
    s = draw(st.integers(1, 76))
    p = draw(st.integers(1, s))
    return f"Decimal({s},{p})"


fixed_string_type = st.builds(lambda p: f"FixedString({p})", st.integers(1, 255))

date_types = st.sampled_from(["Date", "DateTime", "DateTime64(3)"])

# Composite strategy for column types
st_base_column_types = st.one_of(
    int_types,
    float_types,
    st_decimal(),
    date_types,
    st.just("String"),
    fixed_string_type,
    st.just("UUID"),
    st.just("IPv4"),
    st.just("IPv6"),
)


array_types = st.recursive(
    st_base_column_types, lambda s: st.builds(lambda t: f"Array({t})", s), max_leaves=3
)

st_column_types = st.one_of(st_base_column_types, array_types)

nullable = st.builds(
    lambda t: f"Nullable({t})",
    st_column_types,
)

st_column_types = st.one_of(st_column_types, nullable)

# Strategy for column nameswhitespace = ' \t\n\r\v\f'
whitespace = " \t\r\v\f"
ascii_lowercase = "abcdefghijklmnopqrstuvwxyz"
ascii_uppercase = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ascii_letters = ascii_lowercase + ascii_uppercase
digits = "0123456789"
# punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
# excluded: `/\'
punctuation = r"""!"#$%&()*+,-.:;<=>?@[]^_{|}~"""
printable = digits + ascii_letters + punctuation + whitespace

st_column_name = st.text(alphabet=printable, min_size=1, max_size=1000)
st_table_name = st.text(alphabet=printable, min_size=1, max_size=1000)

# Composite strategy for a single column
column_strategy = st.builds(
    lambda name, type: {"name": name, "type": type},
    st_column_name,
    st_column_types,
)

# Strategy for a list of unique columns
st_columns = st.lists(
    column_strategy, min_size=1, max_size=500, unique_by=lambda x: x["name"]
)


order_by = st.one_of(st.just("tuple()"), st.just("1"))
# partition_by = st.one_of(columns)
# sample_by = st.one_of(columns)
primary_key = st.one_of(order_by)
# sample_by = st.one_of()


# Settings
# min_bytes_for_wide_part=0
# min_bytes_for_full_part_storage
# max_file_name_length=8,
# replace_long_file_name_to_hash = 1


class DeepPartsMcWrecker(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()

    @property
    def columns_str(self):
        return ", ".join([f"`{col['name']}` {col['type']}" for col in self.columns])

    def columns_str_no_name(self):
        return ", ".join([f"c{i} {col['type']}" for i, col in enumerate(self.columns)])

    @contextlib.contextmanager
    def validate_data(
        self,
        columns: Optional[list[str]] = None,
        except_columns: Optional[list[str]] = None,
        partitions: Optional[list[str]] = None,
        except_partitions: Optional[list[str]] = None,
        parts: Optional[list[str]] = None,
        except_parts: Optional[list[str]] = None,
    ):
        columns_str = "*"
        if columns:
            columns_str = ", ".join([f"`{col}`" for col in columns])

        if except_columns:
            except_columns_str = ", ".join([f"`{col}`" for col in except_columns])
            columns_str += f" EXCEPT {except_columns_str}"

        def join_(list_):
            return ", ".join([f"'{x}'" for x in list_])

        filter_str = ""
        if partitions or parts or except_partitions or except_parts:
            filter_str = " WHERE "

        if partitions:
            filter_str += f"""_partition_id IN ({join_(partitions)},)"""

        if except_partitions:
            filter_str += f"""_partition_id NOT IN ({join_(except_partitions)},)"""

        if parts:
            filter_str += f"""_part IN ({join_(parts)},)"""

        if except_parts:
            filter_str += f"""_part NOT IN ({join_(except_parts)},)"""

        try:
            before = q(
                f"SELECT groupBitXor(cityHash64({columns_str})) h FROM `{self.table_name}` {filter_str}"
            ).strip()
            before_cnt = q(
                f"SELECT count() FROM `{self.table_name}` {filter_str}"
            ).strip()
            yield
        finally:
            after = q(
                f"SELECT groupBitXor(cityHash64({columns_str})) h FROM `{self.table_name}` {filter_str}"
            ).strip()
            after_cnt = q(
                f"SELECT count() FROM `{self.table_name}` {filter_str}"
            ).strip()

            assert before == after and before_cnt == after_cnt

    def get_uniq_parts(self):
        return TSV(
            q(
                f"""
        SELECT
            distinct name,
        FROM system.parts
        WHERE table = '{self.table_name}' and database = 'test' and active = 1
        ORDER BY name
        """
            )
        ).lines

    def get_partition_of_part(self, part_name):
        return TSV(
            q(
                f"""
        SELECT
            distinct partition_id,
        FROM system.parts
        WHERE table = '{self.table_name}' and database = 'test' and name = '{part_name}'
        ORDER BY partition_id
        """
            )
        ).lines[0]

    def get_partition_of_detached_part(self, part_name):
        return TSV(
            q(
                f"""
        SELECT
            distinct partition_id,
        FROM system.detached_parts
        WHERE table = '{self.table_name}' and database = 'test' and name = '{part_name}'
        ORDER BY partition_id
        """
            )
        ).lines[0]

    def get_uniq_partitions(self):
        return TSV(
            q(
                f"""
        SELECT
            distinct partition,
        FROM system.parts
        WHERE table = '{self.table_name}' and database = 'test' and active = 1
        ORDER BY partition
        """
            )
        ).lines

    detached_partition = Bundle("detached_partition")
    detached_part = Bundle("detached_part")

    @initialize(columns=st_columns, table_name=st_table_name)
    def init(self, columns: list[dict], table_name: str):
        random.seed(int(time.time()))
        self.columns = columns
        self.table_name = table_name

        node.query(
            "DROP DATABASE IF EXISTS test",
        )
        node.query(
            "CREATE DATABASE IF NOT EXISTS test ENGINE = Atomic",
        )

        self.order_by = random.sample(columns, random.randint(0, len(columns)))
        if len(self.order_by) == 0:
            order_by_str = "tuple()"
        else:
            order_by_str = ", ".join([f"`{col['name']}`" for col in self.order_by])

        with_partition_by = random.random() < 0.7
        with_sample_by = random.random() < 0.7
        with_primary_key = random.random() < 0.7 and len(self.order_by) >= 1

        self.partition_key = []
        partition_key_str = ""
        if with_partition_by:
            self.partition_key = random.sample(columns, random.randint(1, len(columns)))

            if (
                len(self.partition_key) == 1
                and "int" in self.partition_key[0]["type"].lower()
            ):
                partition_key_str = f"PARTITION BY `{self.partition_key[0]['name']}`"
            else:
                partition_key = ", ".join(
                    [f"`{col['name']}`" for col in self.partition_key]
                )
                partition_key_str = f"PARTITION BY cityHash64({partition_key}) % {random.randint(0, 10000)}"

        primary_key = None
        primary_key_str = ""
        if with_primary_key:
            primary_key = self.order_by[0 : random.randint(1, len(self.order_by))]
            if len(primary_key) >= 1:
                primary_key = ", ".join([f"`{col['name']}`" for col in primary_key])
                primary_key_str = f"PRIMARY KEY ({primary_key})"

        table_ddl = f"""
        CREATE TABLE `{table_name}` (
        {self.columns_str}
        )
        ENGINE=MergeTree()
        ORDER BY ({order_by_str})
        {primary_key_str}
        {partition_key_str}
        """
        table_ddl = textwrap.dedent(table_ddl).strip()
        q(table_ddl)

    @rule()
    def insert_rows(self):
        q(
            f"""
        INSERT INTO `{self.table_name}`
        SELECT * FROM generateRandom('{self.columns_str_no_name()}', {random.randint(0, 100)}, {random.randint(0, 100)}, {random.randint(0, 10)})
        LIMIT {random.randint(0, 5000)}
        SETTINGS max_partitions_per_insert_block=1000000
        """
        )

    @rule(table_name=st_table_name)
    def rename_table(self, table_name):
        assume(table_name != self.table_name)
        with self.validate_data():
            q(f"""RENAME TABLE `{self.table_name}` TO `{table_name}`""")
            self.table_name = table_name

    @rule()
    def optimize_table(self):
        with self.validate_data():
            q(f"OPTIMIZE TABLE `{self.table_name}`")

    @rule()
    def optimize_final_table(self):
        with self.validate_data():
            q(f"OPTIMIZE TABLE `{self.table_name}` FINAL")

    @rule()
    @precondition(lambda self: len(self.partition_key) >= 1)
    def optimize_table_partition(self):
        partitions = self.get_uniq_partitions()
        if len(partitions) == 0:
            return

        partition = random.choice(partitions)
        with self.validate_data():
            q(f"OPTIMIZE TABLE `{self.table_name}` PARTITION '{partition}'")

    @rule(target=detached_part)
    def detach_part(self):
        parts = self.get_uniq_parts()
        if len(parts) == 0:
            # it means we do not create new bundle
            return multiple()

        part = random.choice(parts)
        with self.validate_data(except_parts=[part]):
            q(f"ALTER TABLE `{self.table_name}` DETACH PART '{part}'")

        return part

    @rule(part=detached_part)
    def attach_part(self, part):
        parts = self.get_uniq_parts()
        # new data can be inserted between detach and attach and new part will be created
        if part in parts:
            raise Exception()

        # exclude partition because part are merged instantly and we cannot filter by it's name
        partition = self.get_partition_of_detached_part(part)
        with self.validate_data(except_partitions=[partition]):
            q(f"ALTER TABLE `{self.table_name}` ATTACH PART '{part}'")

    @rule()
    def detach_attach_part(self):
        parts = self.get_uniq_parts()
        if len(parts) == 0:
            return

        with self.validate_data():
            part = random.choice(parts)
            q(f"ALTER TABLE `{self.table_name}` DETACH PART '{part}'")
            q(f"ALTER TABLE `{self.table_name}` ATTACH PART '{part}'")

    @rule(target=detached_partition)
    @precondition(lambda self: len(self.partition_key) >= 1)
    def detach_partition(self):
        partitions = self.get_uniq_partitions()
        if len(partitions) == 0:
            # it means we do not create new bundle
            return multiple()

        partition = random.choice(partitions)
        with self.validate_data(except_partitions=[partition]):
            q(f"ALTER TABLE `{self.table_name}` DETACH PARTITION '{partition}'")

        return partition

    @rule(partition=consumes(detached_partition))
    def attach_partition(self, partition):
        with self.validate_data(except_partitions=[partition]):
            q(f"ALTER TABLE `{self.table_name}` ATTACH PARTITION '{partition}'")

    @rule()
    @precondition(lambda self: len(self.partition_key) >= 1)
    def detach_attach_partition(self):
        partitions = self.get_uniq_partitions()
        if len(partitions) == 0:
            return

        with self.validate_data():
            partition = random.choice(partitions)
            q(f"ALTER TABLE `{self.table_name}` DETACH PARTITION '{partition}'")
            q(f"ALTER TABLE `{self.table_name}` ATTACH PARTITION '{partition}'")

    #
    # @rule()
    # def clear_index_in_partition(self):
    #     # check data
    #     # ALTER TABLE table_name [ON CLUSTER cluster] CLEAR INDEX index_name IN PARTITION partition_expr
    #     pass


def test_run(started_cluster):
    phases = [
        Phase.explicit,  #: controls whether explicit examples are run.
        Phase.reuse,  #: controls whether previous examples will be reused.
        Phase.generate,  #: controls whether new examples will be generated.
        # Phase.target,  #: controls whether examples will be mutated for targeting.
        # Phase.shrink,   #: controls whether examples will be shrunk.
        # Phase.explain,  #: controls whether Hypothesis attempts to explain test failures.
    ]
    set_ = settings(
        deadline=None,
        report_multiple_bugs=False,
        stateful_step_count=50,
        max_examples=50,
        phases=phases,
    )

    run_state_machine_as_test(DeepPartsMcWrecker, settings=set_)


# def test_example_run(started_cluster):
#     state = DeepPartsMcWrecker()
#     state.init(
#         columns=[
#             {"name": "Zg6%uL", "type": "Array(Int64)"},
#             {"name": "go:[ Q;%XNww", "type": "String"},
#             {"name": "1c3RqQ%tkPu&=a?fdkD:L$W2+pL", "type": "UInt8"},
#             {"name": "+mvC\t", "type": "DateTime64(3)"},
#         ],
#         table_name="qyGncT<-P6#i",
#     )
#     # state.detach_attach_part()
#     state.insert_rows()
#     # state.optimize_final_table()
#     # state.optimize_final_table()
#     detached_part_0 = state.detach_part()
#     # state.detach_attach_part()
#     # state.detach_part()
#     # state.insert_rows()
#     # state.detach_attach_part()
#     # state.attach_part(part=detached_part_0)
#     print("before attach:")
#     print(q(f"SELECT count() h FROM `{state.table_name}`"))
#
#     state.attach_part_cmp_by_partitions(part=detached_part_0)
#
#     print("after attach:")
#     print(q(f"SELECT count() h FROM `{state.table_name}`"))
#
#     state.teardown()
