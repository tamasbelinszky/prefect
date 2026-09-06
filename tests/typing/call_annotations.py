"""Call-site typing contracts for `@task` and `@flow` calls.

Never executed — checked by pyright and mypy in CI. Each `assert_type` pins
how a public call shape resolves; `--verifytypes` cannot see these (#16547
was a pyright call-site regression, #17379 a mypy-only one). Calls that must
STAY errors carry suppression comments: both checkers flag unused
suppressions, so a lost error fails CI too.
"""

# pyright: reportUnnecessaryTypeIgnoreComment=error

from functools import partial

from typing_extensions import assert_type

from prefect import flow, task
from prefect.futures import PrefectFuture, PrefectFutureList
from prefect.states import State
from prefect.task_runners import ThreadPoolTaskRunner


@task
def sync_task(x: int, label: str = "") -> int:
    return x


@task
async def async_task(x: int, label: str = "") -> int:
    return x


@flow
def sync_flow(x: int) -> int:
    return x


@flow
async def async_flow(x: int) -> int:
    return x


@task(name="configured-task")
async def configured_async_task(x: int) -> int:
    return x


@task(retries=1)
def configured_sync_task(x: int) -> int:
    return x


@flow(name="configured-flow")
async def configured_async_flow(x: int) -> int:
    return x


def check_sync_task_calls() -> None:
    assert_type(sync_task(1), int)
    assert_type(sync_task(1, label="a"), int)
    assert_type(sync_task(1, return_state=True), State[int])
    assert_type(sync_task(1, return_state=False), int)
    assert_type(sync_task(1, wait_for=[]), int)
    assert_type(sync_task(1, wait_for=[], return_state=True), State[int])


async def check_async_task_calls() -> None:
    assert_type(await async_task(1), int)
    assert_type(await async_task(1, return_state=False), int)
    assert_type(await async_task(1, wait_for=[]), int)
    assert_type(async_task(1, return_state=True), State[int])


def check_task_submit() -> None:
    assert_type(sync_task.submit(1), PrefectFuture[int])
    assert_type(async_task.submit(1), PrefectFuture[int])
    assert_type(sync_task.submit(1, return_state=True), State[int])


def check_task_map() -> None:
    assert_type(sync_task.map([1, 2]), PrefectFutureList[int])
    assert_type(async_task.map([1, 2]), PrefectFutureList[int])


def check_sync_flow_calls() -> None:
    assert_type(sync_flow(1), int)
    assert_type(sync_flow(1, return_state=True), State[int])


async def check_async_flow_calls() -> None:
    assert_type(await async_flow(1), int)
    assert_type(await async_flow(1, return_state=True), State[int])


async def check_configured_decorator_calls() -> None:
    assert_type(configured_sync_task(1), int)
    assert_type(await configured_async_task(1), int)
    assert_type(configured_async_task(1, return_state=True), State[int])
    assert_type(configured_async_task.submit(1), PrefectFuture[int])
    assert_type(configured_async_task.map([1, 2]), PrefectFutureList[int])
    assert_type(await configured_async_flow(1), int)
    assert_type(await configured_async_flow(1, return_state=True), State[int])


def check_wrong_calls_stay_errors() -> None:
    sync_task("nope")  # type: ignore[call-overload]
    sync_task()  # type: ignore[call-overload]
    sync_flow("nope")  # type: ignore[call-overload]


# --- #16938: a concrete task runner is accepted by `@flow` -------------------
# `ThreadPoolTaskRunner(max_workers=3)` leaves its result type unsolved, so the
# `task_runner` parameter must accept it covariantly.
@flow(task_runner=ThreadPoolTaskRunner(max_workers=3))
def flow_with_runner(x: int) -> int:
    return x


@flow(task_runner=ThreadPoolTaskRunner())
async def async_flow_with_runner(x: int) -> int:
    return x


async def check_task_runner_flows() -> None:
    assert_type(flow_with_runner(1), int)
    assert_type(await async_flow_with_runner(1), int)


# --- #14756: a task stays callable and keeps its parameter names -------------
def check_task_is_callable_and_keeps_signature() -> None:
    # `partial` over the decorated task must resolve; it regressed under mypy
    # 1.11 when `Task.__call__` stopped presenting a single callable signature.
    partial(sync_task, 1)
    partial(sync_task, x=1)

    # Parameter names survive the decorator -- this is what an editor shows on
    # hover, and what keyword calls depend on.
    assert_type(sync_task(x=1, label="a"), int)
    assert_type(sync_flow(x=1), int)
    sync_task(nope=1)  # type: ignore[call-overload]


# --- #17379 / #16770: `@task` / `@flow` on methods ---------------------------
# `__get__` binds the instance at runtime, so the bound signature drops `self`.
class Integrator:
    @task
    def sync_method(self, branch: str) -> int:
        return len(branch)

    @task
    async def async_method(self, branch: str) -> int:
        return len(branch)

    @staticmethod
    @task
    def static_method(x: int) -> int:
        return x

    @flow
    async def run(self, branch: str) -> int:
        # the exact shape reported in #17379 and #16770
        assert_type(self.sync_method(branch), int)
        assert_type(self.sync_method(branch="main"), int)
        assert_type(await self.async_method(branch), int)
        assert_type(self.async_method(branch, return_state=True), State[int])
        return 0


def check_method_tasks(integrator: Integrator) -> None:
    assert_type(integrator.sync_method("main"), int)
    assert_type(integrator.sync_method(branch="main"), int)

    # class-level access keeps `self`, matching runtime
    assert_type(Integrator.sync_method(integrator, "main"), int)

    # staticmethod tasks stay callable through the class
    assert_type(Integrator.static_method(1), int)

    integrator.sync_method()  # type: ignore[call-overload]
    integrator.sync_method(1)  # type: ignore[call-overload]
