import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Status(Enum):
    Passed = "passed"
    Failed = "failed"
    Skipped = "skipped"
    Unimplemented = "unimplemented"


class Action(Enum):
    Run = "run"
    Skip = "skip"


class Case:
    problem_name: str
    case_name: str
    description: str
    points: int
    extra_credit: bool

    def __call__(self) -> None:
        raise ValueError("Cannot call the base class directly")

    def is_close(self, a: float, b: float, tol: float) -> bool:
        return abs(a - b) < tol


@dataclass
class Result:
    case: Case
    status: Status
    message: Optional[str]


@dataclass
class Report:
    results: list[Result] = field(default_factory=list)

    def add_result(self, result: Result) -> None:
        self.results.append(result)


@dataclass
class CollectedCase:
    case: Case
    action: Action


@dataclass
class Grader:
    cases: list[Case]

    def collect(self, problem_name: Optional[str]) -> list[CollectedCase]:
        n_cases_to_run = 0
        collected_cases = []
        for case in self.cases:
            if problem_name is None or case.problem_name == problem_name:
                action = Action.Run
                n_cases_to_run += 1
            else:
                action = Action.Skip
            collected_cases.append(CollectedCase(case=case, action=action))

        if problem_name is not None and n_cases_to_run == 0:
            valid_problem_names = {case.problem_name for case in self.cases}
            raise ValueError(
                f"No problem with name '{problem_name}'. Please select one of {valid_problem_names}"
            )
        return collected_cases

    def run(self, problem_name: Optional[str] = None) -> None:
        collected_cases = self.collect(problem_name)
        report = Report()
        for collected_case in collected_cases:
            case = collected_case.case
            action = collected_case.action
            message = None
            if action == Action.Run:
                try:
                    case()
                    status = Status.Passed
                except NotImplementedError:
                    status = Status.Unimplemented
                except (AssertionError, Exception):
                    status = Status.Failed
                    message = traceback.format_exc()
            else:
                status = Status.Skipped
            result = Result(case=case, status=status, message=message)
            report.add_result(result)

        self.print_report(report)

    def get_status_emoji(self, status: Status) -> str:
        if status == Status.Passed:
            return f"{'✅':2}"
        elif status == Status.Failed:
            return f"{'❌':2}"
        elif status == Status.Skipped:
            return f"{'⏭️':4}"
        elif status == Status.Unimplemented:
            return f"{'⏳':2}"
        raise ValueError(f"Unknown status: {status}")

    def print_report(self, report: Report) -> None:
        print("==========================================")

        for result in report.results:
            if result.status == Status.Failed:
                print(
                    f"<| Exception while testing problem: {result.case.problem_name} |>"
                )
                print(result.message)

        n_earned = 0
        n_possible = 0
        for result in report.results:
            case = result.case
            status = result.status
            status_emoji = self.get_status_emoji(status)
            n_case_earned = case.points if status == Status.Passed else 0
            n_case_possible = case.points if not case.extra_credit else 0
            extra_credit_string = "(extra credit)" if case.extra_credit else ""
            print(
                f"• {status_emoji} {case.problem_name}-{case.case_name} ==> "
                f"[{n_case_earned}/{n_case_possible}] {status.value} {extra_credit_string}"
            )
            if status == Status.Passed:
                n_earned += case.points
            if not case.extra_credit:
                n_possible += case.points

        print("==========================================")
        print(f"Score: {n_earned}/{n_possible}")
        print("==========================================")
