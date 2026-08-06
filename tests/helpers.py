from dataclasses import dataclass
from typing import Mapping

from fastapi import FastAPI

from app.load_balancer_app import (
    LoadBalancerRuntime,
    LoadBalancerSettings,
)
from app.load_balancer_app import (
    create_app as create_load_balancer_app,
)


def monotonic_sequence(*values):
    value_iter = iter(values)

    def fake_monotonic():
        return next(value_iter)

    return fake_monotonic


@dataclass(frozen=True)
class LoadBalancerFixture:
    app: FastAPI
    runtime: LoadBalancerRuntime

    @property
    def settings(self) -> LoadBalancerSettings:
        return self.runtime.settings

    @property
    def dependencies(self):
        return self.runtime.dependencies


def load_balancer_fixture(environ: Mapping[str, str] | None = None) -> LoadBalancerFixture:
    settings = LoadBalancerSettings.from_env(
        {
            "COMPUTE_ENDPOINT_NAMES": "TEST",
            "DASHBOARD_BUCKET_ID": "",
            "DASHBOARD_PREVIEW_MODE": "",
            "SESSION_SHARED_SECRET": "",
            "SESSION_REQUIRE_VERIFIED_HF_TOKEN": "false",
            **(environ or {}),
        }
    )
    app = create_load_balancer_app(settings)
    return LoadBalancerFixture(app=app, runtime=app.state.runtime)
