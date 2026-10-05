from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.core.errors import ConflictError
from app.pilots.service import PilotOperationsService
from app.core.clock import FrozenClock
from app.database import get_connection, transaction


PROTOCOL = {
    "code": "gait-assist",
    "name": "外骨骼步态体验方案",
    "capability": "gait-assist",
    "parameter_schema": {
        "minutes": {"type": "integer", "required": True, "minimum": 1, "maximum": 30},
        "assist_level": {"type": "number", "required": False, "minimum": 0.0, "maximum": 1.0},
        "scene": {"type": "string", "required": True, "choices": ["stairs", "flat"]},
    },
    "default_parameters": {"assist_level": 0.4},
    "max_runtime_seconds": 300,
    "max_attempts": 2,
}


def submit_payload(key: str, *, user: str = "pilot-operator-1", priority: int = 50) -> dict:
    return {
        "protocol_code": "gait-assist",
        "project_code": "expo-health-a",
        "requested_by": user,
        "parameters": {"minutes": 8, "scene": "stairs"},
        "priority": priority,
        "idempotency_key": key,
    }


def create_protocol(client) -> None:
    response = client.post("/api/pilots/protocols?actor=administrator", json=PROTOCOL)
    assert response.status_code == 201, response.text


def test_protocol_submission_idempotency_and_parameter_validation(client):
    create_protocol(client)
    first = client.post("/api/pilots/sessions", json=submit_payload("request-000001"))
    second = client.post("/api/pilots/sessions", json=submit_payload("request-000001"))
    assert first.status_code == second.status_code == 202
    assert first.json()["id"] == second.json()["id"]
    invalid = submit_payload("request-000002")
    invalid["parameters"]["minutes"] = 50
    rejected = client.post("/api/pilots/sessions", json=invalid)
    assert rejected.status_code == 422


def test_priority_capability_claim_and_observation_version(client):
    create_protocol(client)
    low = client.post("/api/pilots/sessions", json=submit_payload("priority-low", priority=10)).json()
    high = client.post("/api/pilots/sessions", json=submit_payload("priority-high", priority=90)).json()
    no_match = client.post("/api/pilots/sessions/claim", json={"site_code": "w0", "capabilities": ["other"], "lease_seconds": 60})
    assert no_match.status_code == 200 and no_match.json()["session"] is None
    claimed = client.post("/api/pilots/sessions/claim", json={"site_code": "w1", "capabilities": ["gait-assist"], "lease_seconds": 60})
    assert claimed.status_code == 200
    assert claimed.json()["session"]["id"] == high["id"]
    completed = client.post(
        f"/api/pilots/sessions/{high['id']}/complete",
        json={"site_code": "w1", "observation": {"value": 3.14}, "metrics": {"seconds": 2}},
    )
    assert completed.status_code == 200
    details = client.get(f"/api/pilots/session-details/{high['id']}").json()
    assert details["status"] == "succeeded"
    assert details["current_observation_version"] == 1
    assert len(details["observations"]) == 1
    assert low["status"] == "queued"


def test_quota_cancel_retry_priority_and_batch_interventions(client):
    create_protocol(client)
    quota = client.put(
        "/api/pilots/quotas?actor=administrator",
        json={"subject_type": "user", "subject_key": "limited", "max_queued": 1, "max_running": 1, "daily_submissions": 2},
    )
    assert quota.status_code == 200
    one = client.post("/api/pilots/sessions", json=submit_payload("quota-one", user="limited")).json()
    blocked = client.post("/api/pilots/sessions", json=submit_payload("quota-two", user="limited"))
    assert blocked.status_code == 409
    cancelled = client.post(f"/api/pilots/sessions/{one['id']}/cancel", json={"actor": "administrator", "reason": "项目暂停"})
    assert cancelled.status_code == 200 and cancelled.json()["status"] == "cancelled"
    retried = client.post(f"/api/pilots/sessions/{one['id']}/retry", json={"actor": "administrator", "reason": "项目恢复", "priority": 95})
    assert retried.status_code == 200 and retried.json()["priority"] == 95
    other = client.post("/api/pilots/sessions", json=submit_payload("batch-other", user="other-user")).json()
    batch = client.post(
        "/api/pilots/sessions/batch",
        json={"session_ids": [one["id"], other["id"]], "operation": "priority", "actor": "administrator", "reason": "临床合作方临时到场", "priority": 99},
    )
    assert batch.status_code == 200
    assert len(batch.json()["succeeded"]) == 2
    details = client.get(f"/api/pilots/session-details/{one['id']}").json()
    assert [item["action"] for item in details["interventions"]] == ["cancel", "retry", "priority"]


def test_failure_backoff_and_expired_lease_recovery(client):
    from app.database import init_db

    init_db()
    clock = FrozenClock(datetime(2026, 9, 26, 2, 0, tzinfo=UTC))
    service = PilotOperationsService(get_connection(), clock)
    service.create_protocol(PROTOCOL, "administrator")
    first = service.submit(submit_payload("failure-000001"))
    claimed = service.claim("site-a", ["gait-assist"], 10)
    assert claimed and claimed["id"] == first["id"]
    failed = service.fail(first["id"], "site-a", "sensor_unstable", "步态传感器读数不稳定", True)
    assert failed["status"] == "queued"
    assert failed["available_at"] > failed["updated_at"]
    clock.advance(seconds=2)
    claimed_again = service.claim("site-a", ["gait-assist"], 10)
    assert claimed_again and claimed_again["attempt_count"] == 2
    clock.advance(seconds=11)
    recovered = service.recover_expired()
    assert recovered["exhausted"] == [first["id"]]
    details = service.get_session(first["id"])
    assert details["status"] == "failed"
    assert details["interventions"][-1]["action"] == "lease_recovery"


def _claim_running(service, key: str, site: str = "site-a", lease: int = 10) -> dict:
    service.submit(submit_payload(key))
    claimed = service.claim(site, ["gait-assist"], lease)
    assert claimed and claimed["status"] == "running"
    return claimed


def test_stop_request_blocks_renewal_and_observations_until_site_confirms(client):
    service = PilotOperationsService(get_connection(), FrozenClock(datetime(2026, 10, 5, 1, 0, tzinfo=UTC)))
    service.create_protocol(PROTOCOL, "administrator")
    session = _claim_running(service, "stop-confirm-001")
    sid = session["id"]

    requested = service.cancel(sid, "safety-officer", "老年观众要求立即停止")
    assert requested["status"] == "cancel_requested"
    assert requested["cancel_requested_by"] == "safety-officer"
    assert requested["cancel_reason"] == "老年观众要求立即停止"
    assert requested["cancel_requested_at"]
    assert requested["finished_at"] is None

    # 停止请求后：续租、观察回执、失败上报全部被阻止。
    for call in (
        lambda: service.heartbeat(sid, "site-a", 10),
        lambda: service.complete(sid, "site-a", {"v": 1}, {}),
        lambda: service.fail(sid, "site-a", "e", "m", True),
    ):
        with pytest.raises(ConflictError):
            call()

    # 其它站点不能代为确认；重复停止也不允许。
    with pytest.raises(ConflictError):
        service.confirm_cancel(sid, "site-b")
    with pytest.raises(ConflictError):
        service.cancel(sid, "safety-officer", "又一条停止")

    confirmed = service.confirm_cancel(sid, "site-a")
    assert confirmed["status"] == "cancelled"
    assert confirmed["termination_kind"] == "cancelled_confirmed"
    assert confirmed["cancel_confirmed_at"]
    assert confirmed["finished_at"] == confirmed["cancel_confirmed_at"]
    assert confirmed["lease_owner"] == ""

    # 终态不可逆：重复确认、迟到完成/心跳都不能改变决定。
    with pytest.raises(ConflictError):
        service.confirm_cancel(sid, "site-a")
    with pytest.raises(ConflictError):
        service.complete(sid, "site-a", {"late": True}, {})
    with pytest.raises(ConflictError):
        service.heartbeat(sid, "site-a", 10)

    details = service.get_session(sid)
    assert details["termination_kind"] == "cancelled_confirmed"
    assert details["termination_label"] == "主动停止"
    assert details["cancellation"]["path"] == "confirmed"
    assert details["cancellation"]["requested_by"] == "safety-officer"
    assert [item["action"] for item in details["interventions"]] == ["cancel", "cancel_confirmed"]


def test_unconfirmed_stop_converges_to_cancelled_on_lease_expiry_without_requeue(client):
    clock = FrozenClock(datetime(2026, 10, 5, 2, 0, tzinfo=UTC))
    service = PilotOperationsService(get_connection(), clock)
    service.create_protocol(PROTOCOL, "administrator")
    session = _claim_running(service, "stop-timeout-001")
    sid = session["id"]
    service.cancel(sid, "safety-officer", "站点无响应的紧急停止")

    # 租约未过期时恢复任务不处理该场次。
    assert service.recover_expired()["cancel_timeouts"] == []
    assert service.get_session(sid)["status"] == "cancel_requested"

    clock.advance(seconds=11)
    first = service.recover_expired()
    assert first == {"recovered": [], "exhausted": [], "cancel_timeouts": [sid]}

    details = service.get_session(sid)
    assert details["status"] == "cancelled"
    assert details["termination_kind"] == "cancelled_timeout"
    assert details["termination_label"] == "超时停止"
    assert details["cancellation"]["path"] == "timeout"
    assert details["cancel_timeout_at"] == details["finished_at"]
    assert details["cancel_requested_by"] == "safety-officer"
    assert details["interventions"][-1]["action"] == "cancel_timeout_recovery"

    # 恢复任务重跑是幂等的，终态不再变化；迟到的完成也无法覆盖。
    rerun = service.recover_expired()
    assert rerun == {"recovered": [], "exhausted": [], "cancel_timeouts": []}
    with pytest.raises(ConflictError):
        service.complete(sid, "site-a", {"late": True}, {})
    assert service.get_session(sid)["termination_kind"] == "cancelled_timeout"


def test_session_details_distinguish_completion_and_stop_paths(client):
    service = PilotOperationsService(get_connection(), FrozenClock(datetime(2026, 10, 5, 3, 0, tzinfo=UTC)))
    service.create_protocol(PROTOCOL, "administrator")

    completed = _claim_running(service, "kind-completed")
    service.complete(completed["id"], "site-a", {"ok": True}, {})
    assert service.get_session(completed["id"])["termination_label"] == "正常完成"

    queued = service.submit(submit_payload("kind-direct"))
    direct = service.cancel(queued["id"], "operator", "排队态直接取消")
    assert direct["termination_kind"] == "cancelled_direct"
    assert service.get_session(queued["id"])["cancellation"]["path"] == "direct"


def test_startup_migration_heals_legacy_cancel_requested(client):
    import sqlite3
    from app.database import database_path, init_db
    from app.core.clock import to_storage

    # client fixture 已按新版 schema 建库；删除试点相关表，改造成只有旧版列的库。
    path = database_path()
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys=OFF")
    connection.executescript(
        """
        DROP TABLE IF EXISTS pilot_observations;
        DROP TABLE IF EXISTS pilot_interventions;
        DROP TABLE IF EXISTS pilot_sessions;
        DROP TABLE IF EXISTS pilot_protocols;
        CREATE TABLE pilot_protocols (
            id INTEGER PRIMARY KEY AUTOINCREMENT, code TEXT UNIQUE, name TEXT, capability TEXT,
            version INTEGER, parameter_schema_json TEXT, default_parameters_json TEXT,
            max_runtime_seconds INTEGER, max_attempts INTEGER, active INTEGER,
            created_by TEXT, created_at TEXT, updated_at TEXT
        );
        CREATE TABLE pilot_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, protocol_id INTEGER, project_code TEXT,
            requested_by TEXT, parameters_json TEXT, parameter_digest TEXT, priority INTEGER,
            idempotency_key TEXT, status TEXT, attempt_count INTEGER, max_attempts INTEGER,
            available_at TEXT, lease_owner TEXT, lease_expires_at TEXT,
            current_observation_version INTEGER, last_error_code TEXT, last_error_message TEXT,
            version INTEGER, started_at TEXT, finished_at TEXT, created_at TEXT, updated_at TEXT,
            UNIQUE(requested_by, idempotency_key)
        );
        CREATE TABLE pilot_observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER, version INTEGER,
            observation_json TEXT, metrics_json TEXT, observation_digest TEXT,
            created_by TEXT, created_at TEXT
        );
        CREATE TABLE pilot_interventions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER, actor TEXT, action TEXT,
            reason TEXT, before_json TEXT, after_json TEXT, batch_key TEXT, created_at TEXT
        );
        """
    )
    now = to_storage(datetime(2026, 10, 4, 0, 0, tzinfo=UTC))
    connection.execute(
        "INSERT INTO pilot_protocols(code,name,capability,version,parameter_schema_json,default_parameters_json,"
        "max_runtime_seconds,max_attempts,active,created_by,created_at,updated_at) VALUES(?,?,?,1,'{}','{}',300,2,1,'x',?,?)",
        ("gait-assist", "方案", "gait-assist", now, now),
    )
    connection.execute(
        "INSERT INTO pilot_sessions(protocol_id,project_code,requested_by,parameters_json,parameter_digest,priority,"
        "idempotency_key,status,attempt_count,max_attempts,available_at,lease_owner,lease_expires_at,"
        "last_error_code,last_error_message,version,started_at,created_at,updated_at) "
        "VALUES(1,'p','u','{}','d',50,'legacy-1','cancel_requested',1,2,?,'site-a',?,'','',1,?,?,?)",
        (now, now, now, now, now),
    )
    connection.execute(
        "INSERT INTO pilot_interventions(session_id,actor,action,reason,before_json,after_json,batch_key,created_at) "
        "VALUES(1,'safety-officer','cancel','旧版遗留停止请求','{}','{}','',?)",
        (now,),
    )
    connection.commit()
    connection.close()

    init_db()  # 启动迁移：补列并把租约已过期的停止请求收敛为超时取消

    service = PilotOperationsService(get_connection())
    details = service.get_session(1)
    assert details["status"] == "cancelled"
    assert details["termination_kind"] == "cancelled_timeout"
    assert details["cancel_requested_by"] == "safety-officer"
    assert details["cancel_reason"] == "旧版遗留停止请求"
    assert details["cancellation"]["path"] == "timeout"
    assert details["interventions"][-1]["action"] == "cancel_timeout_recovery"
    # 再启动一次不应产生重复干预或改动终态。
    intervention_count = len(details["interventions"])
    init_db()
    assert len(service.get_session(1)["interventions"]) == intervention_count
    assert service.get_session(1)["termination_kind"] == "cancelled_timeout"


