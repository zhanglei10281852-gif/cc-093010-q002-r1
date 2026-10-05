from __future__ import annotations

from datetime import UTC, datetime

from app.pilots.service import PilotOperationsService
from app.core.clock import FrozenClock, to_storage
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


def _running_service(clock: FrozenClock, key: str, *, lease: int = 10):
    from app.database import init_db

    init_db()
    service = PilotOperationsService(get_connection(), clock)
    service.create_protocol(PROTOCOL, "administrator")
    submitted = service.submit(submit_payload(key))
    claimed = service.claim("site-a", ["gait-assist"], lease)
    assert claimed and claimed["id"] == submitted["id"]
    return service, submitted["id"]


def test_running_stop_blocks_renewal_and_receipts_then_site_confirms(client):
    clock = FrozenClock(datetime(2026, 10, 5, 1, 0, tzinfo=UTC))
    service, session_id = _running_service(clock, "stop-confirm-001")

    stopped = service.cancel(session_id, "safety-marshal-7", "老年观众要求立即停止")
    assert stopped["status"] == "cancel_requested"
    assert stopped["cancel_requested_by"] == "safety-marshal-7"
    assert stopped["cancel_reason"] == "老年观众要求立即停止"
    assert stopped["cancel_requested_at"] == stopped["updated_at"]
    assert stopped["finished_at"] is None
    assert stopped["cancel_closure_path"] == ""

    import pytest
    from app.core.errors import ConflictError

    with pytest.raises(ConflictError):
        service.heartbeat(session_id, "site-a", 10)
    with pytest.raises(ConflictError):
        service.complete(session_id, "site-a", {"value": 1}, {})
    with pytest.raises(ConflictError):
        service.fail(session_id, "site-a", "late", "迟到失败", True)

    # 其他站点不能代为确认。
    with pytest.raises(ConflictError):
        service.confirm_cancel(session_id, "site-b")

    clock.advance(seconds=1)
    confirmed = service.confirm_cancel(session_id, "site-a")
    assert confirmed["status"] == "cancelled"
    assert confirmed["lease_owner"] == ""
    assert confirmed["cancel_closure_path"] == "confirmed"
    assert confirmed["cancel_confirmed_at"] == confirmed["finished_at"]
    assert confirmed["cancel_requested_by"] == "safety-marshal-7"

    details = service.get_session(session_id)
    assert details["termination"] == {
        "kind": "confirmed",
        "label": "站点确认停止",
        "requested_by": "safety-marshal-7",
        "reason": "老年观众要求立即停止",
        "requested_at": details["cancel_requested_at"],
        "confirmed_at": details["cancel_confirmed_at"],
        "finished_at": details["finished_at"],
    }
    # 被拒绝的观察回执没有落库。
    assert details["observations"] == []
    assert [item["action"] for item in details["interventions"]] == ["cancel", "cancel_confirmed"]

    # 迟到完成 / 重复确认都不能改变最终决定。
    with pytest.raises(ConflictError):
        service.complete(session_id, "site-a", {"value": 2}, {})
    again = service.confirm_cancel(session_id, "site-a")
    assert again["status"] == "cancelled" and again["cancel_closure_path"] == "confirmed"
    details = service.get_session(session_id)
    assert [item["action"] for item in details["interventions"]] == ["cancel", "cancel_confirmed"]


def test_duplicate_stop_keeps_first_requester_and_decision(client):
    clock = FrozenClock(datetime(2026, 10, 5, 2, 0, tzinfo=UTC))
    service, session_id = _running_service(clock, "stop-dup-001")

    first = service.cancel(session_id, "marshal-a", "观众不适需要停止")
    clock.advance(seconds=30)
    second = service.cancel(session_id, "marshal-b", "另一名安全员重复提交")
    assert second["status"] == "cancel_requested"
    # 首位提出者、原因、时间和版本都不被重复停止覆盖。
    assert second["cancel_requested_by"] == "marshal-a"
    assert second["cancel_reason"] == "观众不适需要停止"
    assert second["cancel_requested_at"] == first["cancel_requested_at"]
    assert second["version"] == first["version"]
    details = service.get_session(session_id)
    assert [item["actor"] for item in details["interventions"] if item["action"] == "cancel"] == ["marshal-a"]

    confirmed = service.confirm_cancel(session_id, "site-a")
    assert confirmed["cancel_closure_path"] == "confirmed"
    # 终态后再次停止同样不产生新干预。
    service.cancel(session_id, "marshal-b", "终态后的重复停止")
    details = service.get_session(session_id)
    assert details["status"] == "cancelled"
    assert len([i for i in details["interventions"] if i["action"] == "cancel"]) == 1


def test_unconfirmed_stop_converges_to_timeout_cancel_on_lease_expiry(client):
    clock = FrozenClock(datetime(2026, 10, 5, 3, 0, tzinfo=UTC))
    service, session_id = _running_service(clock, "stop-timeout-001", lease=10)

    service.cancel(session_id, "marshal-a", "站点失联，观众要求停止")
    clock.advance(seconds=5)
    # 租约未到期：恢复不处理。
    assert service.recover_expired()["cancelled"] == []
    assert service.get_session(session_id)["status"] == "cancel_requested"

    clock.advance(seconds=6)
    result = service.recover_expired()
    assert result == {"recovered": [], "exhausted": [], "cancelled": [session_id]}
    details = service.get_session(session_id)
    assert details["status"] == "cancelled"
    assert details["cancel_closure_path"] == "timeout"
    assert details["finished_at"] == to_storage(clock.now())
    assert details["cancel_requested_by"] == "marshal-a"
    assert details["lease_owner"] == ""
    assert details["termination"]["kind"] == "timeout"
    assert details["termination"]["label"] == "超时停止"
    assert details["interventions"][-1]["action"] == "lease_recovery"

    # 恢复任务重跑：终态不被重新排队，也不重复记录干预。
    clock.advance(seconds=60)
    rerun = service.recover_expired()
    assert rerun == {"recovered": [], "exhausted": [], "cancelled": []}
    details = service.get_session(session_id)
    assert details["status"] == "cancelled"
    assert len([i for i in details["interventions"] if i["action"] == "lease_recovery"]) == 1

    # 迟到的站点确认不能把超时停止改写成确认停止。
    import pytest
    from app.core.errors import ConflictError

    with pytest.raises(ConflictError):
        service.confirm_cancel(session_id, "site-a")
    assert service.get_session(session_id)["cancel_closure_path"] == "timeout"


def test_queued_cancel_is_manual_and_termination_views_distinguish_paths(client):
    from app.database import init_db

    clock = FrozenClock(datetime(2026, 10, 5, 4, 0, tzinfo=UTC))
    init_db()
    service = PilotOperationsService(get_connection(), clock)
    service.create_protocol(PROTOCOL, "administrator")

    queued_id = service.submit(submit_payload("manual-cancel-001"))["id"]
    manual = service.cancel(queued_id, "operator", "项目临时取消")
    assert manual["status"] == "cancelled"
    assert manual["cancel_closure_path"] == "manual"
    assert manual["finished_at"] == manual["updated_at"]
    details = service.get_session(queued_id)
    assert details["termination"]["kind"] == "manual"
    assert details["termination"]["label"] == "主动停止"

    completed_id = service.submit(submit_payload("complete-view-001"))["id"]
    service.claim("site-a", ["gait-assist"], 10)
    service.complete(completed_id, "site-a", {"ok": True}, {})
    completed_view = service.get_session(completed_id)["termination"]
    assert completed_view["kind"] == "completed"
    assert completed_view["label"] == "正常完成"


def test_timeout_cancelled_session_can_retry_into_clean_lifecycle(client):
    clock = FrozenClock(datetime(2026, 10, 5, 5, 0, tzinfo=UTC))
    service, session_id = _running_service(clock, "stop-retry-001", lease=10)
    service.cancel(session_id, "marshal-a", "先停止")
    clock.advance(seconds=11)
    service.recover_expired()

    retried = service.retry(session_id, "operator", "排除故障后重试")
    assert retried["status"] == "queued"
    assert retried["cancel_closure_path"] == ""
    assert retried["cancel_requested_by"] == ""
    assert retried["finished_at"] is None

    clock.advance(seconds=1)
    service.claim("site-a", ["gait-assist"], 10)
    service.complete(session_id, "site-a", {"ok": True}, {})
    details = service.get_session(session_id)
    assert details["status"] == "succeeded"
    assert details["termination"]["kind"] == "completed"


def test_legacy_cancel_requested_converged_on_restart(tmp_path):
    import json as _json
    import os
    from app.database import close_connection, get_connection as _get, init_db

    db_path = tmp_path / "legacy.db"
    os.environ["HEALTH_INNOVATION_DATABASE_PATH"] = str(db_path)
    close_connection()

    # 构造旧版（user_version=2）数据库：试点表尚无取消链路字段，且留着一条卡死的 cancel_requested。
    connection = _get()
    connection.executescript(
        """
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
            created_by TEXT, created_at TEXT, UNIQUE(session_id, version)
        );
        CREATE TABLE pilot_interventions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER, actor TEXT, action TEXT,
            reason TEXT, before_json TEXT, after_json TEXT, batch_key TEXT, created_at TEXT
        );
        """
    )
    stamp = "2026-10-04T08:00:00+00:00"
    connection.execute(
        "INSERT INTO pilot_protocols(code,name,capability,version,parameter_schema_json,default_parameters_json,"
        "max_runtime_seconds,max_attempts,active,created_by,created_at,updated_at) VALUES(?,?,?,1,?,?,300,2,1,?,?,?)",
        (
            PROTOCOL["code"], PROTOCOL["name"], PROTOCOL["capability"],
            _json.dumps(PROTOCOL["parameter_schema"], ensure_ascii=False),
            _json.dumps(PROTOCOL["default_parameters"], ensure_ascii=False),
            "administrator", stamp, stamp,
        ),
    )
    connection.execute(
        "INSERT INTO pilot_sessions(protocol_id,project_code,requested_by,parameters_json,parameter_digest,priority,"
        "idempotency_key,status,attempt_count,max_attempts,available_at,lease_owner,lease_expires_at,last_error_code,"
        "last_error_message,version,started_at,created_at,updated_at) "
        "VALUES(1,'expo-health-a','pilot-operator-1','{}','d',50,'legacy-stuck-001','cancel_requested',1,2,?,'site-a',?,'','',1,?,?,?)",
        (stamp, stamp, stamp, stamp, stamp),
    )
    connection.execute(
        "INSERT INTO pilot_interventions(session_id,actor,action,reason,before_json,after_json,batch_key,created_at) "
        "VALUES(1,'marshal-a','cancel','前一天的停止请求','{}','{}','',?)",
        (stamp,),
    )
    connection.execute("PRAGMA user_version=2")
    close_connection()

    # 模拟服务重启：新进程初始化并迁移数据库。
    init_db()
    service = PilotOperationsService(_get(), FrozenClock(datetime(2026, 10, 5, 8, 0, tzinfo=UTC)))
    details = service.get_session(1)
    assert details["status"] == "cancelled"
    assert details["cancel_closure_path"] == "timeout"
    assert details["cancel_requested_by"] == "marshal-a"
    assert details["cancel_reason"] == "前一天的停止请求"
    assert details["finished_at"] is not None
    assert details["interventions"][-1]["action"] == "lease_recovery"

    # 再次重启不会重复收敛或重复记录干预。
    close_connection()
    init_db()
    service = PilotOperationsService(_get())
    details = service.get_session(1)
    assert len([i for i in details["interventions"] if i["action"] == "lease_recovery"]) == 1
    assert _get().execute("PRAGMA user_version").fetchone()[0] == 3
    close_connection()


