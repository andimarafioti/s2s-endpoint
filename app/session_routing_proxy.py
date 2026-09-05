"""Keep an allocator reservation around a private upstream session update."""

import asyncio
import json
import secrets

import httpx

from app.llm_proxy_usage import LLM_PROXY_CALLBACK_AUTH_HEADER

ROUTING_FIELD = "_session_routing"


class SessionRoutingProxy:
    def __init__(self, websocket, grant, callback_key, route_counts, *, client=None):
        if not callback_key:
            raise ValueError("session model updates require LB_CALLBACK_AUTH_TOKEN")
        self.websocket = websocket
        self.grant = grant
        self.pipeline = grant["routing"]["pipeline"]
        self.route_counts = route_counts
        self.url = grant["callback_url"].removesuffix("/event") + "/routing"
        self.client = client or httpx.AsyncClient(timeout=10)
        self.owns_client = client is None
        self.headers = {LLM_PROXY_CALLBACK_AUTH_HEADER: f"Bearer {callback_key}"}
        self.pending = None

    async def close(self):
        if self.owns_client:
            await self.client.aclose()

    def _move_count(self, selected):
        self.route_counts[self.pipeline] -= 1
        if not self.route_counts[self.pipeline]:
            self.route_counts.pop(self.pipeline)
        self.route_counts[selected] = self.route_counts.get(selected, 0) + 1
        self.pipeline = selected

    async def _request(self, action, update_id, **payload):
        response = await self.client.post(
            self.url,
            headers=self.headers,
            json={"session_token": self.grant["session_token"], "action": action, "update_id": update_id, **payload},
        )
        response.raise_for_status()
        return response.json()

    async def _error(self, event_id, message):
        await self.websocket.send_text(
            json.dumps(
                {
                    "type": "error",
                    "event_id": secrets.token_hex(12),
                    "error": {
                        "type": "invalid_request_error",
                        "message": message,
                        **({"event_id": event_id} if isinstance(event_id, str) else {}),
                    },
                }
            )
        )

    async def client_message(self, message):
        try:
            raw = json.loads(message)
        except (ValueError, UnicodeDecodeError):
            return message
        if not isinstance(raw, dict):
            return message
        # Only compute can supply this field to the private upstream listener.
        raw.pop(ROUTING_FIELD, None)
        session = raw.get("session")
        if (
            raw.get("type") != "session.update"
            or not isinstance(session, dict)
            or not ({"model", "models"} & session.keys())
        ):
            return json.dumps(raw)
        models = session.get("models", {})
        if not isinstance(models, dict):
            await self._error(raw.get("event_id"), "models must be a map of stage selections")
            return None
        models = dict(models)
        if "model" in session:
            llm = models.get("llm")
            if "llm" in models and (llm.get("model") if isinstance(llm, dict) else llm) != session["model"]:
                await self._error(raw.get("event_id"), "model and models.llm disagree")
                return None
            models.setdefault("llm", session["model"])
        update_id = secrets.token_hex(16)
        try:
            prepared = await self._request("prepare", update_id, models=models)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code not in (400, 403, 404, 409):
                raise
            await self._error(
                raw.get("event_id"), "The selected models are unavailable or incompatible; retry the update."
            )
            return None
        # Transport failures close the socket; disconnect releases a prepared
        # hold even when the allocator's prepare response was lost.
        self._move_count(prepared["hold"])
        self.pending = (update_id, asyncio.get_running_loop().create_future())
        raw[ROUTING_FIELD] = {"update_id": update_id, "routing": {"id": self.grant["sid"], **prepared["routing"]}}
        return json.dumps(raw)

    async def after_send(self):
        if self.pending is not None:
            _, done = self.pending
            await asyncio.wait_for(asyncio.shield(done), timeout=20)
            self.pending = None

    async def server_message(self, message):
        try:
            raw = json.loads(message)
        except (ValueError, UnicodeDecodeError):
            return message
        if not isinstance(raw, dict) or ROUTING_FIELD not in raw:
            return message
        update_id = raw.pop(ROUTING_FIELD)
        if self.pending is None or self.pending[0] != update_id or raw.get("type") not in {"session.updated", "error"}:
            raise ValueError("unexpected upstream routing acknowledgement")
        finished = await self._request("commit" if raw["type"] == "session.updated" else "abort", update_id)
        self._move_count(finished["pipeline"])
        self.pending[1].set_result(None)
        return json.dumps(raw)
