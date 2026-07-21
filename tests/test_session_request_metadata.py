import asyncio
import unittest

from app.session_request_metadata import reported_hardware_id


class FakeStreamingRequest:
    def __init__(self, chunks, *, headers=None, delay_s=0.0):
        self.chunks = list(chunks)
        self.headers = headers or {"content-type": "application/json"}
        self.delay_s = delay_s
        self.stream_started = False

    async def stream(self):
        self.stream_started = True
        for chunk in self.chunks:
            if self.delay_s:
                await asyncio.sleep(self.delay_s)
            yield chunk


class SessionRequestMetadataTests(unittest.IsolatedAsyncioTestCase):
    async def test_reads_valid_hardware_id_from_fragmented_json(self):
        request = FakeStreamingRequest(
            [b'{"hardware_id": "ABCDEF', b'0123456789"}'],
            headers={"content-type": "application/json; charset=utf-8"},
        )

        self.assertEqual(await reported_hardware_id(request), "abcdef0123456789")

    async def test_rejects_declared_oversized_body_without_streaming_it(self):
        request = FakeStreamingRequest(
            [b'{}'],
            headers={
                "content-type": "application/json",
                "content-length": "4097",
            },
        )

        self.assertIsNone(await reported_hardware_id(request, max_body_bytes=4096))
        self.assertFalse(request.stream_started)

    async def test_stops_after_chunked_body_exceeds_limit(self):
        request = FakeStreamingRequest([b'{"hardware_id":"', b"a" * 4096])

        self.assertIsNone(await reported_hardware_id(request, max_body_bytes=64))

    async def test_ignores_slow_body_after_timeout(self):
        request = FakeStreamingRequest([b'{}'], delay_s=0.05)

        self.assertIsNone(await reported_hardware_id(request, read_timeout_s=0.001))

    async def test_ignores_deep_json_that_exceeds_decoder_recursion(self):
        body = b"[" * 1100 + b"0" + b"]" * 1100
        request = FakeStreamingRequest([body])

        self.assertIsNone(await reported_hardware_id(request))

    async def test_ignores_malformed_json_and_invalid_utf8(self):
        malformed = FakeStreamingRequest([b'{"hardware_id":'])
        invalid_utf8 = FakeStreamingRequest([b"\xff"])

        self.assertIsNone(await reported_hardware_id(malformed))
        self.assertIsNone(await reported_hardware_id(invalid_utf8))


if __name__ == "__main__":
    unittest.main()
