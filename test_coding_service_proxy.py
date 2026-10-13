import hashlib
import tempfile
from pathlib import Path
import unittest

import numpy as np

from competition import eval_coding_service_proxy as service


class CodingServiceProxyTests(unittest.TestCase):
    def test_expected_message_ids_are_deterministic(self):
        first = service.expected_message_ids("target", ["a", "b"])
        second = service.expected_message_ids("target", ["a", "b"])
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)

    def test_cached_backend_rejects_unplanned_text(self):
        backend = object.__new__(service.CachedCodingBackend)
        backend.documents = {hashlib.sha256(b"known").hexdigest(): [(1.0,) + (0.0,) * 2047]}
        backend.queries = {}
        backend.document_calls = 0
        backend.document_inputs = 0
        backend.query_calls = 0
        backend.query_inputs = 0
        self.assertEqual(len(backend.embed_documents(["known"])[0][0]), 2048)
        with self.assertRaisesRegex(ValueError, "Unplanned"):
            backend.embed_documents(["unknown"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
