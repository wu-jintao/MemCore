import unittest

from competition import eval_coding_dense_proxy as dense


class CodingDenseProxyTests(unittest.TestCase):
    def test_fuse_respects_channel_weights_and_ties(self):
        lexical = [("a", 1.0), ("b", .5)]
        code = [("b", 9.0), ("a", 1.0)]
        self.assertEqual(dense.fuse(((1.0, lexical), (0.0, code)))[0][0], "a")
        self.assertEqual(dense.fuse(((1.0, lexical), (4.0, code)))[0][0], "b")

    def test_selected_ids_stops_at_whole_evidence_boundary(self):
        class Row(dict):
            pass
        documents = {
            "a": Row(id="a", role="user", session_id="s", request_id="r",
                     message_index=0, sequence=1, source_timestamp_ms=None,
                     content="x" * 70000),
            "b": Row(id="b", role="user", session_id="s", request_id="r",
                     message_index=1, sequence=2, source_timestamp_ms=None,
                     content="y" * 70000),
        }
        ids, _ = dense.selected_ids([("a", 1), ("b", .5)], documents)
        self.assertEqual(ids, ["a"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
