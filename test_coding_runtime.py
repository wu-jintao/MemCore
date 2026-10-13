import unittest

from competition import verify_coding_runtime as runtime


class CodingRuntimeTests(unittest.TestCase):
    def test_task_payload_has_unique_engineering_signals(self):
        value = runtime.task_payload(23)
        text = value["messages"][0]["content"]
        self.assertIn("worker_23.py", text)
        self.assertIn("Worker23.run", text)
        self.assertIn("TaskError23", text)
        self.assertIn("tests/test_worker_23.py::test_retry_23", text)

    def test_free_port_is_loopback_bindable(self):
        port = runtime.free_port()
        self.assertGreater(port, 0)
        self.assertLess(port, 65536)


if __name__ == "__main__":
    unittest.main(verbosity=2)
