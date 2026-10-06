import json
import unittest
import eval_contracts as contract


class ContractTests(unittest.TestCase):
    def test_answer_whitelist_and_structure(self):
        query = {"query": "Which park?", "question_date": "DATE_SENTINEL",
                 "answer": "GOLD_SENTINEL", "category": "CATEGORY_SENTINEL",
                 "gold_units": ["EVIDENCE_SENTINEL"]}
        messages = contract.build_answer_messages("longmemeval_s", query, "SOURCE_SENTINEL")
        self.assertEqual([m["role"] for m in messages], ["user"])
        text = json.dumps(messages)
        self.assertIn("SOURCE_SENTINEL", text)
        for forbidden in ("DATE_SENTINEL", "GOLD_SENTINEL", "CATEGORY_SENTINEL", "EVIDENCE_SENTINEL"):
            self.assertNotIn(forbidden, text)

    def test_judge_keeps_reference_and_omits_question_date(self):
        text = json.dumps(contract.build_judge_messages("locomo_refined",
                          {"query": "Why?", "question_date": "DATE_SENTINEL"}, "candidate", "reference"))
        self.assertIn("reference", text)
        self.assertNotIn("DATE_SENTINEL", text)

    def test_invalid_or_unsupported_family_fails_closed(self):
        for family in ("beam", "personamem", "scriptmem", "clbench", "streaming"):
            with self.assertRaises(ValueError):
                contract.profile(family)
        with self.assertRaises(ValueError):
            contract.build_answer_messages("longmemeval_s", {"query": "Q", "options": ["A"]}, "")

    def test_label_parser(self):
        self.assertTrue(contract.parse_judgment('{"label":"CORRECT"}')["correct"])
        self.assertFalse(contract.parse_judgment('{"label":"WRONG"}')["correct"])
        for value in ('{"label":"correct"}', '{"label":"CORRECT","extra":1}',
                      '{"correct":true}', 'prefix {"label":"CORRECT"}', '[]', '{"label":null}'):
            with self.assertRaises(ValueError):
                contract.parse_judgment(value)


if __name__ == "__main__":
    unittest.main()
