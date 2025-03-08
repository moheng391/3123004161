import unittest
from main import preprocess_text, calculate_tfidf_similarity, read_file  # 确保 text_processing.py 存在

class TestTextProcessing(unittest.TestCase):
    def test_preprocess_text(self):
        text = "这是一个测试文本。"
        processed_text = preprocess_text(text)
        self.assertEqual(processed_text, "这是 一个 测试 文本")

    def test_calculate_tfidf_similarity(self):
        text1 = "今天天气很好"
        text2 = "今天天气非常好"
        similarity = calculate_tfidf_similarity(preprocess_text(text1), preprocess_text(text2))
        self.assertGreater(similarity, 0.1)  # 避免 min_df 影响

class TestFileHandling(unittest.TestCase):
    def test_read_file(self):
        with open("test.txt", "w", encoding="utf-8") as f:
            f.write("测试内容")
        file_content = read_file("test.txt")
        self.assertIn("测试内容", file_content)

    def test_file_not_found(self):
        with self.assertRaises(SystemExit):
            read_file("non_existing_file.txt")

if __name__ == "__main__":
    unittest.main()
