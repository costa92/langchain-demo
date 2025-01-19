import unittest
from search_tool import get_UID

class TestSearchTool(unittest.TestCase):
    def test_get_UID_with_valid_flower(self):
        """测试 get_UID 函数对有效花名的处理"""
        flower = "牡丹"
        result = get_UID(flower)
        self.assertIsInstance(result, str)  # 确认返回类型为字符串  
    def test_get_UID_with_invalid_flower(self):
        """测试 get_UID 函数对无效花名的处理"""
        flower = ""
        with self.assertRaises(ValueError):
            get_UID(flower)

if __name__ == '__main__':
    unittest.main()