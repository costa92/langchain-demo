import unittest
from weibo_agent import lookup_V  # 确保导入 lookup_V 函数

class TestWeiboAgent(unittest.TestCase):
    def test_lookup_V_with_valid_flower(self):
        """测试 lookup_V 函数对有效花名的处理"""
        flower = "牡丹"
        result = lookup_V(flower)
        self.assertIsInstance(result, str)  # 确认返回类型为字符串
        print(result)

if __name__ == '__main__':
    unittest.main()