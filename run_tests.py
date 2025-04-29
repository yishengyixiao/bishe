#!/usr/bin/env python
import unittest
import os
import sys
import logging
import argparse
from datetime import datetime

# 配置日志
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f"test_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("test_runner")

def run_all_tests():
    """运行所有测试用例"""
    logger.info("开始运行所有测试用例")
    
    # 发现所有测试用例
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(".", pattern="test_*.py")
    
    # 运行测试
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # 记录测试结果
    logger.info(f"测试完成 - 运行: {result.testsRun}, 成功: {result.testsRun - len(result.failures) - len(result.errors)}, 失败: {len(result.failures)}, 错误: {len(result.errors)}")
    
    # 返回成功/失败状态 (0表示成功)
    return 0 if result.wasSuccessful() else 1

def run_specific_test(test_name):
    """运行特定的测试模块"""
    logger.info(f"开始运行特定测试: {test_name}")
    
    # 检查文件是否存在
    if not os.path.exists(test_name) and not os.path.exists(f"{test_name}.py"):
        logger.error(f"测试文件不存在: {test_name}")
        return 2
    
    # 移除.py扩展名（如果有）
    if test_name.endswith('.py'):
        test_name = test_name[:-3]
    
    try:
        # 加载指定的测试模块
        test_module = __import__(test_name)
        
        # 运行测试
        test_loader = unittest.TestLoader()
        test_suite = test_loader.loadTestsFromModule(test_module)
        
        test_runner = unittest.TextTestRunner(verbosity=2)
        result = test_runner.run(test_suite)
        
        # 记录测试结果
        logger.info(f"测试完成 - 运行: {result.testsRun}, 成功: {result.testsRun - len(result.failures) - len(result.errors)}, 失败: {len(result.failures)}, 错误: {len(result.errors)}")
        
        # 返回成功/失败状态
        return 0 if result.wasSuccessful() else 1
    
    except ImportError as e:
        logger.error(f"无法导入测试模块 {test_name}: {str(e)}")
        return 2
    except Exception as e:
        logger.error(f"运行测试时出错: {str(e)}")
        return 2

def run_test_by_name(class_name=None, test_name=None):
    """运行特定的测试类或测试方法"""
    logger.info(f"开始运行特定测试: 类={class_name}, 方法={test_name}")
    
    try:
        # 发现所有测试
        test_loader = unittest.TestLoader()
        
        if class_name and test_name:
            # 运行特定类的特定测试方法
            test_suite = unittest.TestSuite()
            
            # 查找匹配的测试类
            all_tests = test_loader.discover(".", pattern="test_*.py")
            found = False
            
            for suite in all_tests:
                for test_case in suite:
                    if isinstance(test_case, unittest.TestSuite):
                        for test in test_case:
                            if test.__class__.__name__ == class_name:
                                try:
                                    test_method = getattr(test.__class__, test_name)
                                    test_suite.addTest(test.__class__(test_name))
                                    found = True
                                except AttributeError:
                                    pass
            
            if not found:
                logger.error(f"未找到测试类 {class_name} 的测试方法 {test_name}")
                return 2
            
        elif class_name:
            # 运行特定测试类的所有方法
            found = False
            test_suite = unittest.TestSuite()
            
            all_tests = test_loader.discover(".", pattern="test_*.py")
            for suite in all_tests:
                for test_case in suite:
                    if isinstance(test_case, unittest.TestSuite):
                        for test in test_case:
                            if test.__class__.__name__ == class_name:
                                test_suite.addTest(test_loader.loadTestsFromTestCase(test.__class__))
                                found = True
                                break
            
            if not found:
                logger.error(f"未找到测试类 {class_name}")
                return 2
        else:
            logger.error("未指定测试类或方法")
            return 2
        
        # 运行测试
        test_runner = unittest.TextTestRunner(verbosity=2)
        result = test_runner.run(test_suite)
        
        # 记录测试结果
        logger.info(f"测试完成 - 运行: {result.testsRun}, 成功: {result.testsRun - len(result.failures) - len(result.errors)}, 失败: {len(result.failures)}, 错误: {len(result.errors)}")
        
        # 返回成功/失败状态
        return 0 if result.wasSuccessful() else 1
    
    except Exception as e:
        logger.error(f"运行测试时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 2

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="智能问答验证系统测试运行工具")
    
    # 添加命令行参数
    parser.add_argument("-a", "--all", action="store_true", help="运行所有测试")
    parser.add_argument("-m", "--module", help="运行特定测试模块(例如: test_verification)")
    parser.add_argument("-c", "--class", dest="class_name", help="运行特定测试类(例如: TestVerification)")
    parser.add_argument("-t", "--test", help="运行特定测试方法(需要与-c一起使用)")
    parser.add_argument("-v", "--verbose", action="store_true", help="输出详细日志")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 根据参数执行不同的测试
    if args.all:
        return run_all_tests()
    elif args.module:
        return run_specific_test(args.module)
    elif args.class_name:
        return run_test_by_name(args.class_name, args.test)
    else:
        parser.print_help()
        return 0

if __name__ == "__main__":
    # 执行主函数并返回退出码
    sys.exit(main()) 