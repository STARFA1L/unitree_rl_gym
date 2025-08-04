class Debugger:
    def dprint(self,file_name, func_name, *args, **kwargs):
        """
        调试打印函数，格式化输出调试信息
        
        参数:
        file_name -- 文件名
        func_name -- 函数名
        *args -- 要打印的内容（任意数量参数）
        **kwargs -- 打印选项（如 sep, end 等）
        """
        # 创建标题行：6个等号 + 文件名:函数名 + 6个等号
        header = f"{'='*6}{file_name}:{func_name}{'='*6}"
        
        # 打印标题行
        print(header)
        
        # 打印内容（使用原始 print 的行为）
        print(*args, **kwargs)
        
        # 打印与标题行等长的等号分隔线
        print('=' * len(header))