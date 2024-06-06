def remove_consecutive_spaces(file_path):
    # 读取文件内容并逐行替换连续空格
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 创建一个新文件来存储结果
    with open('cleaned_' + file_path, 'w', encoding='utf-8') as file:
        for line in lines:
            # 替换当前行中的连续空格为单个空格
            cleaned_line = ' '.join(line.split())
            # 写入清理后的行，保留换行符
            file.write(cleaned_line + '\n')

# 调用函数，传入文件路径
remove_consecutive_spaces('ans.txt')