import re
import sys
import string
import jieba
import chardet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

"""
这个脚本用于计算原文与抄袭文本之间的相似度。它通过计算TF-IDF（词频-逆文档频率）向量，并使用余弦相似度度量文本间的相似度。
功能包括：
1. 读取原文和抄袭文本；
2. 进行文本预处理（包括去标点、分词等）；
3. 使用TF-IDF计算文本相似度；
4. 将相似度结果保存到输出文件中。
"""

def read_file(file_path):
    """ 自动检测文件编码并读取文件内容 """
    try:
        # 以二进制模式读取文件的前10000字节
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)  # 读取文件前10000字节
            result = chardet.detect(raw_data)  # 使用chardet检测编码
            encoding = result['encoding']  # 获取检测到的编码

        # 使用检测到的编码重新打开文件并读取内容
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read().strip()
    except FileNotFoundError as e:
        print(f"文件未找到: {file_path}, 错误: {e}")
        sys.exit(1)
    except UnicodeDecodeError as e:
        print(f"文件编码错误: {file_path}, 错误: {e}")
        sys.exit(1)
    except OSError as e:
        print(f"操作系统错误: {file_path}, 错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"读取文件失败: {file_path}, 错误: {e}")
        sys.exit(1)

def preprocess_text(text):
    """ 文本预处理：去除标点、分词、转小写 """
    text = text.lower()  # 转小写
    text = re.sub(r'[^\w\s]', '', text)  # 去除所有非字母数字字符
    words = list(jieba.cut(text))  # 使用 jieba 进行中文分词
    return " ".join(words)  # 用空格连接成字符串

def calculate_tfidf_similarity(original_text, plagiarized_text):
    """ 计算 TF-IDF 余弦相似度 """
    if not original_text.strip() or not plagiarized_text.strip():
        return 0.0  # 避免空文本导致报错

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2)  # 使用 1-gram 和 2-gram，提高检测能力
    tfidf_matrix = vectorizer.fit_transform([original_text, plagiarized_text])
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return similarity

def main():
    """ 主函数：读取文件，计算相似度并输出结果 """
    if len(sys.argv) != 4:
        print("使用方式: python script.py 原文文件路径 抄袭版文件路径 输出文件路径")
        sys.exit(1)

    original_file = sys.argv[1]
    plagiarized_file = sys.argv[2]
    output_file = sys.argv[3]

    original_text = preprocess_text(read_file(original_file))
    plagiarized_text = preprocess_text(read_file(plagiarized_file))

    similarity = calculate_tfidf_similarity(original_text, plagiarized_text)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"{similarity:.2f}\n")  # 精确到小数点后两位

    print(f"计算完成，结果已保存到 {output_file}")

if __name__ == "__main__":
    main()
