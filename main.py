import sys
import jieba
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def read_file(file_path):
    """ 读取文件内容 """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"读取文件失败: {file_path}, 错误: {e}")
        sys.exit(1)

def preprocess_text(text):
    """ 文本预处理：去除标点、分词、转小写 """
    text = text.lower()  # 转小写
    text = text.translate(str.maketrans("", "", string.punctuation))  # 去标点
    words = list(jieba.cut(text))  # 使用 jieba 进行中文分词
    return " ".join(words)  # 用空格连接成字符串

def calculate_tfidf_similarity(original_text, plagiarized_text):
    """ 计算 TF-IDF 余弦相似度 """
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2)  # 使用 1-gram 和 2-gram，提高检测能力
    tfidf_matrix = vectorizer.fit_transform([original_text, plagiarized_text])
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return similarity

def main():
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
