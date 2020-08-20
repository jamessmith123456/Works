import sys
from preprocess_data import Question

que = Question()
# 处理问题的方法
def dealquestion(question):
    # 查询知识图谱
    answer=que.question_process(question)
    # answer=12
    return answer


if __name__ == '__main__':
    question = input("请输入问题:")
    print("接收到的问题:", question)
    print("开始寻找答案!")

    answer = dealquestion(question)

    print("得到的答案是：", answer)
    if len(str(answer).strip()) == 0:
        answer = "我也还不知道呢！"
    print("答案已返回!")