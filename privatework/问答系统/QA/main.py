import sys
from similarity import get_alld, get_similraity_index
from query import Query
if __name__ == '__main__':
    all_d = get_alld("question_best.csv")
    queryer = Query()
    question = input("请输入问题:")
    #内平开下悬窗的窗型有什么选择？
    #question = "内平开下悬窗有什么窗型可以选择？"
    #question = "上悬窗有什么优点？"
    print("接收到的问题:", question)
    print("开始寻找答案!")
    target_question, cos_target_question = get_similraity_index(all_d, question)
    print("tfidf对比得到的问题:", target_question)
    print("余弦对比得到的问题:", cos_target_question)
    answer = queryer.run(target_question)
    print("得到的答案是：", answer)
    if len(str(answer).strip()) == 0:
        answer = "我也还不知道呢！"
    print("答案已返回!")